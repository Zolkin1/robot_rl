"""Shared causal-transformer trunk for PPO actor + critic.

Both actor and critic forward an obs-history through a single
:class:`SharedTemporalEncoder` (the expensive part — multi-layer self-attention
over a 25-step window). The encoder output ``z`` is concatenated with the
current observation and fed into a small MLP head per role:

    obs_history → SharedTemporalEncoder → z (d_model)
                                              ↓
    current_obs ─────────────────→ concat[z, current_obs] → Actor MLP
                                                          → Critic MLP

Sharing saves ~50% of transformer parameters (one trunk vs two) and ~50% of
transformer activations (via an identity-keyed forward cache so the encoder
only runs once per PPO minibatch even though actor and critic both ``forward``
on the same batch tensor).

Wiring:
  * :class:`SharedTrunkPPO` (algorithm subclass) constructs the encoder once
    and injects it into both actor and critic cfg dicts before delegating to
    ``PPO.construct_algorithm``. No global singleton.
  * The actor is told ``owns_encoder=True`` and registers the encoder as an
    ``nn.Module`` child, so it appears in ``actor.parameters()`` /
    ``actor.state_dict()`` exactly once (matches what PPO's optimizer +
    checkpointing expect).
  * The critic is told ``owns_encoder=False`` and stores the encoder via a
    non-registered Python attribute, so it doesn't double-count in
    ``critic.parameters()``. Gradients still flow correctly because both heads
    share the same Python encoder object and PPO sums both losses before a
    single backward.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models import MLPModel

from .transformer_network import _sinusoidal_positional_encoding


_ENCODER_KEYS: tuple[str, ...] = (
    "single_obs_dim",
    "history_length",
    "d_model",
    "nhead",
    "num_layers",
    "dim_feedforward",
    "dropout",
)


class SharedTemporalEncoder(nn.Module):
    """Causal transformer that encodes an obs-history to a single latent.

    Input:  ``[B, history_length * single_obs_dim]`` flat observation history.
    Output: ``[B, d_model]`` representation of the most recent token (post
    self-attention so it summarizes the full history).

    Caches the forward output by object identity. PPO calls
    ``actor.forward(obs)`` then ``critic.forward(obs)`` on the same Python
    tensor; the second call returns the cached latent so the transformer is
    only forwarded once per minibatch. The autograd graph stays alive across
    the gap because PPO sums both losses before any backward.
    """

    def __init__(
        self,
        single_obs_dim: int,
        history_length: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ):
        """Initialize the shared temporal encoder.

        Args:
            single_obs_dim: Dimension of one timestep's observation.
            history_length: Number of timesteps in the history.
            d_model: Transformer hidden dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dim_feedforward: Per-layer feedforward dimension.
            dropout: Dropout rate (0.0 is typical for RL).
        """
        super().__init__()
        self.single_obs_dim = single_obs_dim
        self.history_length = history_length
        self.d_model = d_model

        self.input_proj = nn.Linear(single_obs_dim, d_model)
        self.register_buffer(
            "pos_encoding",
            _sinusoidal_positional_encoding(history_length, d_model),
        )
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(history_length),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self._cache_in: torch.Tensor | None = None
        self._cache_out: torch.Tensor | None = None

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the un-cached forward pass."""
        batch_size = x.shape[0]
        x_seq = x.view(batch_size, self.history_length, self.single_obs_dim)
        x_proj = self.input_proj(x_seq) + self.pos_encoding
        out = self.transformer_encoder(
            x_proj, mask=self.causal_mask, is_causal=True
        )
        return out[:, -1, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent for ``x``; reuse cached output if ``x`` is identical.

        Identity check (``x is self._cache_in``) is robust because we hold a
        reference to the cached input. On a new minibatch the runner passes a
        fresh tensor → identity miss → we overwrite both refs, releasing the
        prior tensor for GC.
        """
        if x is self._cache_in and self._cache_out is not None:
            return self._cache_out
        out = self._encode(x)
        self._cache_in = x
        self._cache_out = out
        return out

    def clear_cache(self) -> None:
        """Drop the cached input/output references (e.g. between training runs)."""
        self._cache_in = None
        self._cache_out = None


class SharedTrunkCausalTransformerModel(MLPModel):
    """MLPModel variant that prepends a shared temporal encoder to the MLP head.

    Both actor and critic instantiate this class; the algorithm subclass
    (:class:`SharedTrunkPPO`) injects the SAME ``encoder`` instance into both
    via cfg kwargs. ``owns_encoder`` controls whether this instance registers
    the encoder as an ``nn.Module`` child (True for actor, False for critic).
    """

    def __init__(
        self,
        obs,
        obs_groups,
        role,
        num_outputs,
        *,
        encoder: SharedTemporalEncoder,
        owns_encoder: bool,
        single_obs_dim: int,
        hidden_dims=(256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg=None,
        **_unused,
    ):
        """Initialize a shared-trunk head model.

        Args:
            obs: Observation TensorDict (from rsl_rl).
            obs_groups: Mapping ``role -> [group_names]`` (from rsl_rl cfg).
            role: ``"actor"`` or ``"critic"``.
            num_outputs: Output dimension (num_actions for actor, 1 for critic).
            encoder: The shared encoder instance. The same Python object must
                be passed to both actor and critic.
            owns_encoder: When True, register ``encoder`` as an ``nn.Module``
                child of this model so its parameters live in this model's
                state_dict and parameter list. When False, hold a Python
                reference only (no registration) — used for the critic, since
                the actor already owns the encoder.
            single_obs_dim: Per-step observation dimension (used to slice the
                "current obs" out of the flat history).
            hidden_dims: MLP head hidden dimensions.
            activation: MLP head activation.
            obs_normalization: Forwarded to MLPModel.
            distribution_cfg: Forwarded to MLPModel (actor only).
            **_unused: Swallows any cfg keys not consumed here (e.g. the
                encoder hyperparams that already shaped the injected encoder).
        """
        # Cache the encoder reference BEFORE super().__init__() so that
        # _get_latent_dim() — called inside MLPModel.__init__ to size the
        # first MLP layer — can read encoder.d_model. Use object.__setattr__
        # to bypass nn.Module.__setattr__'s automatic child-registration;
        # owns_encoder controls that explicitly below.
        object.__setattr__(self, "_encoder_obj", encoder)
        self._single_obs_dim = single_obs_dim

        super().__init__(
            obs,
            obs_groups,
            role,
            num_outputs,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        if owns_encoder:
            # Register the encoder as a child module. This is what makes the
            # encoder visible to optimizer construction (chain(actor.parameters(),
            # critic.parameters())) and state_dict() exactly once.
            self.encoder = encoder

    def _get_latent_dim(self) -> int:
        """Size the MLP head input as ``encoder.d_model + single_obs_dim``."""
        return self._encoder_obj.d_model + self._single_obs_dim

    def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
        """Build the head input: ``concat[encoder(history), current_obs]``.

        When there's a single obs group, bypass ``MLPModel.get_latent``'s
        ``torch.cat`` (which always allocates a new tensor) and pass the
        group's tensor straight through. This preserves Python identity
        between ``actor.forward(obs)`` and ``critic.forward(obs)`` so the
        encoder's identity-keyed cache hits on the second call.
        ``nn.Identity`` (used when ``obs_normalization=False``) returns its
        input unchanged, so identity is preserved through the normalizer too.
        Falls back to the parent implementation for multi-group setups.
        """
        if len(self.obs_groups) == 1:
            flat = self.obs_normalizer(obs[self.obs_groups[0]])
        else:
            flat = super().get_latent(obs, masks, hidden_state)
        z = self._encoder_obj(flat)
        current = flat[:, -self._single_obs_dim:]
        return torch.cat([z, current], dim=-1)


class SharedTrunkPPO(PPO):
    """PPO subclass that constructs a shared encoder and injects it into both
    actor and critic before they're built.

    Reads the encoder hyperparameters from the actor cfg (validates the critic
    cfg matches), constructs one :class:`SharedTemporalEncoder`, places it
    plus ``owns_encoder`` (True for actor, False for critic) into the cfg
    dicts, then delegates to ``PPO.construct_algorithm`` which constructs the
    models and optimizer normally.
    """

    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        """Construct PPO with a single shared encoder instance."""
        actor_cfg = cfg["actor"]
        critic_cfg = cfg["critic"]

        # Validate and collect encoder hyperparameters.
        encoder_kwargs: dict = {}
        for key in _ENCODER_KEYS:
            a_val = actor_cfg.get(key)
            c_val = critic_cfg.get(key)
            if a_val is None and c_val is None:
                continue
            if a_val != c_val:
                raise ValueError(
                    f"SharedTrunkPPO: encoder cfg '{key}' must match between "
                    f"actor and critic — got actor={a_val!r}, critic={c_val!r}."
                )
            encoder_kwargs[key] = a_val

        for required in ("single_obs_dim", "history_length"):
            if required not in encoder_kwargs:
                raise ValueError(
                    f"SharedTrunkPPO: actor and critic cfgs must both specify "
                    f"'{required}' for the shared encoder."
                )

        encoder = SharedTemporalEncoder(**encoder_kwargs).to(device)

        # Inject the encoder reference and ownership flag into each model cfg.
        # The model class's **kwargs swallows the encoder hyperparams that
        # already shaped the injected encoder — they're harmless duplicates.
        actor_cfg["encoder"] = encoder
        actor_cfg["owns_encoder"] = True
        critic_cfg["encoder"] = encoder
        critic_cfg["owns_encoder"] = False

        return PPO.construct_algorithm(obs, env, cfg, device)
