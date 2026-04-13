##
## A Causal Transformer network architecture for RL policies.
##
## Each timestep's full observation is treated as a single token.
## A history of observations forms the sequence. Sinusoidal positional
## encoding provides temporal ordering. Causal masking ensures each
## token only attends to itself and past tokens.
##

import math

import torch
import torch.nn as nn

from rsl_rl.models import MLPModel


def _sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Generate fixed sinusoidal positional encoding.

    Standard formulation from "Attention is All You Need" (Vaswani et al., 2017):
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: Maximum sequence length.
        d_model: Embedding dimension.

    Returns:
        Positional encoding tensor of shape [1, seq_len, d_model].
    """
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, seq_len, d_model]


class CausalTransformer(nn.Module):
    """Causal transformer that processes a history of observations.

    The input is a flat vector of shape [batch, history_length * single_obs_dim],
    which is reshaped into a sequence of tokens [batch, history_length, single_obs_dim].
    Each token is projected to the model dimension, positional encoding is added,
    and the sequence is processed by a causal transformer encoder. The last token's
    output is projected to the desired output dimension.

    Args:
        single_obs_dim: Dimension of a single timestep's observation.
        history_length: Number of timesteps in the observation history.
        output_dim: Output dimension (e.g. num_actions or distribution input dim).
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Feedforward dimension in each transformer layer.
        dropout: Dropout rate (0.0 is typical for RL).
    """

    def __init__(
        self,
        single_obs_dim: int,
        history_length: int,
        output_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.single_obs_dim = single_obs_dim
        self.history_length = history_length
        self.d_model = d_model

        # Project each observation token to the model dimension
        self.input_proj = nn.Linear(single_obs_dim, d_model)

        # Fixed sinusoidal positional encoding
        self.register_buffer(
            "pos_encoding",
            _sinusoidal_positional_encoding(history_length, d_model),
        )

        # Causal mask: upper-triangular = -inf so each token only sees past + self
        self.register_buffer(
            "causal_mask",
            nn.Transformer.generate_square_subsequent_mask(history_length),
        )

        # Transformer encoder
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

        # Project last token's representation to output
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Flat observation history [batch, history_length * single_obs_dim].

        Returns:
            Output tensor [batch, output_dim] from the last token.
        """
        batch_size = x.shape[0]

        # Reshape flat input to sequence of tokens
        # [B, H*D] -> [B, H, D]
        x = x.view(batch_size, self.history_length, self.single_obs_dim)

        # Project each token to model dimension
        # [B, H, D] -> [B, H, d_model]
        x = self.input_proj(x)

        # Add positional encoding
        x = x + self.pos_encoding

        # Apply causal transformer
        # [B, H, d_model] -> [B, H, d_model]
        x = self.transformer_encoder(x, mask=self.causal_mask, is_causal=True)

        # Take the last token and project to output
        # [B, d_model] -> [B, output_dim]
        x = x[:, -1, :]
        x = self.output_proj(x)

        return x


class CausalTransformerModel(MLPModel):
    """Drop-in MLPModel replacement that uses a causal transformer internally.

    PPO instantiates this exactly like MLPModel:
        actor = CausalTransformerModel(obs, obs_groups, "actor", num_actions, **cfg)

    The observation history is expected to be flattened into the observation vector
    (via IsaacLab's history_length mechanism). The model reshapes it back into a
    sequence of tokens based on single_obs_dim and history_length.

    Extra kwargs (single_obs_dim, history_length, d_model, etc.) pass through
    because MLPModel accepts **kwargs and ignores unknowns.
    """

    def __init__(
        self,
        obs,
        obs_groups,
        role,
        num_outputs,
        hidden_dims=[256, 128],
        activation="elu",
        # Transformer-specific args
        single_obs_dim: int | None = None,
        history_length: int | None = None,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
        **kwargs,
    ):
        """Initialize the Causal Transformer model.

        Args:
            obs: Observation TensorDict.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            role: Role of this model ("actor" or "critic").
            num_outputs: Number of output dimensions.
            hidden_dims: Hidden dimensions (used by parent MLPModel, replaced by transformer).
            activation: Activation function (used by parent MLPModel).
            single_obs_dim: Dimension of a single timestep's observation. Required.
            history_length: Number of timesteps in the observation history. Required.
            d_model: Transformer hidden dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            dim_feedforward: Feedforward dimension in each transformer layer.
            dropout: Dropout rate.
        """
        if single_obs_dim is None or history_length is None:
            raise ValueError(
                "CausalTransformerModel requires both 'single_obs_dim' and 'history_length'. "
                "These must be set in the model config."
            )

        # Let MLPModel build normalization, distribution, and a temporary self.mlp
        super().__init__(
            obs, obs_groups, role, num_outputs,
            hidden_dims=hidden_dims,
            activation=activation,
            **kwargs,
        )

        # Grab dims from the MLP that MLPModel built
        input_dim = self.mlp[0].in_features
        # Find the output dim of the MLP (accounts for distribution input_dim)
        mlp_output_dim = None
        for module in reversed(list(self.mlp.modules())):
            if isinstance(module, nn.Linear):
                mlp_output_dim = module.out_features
                break

        assert single_obs_dim * history_length == input_dim, (
            f"single_obs_dim ({single_obs_dim}) * history_length ({history_length}) = "
            f"{single_obs_dim * history_length} != input_dim ({input_dim}). "
            f"Ensure all observation terms use the same history_length."
        )

        # Replace the plain MLP with the causal transformer
        self.mlp = CausalTransformer(
            single_obs_dim=single_obs_dim,
            history_length=history_length,
            output_dim=mlp_output_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )


# ─── Quick Test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    batch_size = 4096
    single_obs_dim = 74  # e.g. G1 policy observation
    history_length = 16
    act_dim = 21          # e.g. 21 joint targets

    transformer = CausalTransformer(
        single_obs_dim=single_obs_dim,
        history_length=history_length,
        output_dim=act_dim,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.0,
    )

    x = torch.randn(batch_size, history_length * single_obs_dim)
    y = transformer(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in transformer.parameters()):,}")

    # Verify causal masking: changing a future token should not affect past outputs
    x2 = x.clone()
    x2[:, -single_obs_dim:] = torch.randn(batch_size, single_obs_dim)  # change last token
    y2 = transformer(x2)
    # Outputs for the last token should differ, but this test checks the full output
    # (which only uses the last token anyway)
    print(f"Output differs after changing last token: {not torch.allclose(y, y2)}")
