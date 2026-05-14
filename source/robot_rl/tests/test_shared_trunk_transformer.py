"""Unit tests for the shared-trunk causal-transformer actor/critic.

Covers identity sharing, optimizer parameter dedup, the encoder activation
cache, gradient flow through the trunk, and checkpoint round-tripping.

No Isaac sim needed — the rsl_rl model classes work on plain TensorDict obs.
"""

from __future__ import annotations

import copy

import pytest
import torch
from tensordict import TensorDict

from robot_rl.network.shared_trunk_transformer import (
    SharedTemporalEncoder,
    SharedTrunkCausalTransformerModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_BATCH = 4
_SINGLE_OBS_DIM = 32
_HISTORY = 6
_D_MODEL = 16
_NUM_ACTIONS = 5

_DISTRIBUTION_CFG = {
    "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
    "init_std": 1.0,
    "std_type": "log",
}


def _make_obs() -> TensorDict:
    return TensorDict(
        {"policy": torch.randn(_BATCH, _HISTORY * _SINGLE_OBS_DIM)},
        batch_size=[_BATCH],
    )


def _make_encoder() -> SharedTemporalEncoder:
    return SharedTemporalEncoder(
        single_obs_dim=_SINGLE_OBS_DIM,
        history_length=_HISTORY,
        d_model=_D_MODEL,
        nhead=2,
        num_layers=2,
        dim_feedforward=32,
    )


def _make_models(encoder: SharedTemporalEncoder) -> tuple[
    SharedTrunkCausalTransformerModel, SharedTrunkCausalTransformerModel
]:
    obs = _make_obs()
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = SharedTrunkCausalTransformerModel(
        obs, obs_groups, "actor", _NUM_ACTIONS,
        encoder=encoder, owns_encoder=True,
        single_obs_dim=_SINGLE_OBS_DIM,
        hidden_dims=(32, 16), activation="elu",
        distribution_cfg=copy.deepcopy(_DISTRIBUTION_CFG),
    )
    critic = SharedTrunkCausalTransformerModel(
        obs, obs_groups, "critic", 1,
        encoder=encoder, owns_encoder=False,
        single_obs_dim=_SINGLE_OBS_DIM,
        hidden_dims=(32, 16), activation="elu",
    )
    return actor, critic


# ---------------------------------------------------------------------------
# Identity & registration
# ---------------------------------------------------------------------------


def test_encoder_is_shared_python_object():
    enc = _make_encoder()
    actor, critic = _make_models(enc)
    assert actor._encoder_obj is critic._encoder_obj
    assert actor._encoder_obj is enc


def test_encoder_registered_as_actor_child_only():
    enc = _make_encoder()
    actor, critic = _make_models(enc)
    assert "encoder" in dict(actor.named_modules())
    assert "encoder" not in dict(critic.named_modules())


# ---------------------------------------------------------------------------
# Parameter dedup
# ---------------------------------------------------------------------------


def test_actor_and_critic_have_no_parameter_overlap():
    enc = _make_encoder()
    actor, critic = _make_models(enc)
    actor_ids = {id(p) for p in actor.parameters()}
    critic_ids = {id(p) for p in critic.parameters()}
    assert actor_ids.isdisjoint(critic_ids), (
        "Encoder params must not appear in both actor.parameters() and "
        "critic.parameters() — would cause Adam to update them twice per step."
    )


def test_encoder_params_appear_exactly_once_in_combined_parameters():
    enc = _make_encoder()
    actor, critic = _make_models(enc)
    from itertools import chain
    all_ids = [id(p) for p in chain(actor.parameters(), critic.parameters())]
    enc_ids = {id(p) for p in enc.parameters()}
    enc_appearances = sum(1 for pid in all_ids if pid in enc_ids)
    assert enc_appearances == len(enc_ids), (
        f"Each encoder param should appear exactly once across "
        f"chain(actor, critic).parameters(); got {enc_appearances} appearances "
        f"for {len(enc_ids)} encoder params."
    )


# ---------------------------------------------------------------------------
# Activation cache
# ---------------------------------------------------------------------------


def test_cache_hits_when_actor_and_critic_share_obs():
    enc = _make_encoder()
    actor, critic = _make_models(enc)
    obs = _make_obs()

    call_count = [0]
    orig_encode = enc._encode

    def counted_encode(x):
        call_count[0] += 1
        return orig_encode(x)

    enc._encode = counted_encode

    _ = actor(obs)
    _ = critic(obs)
    assert call_count[0] == 1, (
        f"Encoder should run exactly once when actor and critic share obs; "
        f"got {call_count[0]} calls."
    )


def test_cache_misses_on_different_obs():
    enc = _make_encoder()
    actor, critic = _make_models(enc)
    obs_a = _make_obs()
    obs_b = _make_obs()

    call_count = [0]
    orig_encode = enc._encode

    def counted_encode(x):
        call_count[0] += 1
        return orig_encode(x)

    enc._encode = counted_encode

    _ = actor(obs_a)
    _ = critic(obs_b)
    assert call_count[0] == 2, (
        f"Encoder should run twice with different obs tensors; "
        f"got {call_count[0]} calls."
    )


def test_clear_cache_drops_references():
    enc = _make_encoder()
    obs = _make_obs()
    _ = enc(obs["policy"])
    assert enc._cache_in is not None
    enc.clear_cache()
    assert enc._cache_in is None
    assert enc._cache_out is None


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow_through_shared_encoder():
    enc = _make_encoder()
    actor, critic = _make_models(enc)
    obs = _make_obs()

    out_a = actor(obs)
    out_c = critic(obs)
    loss = out_a.sum() + out_c.sum()
    loss.backward()

    encoder_param_grads = [p.grad for p in enc.parameters()]
    assert all(g is not None for g in encoder_param_grads), \
        "All encoder params must receive gradients."
    grad_total_norm = sum(g.norm().item() for g in encoder_param_grads)
    assert grad_total_norm > 0.0, "Encoder gradients should be non-zero."


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


def test_checkpoint_round_trip_via_actor_state_dict():
    """Encoder weights round-trip via actor.state_dict() since it's an actor child."""
    enc = _make_encoder()
    actor, critic = _make_models(enc)
    obs = _make_obs()

    # Capture pre-load outputs.
    enc.clear_cache()
    out_a_before = actor(obs).detach().clone()
    out_c_before = critic(obs).detach().clone()

    # Save actor state.
    saved = copy.deepcopy(actor.state_dict())

    # Build a fresh trio with random init.
    enc2 = _make_encoder()
    actor2, critic2 = _make_models(enc2)

    # Sanity: fresh encoder produces different output.
    enc2.clear_cache()
    out_a_fresh = actor2(obs).detach().clone()
    assert not torch.allclose(out_a_before, out_a_fresh, atol=1e-5), \
        "Sanity check: freshly initialized encoder should produce different output."

    # Load actor's state -- encoder weights are included because encoder is an actor child.
    actor2.load_state_dict(saved)
    enc2.clear_cache()
    out_a_after = actor2(obs).detach().clone()
    out_c_after = critic2(obs).detach().clone()

    # Actor reproduces.
    assert torch.allclose(out_a_before, out_a_after, atol=1e-5), \
        "After loading saved actor state, actor output should match the original."
    # Critic shares the encoder Python object, so it sees the restored weights too.
    # The critic's MLP head has fresh random weights — but the encoder slice of
    # the latent fed to the critic head is restored. So critic outputs WILL differ
    # from before (different head weights), but the encoder forward should be the
    # exact same as actor2's. Validate via encoder-only forward:
    z_orig = enc(obs["policy"])
    z_loaded = enc2(obs["policy"])
    assert torch.allclose(z_orig, z_loaded, atol=1e-5), \
        "Encoder weights should match after actor.load_state_dict."
    # Output difference for critic comes from the (fresh) critic head only.
    del out_c_before, out_c_after  # silence unused-var lint


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------


def test_forward_shapes():
    enc = _make_encoder()
    actor, critic = _make_models(enc)
    obs = _make_obs()
    out_a = actor(obs)
    out_c = critic(obs)
    assert out_a.shape == (_BATCH, _NUM_ACTIONS)
    assert out_c.shape == (_BATCH, 1)
