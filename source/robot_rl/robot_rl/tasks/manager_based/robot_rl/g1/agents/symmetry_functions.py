"""Sagittal-plane data augmentation for RL training.

Provides two entry points consumed by ``RslRlSymmetryCfg``:

* :func:`symmetric_data_augmentation_episodic` — phase obs unchanged.
* :func:`symmetric_data_augmentation_half_periodic` — phase obs negated.

Internally both delegate to :func:`_symmetric_data_augmentation` which
uses :class:`SagittalReflectionConfig` for config-driven dispatch rather
than hard-coded if/elif branches.
"""

from __future__ import annotations

from typing import Tuple

import torch
import tensordict

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.sagittal_reflector import (
    NamedReflector,
    SagittalReflectionConfig,
)


# ---------------------------------------------------------------------------
# Module-level cache
# ---------------------------------------------------------------------------
# The joint reflector is built once (lazily) because the joint name list is
# only available at runtime from the environment's articulation.

_CACHED_JOINT_REFLECTOR: NamedReflector | None = None
_CACHED_JOINT_NAMES: list[str] | None = None
_REFLECTION_CFG = SagittalReflectionConfig()


def _get_joint_reflector(env) -> NamedReflector:
    """Return (and cache) a joint-level reflector for the robot."""
    global _CACHED_JOINT_REFLECTOR, _CACHED_JOINT_NAMES

    joint_names: list[str] = env.unwrapped.scene["robot"].joint_names
    if _CACHED_JOINT_REFLECTOR is None or joint_names != _CACHED_JOINT_NAMES:
        _CACHED_JOINT_NAMES = joint_names
        _CACHED_JOINT_REFLECTOR = NamedReflector(
            _REFLECTION_CFG, joint_names, device=env.unwrapped.device,
        )
    return _CACHED_JOINT_REFLECTOR


# ---------------------------------------------------------------------------
# Tiling helper (unchanged)
# ---------------------------------------------------------------------------

def _tile_multiplier(multiplier: torch.Tensor, obs_size: int) -> torch.Tensor:
    """Tile a per-timestep multiplier to cover the full (possibly history-stacked) obs slice.

    When history_length > 1, the obs term is flattened as [t0_feat, t1_feat, ..., tH_feat].
    This tiles the single-timestep multiplier H times so element-wise multiply works.

    Args:
        multiplier: 1D tensor for a single timestep (e.g. [-1, 1, -1] for 3-dim obs).
        obs_size: Total flattened size of this obs term (single_dim * history_length).

    Returns:
        Tiled multiplier of shape [obs_size].
    """
    single_dim = multiplier.shape[0]
    n_repeats = obs_size // single_dim
    return multiplier.repeat(n_repeats)


# ---------------------------------------------------------------------------
# Core augmentation (unified)
# ---------------------------------------------------------------------------

def _symmetric_data_augmentation(
    env,
    obs: tensordict.TensorDict,
    actions: torch.Tensor,
    is_half_periodic: bool,
) -> Tuple[tensordict.TensorDict, torch.Tensor]:
    """Augment observations and actions via sagittal-plane reflection.

    Args:
        env: Isaac Lab RL vec env.
        obs: TensorDict of shape ``[batch, num_obs]``.
        actions: Tensor of shape ``[batch, num_actions]``.
        is_half_periodic: When *True*, phase observations are negated
            (half-periodic gait symmetry).  When *False*, they are
            copied unchanged (episodic symmetry).

    Returns:
        Tuple of (augmented obs, augmented actions) each with batch
        size doubled.
    """
    cfg = _REFLECTION_CFG

    if obs is not None:
        device = obs.device
        batch_size = obs.batch_size[0]

        obs_aug = obs.repeat(2)

        cmd = env.unwrapped.command_manager.get_term("traj_ref")
        joint_reflector = _get_joint_reflector(env)

        # Original observations (first half already correct)
        obs_aug["policy"][:batch_size] = obs["policy"][:batch_size]

        # Iterate every obs group present in the env's observation manager.
        # PPO uses ``policy``/``critic``; distillation also uses ``student`` (and
        # sometimes ``unpriv_policy``). Skipping a group present in the obs but
        # missing from this loop would leave the student input un-mirrored.
        active_groups = list(env.unwrapped.observation_manager.active_terms.keys())
        for group in active_groups:
            if group not in obs_aug.keys():
                continue
            obs_idx = 0
            for i, name in enumerate(env.unwrapped.observation_manager.active_terms[group]):
                obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]

                if name in cfg.obs_term_multipliers:
                    multiplier = _tile_multiplier(
                        torch.tensor(cfg.obs_term_multipliers[name], device=device),
                        obs_size,
                    )
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        obs[group][:, obs_idx:obs_idx + obs_size] * multiplier
                    )

                elif name in cfg.joint_obs_terms:
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        joint_reflector.reflect_with_history(
                            obs[group][:, obs_idx:obs_idx + obs_size]
                        )
                    )

                elif name in cfg.pos_traj_obs_terms:
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        cmd.get_symmetric_traj(obs[group][:, obs_idx:obs_idx + obs_size], "pos")
                    )

                elif name in cfg.vel_traj_obs_terms:
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        cmd.get_symmetric_traj(obs[group][:, obs_idx:obs_idx + obs_size], "vel")
                    )

                elif name in cfg.contact_obs_terms:
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        cmd.get_symmetric_contacts(obs[group][:, obs_idx:obs_idx + obs_size])
                    )

                elif name in cfg.phase_obs_terms:
                    if is_half_periodic:
                        obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                            -1 * obs[group][:, obs_idx:obs_idx + obs_size]
                        )
                    else:
                        obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                            obs[group][:, obs_idx:obs_idx + obs_size]
                        )

                # TODO: Add height map support

                obs_idx += obs_size
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]
        joint_reflector = _get_joint_reflector(env)

        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = joint_reflector.reflect(actions)
    else:
        actions_aug = None

    return (obs_aug, actions_aug)


# ---------------------------------------------------------------------------
# Public entry points (signature unchanged for rsl_rl_ppo_cfg.py)
# ---------------------------------------------------------------------------

def symmetric_data_augmentation_episodic(
    env,
    obs: tensordict.TensorDict,
    actions: torch.Tensor,
) -> Tuple[tensordict.TensorDict, torch.Tensor]:
    """Augment data for episodic trajectories (phase obs unchanged)."""
    return _symmetric_data_augmentation(env, obs, actions, is_half_periodic=False)


def symmetric_data_augmentation_half_periodic(
    env,
    obs: tensordict.TensorDict,
    actions: torch.Tensor,
) -> Tuple[tensordict.TensorDict, torch.Tensor]:
    """Augment data for half-periodic trajectories (phase obs negated)."""
    return _symmetric_data_augmentation(env, obs, actions, is_half_periodic=True)
