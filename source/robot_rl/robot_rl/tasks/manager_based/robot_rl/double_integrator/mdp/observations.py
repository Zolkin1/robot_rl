from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def di_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the joint position of the double integrator (shape [N, 1])."""
    return env.scene["robot"].data.joint_pos


def di_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the joint velocity of the double integrator (shape [N, 1])."""
    return env.scene["robot"].data.joint_vel


def di_target_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the target position, which is always 0 (shape [N, 1])."""
    return torch.zeros(env.num_envs, 1, device=env.device)
