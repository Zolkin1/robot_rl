from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def di_out_of_bounds(env: ManagerBasedRLEnv, max_distance: float = 10.0) -> torch.Tensor:
    """Terminate if the double integrator position exceeds max_distance from origin."""
    joint_pos = env.scene["robot"].data.joint_pos[:, 0]
    return torch.abs(joint_pos) > max_distance
