# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def reset_init_config(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset state to a specific initial configuration."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # replicate the init config so the shape is env_ids x dof 
    cmd = env.command_manager.get_term(command_name)
    num_env = len(env_ids)
    base_pos = cmd.init_root_state.unsqueeze(0).expand(num_env, -1).clone()
    base_pos[:,0:3] += env.scene.env_origins[env_ids]
    base_vel = cmd.init_root_vel.unsqueeze(0).expand(num_env, -1)
    joint_pos = cmd.init_joint_pos.unsqueeze(0).expand(num_env, -1)
    joint_vel = torch.zeros_like(joint_pos)
    base_vel = 0.0 * base_vel
#     joint_vel = cmd.init_joint_vel.unsqueeze(0).expand(num_env, -1)

     #quat order wxyz
    # set into the physics simulation
    asset.write_root_pose_to_sim(base_pos, env_ids=env_ids)
    asset.write_root_velocity_to_sim(base_vel, env_ids=env_ids)
    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)