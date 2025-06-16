# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def clf_curriculum(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], update_interval: int = 100
) -> float:
    """Curriculum based on clf value"""
    term_cfg = env.reward_manager.get_term_cfg("clf_decreasing_condition")
    new_clf = term_cfg.params["max_clf_decreasing"]
    if env.common_step_counter  >= update_interval and env.common_step_counter % update_interval == 0:
        
            # buf = env.observation_manager._group_obs_term_history_buffer["critic"]["v"].buffer

            # 2) Compute one global average (over envs *and* time), then clamp
            #    Results in a 0-d tensor; .item() → Python float
            # scale = env.observation_manager.cfg.critic.v.scale
            # global_avg = buf.mean() * 1.0/scale
            # buf = env.command_manager.get_term("hlip_ref").v_buffer
            # global_avg = buf.mean()
            # global_avg = torch.clamp(global_avg, min=1.0, max=100.0)
            # term_cfg = env.reward_manager.get_term_cfg("clf_reward")
            # term_cfg.params["max_clf"] = global_avg.detach().cpu().item()
            # env.reward_manager.set_term_cfg("clf_reward", term_cfg)

            # increase clf decreasing condition weight?
            new_clf = max(new_clf -2,20.0)
            term_cfg.params["max_clf_decreasing"] = new_clf
            env.reward_manager.set_term_cfg("clf_decreasing_condition", term_cfg)
    return new_clf

def terrain_levels(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Curriculum based on distance traveled vs expected distance.
    Promotes to harder terrain if distance > 50% of expected so far.
    Demotes if distance < 50% of expected and not already moving up.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    # # Distance traveled in XY
    # distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)

    # # Time-aware expected distance
    # elapsed_time = env.episode_length_buf[env_ids] * env.step_dt
    # commanded_speed = torch.norm(command[env_ids, :2], dim=1)
    # expected_distance = commanded_speed * elapsed_time

    # # Curriculum decisions
    # # time_gate_down = env.episode_length_buf[env_ids] > 0.4 * env.max_episode_length
    # # time_gate_up = env.episode_length_buf[env_ids] > 0.7 * env.max_episode_length

    # # Logic
    # move_up = (distance > 0.6 * expected_distance) 
    # move_down = (distance < 0.5 * expected_distance) & (~move_up) 
  
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)

    # Return average terrain level (for logging or scaling)
    return torch.mean(terrain.terrain_levels.float())
