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
import warp as wp
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter


def gaits_curriculum(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], update_interval: int = 24000,
    vel_interval: float = 0.1
) -> float:
    """Curriculum based on clf value"""
    cmd_term = env.command_manager.get_term("base_velocity")
    ref_cmd_term = env.command_manager.get_term("hzd_ref")

    max_vel_range = ref_cmd_term.cfg.gait_velocity_ranges
    cur_vel_range = cmd_term.cfg.ranges.lin_vel_x

    # load the gait library velocity range; 
    new_min = cur_vel_range[0] - vel_interval
    new_max = cur_vel_range[1]

    #bound the new vel range
    new_min = max(new_min,max_vel_range[0])
    new_max = min(new_max,max_vel_range[1])

    if env.common_step_counter % update_interval == 0:
        cmd_term.cfg.ranges.lin_vel_x = (new_min,new_max)
        # env.command_manager.set_term_cfg("base_velocity", cmd_term_cfg)
    return cmd_term.cfg.ranges.lin_vel_x[0]

def cmd_vel_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int], max_vel: float, step: float,
                       update_interval: int = 100, first_update: int = 100):
    """Curriculum to increase the commanded velocity."""
    commanded_velocity = env.command_manager.get_term("base_velocity")

    new_vel = commanded_velocity.cfg.ranges.lin_vel_x[1]

    if env.common_step_counter >= first_update and env.common_step_counter % update_interval == 0:
        # Compute new vel
        new_vel = min(new_vel + step, max_vel)

        # Assign vel - create new tuple since tuples are immutable
        commanded_velocity.cfg.ranges.lin_vel_x = (commanded_velocity.cfg.ranges.lin_vel_x[0], new_vel)

    return new_vel

def walk_run_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int], update_interval: int = 80000):
    """Curriculum to switch between walking and running."""
    commanded_velocity = env.command_manager.get_term("base_velocity")

    top_vel = commanded_velocity.cfg.ranges.lin_vel_x[1]

    if env.common_step_counter >= update_interval and env.common_step_counter % update_interval == 0:
        if top_vel == 1.0:
            commanded_velocity.cfg.ranges.lin_vel_x = (1.1, 3.0)
        else:
            commanded_velocity.cfg.ranges.lin_vel_x = (0.0, 1.0)

    return top_vel

def clf_curriculum(
    env: ManagerBasedRLEnv, env_ids: Sequence[int],min_max_err: tuple[float,float] = (0.1,0.25,0.1),
        scale: tuple[float,float] = (0.01,0.01,0.01), update_interval: int = 100,
) -> float:
    """Curriculum based on clf value"""
    term_cfg = env.reward_manager.get_term_cfg("clf_decreasing_condition")
    new_max_eta_err = term_cfg.params["eta_max"]
    new_max_eta_dot_err = term_cfg.params["eta_dot_max"]

    clf_cfg = env.reward_manager.get_term_cfg("clf_reward")
    clf_max_eta_err = clf_cfg.params["max_eta_err"]

    if env.common_step_counter >= update_interval and env.common_step_counter % update_interval == 0:
            # Vdot update
            new_max_eta_err = max(new_max_eta_err - scale[0], min_max_err[0])
            new_max_eta_dot_err = max(new_max_eta_dot_err - scale[1], min_max_err[1])

            term_cfg.params["eta_max"] = new_max_eta_err
            term_cfg.params["eta_dot_max"] = new_max_eta_dot_err
            env.reward_manager.set_term_cfg("clf_decreasing_condition", term_cfg)

            # V update
            clf_max_eta_err = max(clf_max_eta_err - scale[2], min_max_err[2])

            clf_cfg.params["max_eta_err"] = clf_max_eta_err
            env.reward_manager.set_term_cfg("clf_reward", clf_cfg)

    return clf_max_eta_err

def contact_curriculum(env: ManagerBasedRLEnv, env_ids: Sequence[int], max_weight: float, update_amnt: float,
                       update_interval: int = 100) -> float:
    """Curriculum to adjust the weight on the contact penalty."""
    term_cfg = env.reward_manager.get_term_cfg("flight_contact_penalty")
    new_weight = term_cfg.params["weight_scalar"]
    if env.common_step_counter >= update_interval and env.common_step_counter % update_interval == 0:
        # Compute new contact weight
        new_weight = min(new_weight + update_amnt, max_weight)

        term_cfg.params["weight_scalar"] = new_weight
        env.reward_manager.set_term_cfg("flight_contact_penalty", term_cfg)


    return new_weight

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
    distance = torch.norm(wp.to_torch(asset.data.root_pos_w)[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.6
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)

    # Return average terrain level (for logging or scaling)
    return torch.mean(terrain.terrain_levels.float())
