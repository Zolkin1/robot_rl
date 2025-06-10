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
    env: ManagerBasedRLEnv, env_ids: Sequence[int], num_steps: int, original_w: float = -2.0, update_interval: int = 1000
) -> None:
    """Curriculum based on clf value
    """
    # if env._sim_step_counter > num_steps and env._sim_step_counter % update_interval == 0:
            # buf = env.observation_manager._group_obs_term_history_buffer["critic"]["v"].buffer

            # 2) Compute one global average (over envs *and* time), then clamp
            #    Results in a 0-d tensor; .item() → Python float
            # scale = env.observation_manager.cfg.critic.v.scale
            # global_avg = buf.mean() * 1.0/scale
    buf = env.command_manager.get_term("hlip_ref").v_buffer
    global_avg = buf.mean()
    global_avg = torch.clamp(global_avg, min=1.0, max=100.0)
    term_cfg = env.reward_manager.get_term_cfg("clf_reward")
    term_cfg.params["max_clf"] = global_avg.detach().cpu().item()
    env.reward_manager.set_term_cfg("clf_reward", term_cfg)


            # increase clf decreasing condition weight?
            # term_cfg = env.reward_manager.get_term_cfg("clf_decreasing_condition")
            # raw_w = global_avg.detach().cpu().item()/100.0 * original_w
            # term_cfg.weight = max(-4.0, min(-2.0, raw_w))
            # env.reward_manager.set_term_cfg("clf_decreasing_condition", term_cfg)

def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
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
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())