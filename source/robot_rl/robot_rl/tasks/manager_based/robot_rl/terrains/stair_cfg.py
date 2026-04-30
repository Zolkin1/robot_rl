# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from robot_rl.tasks.manager_based.robot_rl.terrains.stair import (
    progressive_x_stairs_terrain,
    single_staircase_terrain,
)


@configclass
class MeshUniformXStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a uniform-rise staircase along +x, centred in its cell."""

    function = single_staircase_terrain

    step_height_range: tuple[float, float] = (0.03, 0.15)
    """Min/max rise of the steps (in m). The actual rise interpolates by ``difficulty``;
    pass equal min/max for a fixed rise."""

    step_width: float = 0.25
    """Tread depth, i.e. the +x extent of each step (in m)."""

    num_steps: int = 30
    """Number of steps in the staircase."""

    tread_lateral_extent: float = 1.0
    """Lateral (y) extent of each step's tread (in m). The remaining ``size[1] -
    tread_lateral_extent`` of the cell is empty space, leaving a gap between
    neighbouring sub-terrain replicas."""

    central_step_index: int | None = None
    """Index of the step whose tread top sits at z = 0 (and on which the env
    origin is placed). Defaults to ``num_steps // 2`` (mid-climb)."""


@configclass
class MeshProgressiveXStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a staircase along +x with linearly increasing step heights."""

    function = progressive_x_stairs_terrain

    step_height_range: tuple[float, float] = (0.03, 0.15)
    """Step rise at the first vs. last step (in m). Heights interpolate linearly
    across the staircase."""

    step_width: float = 0.25
    """Tread depth, i.e. the +x extent of each step (in m)."""

    border_width: float = 0.0
    """Width of the border around the terrain (in m)."""
