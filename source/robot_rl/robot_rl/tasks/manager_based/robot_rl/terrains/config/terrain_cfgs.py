# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from robot_rl.tasks.manager_based.robot_rl.terrains.meta_terrain_generator_cfg import (
    MetaTerrainGeneratorCfg,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.flat_cfg import (
    MeshFlatTerrainCfg,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair_cfg import (
    MeshXStairsDownTerrainCfg,
    MeshXStairsUpTerrainCfg,
)


_STAIR_DIM_OPTIONS = [(0.18, 0.25)]

CUSTOM_STAIR_CFG = MetaTerrainGeneratorCfg(
    size=(10.0, 10.0),
    num_rows=5,
    num_cols=2,
    border_width=0.0,
    sub_terrains={
        "stairs_up": MeshXStairsUpTerrainCfg(
            proportion=0.5,
            size=(5.0, 10.0),
            step_dim_options=_STAIR_DIM_OPTIONS,
            stair_width_range=(2.0, 4.0),
            origin_distance_from_back=1.0,
            float_prob=0.0,
            wall_prob=0.0,
            pole_prob=0.0,
        ),
        "stairs_down": MeshXStairsDownTerrainCfg(
            proportion=0.5,
            size=(5.0, 10.0),
            step_dim_options=_STAIR_DIM_OPTIONS,
            stair_width_range=(2.0, 4.0),
            origin_distance_from_back=1.0,
            float_prob=0.0,
            wall_prob=0.0,
            pole_prob=0.0,
        ),
    },
)


# Single-skill ascending-stairs preset for the G1 stair-tracking env.
# Cell x is sized to fit exactly ``num_steps`` treads of ``step_depth`` so the
# generator tiles cleanly.  With ``start_z_zero=True`` below, the lowest stair
# top sits at z=0, the env origin spawns on that first stair, and the whole
# staircase rises above z=0 — so you can use roughly half the steps you'd need
# under the centred layout to cover the same climb.  10 x 20 = 200 cells
# distribute 4096 envs at ~20 envs/cell, matching the broadphase density the
# default rough cfg is sized for.
_LONG_STAIRS_NUM_STEPS = 60
_LONG_STAIRS_STEP_DEPTH = 0.233
_LONG_STAIRS_STEP_HEIGHT = 0.135

LONG_STAIRS_CFG = MetaTerrainGeneratorCfg(
    curriculum=False,
    size=(_LONG_STAIRS_NUM_STEPS * _LONG_STAIRS_STEP_DEPTH, 2.0),
    border_width=0.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "stairs_up": MeshXStairsUpTerrainCfg(
            proportion=1.0,
            step_dim_options=[(_LONG_STAIRS_STEP_HEIGHT, _LONG_STAIRS_STEP_DEPTH)],
            stair_width_range=(1.5, 1.5),
            # Lowest stair top sits at z=0; spawn the env on stair 0 (ground
            # level) rather than the middle of the staircase.
            start_z_zero=True,
            origin_distance_from_back=0.5 * _LONG_STAIRS_STEP_DEPTH,
            float_prob=0.0,
            wall_prob=0.0,
            pole_prob=0.0,
        ),
    },
)

# Config for both stair climbing and walking, with the two terrain types
# allocated to disjoint column ranges (curriculum=True), separated by a flat
# unspawnable strip. ``num_cols`` is bumped to 22 to absorb the 2 border
# columns (20m strip / 10m cell_y) so the spawnable count stays 20.
STAIR_WALK_CFG = MetaTerrainGeneratorCfg(
    curriculum=True,
    size=(_LONG_STAIRS_NUM_STEPS * _LONG_STAIRS_STEP_DEPTH, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=22,
    inter_column_borders=[("stairs_up", 20.0)],
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "stairs_up": MeshXStairsUpTerrainCfg(
            proportion=0.5,
            step_dim_options=[(_LONG_STAIRS_STEP_HEIGHT, _LONG_STAIRS_STEP_DEPTH)],
            stair_width_range=(1.5, 1.5),
            # Lowest stair top sits at z=0 (default ``start_z_zero=True``);
            # spawn the env on stair 0 so the robot starts at ground level.
            origin_distance_from_back=0.5 * _LONG_STAIRS_STEP_DEPTH,
            float_prob=0.0,
            wall_prob=0.0,
            pole_prob=0.0,
        ),
        # "stairs_down": MeshXStairsDownTerrainCfg(
        #     proportion=0.5,
        #     step_dim_options=[(_LONG_STAIRS_STEP_HEIGHT, _LONG_STAIRS_STEP_DEPTH)],
        #     stair_width_range=(1.5, 1.5),
        #     origin_distance_from_back=(_LONG_STAIRS_NUM_STEPS // 2 + 0.5) * _LONG_STAIRS_STEP_DEPTH,
        #     float_prob=0.0,
        #     wall_prob=0.0,
        #     pole_prob=0.0,
        # ),
        "flat": MeshFlatTerrainCfg(proportion=0.5),
    },
)


ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
"""Rough terrains configuration."""
