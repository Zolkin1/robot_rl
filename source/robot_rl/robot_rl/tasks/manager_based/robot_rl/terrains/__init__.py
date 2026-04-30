# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .stair_cfg import MeshProgressiveXStairsTerrainCfg, MeshUniformXStairsTerrainCfg

# Long uniform staircase preset for G1 stair training.
# Per cell: 30 uniform steps of (d=0.233 m, r=0.135 m) with tread lateral extent
# w=0.777 m, centred inside a 12 x 2 m cell. Empty padding on all sides isolates
# each staircase replica from its neighbours so a height scanner sees only one
# staircase at a time. The env origin sits on the central step at z=0 so a reset
# that places the stance foot at world z=0 spawns the robot mid-climb.
LONG_STAIRS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(12.0, 2.0),
    border_width=0.0,
    # No curriculum and all sub-terrains identical, so a single 1x1 cell suffices.
    # All envs share the same env_origin; each env's physics is its own replica.
    # PLAY configs that want to render multiple robots side-by-side should bump
    # ``terrain_generator.num_cols`` in their __post_init__.
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "uniform_stairs": MeshUniformXStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.135, 0.135),
            step_width=0.233,
            num_steps=30,
            tread_lateral_extent=0.777,
        ),
    },
)
