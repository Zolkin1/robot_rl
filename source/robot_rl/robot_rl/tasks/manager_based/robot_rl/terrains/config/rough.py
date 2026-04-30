# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from robot_rl.tasks.manager_based.robot_rl.terrains.meta_stair_importer import (
    MetaStairTerrainImporterCfg,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.meta_terrain_generator_cfg import (
    MetaTerrainGeneratorCfg,
)
from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair_cfg import (
    MeshXStairsDownTerrainCfg,
    MeshXStairsUpTerrainCfg,
)

# _STAIR_DIM_OPTIONS = [(0.10, 0.30), (0.15, 0.30), (0.18, 0.25)]
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

CUSTOM_STAIR_IMPORTER_CFG = MetaStairTerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=CUSTOM_STAIR_CFG,
    max_init_terrain_level=5,
    collision_group=-1,
    debug_vis=False,
)





ROUGH_FOR_BASIC_LOCOMOTION_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.0005,  
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.45, grid_height_range=(0.0, 0.02), platform_width=5.0
        ),

    },
)

ROBUSTNESS_TEST_FOR_BASIC_LOCOMOTION_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=8,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.0005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.4, grid_width=0.45, grid_height_range=(0.1, 0.15), platform_width=1.0
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.3, platform_width=1.0, slope_range=(0.2, 0.2),
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.3, platform_width=1.0, slope_range=(0.2, 0.2), inverted=True,
        ),
    }
)

ROUGH_SLOPED_FOR_BASIC_LOCOMOTION_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.0005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.5, grid_width=0.45, grid_height_range=(0.0, 0.02), platform_width=5.0
        ),
        "slope_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, platform_width=1.0, slope_range=(0.0, 0.15),
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2, platform_width=1.0, slope_range=(0.0, 0.15), inverted=True,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0, 0.02), noise_step=0.01, border_width=0.25
        ),
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

UP_STAIR_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(6.0, 6.0),
    border_width=6.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.0, 0.15),
        #     step_width=0.25,
        #     platform_width=0.6,
        #     border_width=0.0,
        #     holes=False,
        # ),
        "stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=0.6,
            border_width=0.0,
            holes=False,
        ),
    },
)

"""Rough terrains configuration."""

STAIR_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(6.0, 6.0),
    border_width=5.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.15),
            step_width=0.25,
            platform_width=0.6,
            border_width=0.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.15),
            step_width=0.25,
            platform_width=0.6,
            border_width=0.0,
            holes=False,
        ),
        # "stairs_inv_w_hole": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.0, 0.15),
        #     step_width=0.25,
        #     platform_width=0.6,
        #     border_width=0.0,
        #     holes=True,
        # ),
        # "stairs_w_hole": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.0, 0.15),
        #     step_width=0.25,
        #     platform_width=0.6,
        #     border_width=0.0,
        #     holes=True,
        # ),
        "flat": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.0),
            step_width=0.25,
            platform_width=0.6,
            border_width=0.0,
            holes=False,
        ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.05), platform_width=2.0
        # ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.2, noise_range=(0.0, 0.0), noise_step=0.02, border_width=0.25
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.33, slope_range=(0.0, 0.0), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        # ),
    },
)