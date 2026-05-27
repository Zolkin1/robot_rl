# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from robot_rl.tasks.manager_based.robot_rl.terrains.blocks import (
    BlockChoice,
    CompositeSubTerrainCfg,
    FlatBlockCfg,
    RandomizedCompositeSubTerrainCfg,
    SlopeBlockCfg,
    StairBlockCfg,
)
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


# Demo composite sub-terrain: short stair-up → flat platform → short stair-up,
# all in one sub-terrain cell. Demonstrates per-block skill metadata flow —
# `MetaCompositeTerrainImporter.skill_probs_at` switches between
# ``stair_up`` and ``walk_forward`` as the robot crosses block boundaries, and
# `_project_world` resolves the right stair span. The composer threads
# elevation across blocks: the flat platform sits flush with whichever
# neighbor came before it.
_COMPOSITE_STEP_HEIGHT = 0.135
_COMPOSITE_STEP_DEPTH = 0.233
_COMPOSITE_STAIR_NUM_STEPS = 6
_COMPOSITE_STAIR_SIZE_X = _COMPOSITE_STAIR_NUM_STEPS * _COMPOSITE_STEP_DEPTH
_COMPOSITE_FLAT_SIZE_X = 3.0
START_PLATFORM_LENGTH = 0.4
END_PLATFORM_LENGTH = 0.0

# Per-block sizes. ``StairBlockCfg.size_x`` is the FULL block extent
# (platforms + stair treads); the stairs themselves occupy
# ``size_x - start_platform_length - end_platform_length`` in the middle.
_BLOCK0_SIZE_X = _COMPOSITE_STAIR_SIZE_X + START_PLATFORM_LENGTH + END_PLATFORM_LENGTH
_BLOCK1_SIZE_X = _COMPOSITE_FLAT_SIZE_X
_BLOCK2_SIZE_X = _COMPOSITE_STAIR_SIZE_X
_TOTAL_SIZE_X = _BLOCK0_SIZE_X + 2*_BLOCK1_SIZE_X + 2*_BLOCK2_SIZE_X

STAIR_FLAT_STAIR_CFG = MetaTerrainGeneratorCfg(
    curriculum=False,
    size=(_TOTAL_SIZE_X, 2.0),
    border_width=0.0,
    num_rows=4,
    num_cols=8,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "stair_flat_stair": CompositeSubTerrainCfg(
            proportion=1.0,
            size=(_TOTAL_SIZE_X, 2.0),
            origin_block_index=0,
            blocks=[
                FlatBlockCfg(
                    size_x=_BLOCK1_SIZE_X,
                    skill_probs={"walk_forward": 0.5, "standing": 0.5},
                ),
                StairBlockCfg(
                    size_x=_BLOCK0_SIZE_X,
                    direction="up",
                    skill_probs={"stair_up": 1.0},
                    step_dim_options=[(_COMPOSITE_STEP_HEIGHT, _COMPOSITE_STEP_DEPTH)],
                    stair_width_range=(1.5, 1.5),
                    start_platform_length=START_PLATFORM_LENGTH,
                    end_platform_length=END_PLATFORM_LENGTH,
                ),
                FlatBlockCfg(
                    size_x=_BLOCK1_SIZE_X,
                    skill_probs={"walk_forward": 0.5, "standing": 0.5},
                ),
                StairBlockCfg(
                    size_x=_BLOCK2_SIZE_X,
                    direction="up",
                    skill_probs={"stair_up": 1.0},
                    step_dim_options=[(_COMPOSITE_STEP_HEIGHT, _COMPOSITE_STEP_DEPTH)],
                    stair_width_range=(1.5, 1.5),
                ),
                StairBlockCfg(
                    size_x=_BLOCK2_SIZE_X,
                    direction="down",
                    skill_probs={"stair_down": 1.0},
                    step_dim_options=[(_COMPOSITE_STEP_HEIGHT, _COMPOSITE_STEP_DEPTH)],
                    stair_width_range=(1.5, 1.5),
                ),
            ],
        ),
    },
)


# Randomized multiskill terrain: each sub-terrain cell is a fresh random walk
# along x mixing flat, stair-up (with partial trailing tread), stair-down, and
# slopes. Sub-terrain cells are placed directly adjacent in y; if a gap is
# desired, set the per-block width range smaller than the cell's y extent so
# empty strips appear inside each cell instead of as separate columns.
_RANDOM_SIZE_X = 20.0 #12.0
_RANDOM_SIZE_Y = 10.0
_RANDOM_STAIR_STEP_DIM_OPTIONS = [(0.10, 0.30), (0.14, 0.28), (0.16, 0.32)]
_RANDOM_NUM_COLS = 4

RANDOMIZED_MULTISKILL_TERRAIN_CFG = MetaTerrainGeneratorCfg(
    curriculum=False,
    size=(_RANDOM_SIZE_X, _RANDOM_SIZE_Y),
    border_width=0.0,
    num_rows=4,
    num_cols=_RANDOM_NUM_COLS,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    inter_column_borders=[],
    sub_terrains={
        "randomized": RandomizedCompositeSubTerrainCfg(
            proportion=1.0,
            size=(_RANDOM_SIZE_X, _RANDOM_SIZE_Y),
            origin_block_index=0,
            force_flat_origin=True,
            length_range=(1.0, 3.0),
            trailing_flat_length_range=(1.5, 3.0),
            choices=[
                BlockChoice(
                    cfg=FlatBlockCfg(
                        skill_probs={"walk_forward": 0.5, "running": 0.4, "standing": 0.1},
                        flat_width_range=(1.5, _RANDOM_SIZE_Y),
                    ),
                    weight=2.0,
                ),
                BlockChoice(
                    cfg=StairBlockCfg(
                        direction="up",
                        skill_probs={"stair_up": 1.0},
                        step_dim_options=_RANDOM_STAIR_STEP_DIM_OPTIONS,
                        stair_width_range=(1.5, _RANDOM_SIZE_Y),
                        float_prob=0.0,
                        wall_prob=0.0,
                        pole_prob=0.0,
                    ),
                    weight=1.0,
                    length_range=(1.5, 4.0),
                ),
                BlockChoice(
                    cfg=StairBlockCfg(
                        direction="down",
                        skill_probs={"stair_down": 1.0},
                        step_dim_options=_RANDOM_STAIR_STEP_DIM_OPTIONS,
                        stair_width_range=(1.5, _RANDOM_SIZE_Y),
                        float_prob=0.0,
                        wall_prob=0.0,
                        pole_prob=0.0,
                    ),
                    weight=1.0,
                    length_range=(1.5, 4.0),
                ),
                BlockChoice(
                    cfg=SlopeBlockCfg(
                        direction="up",
                        skill_probs={"walk_forward": 1.0},
                        rise_range=(0.2, 0.8),
                        slope_width_range=(1.5, _RANDOM_SIZE_Y),
                    ),
                    weight=0.5,
                    length_range=(1.5, 3.5),
                ),
                BlockChoice(
                    cfg=SlopeBlockCfg(
                        direction="down",
                        skill_probs={"walk_forward": 1.0},
                        rise_range=(0.2, 0.8),
                        slope_width_range=(1.5, _RANDOM_SIZE_Y),
                    ),
                    weight=0.5,
                    length_range=(1.5, 3.5),
                ),
            ],
        ),
    },
)
"""Randomized multiskill terrain with lateral void gaps."""


# Three-sub-terrain multiskill mix:
#   * ``pure_flat``     — legacy ``MeshFlatTerrainCfg`` (entire cell is one
#     flat plane).
#   * ``pure_stair_up`` — legacy ``MeshXStairsUpTerrainCfg`` (entire cell is
#     a single ascending staircase, full y width).
#   * ``flat_stair_up`` — composite cells that interleave flat and stair-up
#     blocks only (no stair-down, no slopes); the first block is forced to
#     flat so the spawn lands on level ground.
# The pure sub-terrains use the same legacy types as ``STAIR_WALK_CFG``;
# only the mixed sub-terrain needs the randomized composite (the only way to
# interleave block types within one cell).
FLAT_STAIR_MULTISKILL_CFG = MetaTerrainGeneratorCfg(
    curriculum=True,
    size=(_RANDOM_SIZE_X, _RANDOM_SIZE_Y),
    border_width=20.0,
    # num_rows=4,
    # num_cols=_RANDOM_NUM_COLS,
    num_rows=10,
    num_cols=22,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    inter_column_borders=[],
    sub_terrains={
        "pure_flat": MeshFlatTerrainCfg(proportion=0.2),
        "pure_stair_up": MeshXStairsUpTerrainCfg(
            proportion=0.2,
            step_dim_options=[(_LONG_STAIRS_STEP_HEIGHT, _LONG_STAIRS_STEP_DEPTH)],
            # Lowest stair top sits at z=0 (default ``start_z_zero=True``);
            # spawn the env on stair 0 so the robot starts at ground level.
            origin_distance_from_back=0.5 * _LONG_STAIRS_STEP_DEPTH,
            stair_width_range=(1.5, _RANDOM_SIZE_Y),
            float_prob=0.5,
            wall_prob=0.0,
            pole_prob=0.0,
        ),
        "flat_stair_up": RandomizedCompositeSubTerrainCfg(
            proportion=0.6,
            size=(_RANDOM_SIZE_X, _RANDOM_SIZE_Y),
            origin_block_index=0,
            force_flat_origin=True,
            length_range=(1.0, 2.0),
            trailing_flat_length_range=(1.5, 3.0),
            choices=[
                BlockChoice(
                    cfg=FlatBlockCfg(
                        skill_probs={"walk_forward": 0.98, "running": 0.0, "standing": 0.02},
                        flat_width_range=(1.5, _RANDOM_SIZE_Y),
                    ),
                    weight=1.0,
                ),
                BlockChoice(
                    cfg=StairBlockCfg(
                        direction="up",
                        skill_probs={"stair_up": 1.0},
                        step_dim_options=[(_LONG_STAIRS_STEP_HEIGHT, _LONG_STAIRS_STEP_DEPTH)],
                        stair_width_range=(1.5, _RANDOM_SIZE_Y),
                        float_prob=0.5,
                        wall_prob=0.0,
                        pole_prob=0.0,
                    ),
                    weight=1.0,
                    length_range=(0.5, 2.0),
                ),
            ],
        ),
    },
)
"""Three-sub-terrain mix: pure flat (legacy), pure stair-up (legacy), and a
composite flat + stair-up sub-terrain."""


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
