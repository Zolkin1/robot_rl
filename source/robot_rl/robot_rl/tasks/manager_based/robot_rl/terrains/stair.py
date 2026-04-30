# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING

import numpy as np
import trimesh

from isaaclab.terrains.trimesh.utils import make_border

if TYPE_CHECKING:
    from robot_rl.tasks.manager_based.robot_rl.terrains.stair_cfg import (
        MeshProgressiveXStairsTerrainCfg,
        MeshUniformXStairsTerrainCfg,
    )


def progressive_x_stairs_terrain(
    difficulty: float, cfg: "MeshProgressiveXStairsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a staircase along +x with linearly increasing step heights.

    Each step occupies the full lateral (y) extent of the terrain. Step heights
    interpolate linearly from ``cfg.step_height_range[0]`` (first step) to
    ``cfg.step_height_range[1]`` (last step). The :obj:`difficulty` parameter is
    not used here.

    Args:
        difficulty: Unused. Difficulty curriculum is encoded in the height range.
        cfg: Configuration for the progressive staircase terrain.

    Returns:
        A tuple containing the list of trimesh meshes and the origin (in m) of the
        terrain (placed at the +x base of the stairs at floor level).
    """
    terrain_width = cfg.size[1]
    terrain_length = cfg.size[0] - 2 * cfg.border_width
    step_depth = cfg.step_width
    min_h, max_h = cfg.step_height_range

    num_steps = int(terrain_length // step_depth)
    step_heights = np.linspace(min_h, max_h, num_steps)

    meshes_list: list[trimesh.Trimesh] = []
    cum_z = 0.0
    for i in range(num_steps):
        h = step_heights[i]
        pos_x = cfg.border_width + i * step_depth + step_depth / 2
        pos_y = cfg.size[1] / 2
        pos_z = cum_z + h / 2

        box_dims = (step_depth, terrain_width, h)
        box_pos = (pos_x, pos_y, pos_z)
        meshes_list.append(trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos)))
        cum_z += h

    if cfg.border_width > 0.0:
        border_center = [0.5 * cfg.size[0], 0.5 * cfg.size[1], -step_heights[0] / 2]
        inner = (terrain_length, terrain_width)
        meshes_list += make_border(cfg.size, inner, step_heights[0], border_center)

    origin = np.array([cfg.border_width, cfg.size[1] / 2, 0.0])
    return meshes_list, origin


def single_staircase_terrain(
    difficulty: float, cfg: "MeshUniformXStairsTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a uniform staircase along +x, centred inside its sub-terrain cell.

    Geometry per cell:
        - ``num_steps`` solid step boxes laid along +x. Each step's tread spans
          ``step_width`` (depth in x) by ``tread_lateral_extent`` (in y).
        - All step boxes share a common bottom z (so the staircase looks solid
          from the side).
        - Step ``central_step_index`` has its tread top at z = 0; lower steps sit
          below z = 0, higher steps rise above. The default central index is
          ``num_steps // 2`` (mid-climb spawn).
        - The full mesh is centred inside the cell of size ``cfg.size``, leaving
          empty space (no mesh) as padding on all sides — adjacent sub-terrain
          cells are not connected, so a height scanner mounted on the robot does
          not see neighbouring staircase replicas.

    Step rise interpolates ``cfg.step_height_range`` by ``difficulty``; pass an
    equal min/max for a fixed rise.

    Args:
        difficulty: Difficulty in [0, 1]. Selects step height from the range.
        cfg: Configuration for the uniform staircase terrain.

    Returns:
        A tuple of (list of trimesh meshes, terrain origin in m). The origin sits
        on the central step's tread top (z = 0) so a reset that places the stance
        foot at world z = 0 lands the robot on a step.
    """
    step_depth = cfg.step_width
    tread_lateral_extent = cfg.tread_lateral_extent
    num_steps = cfg.num_steps

    min_h, max_h = cfg.step_height_range
    step_height = min_h + difficulty * (max_h - min_h)

    k_center = cfg.central_step_index if cfg.central_step_index is not None else num_steps // 2

    stair_length_x = num_steps * step_depth
    pad_x = (cfg.size[0] - stair_length_x) / 2
    pad_y = (cfg.size[1] - tread_lateral_extent) / 2

    # All step boxes extend down to a common floor z so the staircase reads as a
    # solid block from the side rather than as floating slabs.
    floor_z = -(k_center + 1) * step_height

    meshes_list: list[trimesh.Trimesh] = []
    for k in range(num_steps):
        top_z = (k - k_center) * step_height
        box_height = top_z - floor_z

        pos_x = pad_x + (k + 0.5) * step_depth
        pos_y = pad_y + tread_lateral_extent / 2
        pos_z = (top_z + floor_z) / 2

        box_dims = (step_depth, tread_lateral_extent, box_height)
        box_pos = (pos_x, pos_y, pos_z)
        meshes_list.append(
            trimesh.creation.box(box_dims, trimesh.transformations.translation_matrix(box_pos))
        )

    origin = np.array(
        [
            pad_x + (k_center + 0.5) * step_depth,
            pad_y + tread_lateral_extent / 2,
            0.0,
        ]
    )
    return meshes_list, origin
