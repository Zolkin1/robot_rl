from dataclasses import MISSING
from typing import Literal

import trimesh
import numpy as np
import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_border
from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair import progressive_x_stairs_terrain, single_staircase_terrain


@configclass
class MeshUniformXStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a uniform staircase along +x direction (fixed step height)."""

    function = single_staircase_terrain  # <- make sure this points to your updated function

    step_height_range: tuple[float, float] = (0.03, 0.15)
    """Range used for selecting step height from difficulty."""

    step_width: float = 0.25
    """The depth of each step (in x-direction) in meters."""

    border_width: float = 0.0
    """Border around the terrain in meters."""


@configclass
class MeshProgressiveXStairsTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a long staircase with progressively taller steps along +y."""

    function = progressive_x_stairs_terrain

    step_height_range: tuple[float, float] = (0.03, 0.15)
    """The minimum and maximum height of the steps (in m)."""

    step_width: float = 0.25
    """The depth of each step in y-direction (in m)."""

    border_width: float = 0.0
    """The width of the border around the terrain (in m)."""


import torch
from typing import Union

def get_step_height_at_x(
    x_vals: torch.Tensor, cfg: "MeshProgressiveXStairsTerrainCfg"
) -> torch.Tensor:
    """
    Given a batch of x-coordinates, return the cumulative step height at each x
    for a staircase that increases in +x direction with growing step heights (Torch version).

    Args:
        x_vals: Tensor of x positions (shape: [N]).
        cfg: Stair terrain config.

    Returns:
        Tensor of terrain heights (shape: [N]).
    """
    usable_x = x_vals - cfg.border_width
    step_depth = cfg.step_width
    terrain_length = cfg.size[0] - 2 * cfg.border_width
    num_steps = int(terrain_length // step_depth)

    # Create step heights and cumulative sum
    step_heights = torch.linspace(
        cfg.step_height_range[0],
        cfg.step_height_range[1],
        steps=num_steps,
        dtype=torch.float32,
        device=x_vals.device,
    )
    cum_heights = torch.cumsum(step_heights, dim=0)  # shape: [num_steps]

    # Initialize result
    heights = torch.zeros_like(x_vals)

    # Masks
    mask_below = usable_x < 0
    mask_above = usable_x >= terrain_length
    mask_inside = ~(mask_below | mask_above)

    # Assign heights
    heights[mask_below] = 0.0
    heights[mask_above] = cum_heights[-1]

    if mask_inside.any():
        step_indices = (usable_x[mask_inside] // step_depth).long()
        heights[mask_inside] = cum_heights[step_indices]

    return heights


def get_uniform_stair_step_height_from_env(terrain_origins, cfg: "MeshUniformXStairsTerrainCfg") -> torch.Tensor:
    """
    Estimate step height from terrain origin and known step depth.
    Assumes uniform stair steps and that terrain origin.z = -total_height.
    """
    origin_z = -terrain_origins[:,2]  # negate because origin.z = -cum_z
    num_steps = int((cfg.size[0] - 2 * cfg.border_width) // cfg.step_width)
    return (origin_z / num_steps) 