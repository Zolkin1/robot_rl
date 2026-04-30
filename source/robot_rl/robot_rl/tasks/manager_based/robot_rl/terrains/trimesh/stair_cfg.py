from __future__ import annotations

from dataclasses import MISSING

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair import (
    x_stairs_down_terrain,
    x_stairs_up_terrain,
)


@configclass
class MeshXStairsUpTerrainCfg(SubTerrainBaseCfg):
    """Stairs ascending in +x, filling the sub-terrain and centered around z=0.

    The staircase covers the full x extent of the sub-terrain (no platforms,
    no floor underneath). Each call samples a (height, depth) pair from
    ``step_dim_options`` and a width from ``stair_width_range``. Optional
    sidewalls (solid or pole pillars) can be added on either side.
    """

    function = x_stairs_up_terrain

    step_dim_options: list[tuple[float, float]] = MISSING
    """List of ``(step_height, step_depth)`` tuples sampled uniformly per terrain."""

    stair_width_range: tuple[float, float] = MISSING
    """Range (min, max) for the stair width along y, sampled uniformly."""

    origin_distance_from_back: float = 1.0
    """Approximate x-distance from the back edge (x=0) used to pick the env origin."""

    float_prob: float = 0.0
    """Probability that stairs are floating thin treads instead of solid blocks."""

    float_thick_range: tuple[float, float] = (0.025, 0.075)
    """Thickness range (min, max) for floating treads when ``float_prob`` triggers."""

    wall_prob: float = 0.0
    """Probability of placing sidewalls. If triggered, the wall is solid by default
    and replaced with poles with probability ``pole_prob``."""

    wall_thickness: float = 0.1
    """Thickness in y of each solid sidewall."""

    wall_height: float = 1.5
    """Height of the wall above the highest stair's top surface."""

    pole_prob: float = 0.0
    """Conditional probability (given walls were rolled) of using poles instead
    of a solid wall."""

    pole_thickness_range: tuple[float, float] = (0.05, 0.15)
    """Range (min, max) for each pole's square cross-section side length."""

    pole_spacing_range: tuple[float, float] = (0.3, 0.8)
    """Range (min, max) for the gap (in x) between consecutive poles."""

    pole_height: float = 1.5
    """Height of poles above the highest stair's top surface."""


@configclass
class MeshXStairsDownTerrainCfg(MeshXStairsUpTerrainCfg):
    """Stairs descending in +x, vertically centered at z=0. Same fields as up."""

    function = x_stairs_down_terrain
