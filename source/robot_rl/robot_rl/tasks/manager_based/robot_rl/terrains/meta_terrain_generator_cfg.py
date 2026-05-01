# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

from __future__ import annotations

from dataclasses import field

from isaaclab.utils import configclass
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .meta_terrain_generator import MetaTerrainGenerator


@configclass
class MetaTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Configuration for the terrain generator."""

    class_type: type = MetaTerrainGenerator
    """The class to use for the terrain generator.

    Defaults to :class:`isaaclab.terrains.terrain_generator.TerrainGenerator`.
    """

    inter_column_borders: list[tuple[str, float]] = field(default_factory=list)
    """Configuration for flat border columns inserted between terrain segments.

    Each tuple contains:
    - after_subterrain_name: Name of the sub-terrain after which to insert the border
    - width_in_meters: Width of the border section in meters

    The borders are inserted as additional columns that robots cannot spawn on.
    The number of border columns is computed as ceil(width / size[1]).

    Example: [("wave", 20.0)] inserts a 20m flat border after the "wave" terrain columns.
    """
