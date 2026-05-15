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


@configclass
class MetaTerrainGeneratorCfg(TerrainGeneratorCfg):
    """Configuration for the terrain generator.

    ``class_type`` is a string so importing this cfg does NOT pull in the
    runtime ``MetaTerrainGenerator``. The runtime class is resolved when
    ``cfg.class_type(cfg)`` is called inside the sim, after AppLauncher.
    """

    class_type: type | str = "{DIR}.meta_terrain_generator:MetaTerrainGenerator"
    """The class to use for the terrain generator."""

    inter_column_borders: list[tuple[str, float]] = field(default_factory=list)
    """Configuration for flat border columns inserted between terrain segments.

    Each tuple contains:
    - after_subterrain_name: Name of the sub-terrain after which to insert the border
    - width_in_meters: Width of the border section in meters

    The borders are inserted as additional columns that robots cannot spawn on.
    The number of border columns is computed as ceil(width / size[1]).

    Example: [("wave", 20.0)] inserts a 20m flat border after the "wave" terrain columns.
    """

    inter_column_gaps: bool = False
    """If True, inserts one *void gap column* (no mesh at all) between every
    pair of adjacent content columns that don't already share a fixed
    ``inter_column_borders`` boundary. The robot falls through the gap.

    Each gap column spans ``size[1]`` in y. When True, ``num_cols`` must
    equal ``2 * content_num_cols - 1 - num_fixed_border_boundaries +
    total_fixed_border_cols`` (each content boundary without a fixed border
    consumes exactly one extra gap column). The generator raises a
    ``ValueError`` if the configured ``num_cols`` is inconsistent."""
