# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Only re-export pure-cfg modules here. Runtime importer/generator classes
# import IsaacLab modules that transitively pull in `pxr` (via
# isaaclab.markers.visualization_markers), which must NOT happen before
# AppLauncher runs. Anything that wants those runtime classes can import
# them directly from their leaf module.
from .trimesh.stair_cfg import MeshXStairsDownTerrainCfg, MeshXStairsUpTerrainCfg
