from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import trimesh

from isaaclab.terrains.trimesh.utils import make_plane

if TYPE_CHECKING:
    from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.flat_cfg import (
        MeshFlatTerrainCfg,
    )


def flat_terrain(
    difficulty: float, cfg: "MeshFlatTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict]:
    """Flat plane filling the sub-terrain cell, origin at the cell center on z=0.

    Returns the meta-system 3-tuple (meshes, origin, meta_data) so it plugs
    directly into :class:`MetaTerrainGenerator`. Coordinates are in
    **local-corner** space: x in ``[0, size_x]``, y in ``[0, size_y]`` —
    :class:`MetaTerrainGenerator._get_terrain_mesh` re-centres the mesh and
    origin afterwards.

    Args:
        difficulty: Unused for a flat sub-terrain.
        cfg: The flat sub-terrain configuration.

    Returns:
        Tuple of (meshes, origin, meta_data) in local corner coordinates.
    """
    del difficulty
    mesh = make_plane(cfg.size, height=0.0, center_zero=False)
    origin = np.array([cfg.size[0] / 2.0, cfg.size[1] / 2.0, 0.0])
    meta_data = {
        "needs_projection": False,
        "needs_directional_cmd": False,
        "is_border": False,
        # Sampling probability per skill — must sum to 1.0.
        "skill_probs": {"walk_forward": 0.45, "running": 0.45, "standing": 0.05},
    }
    return [mesh], origin, meta_data
