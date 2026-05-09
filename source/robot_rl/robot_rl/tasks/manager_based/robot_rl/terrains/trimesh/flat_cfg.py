from __future__ import annotations

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.flat import flat_terrain


@configclass
class MeshFlatTerrainCfg(SubTerrainBaseCfg):
    """Flat plane filling the entire sub-terrain cell, centered at z=0.

    Mirrors the ``MeshXStairsUp/Down`` pair: returns the meta-system 3-tuple
    so it plugs directly into :class:`MetaTerrainGenerator` and exposes
    explicit ``meta_data`` (``is_border=False`` so envs can spawn here).
    """

    function = flat_terrain
