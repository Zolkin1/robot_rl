"""Flat terrain block."""

from __future__ import annotations

from dataclasses import field

import numpy as np
import trimesh
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils import configclass

from .base import BlockOutput, TerrainBlock, TerrainBlockCfg, validate_skill_probs


def _default_flat_skill_probs() -> dict[str, float]:
    """Return the legacy ``flat_terrain`` skill distribution."""
    return {"walk_forward": 0.45, "running": 0.45, "standing": 0.10}


class FlatBlock(TerrainBlock):
    """Flat ground plane block."""

    cfg: "FlatBlockCfg"

    def build(
        self,
        local_origin_xy: tuple[float, float],
        subterrain_size_y: float,
        base_z: float = 0.0,
    ) -> BlockOutput:
        """Build a single flat plane at ``z=base_z`` covering the block footprint.

        The composer sets ``base_z`` to the previous block's ``exit_z`` so
        the platform sits flush with whatever neighbor came before it.

        Args:
            local_origin_xy: Bottom-left corner of the block in sub-terrain
                coordinates.
            subterrain_size_y: y extent of the parent sub-terrain.
            base_z: Elevation of the flat plate.

        Returns:
            :class:`BlockOutput` with one mesh, an AABB, the default flat
            skill distribution, and ``entry_z = exit_z = base_z``.
        """
        validate_skill_probs(self.cfg.skill_probs, context=type(self).__name__)

        size_x = float(self.cfg.size_x)
        size_y = float(subterrain_size_y)
        x0, y0 = float(local_origin_xy[0]), float(local_origin_xy[1])

        mesh = make_plane((size_x, size_y), height=0.0, center_zero=False)
        mesh.apply_transform(trimesh.transformations.translation_matrix((x0, y0, base_z)))

        return BlockOutput(
            meshes=[mesh],
            origin=np.array([x0 + size_x / 2.0, y0 + size_y / 2.0, base_z]),
            aabb=(x0, x0 + size_x, y0, y0 + size_y),
            skill_probs=dict(self.cfg.skill_probs),
            needs_projection=False,
            needs_directional_cmd=False,
            entry_z=base_z,
            exit_z=base_z,
            extras={},
        )


@configclass
class FlatBlockCfg(TerrainBlockCfg):
    """Flat slab spanning ``[0, size_x] x [0, parent_size_y]`` at z=0."""

    class_type: type = FlatBlock

    skill_probs: dict[str, float] = field(default_factory=_default_flat_skill_probs)
