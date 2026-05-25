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
        the platform sits flush with whatever neighbor came before it. If
        :attr:`FlatBlockCfg.flat_width_range` is set, the walkable slab is
        narrower than the parent sub-terrain in y: its width is sampled
        uniformly from the range and the slab is centered in y, leaving empty
        strips on either side (mirroring :class:`SlopeBlock`'s
        ``slope_width_range`` pattern).

        Args:
            local_origin_xy: Bottom-left corner of the block in sub-terrain
                coordinates.
            subterrain_size_y: y extent of the parent sub-terrain.
            base_z: Elevation of the flat plate.

        Returns:
            :class:`BlockOutput` with one mesh, an AABB covering the parent
            sub-terrain footprint (matching :class:`SlopeBlock` so block
            membership tests behave consistently across narrower blocks), the
            default flat skill distribution, and ``entry_z = exit_z = base_z``.

        Raises:
            ValueError: If ``flat_width_range`` is set with a non-positive
                lower bound, ``min > max``, or a value exceeding
                ``subterrain_size_y``.
        """
        validate_skill_probs(self.cfg.skill_probs, context=type(self).__name__)

        size_x = float(self.cfg.size_x)
        size_y = float(subterrain_size_y)
        x0, y0 = float(local_origin_xy[0]), float(local_origin_xy[1])

        if self.cfg.flat_width_range is None:
            mesh_size_y = size_y
            mesh_y0 = y0
        else:
            lo, hi = float(self.cfg.flat_width_range[0]), float(self.cfg.flat_width_range[1])
            if not (0.0 < lo <= hi):
                raise ValueError(
                    f"FlatBlockCfg.flat_width_range must satisfy 0 < min <= max; "
                    f"got ({lo}, {hi})."
                )
            if hi > size_y:
                raise ValueError(
                    f"FlatBlockCfg.flat_width_range max ({hi}) exceeds parent "
                    f"sub-terrain size_y ({size_y})."
                )
            mesh_size_y = float(np.random.uniform(lo, hi))
            mesh_y0 = y0 + (size_y - mesh_size_y) / 2.0

        mesh = make_plane((size_x, mesh_size_y), height=0.0, center_zero=False)
        mesh.apply_transform(trimesh.transformations.translation_matrix((x0, mesh_y0, base_z)))

        return BlockOutput(
            meshes=[mesh],
            origin=np.array([x0 + size_x / 2.0, y0 + size_y / 2.0, base_z]),
            aabb=(x0, x0 + size_x, y0, y0 + size_y),
            skill_probs=dict(self.cfg.skill_probs),
            needs_projection=False,
            needs_directional_cmd=False,
            entry_z=base_z,
            exit_z=base_z,
            extras={"flat_width": mesh_size_y},
        )


@configclass
class FlatBlockCfg(TerrainBlockCfg):
    """Flat slab spanning ``[0, size_x] x [0, parent_size_y]`` at z=0."""

    class_type: type = FlatBlock

    skill_probs: dict[str, float] = field(default_factory=_default_flat_skill_probs)

    flat_width_range: tuple[float, float] | None = None
    """Optional range ``(min, max)`` for the slab's walkable y-extent, sampled
    uniformly per block. ``None`` (default) makes the slab span the full
    parent sub-terrain in y. When set, the slab is centered in y with empty
    strips on either side, matching :attr:`SlopeBlockCfg.slope_width_range`.
    The max must not exceed the parent sub-terrain's y size."""
