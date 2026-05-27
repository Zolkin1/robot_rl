"""Slope terrain block (continuous +x ramp)."""

from __future__ import annotations

from dataclasses import MISSING, field
from typing import Literal

import numpy as np
import trimesh
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils import configclass

from .base import BlockOutput, TerrainBlock, TerrainBlockCfg, validate_skill_probs


def _default_slope_skill_probs() -> dict[str, float]:
    """Default skill distribution for slope blocks (normal walking)."""
    return {"walk_forward": 1.0}


def _make_slope_mesh(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    z_left: float,
    z_right: float,
    thickness: float,
) -> trimesh.Trimesh:
    """Build a tilted slab mesh whose top surface is a tilted rectangle.

    The top surface has four corners:
    ``(x0, y0, z_left), (x1, y0, z_right), (x1, y1, z_right), (x0, y1, z_left)``.
    The bottom surface mirrors the top, offset ``thickness`` below in z so the
    slab has a constant thickness measured perpendicular to z.

    Args:
        x0: x of the left edge.
        x1: x of the right edge.
        y0: y of the front (low-y) edge.
        y1: y of the back (high-y) edge.
        z_left: z of the top surface at ``x0``.
        z_right: z of the top surface at ``x1``.
        thickness: distance the bottom sits below the top, in z.

    Returns:
        A closed trimesh slab with 8 vertices and 12 triangles.
    """
    vertices = np.array(
        [
            [x0, y0, z_left],            # 0 top-left-front
            [x1, y0, z_right],           # 1 top-right-front
            [x1, y1, z_right],           # 2 top-right-back
            [x0, y1, z_left],            # 3 top-left-back
            [x0, y0, z_left - thickness],   # 4 bot-left-front
            [x1, y0, z_right - thickness],  # 5 bot-right-front
            [x1, y1, z_right - thickness],  # 6 bot-right-back
            [x0, y1, z_left - thickness],   # 7 bot-left-back
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],   # top (+z)
            [4, 6, 5], [4, 7, 6],   # bottom (-z)
            [0, 4, 5], [0, 5, 1],   # front (-y)
            [2, 6, 7], [2, 7, 3],   # back (+y)
            [3, 7, 4], [3, 4, 0],   # left (-x)
            [1, 5, 6], [1, 6, 2],   # right (+x)
        ],
        dtype=np.int64,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


class SlopeBlock(TerrainBlock):
    """Continuous ramp block ascending or descending in +x."""

    cfg: "SlopeBlockCfg"

    def build(
        self,
        local_origin_xy: tuple[float, float],
        subterrain_size_y: float,
        base_z: float = 0.0,
    ) -> BlockOutput:
        """Build the slope geometry for this block.

        The walkable top surface tilts from ``base_z`` at the block's left
        edge to ``base_z + signed_rise`` at the right edge, where
        ``signed_rise = +rise`` for ``direction == "up"`` and ``-rise`` for
        ``"down"``. Optional flat platforms can be added before / after the
        ramp.

        Args:
            local_origin_xy: Bottom-left corner of the block in sub-terrain
                coordinates.
            subterrain_size_y: y extent of the parent sub-terrain.
            base_z: Walkable-surface z at the block's left edge.

        Returns:
            :class:`BlockOutput` with slope meshes, optional platform meshes,
            AABB, and slope-specific extras.

        Raises:
            ValueError: If platform lengths leave no positive x extent for
                the slope itself.
        """
        validate_skill_probs(self.cfg.skill_probs, context=type(self).__name__)

        size_x = float(self.cfg.size_x)
        size_y = float(subterrain_size_y)
        x0, y0 = float(local_origin_xy[0]), float(local_origin_xy[1])

        start_plat_len = float(self.cfg.start_platform_length)
        end_plat_len = float(self.cfg.end_platform_length)
        if start_plat_len < 0.0 or end_plat_len < 0.0:
            raise ValueError(
                f"SlopeBlock platform lengths must be non-negative; got "
                f"start={start_plat_len}, end={end_plat_len}."
            )
        slope_size_x = size_x - start_plat_len - end_plat_len
        if slope_size_x <= 0.0:
            raise ValueError(
                f"SlopeBlock platforms (start={start_plat_len}, "
                f"end={end_plat_len}) leave no positive x extent for the "
                f"slope (size_x={size_x})."
            )

        slope_x0 = x0 + start_plat_len
        slope_x1 = slope_x0 + slope_size_x

        rise_mag = float(np.random.uniform(*self.cfg.rise_range))
        signed_rise = rise_mag if self.cfg.direction == "up" else -rise_mag
        slope_width = float(np.random.uniform(*self.cfg.slope_width_range))

        y_center = y0 + size_y / 2.0
        slope_y0 = y_center - slope_width / 2.0
        slope_y1 = y_center + slope_width / 2.0

        entry_z = float(base_z)
        exit_z = float(base_z + signed_rise)
        thickness = float(self.cfg.thickness)

        meshes: list[trimesh.Trimesh] = []

        if start_plat_len > 0.0:
            start_mesh = make_plane((start_plat_len, size_y), height=0.0, center_zero=False)
            start_mesh.apply_transform(trimesh.transformations.translation_matrix(
                (x0, y0, entry_z)
            ))
            meshes.append(start_mesh)

        slope_mesh = _make_slope_mesh(
            x0=slope_x0,
            x1=slope_x1,
            y0=slope_y0,
            y1=slope_y1,
            z_left=entry_z,
            z_right=exit_z,
            thickness=thickness,
        )
        meshes.append(slope_mesh)

        end_x0 = slope_x1
        if end_plat_len > 0.0:
            end_mesh = make_plane((end_plat_len, size_y), height=0.0, center_zero=False)
            end_mesh.apply_transform(trimesh.transformations.translation_matrix(
                (end_x0, y0, exit_z)
            ))
            meshes.append(end_mesh)

        origin = np.array(
            [x0 + size_x / 2.0, y_center, (entry_z + exit_z) / 2.0],
            dtype=np.float64,
        )

        return BlockOutput(
            meshes=meshes,
            origin=origin,
            aabb=(x0, x0 + size_x, y0, y0 + size_y),
            walkable_aabb=(x0, x0 + size_x, slope_y0, slope_y1),
            skill_probs=dict(self.cfg.skill_probs),
            needs_projection=False,
            needs_directional_cmd=True,
            entry_z=entry_z,
            exit_z=exit_z,
            extras={
                "is_slope": True,
                "slope_dimension": (rise_mag, slope_size_x, slope_width),
                "direction": self.cfg.direction,
                "start_platform_xmax": slope_x0,
                "end_platform_xmin": end_x0,
                "slope_linear_z": (slope_x0, slope_x1, entry_z, exit_z),
            },
        )


@configclass
class SlopeBlockCfg(TerrainBlockCfg):
    """Continuous ramp block (ascending or descending in +x).

    Mirrors :class:`StairBlockCfg`'s general shape (direction, optional flat
    landings, randomized geometry sampled per build).
    """

    class_type: type = SlopeBlock

    direction: Literal["up", "down"] = "up"
    """``"up"`` ascends in +x; ``"down"`` descends in +x."""

    skill_probs: dict[str, float] = field(default_factory=_default_slope_skill_probs)
    """Per-skill distribution. Defaults to ``{"walk_forward": 1.0}`` until a
    dedicated slope skill is added to the env's skill set."""

    rise_range: tuple[float, float] = MISSING
    """Range ``(min, max)`` for the total z rise across the slope span,
    sampled uniformly per block. Always positive; ``direction`` decides the
    sign."""

    slope_width_range: tuple[float, float] = MISSING
    """Range ``(min, max)`` for the walkable y-extent of the slope, sampled
    uniformly per block."""

    thickness: float = 0.5
    """Slab thickness (m) measured below the top surface in z."""

    start_platform_length: float = 0.0
    """Length (m) of an optional flat platform at the block's left edge,
    sitting at ``entry_z``. Eats into ``size_x``. 0 disables."""

    end_platform_length: float = 0.0
    """Length (m) of an optional flat platform at the block's right edge,
    sitting at ``exit_z``. Eats into ``size_x``. 0 disables."""
