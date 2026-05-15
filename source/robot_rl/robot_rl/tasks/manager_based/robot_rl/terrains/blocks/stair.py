"""Stair terrain block (ascending or descending in +x)."""

from __future__ import annotations

from dataclasses import MISSING, field
from typing import Literal

import numpy as np
import torch
import trimesh
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils import configclass

from ..trimesh.stair import _build_stair_steps
from .base import BlockOutput, TerrainBlock, TerrainBlockCfg, validate_skill_probs


def _default_stair_up_skill_probs() -> dict[str, float]:
    """Stair-up single-skill distribution."""
    return {"stair_up": 1.0}


class StairBlock(TerrainBlock):
    """Staircase block emitting per-tread mesh geometry and metadata."""

    cfg: "StairBlockCfg"

    def build(
        self,
        local_origin_xy: tuple[float, float],
        subterrain_size_y: float,
        base_z: float = 0.0,
    ) -> BlockOutput:
        """Build the staircase geometry for this block.

        Shifts all stair geometry in z so the **leftmost** tread top sits at
        ``base_z``. This lets the composer thread elevation across blocks:
        an upstream stair-up block exits at some height ``H``, and a
        downstream block (flat, stair, …) receives ``base_z = H`` so its left
        edge sits flush.

        Optional flat platforms can be added before and after the stairs
        (``start_platform_length`` / ``end_platform_length``). The platforms
        eat into the block's x extent — the stairs occupy the middle
        ``size_x - start_platform_length - end_platform_length`` and the
        platforms sit at the entry / exit elevation respectively.

        Args:
            local_origin_xy: Bottom-left corner of the block in sub-terrain
                coordinates.
            subterrain_size_y: y extent of the parent sub-terrain.
            base_z: Walkable-surface z at the block's left edge. For
                stair-up this is the lowest tread; for stair-down this is
                the highest tread.

        Returns:
            :class:`BlockOutput` with stair meshes, optional platform meshes,
            AABB, and stair-specific extras. ``entry_z = base_z``; ``exit_z``
            ascends or descends by ``(num_steps - 1) * step_height`` depending
            on direction. Extras include ``start_platform_xmax`` and
            ``end_platform_xmin`` so the importer's projection can route
            platform points to the right z.

        Raises:
            ValueError: If platform lengths leave no positive x extent for
                the stairs themselves.
        """
        validate_skill_probs(self.cfg.skill_probs, context=type(self).__name__)

        size_x = float(self.cfg.size_x)
        size_y = float(subterrain_size_y)
        x0, y0 = float(local_origin_xy[0]), float(local_origin_xy[1])
        going_up = self.cfg.direction == "up"

        start_plat_len = float(self.cfg.start_platform_length)
        end_plat_len = float(self.cfg.end_platform_length)
        if start_plat_len < 0.0 or end_plat_len < 0.0:
            raise ValueError(
                f"StairBlock platform lengths must be non-negative; got "
                f"start={start_plat_len}, end={end_plat_len}."
            )
        stair_size_x = size_x - start_plat_len - end_plat_len
        if stair_size_x <= 0.0:
            raise ValueError(
                f"StairBlock platforms (start={start_plat_len}, "
                f"end={end_plat_len}) leave no positive x extent for the "
                f"stairs (size_x={size_x})."
            )

        # The stairs themselves occupy [stair_x0, stair_x0 + stair_size_x).
        stair_x0 = x0 + start_plat_len

        (
            meshes,
            stair_top_centers,
            step_height,
            step_depth,
            num_steps,
            stair_width,
            partial_step_length,
        ) = _build_stair_steps(
            self.cfg,
            size_x=stair_size_x,
            size_y=size_y,
            going_up=going_up,
            x_offset=stair_x0,
            y_offset=y0,
        )

        # Shift everything in z so the leftmost (index 0) tread top sits at
        # ``base_z``. Works regardless of ``cfg.start_z_zero`` because we read
        # the actual leftmost z from the un-shifted output.
        leftmost_top_z = float(stair_top_centers[0, 2])
        z_shift = base_z - leftmost_top_z
        if z_shift != 0.0:
            transform = trimesh.transformations.translation_matrix((0.0, 0.0, z_shift))
            for mesh in meshes:
                mesh.apply_transform(transform)
            stair_top_centers[:, 2] += z_shift

        # entry_z is the leftmost tread top (now equal to base_z);
        # exit_z is the rightmost tread top.
        entry_z = float(stair_top_centers[0, 2])
        exit_z = float(stair_top_centers[-1, 2])

        # Optional start platform at entry_z.
        if start_plat_len > 0.0:
            start_mesh = make_plane((start_plat_len, size_y), height=0.0, center_zero=False)
            start_mesh.apply_transform(trimesh.transformations.translation_matrix(
                (x0, y0, entry_z)
            ))
            # Insert at the front so the platform's geometry sorts naturally before
            # the stairs (purely cosmetic; trimesh.util.concatenate doesn't care).
            meshes.insert(0, start_mesh)

        # Optional end platform at exit_z.
        end_x0 = stair_x0 + stair_size_x
        if end_plat_len > 0.0:
            end_mesh = make_plane((end_plat_len, size_y), height=0.0, center_zero=False)
            end_mesh.apply_transform(trimesh.transformations.translation_matrix(
                (end_x0, y0, exit_z)
            ))
            meshes.append(end_mesh)

        # Spawn origin tread index, measured from the stair region's xmin
        # (i.e. ignoring the start platform — the user's
        # ``origin_distance_from_back`` is from where the stairs begin).
        origin_idx = int(self.cfg.origin_distance_from_back // step_depth)
        origin_idx = max(0, min(num_steps - 1, origin_idx))
        origin = np.array(stair_top_centers[origin_idx], dtype=np.float64)

        centers_t = torch.from_numpy(stair_top_centers).clone()
        return BlockOutput(
            meshes=meshes,
            origin=origin,
            aabb=(x0, x0 + size_x, y0, y0 + size_y),
            skill_probs=dict(self.cfg.skill_probs),
            needs_projection=True,
            needs_directional_cmd=True,
            entry_z=entry_z,
            exit_z=exit_z,
            extras={
                "is_stair": True,
                "stair_top_centers": centers_t,
                "stair_dimension": (step_height, step_depth, stair_width),
                "num_steps": num_steps,
                "direction": self.cfg.direction,
                "partial_step_length": partial_step_length,
                # Sub-terrain-local x at the boundaries between platforms and
                # stair treads. Used by the projection to route platform
                # points to a flat-z return instead of a tread-center snap.
                # When the corresponding platform length is 0, these equal
                # the stair region's edge so the projection's range checks
                # become no-ops.
                "start_platform_xmax": stair_x0,
                "end_platform_xmin": end_x0,
            },
        )


@configclass
class StairBlockCfg(TerrainBlockCfg):
    """One staircase span inside a composite sub-terrain.

    Parameters mirror :class:`MeshXStairsUpTerrainCfg` minus the redundant
    ``size`` field — the block inherits its y extent from the composite
    parent and uses its own ``size_x`` for the x extent.
    """

    class_type: type = StairBlock

    direction: Literal["up", "down"] = "up"
    """``"up"`` for stair 0 at the lowest tread (ascending in +x); ``"down"``
    for stair 0 at the highest tread (descending in +x)."""

    skill_probs: dict[str, float] = field(default_factory=_default_stair_up_skill_probs)
    """Per-skill distribution. Defaults to ``{"stair_up": 1.0}`` (override to
    ``{"stair_down": 1.0}`` for descending blocks)."""

    step_dim_options: list[tuple[float, float]] = MISSING
    """List of ``(step_height, step_depth)`` tuples sampled uniformly per block."""

    stair_width_range: tuple[float, float] = MISSING
    """Range ``(min, max)`` for stair width along y, sampled uniformly."""

    origin_distance_from_back: float = 1.0
    """x-distance from the block's ``xmin`` used to pick the spawn-origin tread
    when this block is the composite's ``origin_block_index``."""

    start_platform_length: float = 0.0
    """Length (m) of an optional flat platform at the block's left edge,
    sitting at ``entry_z``. The platform eats into the block's x extent so
    the stairs occupy ``size_x - start_platform_length - end_platform_length``
    in the middle. Set to 0 to disable."""

    end_platform_length: float = 0.0
    """Length (m) of an optional flat platform at the block's right edge,
    sitting at ``exit_z`` (the last tread's height). Same x-extent rules as
    :attr:`start_platform_length`. Set to 0 to disable."""

    float_prob: float = 0.0
    """Probability that stairs are floating thin treads instead of solid blocks."""

    float_thick_range: tuple[float, float] = (0.025, 0.075)
    """Thickness range ``(min, max)`` for floating treads."""

    wall_prob: float = 0.0
    """Probability of placing sidewalls along the block."""

    wall_thickness: float = 0.1
    """Thickness in y of each solid sidewall."""

    wall_height: float = 1.5
    """Height of walls above the highest tread."""

    pole_prob: float = 0.0
    """Conditional probability (given walls) of poles instead of solid walls."""

    pole_thickness_range: tuple[float, float] = (0.05, 0.15)
    """Side length range for square pole cross-sections."""

    pole_spacing_range: tuple[float, float] = (0.3, 0.8)
    """Gap range between consecutive poles along x."""

    pole_height: float = 1.5
    """Height of poles above the highest tread."""

    start_z_zero: bool = True
    """If ``True`` (default), the lowest tread top sits at z=0. If ``False``
    the staircase is vertically centred around z=0."""

    allow_partial_last_step: bool = False
    """If ``True``, the sampled ``(step_height, step_depth)`` are taken at
    face value: the stair span is tiled with full-depth treads and the
    trailing leftover (if any) becomes a single partial-depth tread at the
    last position. If ``False`` (default), the nominal ``step_depth`` is
    rescaled so the treads tile the span exactly."""
