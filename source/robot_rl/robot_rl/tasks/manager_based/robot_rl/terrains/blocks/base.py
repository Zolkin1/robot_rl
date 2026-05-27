"""Abstract base classes for composable terrain blocks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING, dataclass, field
from typing import Any

import numpy as np
import trimesh

from isaaclab.utils import configclass


@configclass
class TerrainBlockCfg:
    """Base configuration for one composable terrain block.

    A block is a rectangular slab along the sub-terrain's x-axis. It spans the
    full y extent of the parent sub-terrain and a configurable x extent
    (``size_x``). Subclasses declare their own geometry parameters and bind
    ``class_type`` to the runtime block class.
    """

    class_type: type | str = MISSING
    """The :class:`TerrainBlock` subclass that builds the block geometry."""

    size_x: float = MISSING
    """Block extent along the sub-terrain's x-axis (m)."""

    skill_probs: dict[str, float] = MISSING
    """Per-skill sampling probabilities. Must sum to 1.0 (validated at build)."""

    difficulty_scale: float = 1.0
    """Reserved hook for future per-block difficulty scaling — unused for v1."""


@dataclass
class BlockOutput:
    """Result of :meth:`TerrainBlock.build`.

    All geometry is in sub-terrain-local-corner coordinates: ``x`` and ``y`` in
    ``[0, parent_size_x] x [0, parent_size_y]``. :class:`MetaTerrainGenerator`
    later re-centers the assembled sub-terrain mesh.
    """

    meshes: list[trimesh.Trimesh]
    """Trimesh objects representing the block's geometry."""

    origin: np.ndarray | None
    """Optional spawn point in sub-terrain-local coordinates (3,). The composer
    picks the first non-None origin across blocks (or falls back to the first
    block's center on the ground plane)."""

    aabb: tuple[float, float, float, float]
    """Sub-terrain-local AABB ``(xmin, xmax, ymin, ymax)`` of the block's
    walkable footprint."""

    skill_probs: dict[str, float]
    """Per-skill sampling probabilities for points inside this block."""

    needs_projection: bool
    """Whether MDP terms should snap to a projected point for this block."""

    needs_directional_cmd: bool
    """Whether MDP terms should override the velocity command direction."""

    entry_z: float = 0.0
    """Walkable-surface z at the block's ``xmin`` (left edge). Used by the
    composer to thread elevation across adjacent blocks (the next block's
    ``base_z`` is set to this block's ``exit_z``)."""

    exit_z: float = 0.0
    """Walkable-surface z at the block's ``xmax`` (right edge). For flat
    blocks this equals ``entry_z``; for stair-up it's ``entry_z + (N-1) *
    step_height``; for stair-down it's ``entry_z - (N-1) * step_height``."""

    walkable_aabb: tuple[float, float, float, float] | None = None
    """Optional sub-terrain-local AABB ``(xmin, xmax, ymin, ymax)`` of the
    block's *actual* walkable plate (narrower than :attr:`aabb` when the
    block's width is restricted, e.g. via ``flat_width_range`` /
    ``stair_width_range`` / ``slope_width_range``).  Drives the
    importer's debug-viz outline so the rendered rectangle hugs the real
    block geometry instead of the full sub-terrain y-extent.  ``None``
    falls back to :attr:`aabb` (legacy / full-width blocks)."""

    extras: dict[str, Any] = field(default_factory=dict)
    """Block-type-specific metadata (e.g. stair_top_centers, stair_dimension,
    num_steps, direction)."""


class TerrainBlock(ABC):
    """Abstract base class for a composable terrain block.

    Subclasses implement :meth:`build` to emit meshes, an AABB, and metadata in
    sub-terrain-local coordinates. The composer translates the block's meshes
    so its bottom-left corner sits at ``local_origin_xy``.
    """

    def __init__(self, cfg: TerrainBlockCfg, difficulty: float):
        """Initialize the block.

        Args:
            cfg: The block configuration.
            difficulty: Sub-terrain difficulty scalar in ``[0, 1]``.
        """
        self.cfg = cfg
        self.difficulty = float(difficulty)

    @abstractmethod
    def build(
        self,
        local_origin_xy: tuple[float, float],
        subterrain_size_y: float,
        base_z: float = 0.0,
    ) -> BlockOutput:
        """Emit the block's geometry and metadata.

        Args:
            local_origin_xy: Position of the block's bottom-left (``xmin``,
                ``ymin``) corner in sub-terrain-local coordinates.
            subterrain_size_y: y extent of the parent sub-terrain (m). The
                block must span this in full for the linear-x layout.
            base_z: Walkable-surface z at the block's ``xmin`` (left edge).
                The composer chains this from the previous block's
                ``exit_z`` so the block lattice is continuous in z.

        Returns:
            A :class:`BlockOutput` with meshes, AABB, and metadata in
            sub-terrain-local coordinates.
        """


def validate_skill_probs(probs: dict[str, float], context: str = "", tol: float = 1e-5) -> None:
    """Raise ``ValueError`` if ``probs`` is empty or does not sum to 1.0.

    Args:
        probs: The skill-name → probability mapping.
        context: Optional human-readable string included in the error message.
        tol: Allowed deviation from 1.0.
    """
    if not probs:
        raise ValueError(f"skill_probs is empty{f' ({context})' if context else ''}.")
    total = float(sum(probs.values()))
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"skill_probs sums to {total:.6f}, expected 1.0 (tol={tol})"
            f"{f' ({context})' if context else ''}. skill_probs={probs}."
        )
