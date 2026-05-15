"""Randomized composite sub-terrain: sample block types and lengths at build."""

from __future__ import annotations

import copy
from dataclasses import MISSING, field

import numpy as np
import trimesh
from isaaclab.utils import configclass

from .base import TerrainBlockCfg
from .composite import CompositeSubTerrainCfg, _LAYOUT_TOL, composite_terrain
from .flat import FlatBlockCfg


@configclass
class BlockChoice:
    """A weighted block template used by the randomized sampler.

    The ``cfg`` is treated as a template: at sampling time the generator
    deep-copies it and assigns the sampled ``size_x``. Any per-build
    randomness inside the block (stair widths, slope rise, etc.) still runs
    in the block's own ``build()`` against ``np.random`` as usual.

    Users do not need to set ``size_x`` on the template — it is sampled per
    block and would be discarded anyway. To keep the top-level configclass
    validator happy, ``__post_init__`` pre-fills ``size_x = 0.0`` on the
    template when it would otherwise be left as ``MISSING``.
    """

    cfg: TerrainBlockCfg = MISSING
    """Template block configuration. Its ``size_x`` is overridden per sample."""

    weight: float = 1.0
    """Relative sampling weight. Must be non-negative; weights are
    normalized across the choices list."""

    length_range: tuple[float, float] | None = None
    """Optional per-choice override of the global ``length_range``. ``None``
    falls back to ``RandomizedCompositeSubTerrainCfg.length_range``."""

    def __post_init__(self):
        missing_type = type(MISSING)
        if isinstance(self.cfg, missing_type):
            return
        if isinstance(getattr(self.cfg, "size_x", None), missing_type):
            self.cfg.size_x = 0.0


def _weighted_pick(choices: list[BlockChoice]) -> BlockChoice:
    """Sample one choice from ``choices`` proportional to its ``weight``.

    Args:
        choices: List of weighted block templates.

    Returns:
        The selected :class:`BlockChoice`.

    Raises:
        ValueError: If ``choices`` is empty or all weights are non-positive.
    """
    if not choices:
        raise ValueError("RandomizedCompositeSubTerrainCfg.choices is empty.")
    weights = np.array([float(c.weight) for c in choices], dtype=np.float64)
    if np.any(weights < 0.0):
        raise ValueError(
            f"RandomizedCompositeSubTerrainCfg: negative weight in choices "
            f"({weights.tolist()})."
        )
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(
            "RandomizedCompositeSubTerrainCfg: all choice weights are zero."
        )
    probs = weights / total
    idx = int(np.random.choice(len(choices), p=probs))
    return choices[idx]


def randomized_composite_terrain(
    difficulty: float, cfg: "RandomizedCompositeSubTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict]:
    """Sample a concrete block sequence and delegate to :func:`composite_terrain`.

    Args:
        difficulty: Sub-terrain difficulty scalar in ``[0, 1]``.
        cfg: The randomized composite configuration.

    Returns:
        Same tuple shape as :func:`composite_terrain`:
        ``(meshes, origin, meta_data)``.

    Raises:
        ValueError: If ``choices`` is empty, all weights are zero, or the
            length range is invalid.
    """
    if not cfg.choices:
        raise ValueError("RandomizedCompositeSubTerrainCfg.choices must be non-empty.")

    size_x = float(cfg.size[0])
    if size_x <= 0.0:
        raise ValueError(f"Sub-terrain size_x must be positive; got {size_x}.")

    global_lo, global_hi = float(cfg.length_range[0]), float(cfg.length_range[1])
    if not (0.0 < global_lo <= global_hi):
        raise ValueError(
            f"length_range must satisfy 0 < min <= max; got ({global_lo}, {global_hi})."
        )

    slots: list[TerrainBlockCfg] = []
    remaining = size_x
    while remaining > _LAYOUT_TOL:
        if cfg.force_flat_origin and not slots and cfg.origin_block_index == 0:
            lo, hi = global_lo, global_hi
            length = float(np.random.uniform(lo, hi))
            chosen: TerrainBlockCfg = FlatBlockCfg()
        else:
            choice = _weighted_pick(cfg.choices)
            if choice.length_range is not None:
                lo, hi = float(choice.length_range[0]), float(choice.length_range[1])
                if not (0.0 < lo <= hi):
                    raise ValueError(
                        f"BlockChoice.length_range must satisfy 0 < min <= max; "
                        f"got ({lo}, {hi})."
                    )
            else:
                lo, hi = global_lo, global_hi
            length = float(np.random.uniform(lo, hi))
            chosen = copy.deepcopy(choice.cfg)

        if length >= remaining or (remaining - length) < lo:
            length = remaining
        chosen.size_x = length
        slots.append(chosen)
        remaining -= length

    cfg.blocks = slots
    return composite_terrain(difficulty, cfg)


@configclass
class RandomizedCompositeSubTerrainCfg(CompositeSubTerrainCfg):
    """Composite sub-terrain whose block sequence is sampled at build.

    The sampler walks the sub-terrain's x extent, drawing block lengths
    from ``length_range`` (or the per-choice override) and block types from
    ``choices`` weighted by ``BlockChoice.weight``. The resulting list of
    blocks is written to ``self.blocks`` and the standard
    :func:`composite_terrain` builder is invoked to emit geometry + metadata.

    Randomness uses the global ``np.random`` state so the seed flow from
    :class:`MetaTerrainGenerator` continues to work without extra plumbing.
    """

    function = randomized_composite_terrain

    choices: list[BlockChoice] = MISSING
    """Block templates and their sampling weights."""

    length_range: tuple[float, float] = MISSING
    """Default ``(min, max)`` length sampled per block in meters."""

    force_flat_origin: bool = True
    """If ``True`` and ``origin_block_index == 0``, the first sampled slot
    is forced to a :class:`FlatBlockCfg` so the robot spawns on flat ground
    regardless of which type the categorical sampler picks. Its length is
    sampled from ``length_range`` as usual."""

    blocks: list[TerrainBlockCfg] = field(default_factory=list)
    """Overwritten by the sampler at build time. Leave at the default."""
