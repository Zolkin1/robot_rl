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
from .stair import StairBlockCfg


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

    # Pre-resolve a flat template from ``choices`` (first ``FlatBlockCfg``)
    # so both the optional spawn-block (``force_flat_origin=True``) and the
    # trailing absorber appended at the end of the cell inherit the user's
    # per-choice settings (``skill_probs``, ``flat_width_range``, etc.).  If
    # the spawn-block path or the trailing-absorber path needs a flat
    # template and ``choices`` doesn't have a ``FlatBlockCfg``, surface the
    # inconsistency immediately rather than silently falling back to a
    # default ``FlatBlockCfg()`` (legacy default advertises running).
    flat_template: FlatBlockCfg | None = None
    for choice in cfg.choices:
        if isinstance(choice.cfg, FlatBlockCfg):
            flat_template = choice.cfg
            break

    if cfg.force_flat_origin and cfg.origin_block_index == 0 and flat_template is None:
        raise ValueError(
            "RandomizedCompositeSubTerrainCfg.force_flat_origin=True "
            "requires at least one FlatBlockCfg in ``choices`` so the "
            "spawn block inherits the configured skill_probs / width "
            "range.  Add a FlatBlockCfg to ``choices`` or set "
            "force_flat_origin=False."
        )

    # Pre-sample a reserved trailing-flat length.  The main loop fills only
    # ``size_x - reserved_trailing``; the trailing absorber appended after
    # the loop is then ``reserved_trailing + stair_rounding_leftover``,
    # guaranteeing the cell ends in a usable flat segment of at least the
    # reserved length.  ``None`` disables the reservation and falls back to
    # the legacy behavior (trailing = stair-rounding leftover only, often
    # tiny or zero).
    reserved_trailing = 0.0
    if cfg.trailing_flat_length_range is not None:
        tf_lo = float(cfg.trailing_flat_length_range[0])
        tf_hi = float(cfg.trailing_flat_length_range[1])
        if not (0.0 < tf_lo <= tf_hi):
            raise ValueError(
                f"trailing_flat_length_range must satisfy 0 < min <= max; "
                f"got ({tf_lo}, {tf_hi})."
            )
        if tf_hi >= size_x:
            raise ValueError(
                f"trailing_flat_length_range max ({tf_hi}) must be smaller "
                f"than sub-terrain size_x ({size_x}); leave room for at "
                f"least one randomized block."
            )
        if flat_template is None:
            raise ValueError(
                "RandomizedCompositeSubTerrainCfg.trailing_flat_length_range "
                "requires at least one FlatBlockCfg in ``choices`` to use as "
                "the trailing-flat template."
            )
        reserved_trailing = float(np.random.uniform(tf_lo, tf_hi))

    target_x = size_x - reserved_trailing

    slots: list[TerrainBlockCfg] = []
    remaining = target_x
    while remaining > _LAYOUT_TOL:
        if cfg.force_flat_origin and not slots and cfg.origin_block_index == 0:
            lo, hi = global_lo, global_hi
            length = float(np.random.uniform(lo, hi))
            chosen: TerrainBlockCfg = copy.deepcopy(flat_template)
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

        # Stair blocks require ``size_x`` to be an integer multiple of the
        # chosen ``step_depth``.  Pre-sample (step_height, step_depth) from
        # the choice's ``step_dim_options`` and lock the deepcopy to that
        # single tuple so the builder reuses it.  Round the sampled length
        # *down* to ``N * step_depth``; the residual is absorbed later by
        # the trailing flat absorber appended at the end of the cell.
        if isinstance(chosen, StairBlockCfg):
            step_options = list(chosen.step_dim_options)
            if not step_options:
                raise ValueError(
                    "StairBlockCfg.step_dim_options must contain at least one "
                    "(step_height, step_depth) tuple."
                )
            sh, sd = step_options[int(np.random.randint(len(step_options)))]
            chosen.step_dim_options = [(float(sh), float(sd))]
            sd = float(sd)
            # Round the sampled length down to a multiple of sd; ensure at
            # least one tread, and clamp to the remaining x extent.
            n_treads = max(1, int(length // sd))
            length = n_treads * sd
            if length > remaining + _LAYOUT_TOL:
                n_treads = int(remaining // sd)
                if n_treads < 1:
                    # Not enough room for a single tread of this step_depth.
                    # Break the loop and let the trailing absorber take the
                    # remaining x.
                    break
                length = n_treads * sd
        else:
            # Non-stair: keep the legacy end-of-cell stretch policy so the
            # cell tiles exactly when stair adjustments aren't in play.  The
            # trailing absorber below also covers any residual that this
            # branch can't reach (e.g. when the last block is a stair).
            if length >= remaining or (remaining - length) < lo:
                length = remaining

        chosen.size_x = length
        slots.append(chosen)
        remaining -= length

    # Append a single trailing flat block sized to ``reserved_trailing +
    # stair_rounding_leftover``.  When ``trailing_flat_length_range`` is
    # unset (legacy), reserved_trailing is 0 and the trailing flat covers
    # only the stair-rounding remainder (may be 0 itself, in which case
    # the absorber is skipped).
    total_used = float(sum(b.size_x for b in slots))
    final_trailing = size_x - total_used
    if final_trailing > _LAYOUT_TOL:
        if flat_template is None:
            raise ValueError(
                "RandomizedCompositeSubTerrainCfg sampled a sequence whose "
                "stair rounding left a non-zero trailing remainder, but "
                "``choices`` contains no FlatBlockCfg to absorb it.  Add a "
                "FlatBlockCfg to ``choices``."
            )
        trailing = copy.deepcopy(flat_template)
        trailing.size_x = final_trailing
        slots.append(trailing)

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

    trailing_flat_length_range: tuple[float, float] | None = None
    """Optional ``(min, max)`` length (m) of a guaranteed trailing flat
    segment appended at the very end of every sampled cell.  When set, the
    sampler reserves a uniformly-sampled length from this range at the
    start, fills the rest of the cell with random blocks, and emits a
    final :class:`FlatBlockCfg` (template copied from the first
    ``FlatBlockCfg`` in ``choices``) sized to ``reserved + stair_rounding_leftover``.
    Guarantees a usable flat landing at the cell's right edge, large
    enough for the robot to stand on.  Requires at least one
    ``FlatBlockCfg`` in ``choices``.  ``None`` (default) keeps the legacy
    behavior where the trailing flat is just the stair-rounding leftover
    (often <step_depth and sometimes 0)."""

    blocks: list[TerrainBlockCfg] = field(default_factory=list)
    """Overwritten by the sampler at build time. Leave at the default."""
