"""Terrain importer for composite sub-terrains.

A composite sub-terrain (see :mod:`...terrains.blocks`) is assembled from an
ordered list of terrain blocks, each with its own footprint AABB inside the
sub-terrain cell. This importer exposes the per-block skill distributions
via :meth:`MetaCompositeTerrainImporter.skill_probs_at` so the MDP command
layer can resolve "which skill is valid at world point ``(x, y)``" on a
per-block resolution within a single sub-terrain.

It also generalises the stair projection: a cell may contain zero, one, or
multiple stair spans (one per stair block in the composite), each with its
own ``[xmin, xmax]`` extent and per-tread centers. The single-point
``project(points: (N, 3)) -> (N, 3)`` callable in :attr:`terrain_meta_data`
finds the matching span for each query point and returns the tread top
center.

Sub-terrains using the legacy schema (``MeshFlatTerrainCfg``,
``MeshXStairsUpTerrainCfg``, …) are accepted unchanged: each such cell is
treated as a one-block degenerate composite covering the full cell, so a
generator can mix legacy and composite sub-terrains in the same
``sub_terrains`` dict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from typing_extensions import override

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from .meta_terrain_importer import MetaTerrainImporter

if TYPE_CHECKING:
    from .meta_composite_importer_cfg import MetaCompositeTerrainImporterCfg

_SKILL_PROBS_TOL = 1e-5

_FLAT_OUTLINE_COLOR: tuple[float, float, float] = (0.20, 0.80, 0.30)
_STAIR_OUTLINE_COLOR: tuple[float, float, float] = (0.95, 0.55, 0.15)
_DEFAULT_BLOCK_OUTLINE_COLORS: dict[str, tuple[float, float, float]] = {
    "FlatBlock": _FLAT_OUTLINE_COLOR,
    "StairBlock": _STAIR_OUTLINE_COLOR,
    "LegacyFlat": _FLAT_OUTLINE_COLOR,
    "LegacyStair": _STAIR_OUTLINE_COLOR,
}
_FALLBACK_BLOCK_OUTLINE_COLOR: tuple[float, float, float] = (0.55, 0.55, 0.55)


class MetaCompositeTerrainImporter(MetaTerrainImporter):
    """Importer with per-block skill / projection metadata.

    Mirrors :class:`MetaStairTerrainImporter` but supports composite
    sub-terrains where a single cell can contain multiple blocks (each with
    its own AABB and skill distribution) plus zero or more stair spans.
    Legacy cell-level metadata is accepted and normalised to a single-block
    representation.
    """

    cfg: "MetaCompositeTerrainImporterCfg"

    @override
    def _post_process_meta_data(self):
        """Build per-block / per-span tensors from cell metadata.

        First runs the base post-process to populate ``is_border``,
        ``skill_to_idx``, and the cell-level ``skill_probs`` (unused by this
        importer but kept for callers that read the legacy attribute). Then
        normalises each cell's metadata into a list of blocks, allocates
        padded tensors keyed by ``(row, col, block_idx)`` and ``(row, col,
        span_idx)``, and replaces ``terrain_meta_data`` with the public
        projection API.

        Raises:
            ValueError: A block's ``skill_probs`` doesn't sum to 1.0.
            KeyError: A block declares a skill not in ``self.skill_list``.
        """
        super()._post_process_meta_data()

        nrows, ncols = self.terrain_origins.shape[:2]
        num_skills = len(self.skill_list)
        size_x = float(self.cfg.terrain_generator.size[0])
        size_y = float(self.cfg.terrain_generator.size[1])

        # Pass 1: normalise each cell to a list of per-block dicts and figure
        # out padding extents.
        normalized: dict[tuple[int, int], list[dict]] = {}
        max_blocks = 1
        max_spans = 1
        max_steps_per_span = 1

        for (r, c), md in self.terrain_meta_data.items():
            if "blocks" in md:
                blocks = list(md["blocks"])
            else:
                # Legacy cell — synthesise a single full-cell block.
                blocks = [self._legacy_cell_to_block(md, size_x, size_y)]
            normalized[(r, c)] = blocks

            max_blocks = max(max_blocks, len(blocks))
            n_stair_blocks = 0
            for b in blocks:
                if b["extras"].get("is_stair", False):
                    n_stair_blocks += 1
                    centers = b["extras"]["stair_top_centers"]
                    max_steps_per_span = max(max_steps_per_span, int(centers.shape[0]))
            max_spans = max(max_spans, max(n_stair_blocks, 1))

        # Pass 2: allocate tensors.
        # AABB sentinel: xmin=+inf, xmax=-inf, ymin=+inf, ymax=-inf so unfilled
        # slots never match the membership test.
        self._block_aabbs = torch.empty(
            (nrows, ncols, max_blocks, 4), dtype=torch.float, device=self.device
        )
        self._block_aabbs[..., 0] = float("inf")
        self._block_aabbs[..., 1] = float("-inf")
        self._block_aabbs[..., 2] = float("inf")
        self._block_aabbs[..., 3] = float("-inf")
        # Walkable AABB (per-block actual y-extent — narrower than
        # ``_block_aabbs`` when ``flat_width_range`` / ``stair_width_range``
        # / ``slope_width_range`` restrict the block's plate).  Used by
        # ``_draw_block_outlines`` so the debug-viz rectangle hugs the
        # actual block geometry instead of the full sub-terrain y.
        # Falls back to ``_block_aabbs`` when a block doesn't expose its
        # own walkable AABB (legacy blocks via ``_legacy_cell_to_block``).
        self._block_walkable_aabbs = torch.empty(
            (nrows, ncols, max_blocks, 4), dtype=torch.float, device=self.device
        )
        self._block_walkable_aabbs[..., 0] = float("inf")
        self._block_walkable_aabbs[..., 1] = float("-inf")
        self._block_walkable_aabbs[..., 2] = float("inf")
        self._block_walkable_aabbs[..., 3] = float("-inf")
        self._block_skill_probs = torch.zeros(
            (nrows, ncols, max_blocks, num_skills), dtype=torch.float, device=self.device
        )
        self._block_valid = torch.zeros(
            (nrows, ncols, max_blocks), dtype=torch.bool, device=self.device
        )
        self._block_needs_projection = torch.zeros(
            (nrows, ncols, max_blocks), dtype=torch.bool, device=self.device
        )
        self._block_needs_dir_cmd = torch.zeros(
            (nrows, ncols, max_blocks), dtype=torch.bool, device=self.device
        )
        # Walkable-surface z at each block's left (entry) and right (exit)
        # edges, in sub-terrain-local coordinates. Used by debug viz to place
        # the outline bars at the block's actual elevation.
        self._block_entry_z = torch.zeros(
            (nrows, ncols, max_blocks), dtype=torch.float, device=self.device
        )
        self._block_exit_z = torch.zeros(
            (nrows, ncols, max_blocks), dtype=torch.float, device=self.device
        )

        self._stair_span_xrange = torch.empty(
            (nrows, ncols, max_spans, 2), dtype=torch.float, device=self.device
        )
        self._stair_span_xrange[..., 0] = float("inf")
        self._stair_span_xrange[..., 1] = float("-inf")
        # x boundaries between optional start/end platforms and the stair
        # tread region within each span. Default to the span edges so the
        # platform-region checks in _project_world become no-ops for spans
        # that don't declare platforms.
        self._stair_span_start_plat_xmax = torch.empty(
            (nrows, ncols, max_spans), dtype=torch.float, device=self.device
        )
        self._stair_span_start_plat_xmax[...] = float("-inf")
        self._stair_span_end_plat_xmin = torch.empty(
            (nrows, ncols, max_spans), dtype=torch.float, device=self.device
        )
        self._stair_span_end_plat_xmin[...] = float("inf")
        self._stair_span_step_depth = torch.ones(
            (nrows, ncols, max_spans), dtype=torch.float, device=self.device
        )
        self._stair_span_num_steps = torch.ones(
            (nrows, ncols, max_spans), dtype=torch.long, device=self.device
        )
        self._stair_span_centers = torch.zeros(
            (nrows, ncols, max_spans, max_steps_per_span, 3),
            dtype=torch.float, device=self.device,
        )
        # Connecting-ground elevation at the span's left edge. With the
        # composite ``StairBlock`` now placing its first tread one step
        # *above* base_z (entry riser), this no longer matches
        # ``stair_top_centers[0, 2]`` — read it from the block's
        # ``BlockOutput.entry_z`` instead.  Used in ``_project_world`` to
        # route start-platform points to the actual flat ground (not the
        # first tread top).
        self._stair_span_entry_z = torch.zeros(
            (nrows, ncols, max_spans), dtype=torch.float, device=self.device
        )
        self._stair_span_valid = torch.zeros(
            (nrows, ncols, max_spans), dtype=torch.bool, device=self.device
        )

        is_stair_cell = torch.zeros((nrows, ncols), dtype=torch.bool, device=self.device)

        # Per-block type-name lookup keyed by (row, col, block_idx). Used by the
        # debug outline visualizer to pick a color per block.
        self._block_type_per_cell: dict[tuple[int, int, int], str] = {}

        # Pass 3: fill tensors.
        for (r, c), blocks in normalized.items():
            span_idx = 0
            for b_idx, b in enumerate(blocks):
                self._fill_block(r, c, b_idx, b)
                self._block_type_per_cell[(r, c, b_idx)] = b.get("block_type", "Unknown")
                if b["extras"].get("is_stair", False):
                    self._fill_stair_span(r, c, span_idx, b)
                    is_stair_cell[r, c] = True
                    span_idx += 1

        self._is_stair_cell = is_stair_cell

        # Resolve the block-outline palette: cfg overrides win, then the
        # built-in defaults, then a grey fallback for unknown types.
        cfg_overrides = getattr(self.cfg, "block_outline_colors", None) or {}
        unique_types = sorted({name for name in self._block_type_per_cell.values()})
        self._block_outline_palette: dict[str, tuple[float, float, float]] = {
            name: tuple(cfg_overrides.get(
                name,
                _DEFAULT_BLOCK_OUTLINE_COLORS.get(name, _FALLBACK_BLOCK_OUTLINE_COLOR),
            ))
            for name in unique_types
        }

        # Replay any deferred debug-vis request from the base class's __init__
        # (which calls set_debug_vis before this method runs).
        if getattr(self, "_pending_outline_vis", False):
            self.set_debug_vis(True)

        # Replace the public metadata dict with the projection API.
        self.terrain_meta_data = {
            "project": self._project_world,
            "is_stair": is_stair_cell,
            "stair_top_centers": self._stair_span_centers,
            "stair_step_depths": self._stair_span_step_depth,
            "stair_num_steps": self._stair_span_num_steps,
            "stair_span_valid": self._stair_span_valid,
        }

    def _legacy_cell_to_block(self, md: dict, size_x: float, size_y: float) -> dict:
        """Wrap a legacy cell-level metadata dict as a single full-cell block."""
        extras: dict = {}
        is_stair = "stair_top_centers" in md
        if is_stair:
            extras = {
                "is_stair": True,
                "stair_top_centers": md["stair_top_centers"],
                "stair_dimension": md.get("stair_dimension"),
                "num_steps": md.get("num_steps"),
                "direction": md.get("direction"),
            }
        return {
            "aabb": (0.0, size_x, 0.0, size_y),
            "skill_probs": dict(md.get("skill_probs", {})),
            "needs_projection": bool(md.get("needs_projection", False)),
            "needs_directional_cmd": bool(md.get("needs_directional_cmd", False)),
            "entry_z": 0.0,
            "exit_z": 0.0,
            "extras": extras,
            "block_type": "LegacyStair" if is_stair else "LegacyFlat",
        }

    def _fill_block(self, r: int, c: int, b_idx: int, b: dict) -> None:
        """Write one block's AABB / skill_probs / flags into the tensors."""
        aabb = b["aabb"]
        self._block_aabbs[r, c, b_idx, 0] = float(aabb[0])
        self._block_aabbs[r, c, b_idx, 1] = float(aabb[1])
        self._block_aabbs[r, c, b_idx, 2] = float(aabb[2])
        self._block_aabbs[r, c, b_idx, 3] = float(aabb[3])
        # Walkable AABB (narrower than ``aabb`` when the block restricts
        # its y-extent).  Composite-builder blocks emit this in their
        # metadata; legacy blocks (via ``_legacy_cell_to_block``) fall
        # back to the full ``aabb`` so the outline matches the cell.
        walkable = b.get("walkable_aabb", aabb)
        self._block_walkable_aabbs[r, c, b_idx, 0] = float(walkable[0])
        self._block_walkable_aabbs[r, c, b_idx, 1] = float(walkable[1])
        self._block_walkable_aabbs[r, c, b_idx, 2] = float(walkable[2])
        self._block_walkable_aabbs[r, c, b_idx, 3] = float(walkable[3])
        self._block_valid[r, c, b_idx] = True
        self._block_needs_projection[r, c, b_idx] = bool(b["needs_projection"])
        self._block_needs_dir_cmd[r, c, b_idx] = bool(b["needs_directional_cmd"])
        self._block_entry_z[r, c, b_idx] = float(b.get("entry_z", 0.0))
        self._block_exit_z[r, c, b_idx] = float(b.get("exit_z", 0.0))

        sp = b["skill_probs"]
        if not sp:
            return  # Border/etc with no skill info — leave the column at zero.
        total = float(sum(sp.values()))
        if abs(total - 1.0) > _SKILL_PROBS_TOL:
            raise ValueError(
                f"Block ({r}, {c})[{b_idx}] skill_probs sum to {total:.6f}, "
                f"expected 1.0 (tol={_SKILL_PROBS_TOL}). skill_probs={sp}."
            )
        unknown = [s for s in sp if s not in self.skill_to_idx]
        if unknown:
            raise KeyError(
                f"Block ({r}, {c})[{b_idx}] declares unknown skills {unknown}. "
                f"skill_list={self.skill_list}."
            )
        for skill, prob in sp.items():
            self._block_skill_probs[r, c, b_idx, self.skill_to_idx[skill]] = prob

    def _fill_stair_span(self, r: int, c: int, span_idx: int, b: dict) -> None:
        """Write one block's stair span into the tensors."""
        aabb = b["aabb"]
        extras = b["extras"]
        centers = extras["stair_top_centers"].to(self.device, dtype=torch.float)
        ns = int(centers.shape[0])
        step_depth = float(extras["stair_dimension"][1])

        # The span's xrange covers the FULL block (including any platforms);
        # the platform-vs-stair distinction happens via the platform boundary
        # tensors below. Spans without platforms default those boundaries to
        # the span edges so the membership tests fall through to the
        # stair-tread branch.
        block_xmin = float(aabb[0])
        block_xmax = float(aabb[1])
        self._stair_span_xrange[r, c, span_idx, 0] = block_xmin
        self._stair_span_xrange[r, c, span_idx, 1] = block_xmax
        self._stair_span_start_plat_xmax[r, c, span_idx] = float(
            extras.get("start_platform_xmax", block_xmin)
        )
        self._stair_span_end_plat_xmin[r, c, span_idx] = float(
            extras.get("end_platform_xmin", block_xmax)
        )
        self._stair_span_step_depth[r, c, span_idx] = step_depth
        self._stair_span_num_steps[r, c, span_idx] = ns
        self._stair_span_centers[r, c, span_idx, :ns] = centers
        # Connecting-ground z (composer-threaded). Falls back to the first
        # tread top z for legacy / hand-written blocks that don't expose
        # ``entry_z`` — that keeps the pre-entry-riser behavior for any
        # block that hasn't migrated yet.
        self._stair_span_entry_z[r, c, span_idx] = float(
            b.get("entry_z", float(centers[0, 2]))
        )
        self._stair_span_valid[r, c, span_idx] = True

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    @override
    def skill_probs_at(self, xy_w: torch.Tensor) -> torch.Tensor:
        """Return the per-block skill distribution at each world ``(x, y)``.

        Args:
            xy_w: World ``(x, y)`` positions with last dimension 2 and
                arbitrary leading shape.

        Returns:
            Tensor of shape ``xy_w.shape[:-1] + (num_skills,)``. For points
            outside any block (impossible in v1 since the linear layout fills
            the cell) the result is the zero vector.
        """
        rows, cols, local_x, local_y = self._cell_and_local(xy_w)
        aabbs = self._block_aabbs[rows, cols]  # (..., max_blocks, 4)
        hit = (
            (local_x.unsqueeze(-1) >= aabbs[..., 0])
            & (local_x.unsqueeze(-1) < aabbs[..., 1])
            & (local_y.unsqueeze(-1) >= aabbs[..., 2])
            & (local_y.unsqueeze(-1) < aabbs[..., 3])
        )
        block_idx = hit.float().argmax(dim=-1)
        return self._block_skill_probs[rows, cols, block_idx]

    def eligible_skills_at(self, xy_w: torch.Tensor) -> torch.Tensor:
        """Return a bool mask of skills with non-zero probability at each point."""
        return self.skill_probs_at(xy_w) > 0.0

    def _project_world(self, points: torch.Tensor) -> torch.Tensor:
        """Project a batch of world foot poses ``(x, y, z)`` to stair ``(x, y, z)``.

        The xy selects the **span** (which staircase — the one the point is
        inside, else the nearest by ``|local_x − span_center|``).  The query
        **z selects the step within that span**: the tread (or the
        fictitious entry/exit level) whose top-z is nearest ``z``.  This
        snaps the reference to the step the foot is physically resting on,
        even when the ankle's xy still sits over the flat below a riser —
        an xy-only tread index would land a whole step too low there.

        Routing:
          * Height nearest a real tread → ``(tread_center_x, ·, tread_top_z)``.
          * Height nearest the entry level (one step below tread 0, at the
            riser) → ``(span_xmin − sd/2, ·, entry_z)``, or the foot's own x
            if it is on a real start platform (in-span).
          * ``local_x`` past ``span_xmax`` → exit-side fictitious tread
            ``(span_xmax + sd/2, ·, exit_z)`` (its z equals the last tread's,
            so height alone can't pick it — gated on xy).
          * Cell with no stair span at all → passthrough ``(x, y, 0)``.

        Always takes the full ``(x, y, z)`` because its sole caller
        (:func:`apply_terrain_aware_ref`) places the reference at a contact
        gate, where the landed foot's height is always available and is the
        unambiguous signal for which step it is on.

        Args:
            points: World-frame foot positions of shape ``(N, 3)``.

        Returns:
            Tensor of shape ``(N, 3)``.
        """
        if points.dim() != 2 or points.shape[-1] != 3:
            raise ValueError(
                f"_project_world expects points of shape (N, 3); "
                f"got {tuple(points.shape)}"
            )
        query_z = points[..., 2]

        rows, cols, local_x, local_y = self._cell_and_local(points[..., :2])
        size_x = float(self.cfg.terrain_generator.size[0])
        size_y = float(self.cfg.terrain_generator.size[1])
        nrows, ncols = self.terrain_origins.shape[:2]

        spans = self._stair_span_xrange[rows, cols]      # (N, max_spans, 2)
        valid = self._stair_span_valid[rows, cols]        # (N, max_spans)
        in_span = (
            (local_x.unsqueeze(-1) >= spans[..., 0])
            & (local_x.unsqueeze(-1) < spans[..., 1])
            & valid
        )
        has_hit = in_span.any(dim=-1)
        cell_has_stair = valid.any(dim=-1)
        in_span_idx = in_span.float().argmax(dim=-1)

        # For points outside any span, pick the nearest valid span by
        # ``|local_x − span_center|``.  Invalid spans get +inf distance so
        # they're never chosen.  For cells with no valid span at all the
        # argmin is arbitrary (all +inf), but those points are routed
        # through the passthrough branch below regardless.
        span_centers = 0.5 * (spans[..., 0] + spans[..., 1])
        span_dist = (local_x.unsqueeze(-1) - span_centers).abs()
        span_dist = torch.where(
            valid, span_dist, torch.full_like(span_dist, float("inf"))
        )
        nearest_span_idx = span_dist.argmin(dim=-1)

        # Unified span_idx: in-span points use the span they're inside;
        # outside-span points use the nearest span.
        span_idx = torch.where(has_hit, in_span_idx, nearest_span_idx)

        step_depth = self._stair_span_step_depth[rows, cols, span_idx].clamp_min(1e-6)
        num_steps = self._stair_span_num_steps[rows, cols, span_idx]
        # ``entry_z`` is the connecting-ground z (from ``BlockOutput.entry_z``)
        # — for composite ``StairBlock`` this is ``base_z``, one full step
        # *below* the first tread top (the left edge is a riser).  ``exit_z``
        # is the last tread top, which doubles as the top-platform z.
        entry_z = self._stair_span_entry_z[rows, cols, span_idx]
        last_idx = (num_steps - 1).clamp_min(0)
        exit_z = self._stair_span_centers[rows, cols, span_idx, last_idx, 2]
        span_xmin = self._stair_span_xrange[rows, cols, span_idx, 0]
        span_xmax = self._stair_span_xrange[rows, cols, span_idx, 1]

        non_stair = torch.stack(
            [local_x, local_y, torch.zeros_like(local_x)], dim=-1
        )

        # Among the real treads of the chosen span, pick the one whose top-z
        # is nearest the query z; weigh that against the fictitious entry
        # level (whichever is closer in z wins).
        centers_all_z = self._stair_span_centers[rows, cols, span_idx, :, 2]  # (N, S)
        n_step_slots = centers_all_z.shape[1]
        step_ids = torch.arange(n_step_slots, device=points.device)
        tread_valid = step_ids.unsqueeze(0) < num_steps.unsqueeze(1)          # (N, S)
        d_tread = torch.where(
            tread_valid,
            (centers_all_z - query_z.unsqueeze(1)).abs(),
            torch.full_like(centers_all_z, float("inf")),
        )
        best_tread_d, best_tread_k = d_tread.min(dim=1)                        # (N,)
        tread_sel = self._stair_span_centers[rows, cols, span_idx, best_tread_k]  # (N, 3)

        # Fictitious entry level (at the riser).  Keep the foot's xy when it
        # is on a real start platform (in-span); else snap just before the riser.
        d_entry = (entry_z - query_z).abs()                                   # (N,)
        entry_x = torch.where(has_hit, local_x, span_xmin - 0.5 * step_depth)
        entry_sel = torch.stack([entry_x, local_y, entry_z], dim=-1)          # (N, 3)

        use_entry = (d_entry < best_tread_d).unsqueeze(-1)
        z_sel = torch.where(use_entry, entry_sel, tread_sel)

        # Exit-side fictitious level shares the top tread's z, so height
        # can't distinguish it — gate it on xy being past the span.
        exit_x = span_xmax + 0.5 * step_depth
        exit_sel = torch.stack([exit_x, local_y, exit_z], dim=-1)             # (N, 3)
        past_exit = (local_x >= span_xmax).unsqueeze(-1)
        z_sel = torch.where(past_exit, exit_sel, z_sel)

        local_out = torch.where(cell_has_stair.unsqueeze(-1), z_sel, non_stair)

        offset_x = rows.to(local_out.dtype) * size_x - nrows * size_x / 2.0
        offset_y = cols.to(local_out.dtype) * size_y - ncols * size_y / 2.0
        return torch.stack([
            local_out[..., 0] + offset_x,
            local_out[..., 1] + offset_y,
            local_out[..., 2],
        ], dim=-1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cell_and_local(
        self, xy_w: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map world ``(x, y)`` to ``(row, col, local_x, local_y)``.

        ``local_x`` and ``local_y`` are sub-terrain-local-corner coordinates
        (``0 <= local < size``).
        """
        size_x = float(self.cfg.terrain_generator.size[0])
        size_y = float(self.cfg.terrain_generator.size[1])
        nrows, ncols = self.terrain_origins.shape[:2]

        rows, cols = self.world_xy_to_cell(xy_w)
        rel_x = xy_w[..., 0] + nrows * size_x / 2.0
        rel_y = xy_w[..., 1] + ncols * size_y / 2.0
        local_x = rel_x - rows.to(rel_x.dtype) * size_x
        local_y = rel_y - cols.to(rel_y.dtype) * size_y
        return rows, cols, local_x, local_y

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    @override
    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Toggle frame markers (base) and per-block outline bars.

        Extends :meth:`TerrainImporter.set_debug_vis` to additionally draw a
        thin colored rectangle around every valid block in the grid, colored
        by block type (``FlatBlock``, ``StairBlock``, …). Colors come from the
        cfg override or the importer's built-in palette.

        ``TerrainImporter.__init__`` calls this method before
        :meth:`_post_process_meta_data` populates the block tensors, so the
        outline visualizer is deferred until the palette exists. The deferred
        state is replayed from the tail of :meth:`_post_process_meta_data`.

        Args:
            debug_vis: Whether to enable both visualizers.

        Returns:
            ``True`` (always succeeds).
        """
        super().set_debug_vis(debug_vis)

        # Remember the request; it may have arrived before block metadata is
        # ready (first call from ``TerrainImporter.__init__``).
        self._pending_outline_vis = debug_vis

        if not hasattr(self, "_block_outline_palette"):
            return True

        if debug_vis:
            if not hasattr(self, "_block_outline_visualizer"):
                self._block_outline_visualizer = VisualizationMarkers(
                    cfg=self._build_block_outline_markers_cfg()
                )
                self._draw_block_outlines()
            self._block_outline_visualizer.set_visibility(True)
        else:
            if hasattr(self, "_block_outline_visualizer"):
                self._block_outline_visualizer.set_visibility(False)

        return True

    def _build_block_outline_markers_cfg(self) -> VisualizationMarkersCfg:
        """Create one unit-cube prototype per block type, colored from the palette.

        Per-instance ``scales`` in :meth:`VisualizationMarkers.visualize`
        stretch each unit cube into a thin bar of the desired length.
        """
        # Stable insertion order so ``_block_type_marker_index`` is well-defined.
        self._block_type_marker_index = {
            name: i for i, name in enumerate(sorted(self._block_outline_palette))
        }
        markers = {}
        for name in self._block_type_marker_index:
            color = self._block_outline_palette[name]
            markers[name] = sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=color, roughness=1.0
                ),
            )
        return VisualizationMarkersCfg(
            prim_path="/Visuals/TerrainBlockOutlines",
            markers=markers,
        )

    def _draw_block_outlines(self) -> None:
        """Compute and submit all bar instances for every valid block.

        Each block contributes four thin bars (bottom, top, left, right). Bars
        are sized so corners overlap by ``bar_width``, giving clean joins.
        """
        if self.terrain_origins is None:
            return

        size_x = float(self.cfg.terrain_generator.size[0])
        size_y = float(self.cfg.terrain_generator.size[1])
        bar_w = float(self.cfg.block_outline_bar_width)
        bar_h = float(self.cfg.block_outline_bar_height)
        z_off = float(self.cfg.block_outline_z_offset)

        valid_cpu = self._block_valid.detach().cpu().numpy()
        # Outline uses the *walkable* AABB so narrow blocks (those with
        # ``flat_width_range`` / ``stair_width_range`` / ``slope_width_range``
        # set) render an outline that hugs the actual block plate rather
        # than the full sub-terrain y-extent.  Falls back to the full AABB
        # for legacy blocks (cell-wide).
        aabbs_cpu = self._block_walkable_aabbs.detach().cpu().numpy()
        entry_z_cpu = self._block_entry_z.detach().cpu().numpy()
        exit_z_cpu = self._block_exit_z.detach().cpu().numpy()

        translations: list[tuple[float, float, float]] = []
        scales: list[tuple[float, float, float]] = []
        marker_indices: list[int] = []

        nrows, ncols, max_blocks = valid_cpu.shape
        # Cell bottom-left in world frame, matching ``world_xy_to_cell``:
        # world_x in [r*size_x - nrows*size_x/2, ...], i.e. the grid is
        # centered at the origin.
        grid_origin_x = -nrows * size_x / 2.0
        grid_origin_y = -ncols * size_y / 2.0

        for r in range(nrows):
            for c in range(ncols):
                corner_x = grid_origin_x + r * size_x
                corner_y = grid_origin_y + c * size_y

                for b_idx in range(max_blocks):
                    if not valid_cpu[r, c, b_idx]:
                        continue
                    name = self._block_type_per_cell.get((r, c, b_idx), "Unknown")
                    marker_idx = self._block_type_marker_index.get(name)
                    if marker_idx is None:
                        # Should be impossible — palette was built from
                        # exactly the set of names in ``_block_type_per_cell``.
                        continue

                    xmin, xmax, ymin, ymax = aabbs_cpu[r, c, b_idx]
                    block_xmin_w = corner_x + float(xmin)
                    block_xmax_w = corner_x + float(xmax)
                    block_ymin_w = corner_y + float(ymin)
                    block_ymax_w = corner_y + float(ymax)
                    block_w = block_xmax_w - block_xmin_w
                    block_h = block_ymax_w - block_ymin_w
                    cx = 0.5 * (block_xmin_w + block_xmax_w)
                    cy = 0.5 * (block_ymin_w + block_ymax_w)

                    # Bar z = low end of the block's walkable surface, plus
                    # the configured offset, plus half the bar height so the
                    # bar's base sits at ``surface_z + z_off``. For flat
                    # blocks entry_z == exit_z so the bars trace the platform
                    # cleanly. For stair blocks min(entry, exit) is the bottom
                    # of the staircase, so the bars sit at the lower walkway
                    # level — useful for spotting the block from below /
                    # without the bars getting hidden inside taller geometry.
                    surface_z = min(
                        float(entry_z_cpu[r, c, b_idx]),
                        float(exit_z_cpu[r, c, b_idx]),
                    )
                    z_center = surface_z + z_off + bar_h / 2.0

                    # Bottom edge (y = ymin).
                    translations.append((cx, block_ymin_w, z_center))
                    scales.append((block_w + bar_w, bar_w, bar_h))
                    marker_indices.append(marker_idx)
                    # Top edge (y = ymax).
                    translations.append((cx, block_ymax_w, z_center))
                    scales.append((block_w + bar_w, bar_w, bar_h))
                    marker_indices.append(marker_idx)
                    # Left edge (x = xmin).
                    translations.append((block_xmin_w, cy, z_center))
                    scales.append((bar_w, block_h + bar_w, bar_h))
                    marker_indices.append(marker_idx)
                    # Right edge (x = xmax).
                    translations.append((block_xmax_w, cy, z_center))
                    scales.append((bar_w, block_h + bar_w, bar_h))
                    marker_indices.append(marker_idx)

        if not translations:
            return

        translations_np = np.asarray(translations, dtype=np.float32)
        scales_np = np.asarray(scales, dtype=np.float32)
        # Identity quaternion (x, y, z, w).
        orientations_np = np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (translations_np.shape[0], 1),
        )
        marker_indices_np = np.asarray(marker_indices, dtype=np.int32)

        self._block_outline_visualizer.visualize(
            translations=translations_np,
            orientations=orientations_np,
            scales=scales_np,
            marker_indices=marker_indices_np,
        )
