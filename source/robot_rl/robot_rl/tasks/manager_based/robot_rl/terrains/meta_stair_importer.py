"""Terrain importer that exposes a vectorized stair projection in world coords.

Stair sub-terrains generated via :mod:`robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair`
attach the per-stair geometry to their metadata (``stair_top_centers``,
``stair_dimension``). This importer collects that information across the grid
and exposes a single ``project`` callable on
:attr:`MetaTerrainImporter.terrain_meta_data` that operates in **world**
coordinates, hiding the per-cell local-to-world frame conversion from
downstream consumers (rewards, commands, ...).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from typing_extensions import override

from .meta_terrain_importer import MetaTerrainImporter

if TYPE_CHECKING:
    pass


class MetaStairTerrainImporter(MetaTerrainImporter):
    """Importer that builds a global stair-top projection from per-cell metadata.

    Replaces ``self.terrain_meta_data`` with ``{"project": <callable>}`` after
    construction. The callable signature is::

        project(points: Tensor[N, k, 2]) -> Tensor[N, 3]

    where ``points`` are world-frame ``(x, y)`` and the output is the world
    ``(x, y, z)`` of the top-surface center of the most-forward stair (largest
    stair index in ``+x``) covered by the ``k`` query points.

    Cells that did not produce stair metadata (e.g. border or non-stair
    sub-terrains) are treated as flat at ``z = 0`` and just pass through
    ``(x, y, 0)``.
    """

    @override
    def _post_process_meta_data(self):
        num_rows, num_cols = self.terrain_origins.shape[:2]
        size_x = float(self.cfg.terrain_generator.size[0])
        size_y = float(self.cfg.terrain_generator.size[1])

        # Find the largest staircase so we can pad to a single tensor.
        max_steps = 1
        for md in self.terrain_meta_data.values():
            if "stair_top_centers" in md:
                max_steps = max(max_steps, int(md["stair_top_centers"].shape[0]))

        is_stair = torch.zeros(num_rows, num_cols, dtype=torch.bool, device=self.device)
        step_depths = torch.ones(num_rows, num_cols, dtype=torch.float, device=self.device)
        num_steps_t = torch.ones(num_rows, num_cols, dtype=torch.long, device=self.device)
        # Default centers: a single (0, 0, 0) entry per cell so the gather is safe
        # even for non-stair cells (the result is overridden below).
        centers = torch.zeros(num_rows, num_cols, max_steps, 3, dtype=torch.float, device=self.device)

        for (r, c), md in self.terrain_meta_data.items():
            if "stair_top_centers" not in md:
                continue
            top_centers = md["stair_top_centers"].to(self.device, dtype=torch.float)
            ns = int(top_centers.shape[0])
            is_stair[r, c] = True
            step_depths[r, c] = float(md["stair_dimension"][1])
            num_steps_t[r, c] = ns
            centers[r, c, :ns] = top_centers

        self._stair_is_stair = is_stair
        self._stair_step_depths = step_depths
        self._stair_num_steps = num_steps_t
        self._stair_centers = centers
        self._stair_size = torch.tensor([size_x, size_y], device=self.device)
        self._stair_grid = torch.tensor([num_rows, num_cols], device=self.device, dtype=torch.long)

        # Replace metadata with the public-facing API.
        self.terrain_meta_data = {
            "project": self._project_world,
            "is_stair": is_stair,
            "stair_top_centers": centers,
            "stair_step_depths": step_depths,
            "stair_num_steps": num_steps_t,
        }

    def _world_to_subterrain(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map world ``(x, y)`` to ``(row, col)`` indices and per-cell local coords.

        The full grid is centered at the world origin (see
        :meth:`isaaclab.terrains.terrain_generator.TerrainGenerator.__init__`),
        so sub-terrain ``(r, c)`` spans
        ``[r*size_x - num_rows*size_x/2, (r+1)*size_x - num_rows*size_x/2]`` in x.
        """
        size_x, size_y = self._stair_size[0], self._stair_size[1]
        nrows, ncols = int(self._stair_grid[0]), int(self._stair_grid[1])

        rel_x = xy[..., 0] + nrows * size_x / 2.0
        rel_y = xy[..., 1] + ncols * size_y / 2.0

        r = torch.div(rel_x, size_x, rounding_mode="floor").long().clamp(0, nrows - 1)
        c = torch.div(rel_y, size_y, rounding_mode="floor").long().clamp(0, ncols - 1)

        local_x = rel_x - r.to(rel_x.dtype) * size_x
        local_y = rel_y - c.to(rel_y.dtype) * size_y
        return r, c, local_x, local_y

    def _project_world(self, points: torch.Tensor) -> torch.Tensor:
        """Project a batch of ``(N, k, 2)`` world points to leading-stair centers.

        For each row ``n``, picks the ``k``-index whose stair index is largest
        in ``+x`` (the "most-forward foot"), then returns the world ``(x, y, z)``
        center of that stair's top surface.
        """
        if points.dim() != 3 or points.shape[-1] != 2:
            raise ValueError(f"Expected points of shape (N, k, 2); got {tuple(points.shape)}")

        size_x, size_y = self._stair_size[0], self._stair_size[1]
        nrows, ncols = int(self._stair_grid[0]), int(self._stair_grid[1])

        r, c, local_x, _ = self._world_to_subterrain(points)

        step_depth_pk = self._stair_step_depths[r, c].clamp_min(1e-6)
        num_steps_pk = self._stair_num_steps[r, c]
        is_stair_pk = self._stair_is_stair[r, c]

        # Per-point stair index (clamped to the cell's range).
        stair_idx = torch.div(local_x, step_depth_pk, rounding_mode="floor").long()
        stair_idx = torch.minimum(stair_idx, num_steps_pk - 1).clamp_min(0)
        # Non-stair cells: pretend index 0 (centers default to zeros there).
        stair_idx = torch.where(is_stair_pk, stair_idx, torch.zeros_like(stair_idx))

        # Pick the most-forward k per N. Use stair_idx as the criterion; for
        # non-stair cells fall back to local_x so we don't bias toward index 0.
        criterion = torch.where(is_stair_pk, stair_idx.to(local_x.dtype), local_x)
        leading_k = criterion.argmax(dim=-1)

        n_idx = torch.arange(points.shape[0], device=points.device)
        r_lead = r[n_idx, leading_k]
        c_lead = c[n_idx, leading_k]
        idx_lead = stair_idx[n_idx, leading_k]
        is_stair_lead = is_stair_pk[n_idx, leading_k]
        local_x_lead = local_x[n_idx, leading_k]
        local_y_lead = points[n_idx, leading_k, 1] + ncols * size_y / 2.0
        local_y_lead = local_y_lead - c_lead.to(local_y_lead.dtype) * size_y

        local_center = self._stair_centers[r_lead, c_lead, idx_lead]
        # For non-stair cells, fall back to the input point at z = 0.
        non_stair_center = torch.stack(
            [local_x_lead, local_y_lead, torch.zeros_like(local_x_lead)], dim=-1
        )
        local_out = torch.where(is_stair_lead.unsqueeze(-1), local_center, non_stair_center)

        # Local (corner-coords) → world by re-adding the cell corner offset.
        offset_x = r_lead.to(local_out.dtype) * size_x - nrows * size_x / 2.0
        offset_y = c_lead.to(local_out.dtype) * size_y - ncols * size_y / 2.0
        world_x = local_out[..., 0] + offset_x
        world_y = local_out[..., 1] + offset_y
        world_z = local_out[..., 2]
        return torch.stack([world_x, world_y, world_z], dim=-1)


