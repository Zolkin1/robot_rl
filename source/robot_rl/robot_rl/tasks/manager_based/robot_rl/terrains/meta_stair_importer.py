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

        project(points: Tensor[N, 2]) -> Tensor[N, 3]

    where ``points`` are world-frame ``(x, y)`` (typically the trajectory
    reference pose / stance-foot pose) and the output is the world
    ``(x, y, z)`` of the top-surface center of the stair the point sits over.

    Cells that did not produce stair metadata (e.g. border or non-stair
    sub-terrains) are treated as flat at ``z = 0`` and just pass through
    ``(x, y, 0)``.
    """

    @override
    def _post_process_meta_data(self):
        # First run the base post-process so ``is_border``, ``skill_probs``,
        # and ``skill_to_idx`` are populated. The stair-specific projection
        # build below relies on the per-cell metadata dict, which the base
        # impl iterates but does not mutate, so the order is safe.
        super()._post_process_meta_data()

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

        Row/col come from the shared :meth:`MetaTerrainImporter.world_xy_to_cell`
        utility; this method adds the stair-specific local-corner ``(local_x,
        local_y)`` needed by :meth:`_project_world`.
        """
        size_x, size_y = self._stair_size[0], self._stair_size[1]
        nrows, ncols = int(self._stair_grid[0]), int(self._stair_grid[1])

        r, c = self.world_xy_to_cell(xy)
        rel_x = xy[..., 0] + nrows * size_x / 2.0
        rel_y = xy[..., 1] + ncols * size_y / 2.0
        local_x = rel_x - r.to(rel_x.dtype) * size_x
        local_y = rel_y - c.to(rel_y.dtype) * size_y
        return r, c, local_x, local_y

    def _project_world(self, points: torch.Tensor) -> torch.Tensor:
        """Project a batch of ``(N, 2)`` world points to per-stair tread centers.

        For each query point, computes its sub-terrain cell ``(row, col)``,
        finds the tread index within that cell's staircase via integer
        division of the local x coordinate by the cell's step depth, and
        returns the world ``(x, y, z)`` of the tread top center. Points in
        non-stair cells pass through as ``(x, y, 0)``.
        """
        if points.dim() != 2 or points.shape[-1] != 2:
            raise ValueError(f"Expected points of shape (N, 2); got {tuple(points.shape)}")

        size_x, size_y = self._stair_size[0], self._stair_size[1]
        nrows, ncols = int(self._stair_grid[0]), int(self._stair_grid[1])

        r, c, local_x, local_y = self._world_to_subterrain(points)

        step_depth = self._stair_step_depths[r, c].clamp_min(1e-6)
        num_steps = self._stair_num_steps[r, c]
        is_stair = self._stair_is_stair[r, c]

        stair_idx = torch.div(local_x, step_depth, rounding_mode="floor").long()
        stair_idx = torch.minimum(stair_idx, num_steps - 1).clamp_min(0)
        # Non-stair cells: pretend index 0 (centers default to zeros there).
        stair_idx = torch.where(is_stair, stair_idx, torch.zeros_like(stair_idx))

        local_center = self._stair_centers[r, c, stair_idx]
        non_stair_center = torch.stack(
            [local_x, local_y, torch.zeros_like(local_x)], dim=-1
        )
        local_out = torch.where(is_stair.unsqueeze(-1), local_center, non_stair_center)

        # Local (corner-coords) → world by re-adding the cell corner offset.
        offset_x = r.to(local_out.dtype) * size_x - nrows * size_x / 2.0
        offset_y = c.to(local_out.dtype) * size_y - ncols * size_y / 2.0
        return torch.stack([
            local_out[..., 0] + offset_x,
            local_out[..., 1] + offset_y,
            local_out[..., 2],
        ], dim=-1)


