from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

if TYPE_CHECKING:
    from robot_rl.tasks.manager_based.robot_rl.terrains.trimesh.stair_cfg import (
        MeshXStairsDownTerrainCfg,
        MeshXStairsUpTerrainCfg,
    )


def _make_project_fn(stair_top_centers: torch.Tensor, step_depth: float, num_steps: int):
    """Build the metadata projection closure for a stair sub-terrain.

    The closure maps batches of (x, y) points in **local terrain corner
    coordinates** (x in [0, size_x], y in [0, size_y]) to the top-surface
    center of the most-forward (largest x) stair the points cover.

    Args:
        stair_top_centers: Tensor of shape (num_steps, 3). Per-stair top center
            (x_c, y_c, z_top) in local corner coordinates.
        step_depth: Step depth in x (m). Used for the integer-division lookup.
        num_steps: Number of stairs along x.

    Returns:
        Callable that takes a tensor of shape (N, k, 2) and returns a tensor
        of shape (N, 3).
    """

    def project(points: torch.Tensor) -> torch.Tensor:
        if points.dim() != 3 or points.shape[-1] != 2:
            raise ValueError(
                f"project expects points of shape (N, k, 2); got {tuple(points.shape)}"
            )
        centers = stair_top_centers.to(points.device)
        x = points[..., 0]
        idx = torch.div(x, step_depth, rounding_mode="floor").long()
        idx = idx.clamp(0, num_steps - 1)
        leading = idx.max(dim=-1).values
        return centers[leading]

    return project


def _add_walls(
    meshes: list[trimesh.Trimesh],
    cfg,
    *,
    size_x: float,
    y_center: float,
    stair_width: float,
    z_max: float,
    z_min: float,
    step_height: float,
) -> None:
    """Append solid sidewall boxes to ``meshes`` (full stair length)."""
    wall_z_bot = z_min - step_height
    wall_z_top = z_max + cfg.wall_height
    wall_h = wall_z_top - wall_z_bot
    wall_cz = (wall_z_top + wall_z_bot) / 2.0

    def make_one(left: bool) -> None:
        sgn = -1.0 if left else 1.0
        cy = y_center + sgn * (stair_width / 2.0 + cfg.wall_thickness / 2.0)
        dims = (size_x, cfg.wall_thickness, wall_h)
        pos = (size_x / 2.0, cy, wall_cz)
        meshes.append(trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos)))

    if np.random.random() < 0.5:
        make_one(True)
        make_one(False)
    else:
        make_one(np.random.random() < 0.5)


def _add_poles(
    meshes: list[trimesh.Trimesh],
    cfg,
    *,
    size_x: float,
    y_center: float,
    stair_width: float,
    z_max: float,
    z_min: float,
    step_height: float,
) -> None:
    """Append vertical pole pillars along the stair edges."""
    pole_z_bot = z_min - step_height
    pole_z_top = z_max + cfg.pole_height
    pole_h = pole_z_top - pole_z_bot
    pole_cz = (pole_z_top + pole_z_bot) / 2.0

    def march(left: bool) -> None:
        sgn = -1.0 if left else 1.0
        current_x = 0.0
        while current_x < size_x:
            thickness = float(np.random.uniform(*cfg.pole_thickness_range))
            spacing = float(np.random.uniform(*cfg.pole_spacing_range))
            if current_x + thickness > size_x:
                break
            cx = current_x + thickness / 2.0
            cy = y_center + sgn * (stair_width / 2.0 + thickness / 2.0)
            dims = (thickness, thickness, pole_h)
            pos = (cx, cy, pole_cz)
            meshes.append(
                trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))
            )
            current_x += thickness + spacing

    if np.random.random() < 0.5:
        march(True)
        march(False)
    else:
        march(np.random.random() < 0.5)


def _build_stairs(
    cfg,
    going_up: bool,
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict]:
    """Shared body for the up and down stair generators.

    The function works in **local corner coordinates** (x in [0, size_x], y in
    [0, size_y]); :class:`MetaTerrainGenerator` re-centers the mesh and origin
    afterwards. The projection callable in metadata also operates in this
    frame.
    """
    size_x = float(cfg.size[0])
    size_y = float(cfg.size[1])

    # 1. Sample step (height, depth) and stair width.
    options = list(cfg.step_dim_options)
    if not options:
        raise ValueError("step_dim_options must contain at least one (height, depth) tuple")
    step_height, step_depth_nominal = options[np.random.randint(len(options))]
    step_height = float(step_height)
    stair_width = float(np.random.uniform(*cfg.stair_width_range))

    # 2. Stairs fill the whole x; rescale step_depth so they tile exactly.
    num_steps = max(1, int(round(size_x / float(step_depth_nominal))))
    step_depth = size_x / num_steps

    # 3. Vertical anchoring.  Default: centre the staircase around z = 0.
    # If ``start_z_zero`` is set on the cfg, anchor the lowest stair's top at
    # z = 0 so the whole staircase rises above z = 0 (and the env origin can
    # spawn at ground level on the first stair).
    if getattr(cfg, "start_z_zero", False):
        z_lowest_top = 0.0
        z_highest_top = (num_steps - 1) * step_height
    else:
        half_range = (num_steps - 1) * step_height / 2.0
        z_lowest_top = -half_range
        z_highest_top = half_range
    y_center = size_y / 2.0

    # 4. Floating treads vs solid step boxes.
    is_floating = np.random.random() < cfg.float_prob
    floating_thickness = (
        float(np.random.uniform(*cfg.float_thick_range)) if is_floating else 0.0
    )

    meshes: list[trimesh.Trimesh] = []
    stair_top_centers = np.zeros((num_steps, 3), dtype=np.float32)

    for i in range(num_steps):
        # Order along +x: stair 0 is at x=0. For up stairs stair 0 is lowest; for
        # down stairs stair 0 is highest. The middle stair always sits at z=0.
        if going_up:
            top_z = z_lowest_top + i * step_height
        else:
            top_z = z_highest_top - i * step_height

        cx = (i + 0.5) * step_depth
        cy = y_center
        stair_top_centers[i] = (cx, cy, top_z)

        if is_floating:
            dims = (step_depth, stair_width, floating_thickness)
            cz = top_z - floating_thickness / 2.0
        else:
            # Solid step: extend down to one step below the lowest stair top so
            # neighbouring boxes always overlap with no visible gap underneath.
            box_bottom = z_lowest_top - step_height
            box_h = top_z - box_bottom
            dims = (step_depth, stair_width, box_h)
            cz = (top_z + box_bottom) / 2.0

        meshes.append(
            trimesh.creation.box(dims, trimesh.transformations.translation_matrix((cx, cy, cz)))
        )

    # 5. Walls / poles (mutually exclusive — wall takes priority).
    wall_kwargs = dict(
        size_x=size_x,
        y_center=y_center,
        stair_width=stair_width,
        z_max=z_highest_top,
        z_min=z_lowest_top,
        step_height=step_height,
    )
    if np.random.random() < cfg.wall_prob:
        if np.random.random() < cfg.pole_prob:
            _add_poles(meshes, cfg, **wall_kwargs)
        else:
            _add_walls(meshes, cfg, **wall_kwargs)

    # 6. Env origin — middle of a stair top, ~origin_distance_from_back from x=0.
    origin_idx = int(cfg.origin_distance_from_back // step_depth)
    origin_idx = max(0, min(num_steps - 1, origin_idx))
    origin = np.array(stair_top_centers[origin_idx], dtype=np.float64)

    # 7. Metadata.
    centers_t = torch.from_numpy(stair_top_centers).clone()
    meta_data = {
        "needs_projection": False,
        "needs_directional_cmd": False,
        "is_border": False,
        "stair_dimension": (step_height, step_depth, stair_width),
        "num_steps": num_steps,
        "direction": "up" if going_up else "down",
        "stair_top_centers": centers_t,
        "project": _make_project_fn(centers_t, step_depth, num_steps),
        # Sampling probability per skill — must sum to 1.0.
        "skill_probs": {"stair_up": 1.0} if going_up else {"stair_down": 1.0},
    }
    return meshes, origin, meta_data


def x_stairs_up_terrain(
    difficulty: float, cfg: "MeshXStairsUpTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict]:
    """Generate a staircase ascending in +x, vertically centered at z=0.

    Args:
        difficulty: Unused; kept for the standard sub-terrain function signature.
        cfg: Up-stairs configuration.

    Returns:
        Tuple of (meshes, origin, metadata) in local corner coordinates.
    """
    del difficulty
    return _build_stairs(cfg, going_up=True)


def x_stairs_down_terrain(
    difficulty: float, cfg: "MeshXStairsDownTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict]:
    """Generate a staircase descending in +x, vertically centered at z=0.

    Args:
        difficulty: Unused; kept for the standard sub-terrain function signature.
        cfg: Down-stairs configuration.

    Returns:
        Tuple of (meshes, origin, metadata) in local corner coordinates.
    """
    del difficulty
    return _build_stairs(cfg, going_up=False)
