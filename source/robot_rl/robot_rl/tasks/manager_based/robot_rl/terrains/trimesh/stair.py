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

_LAYOUT_TOL = 1e-5


def _make_project_fn(stair_top_centers: torch.Tensor, step_depth: float, num_steps: int):
    """Build the metadata projection closure for a stair sub-terrain.

    The closure maps a batch of (x, y) points in **local terrain corner
    coordinates** (x in [0, size_x], y in [0, size_y]) to the top-surface
    center of the stair the point sits over.

    Args:
        stair_top_centers: Tensor of shape (num_steps, 3). Per-stair top center
            (x_c, y_c, z_top) in local corner coordinates.
        step_depth: Step depth in x (m). Used for the integer-division lookup.
        num_steps: Number of stairs along x.

    Returns:
        Callable that takes a tensor of shape (N, 2) and returns a tensor of
        shape (N, 3).
    """

    def project(points: torch.Tensor) -> torch.Tensor:
        if points.dim() != 2 or points.shape[-1] != 2:
            raise ValueError(
                f"project expects points of shape (N, 2); got {tuple(points.shape)}"
            )
        centers = stair_top_centers.to(points.device)
        x = points[..., 0]
        idx = torch.div(x, step_depth, rounding_mode="floor").long()
        idx = idx.clamp(0, num_steps - 1)
        return centers[idx]

    return project


def _add_walls(
    meshes: list[trimesh.Trimesh],
    cfg,
    *,
    size_x: float,
    x_offset: float,
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
        pos = (x_offset + size_x / 2.0, cy, wall_cz)
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
    x_offset: float,
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
            cx = x_offset + current_x + thickness / 2.0
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


def _build_stair_steps(
    cfg,
    *,
    size_x: float,
    size_y: float,
    going_up: bool,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    step_dim_override: tuple[float, float] | None = None,
) -> tuple[list[trimesh.Trimesh], np.ndarray, float, float, int, float, float | None]:
    """Build the per-step mesh + tread-center geometry for one staircase span.

    Shared by the legacy ``_build_stairs`` (whole sub-terrain) and by
    :class:`StairBlock` (per-block span inside a composite sub-terrain). The
    function works in **sub-terrain-local corner coordinates** and shifts all
    geometry by ``(x_offset, y_offset)`` so callers can place the span at an
    arbitrary x-position within the parent sub-terrain.

    Args:
        cfg: Object with the stair geometry parameters (``step_dim_options``,
            ``stair_width_range``, ``start_z_zero``, ``float_prob``,
            ``float_thick_range``, ``wall_prob``, ``pole_prob``,
            ``wall_thickness``, ``wall_height``, ``pole_thickness_range``,
            ``pole_spacing_range``, ``pole_height``,
            ``allow_partial_last_step``).
        size_x: x extent of the stair span (m).
        size_y: y extent of the stair span (m).
        going_up: ``True`` for stair 0 lowest (ascending in +x); ``False`` for
            stair 0 highest.
        x_offset: x coordinate of the span's bottom-left corner.
        y_offset: y coordinate of the span's bottom-left corner.

    Returns:
        ``(meshes, stair_top_centers, step_height, step_depth, num_steps,
        stair_width, partial_step_length)``. ``stair_top_centers`` is shape
        ``(num_steps, 3)`` with coordinates already shifted by
        ``(x_offset, y_offset, 0)``. ``partial_step_length`` is ``None`` when
        the trailing tread has the same depth as every other tread, or a
        positive float ``< step_depth`` when ``allow_partial_last_step`` is
        set on the cfg and the span doesn't tile exactly.
    """
    # 1. Sample step (height, depth) and stair width.  Callers that have
    # already pre-sampled the (h, d) pair (e.g. the composite ``StairBlock``
    # builder, which needs to validate the block's ``size_x`` against the
    # chosen step_depth before delegating here) can pass it via
    # ``step_dim_override`` to skip the resample.
    if step_dim_override is not None:
        step_height, step_depth_nominal = step_dim_override
    else:
        options = list(cfg.step_dim_options)
        if not options:
            raise ValueError("step_dim_options must contain at least one (height, depth) tuple")
        step_height, step_depth_nominal = options[np.random.randint(len(options))]
    step_height = float(step_height)
    step_depth_nominal = float(step_depth_nominal)
    stair_width = float(np.random.uniform(*cfg.stair_width_range))

    # 2. Pick step depth + step count to fill the span.
    if getattr(cfg, "allow_partial_last_step", False):
        step_depth = step_depth_nominal
        num_full = int(size_x // step_depth)
        leftover = size_x - num_full * step_depth
        has_partial = leftover > _LAYOUT_TOL
        num_steps = num_full + (1 if has_partial else 0)
        if num_steps == 0:
            num_steps = 1
            step_depth = size_x
            has_partial = False
            leftover = 0.0
    else:
        num_steps = max(1, int(round(size_x / step_depth_nominal)))
        step_depth = size_x / num_steps
        has_partial = False
        leftover = 0.0
    partial_step_length: float | None = float(leftover) if has_partial else None

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
    y_center = y_offset + size_y / 2.0

    # 4. Floating treads vs solid step boxes.
    is_floating = np.random.random() < cfg.float_prob
    floating_thickness = (
        float(np.random.uniform(*cfg.float_thick_range)) if is_floating else 0.0
    )

    meshes: list[trimesh.Trimesh] = []
    stair_top_centers = np.zeros((num_steps, 3), dtype=np.float32)

    for i in range(num_steps):
        # Order along +x: stair 0 is at the span's xmin. For up stairs stair 0
        # is lowest; for down stairs stair 0 is highest.
        if going_up:
            top_z = z_lowest_top + i * step_height
        else:
            top_z = z_highest_top - i * step_height

        is_partial = has_partial and i == num_steps - 1
        this_depth = leftover if is_partial else step_depth
        if is_partial:
            cx_local = num_full * step_depth + this_depth / 2.0
        else:
            cx_local = (i + 0.5) * step_depth
        cx = x_offset + cx_local
        cy = y_center
        stair_top_centers[i] = (cx, cy, top_z)

        if is_floating:
            dims = (this_depth, stair_width, floating_thickness)
            cz = top_z - floating_thickness / 2.0
        else:
            # Solid step: extend down to one step below the lowest stair top so
            # neighbouring boxes always overlap with no visible gap underneath.
            box_bottom = z_lowest_top - step_height
            box_h = top_z - box_bottom
            dims = (this_depth, stair_width, box_h)
            cz = (top_z + box_bottom) / 2.0

        meshes.append(
            trimesh.creation.box(dims, trimesh.transformations.translation_matrix((cx, cy, cz)))
        )

    # 5. Walls / poles (mutually exclusive — wall takes priority).
    wall_kwargs = dict(
        size_x=size_x,
        x_offset=x_offset,
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

    return (
        meshes,
        stair_top_centers,
        step_height,
        step_depth,
        num_steps,
        stair_width,
        partial_step_length,
    )


def _build_stairs(
    cfg,
    going_up: bool,
) -> tuple[list[trimesh.Trimesh], np.ndarray, dict]:
    """Shared body for the up and down legacy stair generators.

    The function works in **local corner coordinates** (x in [0, size_x], y in
    [0, size_y]); :class:`MetaTerrainGenerator` re-centers the mesh and origin
    afterwards. The projection callable in metadata also operates in this
    frame.
    """
    size_x = float(cfg.size[0])
    size_y = float(cfg.size[1])

    meshes, stair_top_centers, step_height, step_depth, num_steps, stair_width, _ = (
        _build_stair_steps(cfg, size_x=size_x, size_y=size_y, going_up=going_up)
    )

    # Env origin — middle of a stair top, ~origin_distance_from_back from x=0.
    origin_idx = int(cfg.origin_distance_from_back // step_depth)
    origin_idx = max(0, min(num_steps - 1, origin_idx))
    origin = np.array(stair_top_centers[origin_idx], dtype=np.float64)

    # Metadata.
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
