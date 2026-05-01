"""Depth-camera randomization events.

Ported from legged_locomotion_rl, adapted to IsaacLab v3.0.0-beta's
:class:`isaaclab.sensors.ray_caster.RayCasterCamera` API
(``set_intrinsic_matrices`` + per-env ``_data.intrinsic_matrices``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_camera_intrinsics(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    sensor_cfg: SceneEntityCfg,
    focal_length_distribution_params: tuple[float, float] | None = None,
    principal_point_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "gaussian"] = "uniform",
):
    """Perturb pinhole-camera intrinsics on reset.

    Operates on a :class:`MultiMeshRayCasterCamera` (or any
    :class:`RayCasterCamera`) configured with a
    :class:`PinholeCameraPatternCfg`. Snapshots the nominal K matrix on first
    use so successive resets compose against the original intrinsics, not the
    previously-jittered ones.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize. If ``None``, all envs.
        sensor_cfg: Scene entity pointing at the depth camera.
        focal_length_distribution_params: ``(low, high)`` for fx/fy. With
            ``operation="scale"``, ``(0.98, 1.02)`` ≈ ±2 %. ``None`` skips fx/fy.
        principal_point_distribution_params: ``(low, high)`` for cx/cy.
        operation: ``"scale"`` multiplies, ``"add"`` adds, ``"abs"`` replaces.
        distribution: ``"uniform"`` or ``"gaussian"``.
    """
    sensor = env.scene[sensor_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(sensor._view.count, device=sensor.device)
    n = len(env_ids)

    # Snapshot the nominal K matrix on first call so subsequent randomisations
    # compose against the original (not the most recently sampled) intrinsics.
    if not hasattr(sensor, "_nominal_intrinsic_matrices"):
        sensor._nominal_intrinsic_matrices = sensor._data.intrinsic_matrices.clone()

    K = sensor._nominal_intrinsic_matrices[env_ids].clone()

    if focal_length_distribution_params is not None:
        low, high = focal_length_distribution_params
        noise = _sample_noise(n, 2, low, high, distribution, sensor.device)
        K[:, 0, 0] = _apply_op(operation, K[:, 0, 0], noise[:, 0])  # fx
        K[:, 1, 1] = _apply_op(operation, K[:, 1, 1], noise[:, 1])  # fy

    if principal_point_distribution_params is not None:
        low, high = principal_point_distribution_params
        noise = _sample_noise(n, 2, low, high, distribution, sensor.device)
        K[:, 0, 2] = _apply_op(operation, K[:, 0, 2], noise[:, 0])  # cx
        K[:, 1, 2] = _apply_op(operation, K[:, 1, 2], noise[:, 1])  # cy

    # NOTE: ``env_ids`` is keyword-only in
    # ``RayCasterCamera.set_intrinsic_matrices(matrices, focal_length=1.0, env_ids=None)``.
    # Passing it positionally binds it to ``focal_length`` instead.
    sensor.set_intrinsic_matrices(K, env_ids=env_ids)


def _sample_noise(n: int, dim: int, low: float, high: float, distribution: str, device) -> torch.Tensor:
    if distribution == "uniform":
        return torch.empty(n, dim, device=device).uniform_(low, high)
    if distribution == "gaussian":
        mean = (low + high) / 2.0
        std = (high - low) / 2.0
        return torch.normal(mean, std, size=(n, dim), device=device)
    raise ValueError(f"Unsupported distribution: {distribution!r}")


def _apply_op(operation: str, base: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    if operation == "scale":
        return base * noise
    if operation == "add":
        return base + noise
    if operation == "abs":
        return noise
    raise ValueError(f"Unsupported operation: {operation!r}")
