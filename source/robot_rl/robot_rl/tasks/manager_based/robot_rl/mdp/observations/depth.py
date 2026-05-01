"""Depth observation wrappers around ``MultiMeshRayCasterCamera``.

These functions consume the camera's native ``distance_to_image_plane`` data
type — already projected onto the camera z-axis — and apply the same
``(d - offset) * scale`` normalisation used by the legged_locomotion_rl
training pipeline. The normalisation lives on the obs term (rather than on
the sensor) so the matching :class:`DepthNoiseCfg` can compose with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def depth_image(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.75,
    scale: float = 0.5,
    min_dist: float = 0.2,
    max_dist: float = 9.0,
    min_clip_val: float = 0.0,
    max_clip_val: float = 0.0,
) -> torch.Tensor:
    """Flattened depth scan of shape ``(B, H*W)``.

    Pixels closer than ``min_dist`` are replaced with ``min_clip_val`` and
    pixels farther than ``max_dist`` (or NaN/inf) with ``max_clip_val``,
    *before* the ``(d - offset) * scale`` transform.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    d = sensor.data.output["distance_to_image_plane"].squeeze(-1)  # (B, H, W)
    d = d.clone()
    d[d < min_dist] = min_clip_val
    bad = (d > max_dist) | torch.isnan(d) | torch.isinf(d)
    d[bad] = max_clip_val
    return ((d - offset) * scale).reshape(env.num_envs, -1)


def depth_image_4d(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.75,
    scale: float = 0.5,
    min_dist: float = 0.2,
    max_dist: float = 9.0,
    min_clip_val: float = 0.0,
    max_clip_val: float = 0.0,
    height: int = 26,
    width: int = 30,
) -> torch.Tensor:
    """Depth image of shape ``(B, 1, H, W)`` for CNN backbones."""
    flat = depth_image(
        env,
        sensor_cfg,
        offset=offset,
        scale=scale,
        min_dist=min_dist,
        max_dist=max_dist,
        min_clip_val=min_clip_val,
        max_clip_val=max_clip_val,
    )
    return flat.reshape(env.num_envs, 1, height, width)
