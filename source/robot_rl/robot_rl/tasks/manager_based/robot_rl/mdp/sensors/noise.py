"""Depth-camera noise model: plane tilt + per-pixel noise + Gaussian blur + dropout.

Ported from legged_locomotion_rl. Plugs into IsaacLab's :class:`NoiseModel`
machinery so it can be attached as the ``noise`` field on an
:class:`isaaclab.managers.ObservationTermCfg`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
import torch.nn.functional as F

from isaaclab.utils import configclass
from isaaclab.utils.noise.noise_cfg import NoiseCfg, NoiseModelCfg, UniformNoiseCfg
from isaaclab.utils.noise.noise_model import NoiseModel


class DepthNoise(NoiseModel):
    """Composite noise applied to depth images.

    Per-environment effects (re-sampled on reset):

    1. Plane bias: adds ``a*x + b*y + c`` over the image with ``max(|a*x+b*y+c|)``
       drawn uniformly from ``[0, max_plane_deviation]``.
    2. Per-pixel uniform noise from ``noise_cfg``.
    3. 3x3 Gaussian blur whose ``sigma`` is sampled from ``sigma_noise_cfg``.
    4. Random pixel dropout (set to ``(dropout_val - offset) * scale``).

    Accepts both flat ``(B, H*W)`` and image ``(B, 1, H, W)`` inputs and returns
    the same shape.
    """

    def __init__(self, cfg: DepthNoiseCfg, num_envs: int, device: str):
        super().__init__(cfg, num_envs, device)
        self.device = device
        self.cfg = cfg
        self._H = self.cfg.height
        self._W = self.cfg.width
        self.num_envs = num_envs

        self._sigma = torch.zeros((num_envs, 1), device=self.device)

        ax = torch.arange(-1, 2, dtype=torch.float, device=self.device)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        self._dist_sq = (xx ** 2 + yy ** 2).reshape(1, 9)
        self._kernel = torch.zeros((num_envs, 1, 9), device=self.device)

        ys = torch.linspace(-1, 1, self._H, device=self.device)
        xs = torch.linspace(-1, 1, self._W, device=self.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        self._grid_x = xx.reshape(1, -1)
        self._grid_y = yy.reshape(1, -1)
        self._plane = torch.zeros((self.num_envs, self._H * self._W), device=self.device)
        self._plane_coeffs = torch.zeros((self.num_envs, 3), device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)

        # Resample blur kernel.
        self._sigma[env_ids] = self.cfg.sigma_noise_cfg.func(self._sigma[env_ids], self.cfg.sigma_noise_cfg)
        sigma_sq = (self._sigma[env_ids] ** 2).unsqueeze(1).clamp(min=1e-12)
        kernel = torch.exp(-self._dist_sq / (2 * sigma_sq))
        kernel = kernel / kernel.sum(dim=-1, keepdim=True)
        self._kernel[env_ids] = kernel

        # Resample plane coefficients with constrained max deviation.
        n = self._plane[env_ids].shape[0]
        coeffs = torch.randn(n, 3, device=self.device)
        max_dev = coeffs.abs().sum(dim=-1, keepdim=True).clamp(min=1e-12)
        target = torch.rand(n, 1, device=self.device) * self.cfg.max_plane_deviation
        coeffs = coeffs * (target / max_dev)
        self._plane_coeffs[env_ids] = coeffs
        a, b, c = coeffs[:, 0:1], coeffs[:, 1:2], coeffs[:, 2:3]
        self._plane[env_ids] = a * self._grid_x + b * self._grid_y + c

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        is_4d = data.dim() == 4
        if is_4d:
            orig_shape = data.shape
            data = data.reshape(data.shape[0], -1)

        data = data + self._plane
        data = self.cfg.noise_cfg.func(data, self.cfg.noise_cfg)

        # 3x3 Gaussian blur, applied per-env via grouped conv2d.
        data = F.conv2d(
            data.reshape(1, self.num_envs, self._H, self._W),
            self._kernel.reshape(self.num_envs, 1, 3, 3),
            padding=1,
            groups=self.num_envs,
        ).reshape(self.num_envs, -1)

        mask = torch.rand(data.shape, device=self.device) < self.cfg.dropout_prob
        data[mask] = (self.cfg.dropout_val - self.cfg.offset) * self.cfg.scale

        if is_4d:
            data = data.reshape(orig_shape)
        return data


@configclass
class DepthNoiseCfg(NoiseModelCfg):
    """Config for :class:`DepthNoise`."""

    class_type: type = DepthNoise

    noise_cfg: NoiseCfg = UniformNoiseCfg(n_min=-0.015, n_max=0.015, operation="add")
    """Per-pixel noise applied after the plane bias."""

    max_plane_deviation: float = 0.1
    """Maximum absolute value of the per-env plane bias ``a*x + b*y + c``."""

    sigma_noise_cfg: NoiseCfg = UniformNoiseCfg(n_min=0.0, n_max=0.33, operation="abs")
    """Distribution that samples the Gaussian blur sigma per environment on reset."""

    dropout_prob: float = 0.025
    """Per-pixel dropout probability."""

    dropout_val: float = 0.0
    """Pre-normalisation value assigned to dropped pixels."""

    height: int = MISSING
    """Image height in pixels (must be set to match the depth camera)."""

    width: int = MISSING
    """Image width in pixels (must be set to match the depth camera)."""

    offset: float = MISSING
    """``offset`` parameter from the matching observation term, used to compute
    the post-normalisation dropout value."""

    scale: float = MISSING
    """``scale`` parameter from the matching observation term, used to compute
    the post-normalisation dropout value."""
