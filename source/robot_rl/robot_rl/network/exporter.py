# Copyright 2026 Zachary Olkin. All rights reserved.

# my_modules/policy_export.py
#
# Usage
# -----
#   from my_modules.policy_export import export_policy_as_jit, export_policy_as_onnx
#
#   policy = ActorCriticCNN(...)
#   normalizer = ...               # or None
#   export_policy_as_jit(policy, normalizer, "./export")
#   export_policy_as_onnx(policy, "./export", normalizer)
#
# ------------------------------------------------------------
from __future__ import annotations
import copy, os, torch
from torch import nn
from typing import Optional

# --------------------------------------------------------------------- #
#                          Public convenience                           #
# --------------------------------------------------------------------- #

def export_policy_as_jit(
    policy: nn.Module,
    normalizer: Optional[nn.Module],
    path: str,
    filename: str = "policy.pt",
) -> None:
    """Trace-and-save policy as TorchScript (.pt)."""
    print("Exporting policy via custom CNN exporter")
    _TorchExporterCNN(policy, normalizer).export(path, filename)


def export_policy_as_onnx(
    policy: nn.Module,
    path: str,
    normalizer: Optional[nn.Module] = None,
    filename: str = "policy.onnx",
    verbose: bool = False,
) -> None:
    """Export policy as ONNX (.onnx)."""
    _OnnxExporterCNN(policy, normalizer, verbose).export(path, filename)

# --------------------------------------------------------------------- #
#                             Private bits                              #
# --------------------------------------------------------------------- #

class _BaseExporterCNN(nn.Module):
    """Common logic for JIT / ONNX exporters that unwrap the CNN trunk."""

    def __init__(self, policy: nn.Module, normalizer: Optional[nn.Module]):
        super().__init__()

        if getattr(policy, "is_recurrent", False):
            raise ValueError("This exporter handles only non-recurrent policies.")

        # --------- copy objects we actually need -------------
        self.hmap_flat        = policy.hmap_flat
        self.height_map_shape = policy.height_map_shape            # (C, H, W)

        # CNN trunk & actor head
        self.cnn   = copy.deepcopy(policy.cnn)
        self.actor = copy.deepcopy(policy.actor)
        self.cnn_out = policy.cnn_out
        # optional obs normaliser (copy so the original graph stays untouched)
        self.normalizer = copy.deepcopy(normalizer) if normalizer else nn.Identity()

        # freeze parameters (safer)
        for p in self.parameters():
            p.requires_grad_(False)

    # ----------------------------------------------------------- #
    #                 helper for both forward paths               #
    # ----------------------------------------------------------- #
    def _forward_detached(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        obs : (B, N_obs)  – **flattened** observation where the first
               `hmap_flat` elements encode the height map row-major.
        """
        B = obs.shape[0]
        hmap = obs[:, : self.hmap_flat].reshape(B, *self.height_map_shape)
        prop = obs[:, self.hmap_flat :]

        # normalise proprio only (height map is already ‘raw’)
        prop = self.normalizer(prop)

        feats = torch.cat([self.cnn(hmap), prop], dim=-1)
        return self.actor(feats)


class _TorchExporterCNN(_BaseExporterCNN):
    """Turns the CNN policy into a TorchScript graph."""

    def export(self, path: str, filename: str):
        os.makedirs(path, exist_ok=True)
        self.eval().cpu()

        scripted = torch.jit.script(self)
        scripted.save(os.path.join(path, filename))

    # TorchScript needs a callable object; expose forward explicitly
    def forward(self, obs: torch.Tensor) -> torch.Tensor:          # noqa: D401,E501
        return self._forward_detached(obs)


class _OnnxExporterCNN(_BaseExporterCNN):
    """Export the CNN policy to ONNX (opset-11, fixed input size)."""

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__(policy, normalizer)
        self.verbose = verbose

    def export(self, path: str, filename: str):
        self.eval().cpu()
        os.makedirs(path, exist_ok=True)

        # dummy input (batch-1) just for tracing shapes
        obs = torch.zeros(1, self.hmap_flat + self.actor[0].in_features - self.cnn_out)

        torch.onnx.export(
            self,                                                # callable module
            obs,                                                 # example input
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},                                     # fixed-shape export
        )

    # needed by torch.onnx.export
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self._forward_detached(obs)
