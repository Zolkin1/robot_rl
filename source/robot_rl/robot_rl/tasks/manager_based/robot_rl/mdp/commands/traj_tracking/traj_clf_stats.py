"""Per-trajectory CLF (Lyapunov) statistics tracker.

Aggregates per-step CLF values (``V``) into per-trajectory statistics that
can drive eval reporting and adaptive sampling during training. Two modes
are supported:

- ``ema``: exponential moving average of ``V`` per trajectory. Cheap and
  stable, intended for the live training-side tracker that feeds adaptive
  sampling.
- ``mean``: plain running mean of ``V`` per trajectory (sum / count).
  Intended for the eval script, where temporal weighting is unnecessary
  and a clean per-rollout average is easier to interpret.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor


class TrajectoryCLFStats:
    """Per-trajectory aggregator for CLF values."""

    def __init__(
        self,
        num_trajectories: int,
        device: str | torch.device,
        mode: Literal["ema", "mean"] = "ema",
        alpha: float = 0.005,
    ) -> None:
        """Create an empty per-trajectory stats tracker.

        Args:
            num_trajectories: Number of trajectories ``T`` to track.
            device: Torch device for the internal tensors.
            mode: ``"ema"`` (training) or ``"mean"`` (eval).
            alpha: EMA smoothing factor in ``(0, 1]``. Only used when
                ``mode == "ema"``. Higher values weight recent samples
                more. Default ``0.005`` matches
                :attr:`MultiSkillManager.skill_v_logs`.
        """
        if mode not in ("ema", "mean"):
            raise ValueError(f"mode must be 'ema' or 'mean', got {mode!r}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")

        self.num_trajectories = int(num_trajectories)
        self.device = torch.device(device)
        self.mode = mode
        self.alpha = float(alpha)

        self._mean_v = torch.zeros(self.num_trajectories, device=self.device)
        self._sum_v = torch.zeros(self.num_trajectories, device=self.device)
        self._count = torch.zeros(self.num_trajectories, device=self.device, dtype=torch.long)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        traj_idx: Tensor,
        v: Tensor,
        active: Tensor | None = None,
    ) -> None:
        """Fold one batch of per-env CLF samples into the running stats.

        Args:
            traj_idx: ``[N]`` long tensor of global trajectory indices, one
                per env in this batch.
            v: ``[N]`` float tensor of CLF values for the same envs.
            active: Optional ``[N]`` bool tensor. Envs with ``active=False``
                are skipped (e.g. recently reset, mid-skill-transition).
                When ``None`` all envs contribute.
        """
        if traj_idx.shape != v.shape:
            raise ValueError(
                f"traj_idx and v must have the same shape, got {traj_idx.shape} vs {v.shape}"
            )

        if active is not None:
            if active.shape != v.shape:
                raise ValueError(
                    f"active mask shape {active.shape} != v shape {v.shape}"
                )
            traj_idx = traj_idx[active]
            v = v[active]

        if traj_idx.numel() == 0:
            return

        traj_idx = traj_idx.to(device=self.device, dtype=torch.long)
        v = v.to(device=self.device, dtype=torch.float32).detach()

        # Per-bin batch statistics for this update.
        batch_sum = torch.zeros_like(self._sum_v)
        batch_count = torch.zeros_like(self._sum_v)
        batch_sum.scatter_add_(0, traj_idx, v)
        batch_count.scatter_add_(0, traj_idx, torch.ones_like(v))

        self._sum_v += batch_sum
        self._count += batch_count.to(torch.long)

        if self.mode == "ema":
            valid = batch_count > 0
            if valid.any():
                batch_mean = torch.zeros_like(self._mean_v)
                batch_mean[valid] = batch_sum[valid] / batch_count[valid]
                # First sample on a previously-empty bin: seed with the batch mean
                # rather than blending against the zero initial value, so cold-start
                # estimates aren't dragged toward zero.
                fresh = valid & (self._count == batch_count.to(torch.long))
                self._mean_v[fresh] = batch_mean[fresh]
                blend = valid & ~fresh
                self._mean_v[blend] = (
                    (1.0 - self.alpha) * self._mean_v[blend]
                    + self.alpha * batch_mean[blend]
                )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_means(self) -> Tensor:
        """Return current per-trajectory mean V, shape ``[T]``.

        Trajectories that have never been seen return ``0.0``.
        """
        if self.mode == "ema":
            return self._mean_v.clone()
        # mode == "mean"
        denom = self._count.clamp(min=1).to(self._sum_v.dtype)
        return self._sum_v / denom

    def get_counts(self) -> Tensor:
        """Return per-trajectory sample counts, shape ``[T]`` (long)."""
        return self._count.clone()

    def top_k_hardest(self, k: int) -> tuple[Tensor, Tensor]:
        """Return the ``k`` hardest trajectories by mean V.

        Args:
            k: Number of trajectories to return.

        Returns:
            ``(indices, values)`` where ``indices`` is a long tensor of
            global trajectory indices and ``values`` is the corresponding
            mean V, both shape ``[k]`` (clamped to ``num_trajectories``).
        """
        k = min(int(k), self.num_trajectories)
        means = self.get_means()
        values, indices = torch.topk(means, k=k, largest=True, sorted=True)
        return indices, values

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Zero all accumulators (used between eval runs)."""
        self._mean_v.zero_()
        self._sum_v.zero_()
        self._count.zero_()

    def state_dict(self) -> dict[str, Tensor]:
        """Serialise the tracker state."""
        return {
            "mean_v": self._mean_v.clone(),
            "sum_v": self._sum_v.clone(),
            "count": self._count.clone(),
        }

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        """Load tracker state previously produced by :meth:`state_dict`."""
        self._mean_v.copy_(state["mean_v"].to(self.device))
        self._sum_v.copy_(state["sum_v"].to(self.device))
        self._count.copy_(state["count"].to(self.device))
