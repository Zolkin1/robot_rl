"""Sagittal-plane reflection utility for bilateral robot symmetry.

Provides a robot-agnostic, precomputed reflector that swaps left/right
quantities and negates the appropriate axes. All string-based lookups
happen once at construction; runtime reflection is a single gather +
element-wise multiply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SagittalReflectionConfig:
    """Robot-agnostic description of sagittal-plane symmetry.

    Names follow the trajectory output convention used throughout the
    codebase (e.g. ``"left_ankle_roll_link:pos_y"``,
    ``"joint:left_hip_roll_joint"``).
    """

    # Substring pair used for left/right swapping.
    left_token: str = "left"
    right_token: str = "right"

    # Axis suffixes (after the ``:``) that are negated under reflection.
    negated_axis_suffixes: tuple[str, ...] = (
        "pos_y",
        "ori_x",
        "ori_z",
    )

    # Joint-name substrings that are negated under reflection.
    # Use sufficiently specific strings to avoid false positives on body
    # names that happen to contain "roll" (e.g. ``ankle_roll_link``).
    negated_joint_keywords: tuple[str, ...] = (
        "roll_joint",
        "yaw_joint",
    )

    # ---- observation augmentation classification ----

    # Per-observation-term sign multipliers (applied via tiling).
    obs_term_multipliers: dict[str, list[float]] = field(default_factory=lambda: {
        "base_ang_vel": [-1.0, 1.0, -1.0],
        "base_lin_vel": [1.0, -1.0, 1.0],
        "projected_gravity": [1.0, -1.0, 1.0],
        "velocity_commands": [1.0, -1.0, -1.0],
        "root_quat": [-1.0, 1.0, -1.0, 1.0],
    })

    # Observation terms whose reflection is a joint-level swap+sign.
    joint_obs_terms: tuple[str, ...] = ("joint_pos", "joint_vel", "actions")

    # Observation terms that are position / velocity trajectories.
    pos_traj_obs_terms: tuple[str, ...] = ("ref_traj", "act_traj")
    vel_traj_obs_terms: tuple[str, ...] = ("ref_traj_vel", "act_traj_vel")

    # Observation terms that are contact states.
    contact_obs_terms: tuple[str, ...] = ("contact_state",)

    # Observation terms that carry phase info (copied in episodic,
    # negated in half-periodic).
    phase_obs_terms: tuple[str, ...] = ("sin_phase", "cos_phase")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def swap_left_right(name: str, cfg: SagittalReflectionConfig | None = None) -> str:
    """Swap *left* and *right* tokens in *name*.

    Args:
        name: The string to transform.
        cfg: Optional config supplying the token pair.  Uses ``"left"``
            / ``"right"`` when *None*.

    Returns:
        The name with left/right swapped.  If neither token is present
        the original string is returned unchanged.
    """
    left = cfg.left_token if cfg is not None else "left"
    right = cfg.right_token if cfg is not None else "right"
    _SENTINEL = "\x00SWAP_SENTINEL\x00"
    return name.replace(left, _SENTINEL).replace(right, left).replace(_SENTINEL, right)


def _should_negate(name: str, cfg: SagittalReflectionConfig) -> bool:
    """Return *True* if *name* should be sign-negated under reflection."""
    for suffix in cfg.negated_axis_suffixes:
        if name.endswith(suffix):
            return True
    for kw in cfg.negated_joint_keywords:
        if kw in name:
            return True
    return False


# ---------------------------------------------------------------------------
# NamedReflector
# ---------------------------------------------------------------------------

class NamedReflector:
    """Precomputed sagittal-plane reflector for a named set of quantities.

    At construction the string-based logic runs once.  At runtime,
    :meth:`reflect` is a single advanced-index gather plus an element-wise
    multiply — no string operations, no Python loops.

    Args:
        cfg: Reflection configuration.
        names: Ordered list of quantity names (e.g. joint names,
            trajectory output names, contact body names).
        device: Torch device for the precomputed tensors.
    """

    def __init__(
        self,
        cfg: SagittalReflectionConfig,
        names: list[str],
        device: torch.device | str = "cpu",
    ) -> None:
        n = len(names)
        perm = list(range(n))
        sign = [1.0] * n

        name_to_idx: dict[str, int] = {name: i for i, name in enumerate(names)}

        for i, name in enumerate(names):
            mirror = swap_left_right(name, cfg)
            if mirror in name_to_idx and mirror != name:
                perm[i] = name_to_idx[mirror]
            # else: perm[i] stays as i (no swap)

            if _should_negate(name, cfg):
                sign[i] = -1.0

        self._perm = torch.tensor(perm, dtype=torch.long, device=device)
        self._sign = torch.tensor(sign, dtype=torch.float32, device=device)
        self._names = list(names)
        self._device = device

    # -- public API --------------------------------------------------------

    def reflect(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply precomputed sagittal reflection.

        Args:
            tensor: ``[batch, N]`` tensor whose columns correspond to
                *names* in construction order.

        Returns:
            Reflected tensor of the same shape.
        """
        return self._sign * tensor[:, self._perm]

    def reflect_with_history(self, tensor: torch.Tensor, single_dim: int | None = None) -> torch.Tensor:
        """Apply reflection to a possibly history-stacked tensor.

        When ``tensor.shape[-1] > len(names)``, the tensor is assumed to
        be a flattened history ``[t0, t1, ..., tH]`` each of width
        *single_dim*.  The reflection is applied to each timestep
        independently.

        Args:
            tensor: ``[batch, history_length * single_dim]`` tensor.
            single_dim: Width of one timestep.  Defaults to ``len(names)``.

        Returns:
            Reflected tensor of the same shape.
        """
        if single_dim is None:
            single_dim = len(self._names)
        obs_size = tensor.shape[-1]
        if obs_size == single_dim:
            return self.reflect(tensor)
        batch = tensor.shape[0]
        reshaped = tensor.reshape(-1, single_dim)
        reflected = self.reflect(reshaped)
        return reflected.reshape(batch, obs_size)

    def build_relabel_matrix(self) -> torch.Tensor:
        """Build the ``[N, N]`` relabeling matrix *R*.

        ``R @ x`` is equivalent to :meth:`reflect` applied to ``x`` (as
        a single row-vector).

        Returns:
            Signed permutation matrix on :attr:`device`.
        """
        n = len(self._names)
        R = torch.zeros(n, n, device=self._device)
        R[torch.arange(n, device=self._device), self._perm] = self._sign
        return R

    def build_relabel_matrix_numpy(self) -> np.ndarray:
        """Convenience wrapper returning the relabel matrix as a NumPy array."""
        return self.build_relabel_matrix().cpu().numpy()

    # -- introspection -----------------------------------------------------

    @property
    def perm_indices(self) -> torch.Tensor:
        """Integer permutation tensor (read-only view)."""
        return self._perm

    @property
    def sign_vector(self) -> torch.Tensor:
        """Sign-flip tensor (read-only view)."""
        return self._sign

    @property
    def names(self) -> list[str]:
        """The ordered names this reflector was built from."""
        return list(self._names)
