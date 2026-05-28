"""Per-env state-object containers for the multi-skill command stack.

These tiny dataclass-like helpers consolidate the per-env tensors that
travel together for the deferred-skill-change ("pending") and post-
commit cross-fade ("transition") features, and own their lifecycle
(enqueue/take/clear and start/step/clear).  The goal is to keep all of
the bookkeeping for a given feature in one place so adding or removing
fields doesn't require touching scattered call sites.
"""

from __future__ import annotations

from typing import Tuple

import torch


class PendingSkillChange:
    """Per-env queue for skill changes deferred to a contact-gate fire.

    A row is "active" while the trajectory cmd's bucket-derived desired
    skill differs from its active skill (and gate-on-contact is on).
    The trajectory cmd drains active rows when a contact gate fires,
    flipping its active ``skill_id`` and kicking off a cross-fade.

    Only the skill index is queued — velocity/heading are not deferred
    (they live live on the velocity cmd's ``vel_target_b``, which the
    policy is already ramping toward).
    """

    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        self.skill_id = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def enqueue(self, env_ids: torch.Tensor, skill: torch.Tensor) -> None:
        """Stash a pending skill for ``env_ids`` (overrides any prior entry)."""
        self.skill_id[env_ids] = skill
        self.active[env_ids] = True

    def clear(self, env_ids: torch.Tensor) -> None:
        """Drop the active flag for ``env_ids`` without committing."""
        self.active[env_ids] = False

    def take(
        self, env_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hand the active payload for ``env_ids`` to the caller and clear.

        Returns ``(active_mask, skill)`` where ``active_mask`` is aligned
        to ``env_ids`` and ``skill`` is compacted to the active subset
        (caller does the live-buffer write).  ``self.active`` is cleared
        for that subset; already-inactive rows pass through untouched.
        """
        mask = self.active[env_ids]
        ids = env_ids[mask]
        out = (mask, self.skill_id[ids].clone())
        self.active[ids] = False
        return out


class CrossfadeState:
    """Per-env cross-fade state for the post-skill-commit blend.

    A row is "active" between the moment a contact-gate-driven commit
    fires and the moment ``phi_elapsed`` reaches ``blend_end_phi``.
    While active, ``_transform_desired_outputs`` blends the new
    trajectory's output (at ``manager.phase``) with the old
    trajectory's output (at ``old_phase``, which advances on its own
    clock so periodic-to-perpetual transitions can still progress).

    Per-env fields:

    - ``active``: True while the env is mid-fade.
    - ``old_traj_idx``: which trajectory to fade out from.
    - ``old_phase``: phase used to evaluate the old trajectory.  Set
      at ``start()`` to whatever ``manager.phase`` was at commit time,
      then advanced each step using the old trajectory's own
      phase-advance rule via
      :meth:`MultiSkillManager.advance_phase_for_traj`.  Decoupled
      from ``manager.phase`` so the blend keeps morphing even when
      the new trajectory is perpetual (its phase locked at 0).
    - ``last_old_phase_delta``: most recent wrap-aware delta from
      ``old_phase`` advancement, used by ``_transform_desired_outputs``
      to drive ``phi_elapsed`` when the new trajectory's delta would
      be zero (perpetual new) or negative (perpetual-snap first step).
    - ``phi_elapsed``: accumulator the blend alpha is derived from.
    """

    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.old_traj_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.phi_elapsed = torch.zeros(num_envs, device=device)
        self.old_phase = torch.zeros(num_envs, device=device)
        self.last_old_phase_delta = torch.zeros(num_envs, device=device)

    def start(
        self,
        env_ids: torch.Tensor,
        old_traj_idx: torch.Tensor,
        initial_old_phase: torch.Tensor,
    ) -> None:
        """Begin a fresh transition for ``env_ids`` from ``old_traj_idx``.

        Args:
            env_ids: Envs whose blend is starting now.
            old_traj_idx: Global trajectory index of each env's old
                (fading-out) trajectory.
            initial_old_phase: Phase value to seed ``old_phase`` with
                — typically ``manager.phase[env_ids]`` at commit time
                so the old-trajectory clock continues from wherever
                the env was.
        """
        self.old_traj_idx[env_ids] = old_traj_idx
        self.phi_elapsed[env_ids] = 0.0
        self.old_phase[env_ids] = initial_old_phase
        self.last_old_phase_delta[env_ids] = 0.0
        self.active[env_ids] = True

    def clear(self, env_ids: torch.Tensor) -> None:
        """Force-clear the transition for ``env_ids`` (e.g. on episode reset)."""
        self.active[env_ids] = False
        self.phi_elapsed[env_ids] = 0.0
        self.old_phase[env_ids] = 0.0
        self.last_old_phase_delta[env_ids] = 0.0

    def alpha(self, env_ids: torch.Tensor, blend_end_phi: float) -> torch.Tensor:
        """Return the per-env blend weight in [0, 1] for ``env_ids``."""
        return (self.phi_elapsed[env_ids] / blend_end_phi).clamp(0.0, 1.0)

    def step(
        self, env_ids: torch.Tensor, phi_delta: torch.Tensor, blend_end_phi: float
    ) -> None:
        """Advance ``phi_elapsed`` by ``phi_delta`` and end saturated transitions."""
        self.phi_elapsed[env_ids] = self.phi_elapsed[env_ids] + phi_delta
        done = self.phi_elapsed[env_ids] >= blend_end_phi
        if done.any():
            self.active[env_ids[done]] = False


class ForcedTrajLatch:
    """Per-env one-stride override of the manager's velocity-based
    trajectory selection.

    Set at a contact gate, consumed by
    :meth:`MultiSkillManager._select_trajectories` (which overlays the
    forced index on top of its velocity ``argmin``), and cleared at the
    *next* gate fire.  Used for terrain-approach "slow-down" steps: the
    env stays in its current skill but is pinned to a specific (shorter)
    trajectory whose end-of-stride swing displacement lands the foot in
    front of the terrain feature.  The commanded velocity is deliberately
    NOT modified, so this latch is the only thing breaking the normal
    nearest-by-velocity selection.

    Per-env fields:

    - ``active``: True while the env's selection is pinned.
    - ``traj_idx``: the global trajectory index to force (meaningful only
      where ``active`` is True).  The caller must guarantee the forced
      index belongs to the env's current skill, since the overlay wins
      *after* the skill mask in ``_select_trajectories``.
    """

    def __init__(self, num_envs: int, device: torch.device | str = "cpu"):
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.traj_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

    def set(self, env_ids: torch.Tensor, traj_idx: torch.Tensor) -> None:
        """Pin ``env_ids`` to ``traj_idx`` (overrides any prior latch)."""
        self.traj_idx[env_ids] = traj_idx
        self.active[env_ids] = True

    def clear(self, env_ids: torch.Tensor) -> None:
        """Release the latch for ``env_ids`` (back to velocity selection)."""
        self.active[env_ids] = False
