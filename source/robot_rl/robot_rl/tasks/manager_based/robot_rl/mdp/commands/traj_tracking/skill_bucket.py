"""Pure-function helpers for the velocity-bucket → skill-id state machine.

The trajectory cmd's "active skill" is derived per-step from where the
live ``vel_target_b`` sits relative to the configured velocity buckets,
constrained to skills the env's current terrain block allows. These
helpers are kept side-effect-free and tensor-in/tensor-out so the unit
tests can hit them without instantiating any IsaacLab command term.

Naming convention used below:

- ``num_skills`` (``S``): number of skills in ``terrain.skill_list``.
- ``K``: number of envs the call is operating over.
- ``lin_x``, ``lin_y``, ``ang_z``: ``[S, 2]`` bucket tables of
  ``[min, max]`` ranges per skill in ``skill_list`` order.
- ``eligible``: ``[S, K]`` bool mask of which skills the env's current
  cell (or per-block region) is allowed to sample (from
  ``terrain.skill_probs_at(xy) > 0``).
"""

from __future__ import annotations

from typing import Tuple

import torch


def bucket_for_velocity(
    vel: torch.Tensor,
    eligible: torch.Tensor,
    lin_x: torch.Tensor,
    lin_y: torch.Tensor,
    ang_z: torch.Tensor,
) -> torch.Tensor:
    """Return per-env desired skill_id by box-distance to the buckets.

    For each skill ``s``, define the *edge distance* from ``vel`` to the
    closed axis-aligned bucket box ``[lo_x, hi_x] × [lo_y, hi_y] ×
    [lo_w, hi_w]`` as the L2 distance from ``vel`` to the nearest point
    on the box. This is zero when ``vel`` is inside the box and the
    Euclidean distance to the nearest edge otherwise. The returned skill
    is the ``argmin`` of the edge distance over eligible skills:

    * **``vel`` lies in one or more eligible buckets** → dist = 0 for
      those; the lowest-index one wins. Preserves the seam-closed
      semantics where a velocity equal to two buckets' shared boundary
      picks the lower-index skill (e.g. v=1.5 with walk_forward
      (0.1, 1.5) and running (1.5, 3.7) → walk_forward).
    * **``vel`` is outside every eligible bucket** → the eligible bucket
      with the nearest edge (min or max along the axes) wins. This is
      the path that fires when the stance foot lands on a block where
      the previous skill is no longer eligible: an eligible skill is
      picked regardless of where the velocity ramp currently sits, so
      the trajectory cmd can enqueue pending and the next contact gate
      can drive the skill change without waiting for the velocity cmd
      to resample on the new block.

    Args:
        vel: ``[K, 3]`` velocities (vx, vy, wz) per env.
        eligible: ``[S, K]`` bool mask of which skills are allowed per env.
        lin_x: ``[S, 2]`` (min, max) lin-x bucket per skill.
        lin_y: ``[S, 2]`` (min, max) lin-y bucket per skill.
        ang_z: ``[S, 2]`` (min, max) ang-z bucket per skill.

    Returns:
        ``[K]`` long tensor of skill indices into ``skill_list``.
    """
    vx = vel[:, 0:1]                                            # [K, 1]
    vy = vel[:, 1:2]
    wz = vel[:, 2:3]

    lo_x, hi_x = lin_x[:, 0].unsqueeze(0), lin_x[:, 1].unsqueeze(0)  # [1, S]
    lo_y, hi_y = lin_y[:, 0].unsqueeze(0), lin_y[:, 1].unsqueeze(0)
    lo_w, hi_w = ang_z[:, 0].unsqueeze(0), ang_z[:, 1].unsqueeze(0)

    # Nearest point on each bucket box: per-axis clamp of vel into [lo, hi].
    clamped_x = torch.minimum(torch.maximum(vx, lo_x), hi_x)         # [K, S]
    clamped_y = torch.minimum(torch.maximum(vy, lo_y), hi_y)
    clamped_w = torch.minimum(torch.maximum(wz, lo_w), hi_w)

    dist2 = (
        (vx - clamped_x).pow(2)
        + (vy - clamped_y).pow(2)
        + (wz - clamped_w).pow(2)
    )                                                                # [K, S]
    dist2 = dist2.masked_fill(~eligible.T, float("inf"))

    # argmin returns the lowest index on ties — preserves the
    # seam-closed lowest-index-wins semantics for the dist=0 case.
    return dist2.argmin(dim=1)                                       # [K]


def step_skill_pending(
    desired: torch.Tensor,
    active: torch.Tensor,
    pending_active: torch.Tensor,
    pending_skill: torch.Tensor,
    gate_on_contact: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the per-env pending-skill state machine one tick.

    Behaviour by mode:

    - ``gate_on_contact=True`` (deferred):
        * ``desired == active`` and ``pending_active``: clear pending
          (velocity ramped back into the active bucket before any gate
          fired — abandon the queued change).
        * ``desired != active`` and (``not pending_active`` OR
          ``desired != pending_skill``): overwrite pending with desired.
        * Otherwise: no state change.
    - ``gate_on_contact=False`` (instant):
        * ``desired != active``: write ``active = desired`` and signal
          ``transition_clear_mask = True`` so the caller can also clear
          any leftover cross-fade state.  Pending is never used.

    Args:
        desired: ``[K]`` long bucket-derived skill for these envs.
        active: ``[K]`` long current active skill.
        pending_active: ``[K]`` bool of current pending state.
        pending_skill: ``[K]`` long of currently queued skill (only
            meaningful where ``pending_active`` is True).
        gate_on_contact: cfg flag.

    Returns:
        Tuple ``(active', pending_active', pending_skill',
        transition_clear_mask)`` — all aligned to the input shape ``[K]``.
        ``transition_clear_mask`` is only non-zero in the gate-off
        instant-flip mode.
    """
    active_out = active.clone()
    pending_active_out = pending_active.clone()
    pending_skill_out = pending_skill.clone()
    transition_clear = torch.zeros_like(active, dtype=torch.bool)

    if gate_on_contact:
        ramp_back = pending_active & (desired == active)
        if ramp_back.any():
            pending_active_out = pending_active_out & ~ramp_back

        diverging = desired != active
        already_queued = pending_active_out & (pending_skill_out == desired)
        need_enqueue = diverging & ~already_queued
        if need_enqueue.any():
            pending_skill_out = torch.where(need_enqueue, desired, pending_skill_out)
            pending_active_out = pending_active_out | need_enqueue
    else:
        flip = desired != active
        if flip.any():
            active_out = torch.where(flip, desired, active_out)
            transition_clear = flip
        # Pending channel is unused in instant mode — leave as-is.

    return active_out, pending_active_out, pending_skill_out, transition_clear


def commit_pending_at_fire(
    fire_mask: torch.Tensor,
    pending_active: torch.Tensor,
    pending_skill: torch.Tensor,
    active: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drain the pending queue for envs whose contact gate fired.

    For each env where ``fire_mask`` is True AND ``pending_active`` is
    True: write ``active = pending_skill`` and clear ``pending_active``.
    Envs where the gate fired but no pending entry was queued pass
    through untouched.

    Args:
        fire_mask: ``[K]`` bool — gate fired this step.
        pending_active: ``[K]`` bool.
        pending_skill: ``[K]`` long.
        active: ``[K]`` long current active skill.

    Returns:
        Tuple ``(active', pending_active', commit_mask)``.
        ``commit_mask`` is ``[K]`` bool, True for envs that actually had
        a commit happen this tick (caller uses it to start the
        cross-fade).
    """
    commit_mask = fire_mask & pending_active
    active_out = torch.where(commit_mask, pending_skill, active)
    pending_active_out = pending_active & ~commit_mask
    return active_out, pending_active_out, commit_mask
