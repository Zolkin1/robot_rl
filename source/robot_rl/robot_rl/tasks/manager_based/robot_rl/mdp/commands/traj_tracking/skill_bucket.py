"""Pure-function helpers for the velocity-bucket → skill-id state machine.

The trajectory cmd's "active skill" is derived from which configured
velocity bucket the live ``vel_target_b`` currently sits in.  These
helpers are kept side-effect-free and tensor-in/tensor-out so the unit
tests can hit them without instantiating any IsaacLab command term.

Naming convention used below:

- ``num_skills`` (``S``): number of skills in ``terrain.skill_list``.
- ``K``: number of envs the call is operating over.
- ``lin_x``, ``lin_y``, ``ang_z``: ``[S, 2]`` bucket tables of
  ``[min, max]`` ranges per skill in ``skill_list`` order.
- ``eligible``: ``[S, K]`` bool mask of which skills the env's current
  cell (or per-block region) is allowed to sample (from
  ``terrain.skill_probs_at(xy) > 0``).  This
  disambiguates buckets that overlap across skills (e.g.  stair_up vs
  walk_forward both containing 0.4 m/s).
- ``fallback``: ``[K]`` long skill_id to fall back to when no bucket
  matches.  Typically the env's current active skill.
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
    fallback: torch.Tensor,
) -> torch.Tensor:
    """Return per-env skill_id of whichever bucket contains ``vel``.

    A bucket *contains* ``vel`` iff each component (vx, vy, wz) lies in
    the bucket's closed ``[min, max]`` interval.  When multiple skills'
    buckets contain ``vel`` for an env, the ``eligible`` mask filters
    the search to skills the env's current cell is allowed to sample;
    among remaining ties, the lowest skill index wins (deterministic).

    Args:
        vel: ``[K, 3]`` velocities (vx, vy, wz) per env.
        eligible: ``[S, K]`` bool mask of which skills are allowed per env.
        lin_x: ``[S, 2]`` (min, max) lin-x bucket per skill.
        lin_y: ``[S, 2]`` (min, max) lin-y bucket per skill.
        ang_z: ``[S, 2]`` (min, max) ang-z bucket per skill.
        fallback: ``[K]`` long skill_id used for envs where no eligible
            bucket contains ``vel`` (defensive — well-formed cfgs whose
            buckets span each cell's reachable velocity won't trip this).

    Returns:
        ``[K]`` long tensor of skill indices into ``skill_list``.
    """
    vx = vel[:, 0].unsqueeze(0)                            # [1, K]
    vy = vel[:, 1].unsqueeze(0)
    wz = vel[:, 2].unsqueeze(0)

    in_x = (vx >= lin_x[:, 0:1]) & (vx <= lin_x[:, 1:2])   # [S, K]
    in_y = (vy >= lin_y[:, 0:1]) & (vy <= lin_y[:, 1:2])
    in_w = (wz >= ang_z[:, 0:1]) & (wz <= ang_z[:, 1:2])

    in_bucket = in_x & in_y & in_w & eligible              # [S, K]
    has_match = in_bucket.any(dim=0)                       # [K]
    # argmax on a bool tensor returns the index of the first True (or 0
    # when all False) — broadcasts cleanly with the where-fallback below.
    picked = in_bucket.long().argmax(dim=0)                # [K]
    return torch.where(has_match, picked, fallback)


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
