"""Terrain-entry "approach" planners for the multi-skill command.

When a contact gate is about to commit a terrain-aware skill (e.g.
``stair_up``) the robot may still be mid-approach with its stance foot far
from where that skill wants it.  Committing directly snaps the stance
anchor onto the terrain and drags the foot — an awkward over-extended step.
Instead, this module decides whether to insert one shortened "slow-down"
step in the *current* skill that lands the swing foot at the terrain
skill's stance reference, deferring the terrain-skill commit by one stride.

Design mirrors the existing pure-helper pattern (``skill_bucket.py``,
``terrain_aware_ref.py``):

- :func:`plan_slowdown_step` is the side-effect-free decision core
  (tensor-in / tensor-out), unit-tested without IsaacLab.
- :class:`TerrainApproachPlanner` is a per-terrain-skill strategy; the
  command dispatches to whichever planner is registered for the skill that
  is about to commit, so new terrain types (descending stairs, gaps) plug
  in with zero new branches in the command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Tuple, runtime_checkable

import torch

from isaaclab.utils.math import quat_apply, yaw_quat

from .terrain_aware_ref import stair_stance_ref

if TYPE_CHECKING:
    from .batched_multiskill_cmd import BatchedMultiSkillCommand


def plan_slowdown_step(
    target_dist: torch.Tensor,
    step_lengths: torch.Tensor,
    eligible_mask: torch.Tensor,
    approach_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decide, per env, whether to take a slow-down step and which traj.

    For each env the slow-down step is taken iff (a) the stance foot is
    farther than ``approach_threshold`` from the terrain skill's stance
    reference (``target_dist`` measures that gap along the travel
    direction), and (b) there exists an eligible (current-skill)
    trajectory whose forward step length does not overshoot ``target_dist``
    — the largest such step is chosen so the swing foot lands as close to
    the stance reference as possible without stepping past it onto the
    terrain.  When no eligible step fits (target shorter than the shortest
    available step), the env declines the slow-down (returns ``-1``) so the
    caller commits the terrain skill normally.

    Args:
        target_dist: ``[K]`` forward distance from the current stance foot
            to the terrain skill's snapped stance reference.
        step_lengths: ``[T]`` per-trajectory forward swing reach
            (``inf`` for gate-less trajectories, which never fit).
        eligible_mask: ``[K, T]`` bool — trajectory ``t`` is in env ``k``'s
            current skill (the only trajectories a same-skill slow-down may
            select).
        approach_threshold: distance (m) below which the stance is treated
            as "already aligned" and no slow-down is inserted.

    Returns:
        ``(do_approach, forced_traj_idx)`` each ``[K]``.  ``do_approach``
        is bool; ``forced_traj_idx`` is the chosen global trajectory index
        where ``do_approach`` else ``-1``.
    """
    K = target_dist.shape[0]
    T = step_lengths.shape[0]
    fits = (step_lengths.unsqueeze(0) <= target_dist.unsqueeze(1)) & eligible_mask  # [K, T]
    neg_inf = torch.full((K, T), float("-inf"), device=step_lengths.device)
    score = torch.where(fits, step_lengths.unsqueeze(0).expand(K, T), neg_inf)
    best_val, best_idx = score.max(dim=1)                                          # [K]
    has_fit = best_val > float("-inf")
    cond2 = target_dist > approach_threshold
    do_approach = cond2 & has_fit
    forced = torch.where(
        do_approach, best_idx, torch.full_like(best_idx, -1)
    )
    return do_approach, forced


@dataclass
class ApproachContext:
    """Inputs handed to a :class:`TerrainApproachPlanner` at a gate fire.

    ``env_ids`` are the *candidate* envs (gate fired, pending skill ==
    the planner's ``target_skill``).  Poses are the just-landed stance
    foot's world pose.
    """

    cmd: "BatchedMultiSkillCommand"
    env_ids: torch.Tensor      # [K] candidate env indices
    stance_pos: torch.Tensor   # [K, 3] just-landed stance foot world xyz
    stance_quat: torch.Tensor  # [K, 4] just-landed stance foot world quat
    predicted_xy: torch.Tensor  # [K, 2] predicted next-gate touchdown world xy


@dataclass
class ApproachDecision:
    """Per-candidate planner output."""

    do_approach: torch.Tensor    # [K] bool
    forced_traj_idx: torch.Tensor  # [K] long, -1 where no slow-down


@runtime_checkable
class TerrainApproachPlanner(Protocol):
    """Strategy that decides the approach behaviour for one terrain skill."""

    target_skill: str

    def plan(self, ctx: ApproachContext) -> ApproachDecision:
        ...


_PLANNERS: dict[str, TerrainApproachPlanner] = {}


def register_approach_planner(planner: TerrainApproachPlanner) -> None:
    """Register ``planner`` as the handler for ``planner.target_skill``."""
    _PLANNERS[planner.target_skill] = planner


def get_approach_planner(skill_name: str) -> TerrainApproachPlanner | None:
    """Return the planner registered for ``skill_name`` (or ``None``)."""
    return _PLANNERS.get(skill_name)


def registered_skills() -> list[str]:
    """Return the skill names that currently have an approach planner."""
    return list(_PLANNERS.keys())


class StairUpApproachPlanner:
    """Approach planner for the ``stair_up`` terrain skill.

    Computes the stair trajectory's snapped stance reference for each
    candidate env (reusing :func:`stair_stance_ref`), measures how far the
    current stance foot is from it along the travel direction, and defers
    to :func:`plan_slowdown_step` for the threshold + step-length choice.
    """

    target_skill = "stair_up"

    def plan(self, ctx: ApproachContext) -> ApproachDecision:
        cmd = ctx.cmd
        mgr = cmd.manager
        env_ids = ctx.env_ids
        K = env_ids.numel()
        device = cmd.device

        # The stair trajectory each candidate env would commit to (nearest
        # by velocity within stair_up), and its domain at the inherited
        # phase, so we can snap its stance reference exactly as
        # ``apply_terrain_aware_ref`` would at commit.
        stair_owner_idx = cmd._skill_list.index(self.target_skill)
        stair_skill_ids = torch.full((K,), stair_owner_idx, dtype=torch.long, device=device)
        stair_traj = mgr.nearest_traj_in_skill(stair_skill_ids, env_ids)   # [K]
        phase = mgr.phase[env_ids]
        domain = mgr._get_domain_indices(phase, stair_traj)
        project = cmd._terrain.terrain_meta_data["project"]
        stair_ref = stair_stance_ref(
            mgr, project, stair_traj, ctx.stance_pos, domain,
        )                                                                  # [K, 3]

        # Forward (travel-direction) distance from the current stance foot
        # to the stair stance reference, using the stance-foot yaw (matches
        # the yaw-only convention in ``_ref_xy_at_next_gate``).
        fwd = quat_apply(
            yaw_quat(ctx.stance_quat),
            torch.tensor([1.0, 0.0, 0.0], device=device).expand(K, 3),
        )                                                                  # [K, 3]
        delta = stair_ref[:, :2] - ctx.stance_pos[:, :2]                   # [K, 2]
        target_dist = (delta * fwd[:, :2]).sum(dim=1)                      # [K]

        # Slow-down step must stay in the env's current skill: eligible
        # trajectories are those whose manager skill matches each env's
        # current owner skill.
        cur_owner_skill = cmd._skill_id[env_ids]                           # [K]
        cur_mgr_skill = mgr._vel_skill_to_traj_skill[cur_owner_skill]      # [K]
        eligible = (
            mgr.data["skill_idx"].unsqueeze(0) == cur_mgr_skill.unsqueeze(1)
        )                                                                  # [K, T]

        do_approach, forced = plan_slowdown_step(
            target_dist,
            cmd._traj_step_length,
            eligible,
            float(cmd.cfg.approach_threshold),
        )
        return ApproachDecision(do_approach=do_approach, forced_traj_idx=forced)


# Register the built-in planners at import time.
register_approach_planner(StairUpApproachPlanner())
