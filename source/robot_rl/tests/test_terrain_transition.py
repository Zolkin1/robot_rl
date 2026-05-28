"""Tests for the terrain-approach slow-down decision core.

Pure tensor-level tests on :func:`plan_slowdown_step` in
``robot_rl...traj_tracking.terrain_transition``.  No IsaacLab, no YAML —
every case hand-builds a per-trajectory step-length table, a per-env
eligibility mask, and a target distance, then asserts the expected
``(do_approach, forced_traj_idx)`` per env.
"""

from __future__ import annotations

import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.terrain_transition import (
    plan_slowdown_step,
)

DEVICE = "cpu"

# A trajectory pool: 4 walk steps of increasing reach + one perpetual
# (gate-less) standing trajectory whose step length is inf.
#   idx:   0     1     2     3     4(standing)
STEP_LENGTHS = torch.tensor(
    [0.20, 0.35, 0.50, 0.70, float("inf")], device=DEVICE
)
WALK_IDS = [0, 1, 2, 3]
STANDING_ID = 4


def _walk_eligible(K: int) -> torch.Tensor:
    """[K, T] mask: the 4 walk trajectories eligible, standing excluded."""
    mask = torch.zeros(K, STEP_LENGTHS.numel(), dtype=torch.bool, device=DEVICE)
    mask[:, WALK_IDS] = True
    return mask


def test_overshoot_picks_largest_step_not_exceeding_target():
    """Stance far from the stair ref -> slow down with the largest step
    whose reach does not overshoot the target."""
    target = torch.tensor([0.60], device=DEVICE)  # between step 2 (.50) and 3 (.70)
    do, forced = plan_slowdown_step(
        target, STEP_LENGTHS, _walk_eligible(1), approach_threshold=0.15
    )
    assert bool(do[0]) is True
    assert int(forced[0]) == 2  # 0.50 is the largest <= 0.60


def test_exact_match_is_eligible():
    """A step length exactly equal to the target fits (<=)."""
    target = torch.tensor([0.50], device=DEVICE)
    do, forced = plan_slowdown_step(
        target, STEP_LENGTHS, _walk_eligible(1), approach_threshold=0.15
    )
    assert bool(do[0]) is True
    assert int(forced[0]) == 2


def test_already_aligned_declines():
    """Stance within the threshold of the stair ref -> commit directly."""
    target = torch.tensor([0.10], device=DEVICE)  # below threshold 0.15
    do, forced = plan_slowdown_step(
        target, STEP_LENGTHS, _walk_eligible(1), approach_threshold=0.15
    )
    assert bool(do[0]) is False
    assert int(forced[0]) == -1


def test_target_shorter_than_shortest_step_declines():
    """No walk step fits under the target -> decline (commit stair)."""
    # target above threshold (so condition 2 holds) but below the shortest
    # available step (0.20) -> no fit -> decline.
    target = torch.tensor([0.18], device=DEVICE)
    do, forced = plan_slowdown_step(
        target, STEP_LENGTHS, _walk_eligible(1), approach_threshold=0.15
    )
    assert bool(do[0]) is False
    assert int(forced[0]) == -1


def test_perpetual_traj_never_selected():
    """A gate-less (inf step length) trajectory is never chosen even if it
    is the only eligible one."""
    K = 1
    elig = torch.zeros(K, STEP_LENGTHS.numel(), dtype=torch.bool, device=DEVICE)
    elig[:, STANDING_ID] = True
    target = torch.tensor([5.0], device=DEVICE)
    do, forced = plan_slowdown_step(
        target, STEP_LENGTHS, elig, approach_threshold=0.15
    )
    assert bool(do[0]) is False
    assert int(forced[0]) == -1


def test_mixed_batch():
    """Several envs at once exercise every branch."""
    targets = torch.tensor(
        [0.60, 0.10, 0.18, 0.80], device=DEVICE
    )
    #   env0: overshoot -> step 2 (0.50)
    #   env1: aligned (below threshold) -> decline
    #   env2: above threshold but below shortest step -> decline
    #   env3: large target -> largest step 3 (0.70)
    do, forced = plan_slowdown_step(
        targets, STEP_LENGTHS, _walk_eligible(4), approach_threshold=0.15
    )
    assert do.tolist() == [True, False, False, True]
    assert forced.tolist() == [2, -1, -1, 3]


def test_eligibility_restricts_choice():
    """Only eligible (current-skill) trajectories may be chosen."""
    K = 1
    # Only the two shortest walk steps eligible; target would otherwise
    # prefer step 3 (0.70).
    elig = torch.zeros(K, STEP_LENGTHS.numel(), dtype=torch.bool, device=DEVICE)
    elig[:, [0, 1]] = True
    target = torch.tensor([0.90], device=DEVICE)
    do, forced = plan_slowdown_step(
        target, STEP_LENGTHS, elig, approach_threshold=0.15
    )
    assert bool(do[0]) is True
    assert int(forced[0]) == 1  # 0.35 is the largest eligible <= 0.90
