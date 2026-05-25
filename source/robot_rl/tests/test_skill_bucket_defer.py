"""Tests for the velocity-bucket → skill-id state machine.

Pure tensor-level tests on the helpers in
``robot_rl...traj_tracking.skill_bucket``.  No IsaacLab, no YAML — every
case constructs hand-built bucket tables, an eligibility mask, and a
velocity vector and asserts the expected per-env outputs.
"""

from __future__ import annotations

import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.skill_bucket import (
    bucket_for_velocity,
    commit_pending_at_fire,
    step_skill_pending,
)

DEVICE = "cpu"

# Skill order used by every test below: ``["walk_forward", "running",
# "standing"]`` — matches the layout in g1_walk_run_env_cfg.py.
WF, RN, ST = 0, 1, 2
# Bonus skill row for the eligibility-disambiguation case.
SU = 3  # "stair_up"


def _flat_buckets() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bucket tables matching the walk_run env cfg (3 skills, lin_x only)."""
    lin_x = torch.tensor(
        [
            [0.1, 1.5],  # walk_forward
            [1.5, 3.7],  # running
            [0.0, 0.1],  # standing
        ],
        device=DEVICE,
    )
    lin_y = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        device=DEVICE,
    )
    ang_z = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
        device=DEVICE,
    )
    return lin_x, lin_y, ang_z


def _all_eligible(K: int, S: int = 3) -> torch.Tensor:
    return torch.ones(S, K, dtype=torch.bool, device=DEVICE)


class TestBucketLookup:
    """``bucket_for_velocity``: pure velocity → skill_id resolution."""

    def test_unique_bucket(self) -> None:
        """vx inside exactly one bucket → that bucket wins."""
        lin_x, lin_y, ang_z = _flat_buckets()
        vel = torch.tensor([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]], device=DEVICE)
        eligible = _all_eligible(K=2)

        out = bucket_for_velocity(vel, eligible, lin_x, lin_y, ang_z)
        assert out.tolist() == [WF, RN]

    def test_standing_bucket(self) -> None:
        """vx near zero lands in standing."""
        lin_x, lin_y, ang_z = _flat_buckets()
        vel = torch.tensor([[0.05, 0.0, 0.0]], device=DEVICE)
        eligible = _all_eligible(K=1)

        out = bucket_for_velocity(vel, eligible, lin_x, lin_y, ang_z)
        # 0.05 is in standing (0.0, 0.1) but NOT in walk_forward (0.1, 1.5).
        assert out.tolist() == [ST]

    def test_eligibility_disambiguates_overlap(self) -> None:
        """When two skills' buckets contain vel, eligibility picks one."""
        # Add a stair_up bucket that overlaps walk_forward at 0.4 m/s.
        lin_x = torch.tensor(
            [
                [0.1, 1.5],  # walk_forward
                [1.5, 3.7],  # running
                [0.0, 0.1],  # standing
                [0.4, 0.4],  # stair_up
            ],
            device=DEVICE,
        )
        lin_y = torch.zeros(4, 2, device=DEVICE)
        ang_z = torch.zeros(4, 2, device=DEVICE)
        vel = torch.tensor([[0.4, 0.0, 0.0], [0.4, 0.0, 0.0]], device=DEVICE)

        # Env 0: only walk_forward eligible. Env 1: only stair_up eligible.
        eligible = torch.tensor(
            [
                [True, False],
                [False, False],
                [False, False],
                [False, True],
            ],
            device=DEVICE,
        )

        out = bucket_for_velocity(vel, eligible, lin_x, lin_y, ang_z)
        assert out.tolist() == [WF, SU]

    def test_no_match_returns_nearest_eligible_edge(self) -> None:
        """Velocity outside every eligible bucket → nearest-edge eligible skill.

        With flat buckets and ``vx = -1.0`` (below every bucket's min), the
        edge distances are |−1 − 0.1| = 1.1 (walk_forward), |−1 − 1.5| = 2.5
        (running), |−1 − 0.0| = 1.0 (standing). Standing wins.
        """
        lin_x, lin_y, ang_z = _flat_buckets()
        vel = torch.tensor([[-1.0, 0.0, 0.0]], device=DEVICE)
        eligible = _all_eligible(K=1)

        out = bucket_for_velocity(vel, eligible, lin_x, lin_y, ang_z)
        assert out.tolist() == [ST]

    def test_closed_interval_seam_lower_index(self) -> None:
        """v exactly on a seam (1.5) resolves to the lower-index skill.

        walk_forward (0.1, 1.5) and running (1.5, 3.7) both contain 1.5;
        skill_list order puts walk_forward (idx 0) before running (idx 1).
        Both have dist=0, argmin breaks ties to the lower index.
        """
        lin_x, lin_y, ang_z = _flat_buckets()
        vel = torch.tensor([[1.5, 0.0, 0.0]], device=DEVICE)
        eligible = _all_eligible(K=1)

        out = bucket_for_velocity(vel, eligible, lin_x, lin_y, ang_z)
        assert out.tolist() == [WF]

    def test_current_ineligible_single_eligible_picks_it(self) -> None:
        """Stepping onto a stair-only block while vel is in walk_forward range.

        Only ``stair_up`` is eligible. ``vel = 1.0`` doesn't fall in the
        stair_up bucket (0.4, 0.4) — it doesn't matter; stair_up is the
        only eligible skill so it's picked unambiguously.
        """
        lin_x = torch.tensor(
            [
                [0.1, 1.5],  # walk_forward
                [1.5, 3.7],  # running
                [0.0, 0.1],  # standing
                [0.4, 0.4],  # stair_up
            ],
            device=DEVICE,
        )
        lin_y = torch.zeros(4, 2, device=DEVICE)
        ang_z = torch.zeros(4, 2, device=DEVICE)
        vel = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)
        eligible = torch.tensor(
            [[False], [False], [False], [True]], device=DEVICE
        )

        out = bucket_for_velocity(vel, eligible, lin_x, lin_y, ang_z)
        assert out.tolist() == [SU]

    def test_current_ineligible_two_eligible_edge_tiebreak(self) -> None:
        """Two eligible skills, vel outside both — closer edge wins.

        Flat block (walk_forward + running + standing all eligible);
        ``vel = 0.05`` is in standing (0.0, 0.1) — dist 0, standing wins
        even though walk_forward (0.1, 1.5) is only 0.05 away on its
        lower edge.
        """
        lin_x, lin_y, ang_z = _flat_buckets()
        vel = torch.tensor([[0.05, 0.0, 0.0]], device=DEVICE)
        # Pretend running became ineligible to stress the eligibility mask.
        eligible = torch.tensor(
            [[True], [False], [True]], device=DEVICE
        )

        out = bucket_for_velocity(vel, eligible, lin_x, lin_y, ang_z)
        assert out.tolist() == [ST]

        # Now move vel to where standing's edge is farther than walking's:
        # vel = 0.3 is in walk_forward (dist 0); standing edge at 0.1 is
        # 0.2 away. Walk_forward wins by dist=0.
        vel = torch.tensor([[0.3, 0.0, 0.0]], device=DEVICE)
        out = bucket_for_velocity(vel, eligible, lin_x, lin_y, ang_z)
        assert out.tolist() == [WF]

    def test_in_bucket_beats_far_eligible_neighbor(self) -> None:
        """vel inside one eligible bucket → that bucket wins regardless of
        whether another eligible bucket's *center* is closer.
        """
        # Construct buckets where running's center (2.6) is closer to vel
        # than standing's center (0.05), but vel still sits inside standing.
        lin_x = torch.tensor(
            [
                [0.1, 1.5],  # walk_forward
                [1.5, 3.7],  # running    center 2.6
                [0.0, 0.1],  # standing   center 0.05
            ],
            device=DEVICE,
        )
        lin_y = torch.zeros(3, 2, device=DEVICE)
        ang_z = torch.zeros(3, 2, device=DEVICE)
        # vel = 0.05 is inside standing (dist 0).
        vel = torch.tensor([[0.05, 0.0, 0.0]], device=DEVICE)
        eligible = _all_eligible(K=1)

        out = bucket_for_velocity(vel, eligible, lin_x, lin_y, ang_z)
        assert out.tolist() == [ST]


class TestPendingStateMachine:
    """``step_skill_pending``: per-tick bucket-crossing detection."""

    def test_no_change_when_desired_matches_active(self) -> None:
        desired = torch.tensor([WF, RN], dtype=torch.long, device=DEVICE)
        active = torch.tensor([WF, RN], dtype=torch.long, device=DEVICE)
        pa = torch.tensor([False, False], device=DEVICE)
        ps = torch.zeros(2, dtype=torch.long, device=DEVICE)

        a, pa_o, ps_o, tc = step_skill_pending(desired, active, pa, ps, True)
        assert a.tolist() == [WF, RN]
        assert pa_o.tolist() == [False, False]
        assert tc.any().item() is False

    def test_cross_bucket_enqueues_pending(self) -> None:
        """desired != active, no prior pending → pending = desired."""
        desired = torch.tensor([RN], dtype=torch.long, device=DEVICE)
        active = torch.tensor([WF], dtype=torch.long, device=DEVICE)
        pa = torch.tensor([False], device=DEVICE)
        ps = torch.tensor([0], dtype=torch.long, device=DEVICE)

        a, pa_o, ps_o, tc = step_skill_pending(desired, active, pa, ps, True)
        assert a.tolist() == [WF]                # active unchanged
        assert pa_o.tolist() == [True]           # pending now active
        assert ps_o.tolist() == [RN]             # queued skill = desired
        assert tc.any().item() is False          # no transition clear in deferred mode

    def test_ramp_back_clears_pending(self) -> None:
        """desired == active and pending was active → pending cleared."""
        desired = torch.tensor([WF], dtype=torch.long, device=DEVICE)
        active = torch.tensor([WF], dtype=torch.long, device=DEVICE)
        pa = torch.tensor([True], device=DEVICE)
        ps = torch.tensor([RN], dtype=torch.long, device=DEVICE)

        a, pa_o, ps_o, _ = step_skill_pending(desired, active, pa, ps, True)
        assert a.tolist() == [WF]
        assert pa_o.tolist() == [False]

    def test_fast_bucket_hop_overwrites_pending(self) -> None:
        """desired != active and != pending_skill → pending overwrites."""
        desired = torch.tensor([ST], dtype=torch.long, device=DEVICE)
        active = torch.tensor([WF], dtype=torch.long, device=DEVICE)
        pa = torch.tensor([True], device=DEVICE)
        ps = torch.tensor([RN], dtype=torch.long, device=DEVICE)

        a, pa_o, ps_o, _ = step_skill_pending(desired, active, pa, ps, True)
        assert a.tolist() == [WF]
        assert pa_o.tolist() == [True]
        assert ps_o.tolist() == [ST]

    def test_already_queued_is_idempotent(self) -> None:
        """desired matches existing pending → no re-enqueue, no clear."""
        desired = torch.tensor([RN], dtype=torch.long, device=DEVICE)
        active = torch.tensor([WF], dtype=torch.long, device=DEVICE)
        pa = torch.tensor([True], device=DEVICE)
        ps = torch.tensor([RN], dtype=torch.long, device=DEVICE)

        a, pa_o, ps_o, _ = step_skill_pending(desired, active, pa, ps, True)
        assert a.tolist() == [WF]
        assert pa_o.tolist() == [True]
        assert ps_o.tolist() == [RN]

    def test_gate_off_instant_flip(self) -> None:
        """gate_on_contact=False → active flips immediately."""
        desired = torch.tensor([RN, WF], dtype=torch.long, device=DEVICE)
        active = torch.tensor([WF, WF], dtype=torch.long, device=DEVICE)
        pa = torch.tensor([False, False], device=DEVICE)
        ps = torch.zeros(2, dtype=torch.long, device=DEVICE)

        a, pa_o, _, tc = step_skill_pending(desired, active, pa, ps, False)
        assert a.tolist() == [RN, WF]            # env 0 flipped, env 1 stayed
        assert pa_o.tolist() == [False, False]
        assert tc.tolist() == [True, False]      # transition_clear only on flip

    def test_mixed_batch_deferred(self) -> None:
        """Per-env independence in a single batched call."""
        desired = torch.tensor([RN, WF, ST], dtype=torch.long, device=DEVICE)
        active = torch.tensor([WF, WF, WF], dtype=torch.long, device=DEVICE)
        pa = torch.tensor([False, True, False], device=DEVICE)
        ps = torch.tensor([0, RN, 0], dtype=torch.long, device=DEVICE)

        a, pa_o, ps_o, _ = step_skill_pending(desired, active, pa, ps, True)
        # env 0: cross-bucket enqueue
        # env 1: ramp back (desired==active) clears its pending
        # env 2: cross-bucket enqueue
        assert a.tolist() == [WF, WF, WF]
        assert pa_o.tolist() == [True, False, True]
        assert ps_o.tolist() == [RN, RN, ST]


class TestGateCommit:
    """``commit_pending_at_fire``: drain pending on contact-gate fire."""

    def test_fire_with_pending_commits(self) -> None:
        fire = torch.tensor([True], device=DEVICE)
        pa = torch.tensor([True], device=DEVICE)
        ps = torch.tensor([RN], dtype=torch.long, device=DEVICE)
        active = torch.tensor([WF], dtype=torch.long, device=DEVICE)

        a_o, pa_o, commit = commit_pending_at_fire(fire, pa, ps, active)
        assert a_o.tolist() == [RN]
        assert pa_o.tolist() == [False]
        assert commit.tolist() == [True]

    def test_fire_without_pending_is_noop(self) -> None:
        fire = torch.tensor([True], device=DEVICE)
        pa = torch.tensor([False], device=DEVICE)
        ps = torch.tensor([RN], dtype=torch.long, device=DEVICE)
        active = torch.tensor([WF], dtype=torch.long, device=DEVICE)

        a_o, pa_o, commit = commit_pending_at_fire(fire, pa, ps, active)
        assert a_o.tolist() == [WF]
        assert pa_o.tolist() == [False]
        assert commit.tolist() == [False]

    def test_pending_without_fire_is_noop(self) -> None:
        fire = torch.tensor([False], device=DEVICE)
        pa = torch.tensor([True], device=DEVICE)
        ps = torch.tensor([RN], dtype=torch.long, device=DEVICE)
        active = torch.tensor([WF], dtype=torch.long, device=DEVICE)

        a_o, pa_o, commit = commit_pending_at_fire(fire, pa, ps, active)
        assert a_o.tolist() == [WF]
        assert pa_o.tolist() == [True]           # still queued, untouched
        assert commit.tolist() == [False]

    def test_mixed_batch(self) -> None:
        """Per-env commit independence in one batched call."""
        fire = torch.tensor([True, True, False, True], device=DEVICE)
        pa = torch.tensor([True, False, True, True], device=DEVICE)
        ps = torch.tensor([RN, ST, RN, ST], dtype=torch.long, device=DEVICE)
        active = torch.tensor([WF, WF, WF, RN], dtype=torch.long, device=DEVICE)

        a_o, pa_o, commit = commit_pending_at_fire(fire, pa, ps, active)
        # env 0: fire+pending → commit RN
        # env 1: fire, no pending → noop
        # env 2: pending but no fire → still queued
        # env 3: fire+pending → commit ST
        assert a_o.tolist() == [RN, WF, WF, ST]
        assert pa_o.tolist() == [False, False, True, False]
        assert commit.tolist() == [True, False, False, True]


class TestResetPath:
    """``bucket_for_velocity`` used to snap active skill on reset.

    This isn't a separate function — it's the contract that the reset
    path on the trajectory cmd will use.  Test the inputs/outputs to
    document the expected behaviour.
    """

    def test_reset_snaps_to_velocity_bucket(self) -> None:
        """After reset the active skill equals bucket_for_velocity(vel)."""
        lin_x, lin_y, ang_z = _flat_buckets()
        vel = torch.tensor([[1.0, 0.0, 0.0], [2.5, 0.0, 0.0]], device=DEVICE)
        eligible = _all_eligible(K=2)

        new_active = bucket_for_velocity(vel, eligible, lin_x, lin_y, ang_z)
        assert new_active.tolist() == [WF, RN]

    def test_reset_with_eligibility_constraint(self) -> None:
        """Reset on a stair-only cell snaps active to stair_up."""
        lin_x = torch.tensor(
            [
                [0.1, 1.5],
                [1.5, 3.7],
                [0.0, 0.1],
                [0.4, 0.4],
            ],
            device=DEVICE,
        )
        lin_y = torch.zeros(4, 2, device=DEVICE)
        ang_z = torch.zeros(4, 2, device=DEVICE)
        vel = torch.tensor([[0.4, 0.0, 0.0]], device=DEVICE)
        eligible = torch.tensor([[False], [False], [False], [True]], device=DEVICE)

        out = bucket_for_velocity(vel, eligible, lin_x, lin_y, ang_z)
        assert out.tolist() == [SU]
