"""Tests for TrajectoryManager.

TODO: Trim this file to parser-level tests (YAML loading, expanded vs
original domain counts, contact body lists, ref-frame parsing) when the
runtime-eval API on :class:`TrajectoryManager` is stripped. Tests for
``get_output``, ``get_phasing_var``, ``get_acceleration``,
``get_current_domains``, ``get_domain_times``, ``get_contact_state``,
``get_ref_frames_in_use`` exercise removed-from-runtime methods and can
go once :class:`MultiSkillManager` is the only runtime path.
"""

import pytest
import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_manager import (
    TrajectoryManager,
    TrajectoryType,
)
from conftest import STANDING_YAML, WALKING_YAML, RUNNING_YAML, DEVICE, TEST_DATA


# ---------------------------------------------------------------------------
# Group A: Construction & Metadata
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_standing_loads(self, standing_manager):
        assert standing_manager is not None

    def test_walking_loads(self, walking_manager):
        assert walking_manager is not None

    def test_running_loads(self, running_manager):
        assert running_manager is not None

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            TrajectoryManager("nonexistent.yaml", None, DEVICE)

    def test_standing_metadata(self, standing_manager):
        td = standing_manager.traj_data
        assert td.trajectory_type == TrajectoryType.FULL_PERIODIC
        assert standing_manager.num_domains == 1
        assert standing_manager.expanded_num_domains == 1
        assert td.total_time == pytest.approx(0.4)
        assert td.conditioner == [-0.1, 0.1]

    def test_walking_metadata(self, walking_manager):
        td = walking_manager.traj_data
        assert td.trajectory_type == TrajectoryType.HALF_PERIODIC
        assert walking_manager.num_domains == 1
        assert walking_manager.expanded_num_domains == 2
        assert td.total_time == pytest.approx(0.46)

    def test_running_metadata(self, running_manager):
        td = running_manager.traj_data
        assert td.trajectory_type == TrajectoryType.HALF_PERIODIC
        assert running_manager.num_domains == 2
        assert running_manager.expanded_num_domains == 4
        assert td.domain_order == ["single_support", "flight_phase"]
        assert td.total_time == pytest.approx(0.299)

    def test_pos_vel_output_count_difference(self, running_manager):
        """num_pos_outputs - num_vel_outputs should equal the number of ori_w entries."""
        td = running_manager.traj_data
        ori_w_count = sum(1 for n in td.pos_output_names if "ori_w" in n)
        assert td.num_pos_outputs - td.num_vel_outputs == ori_w_count

    def test_ori_w_only_in_pos_names(self, running_manager):
        td = running_manager.traj_data
        for name in td.pos_output_names:
            if "ori_w" in name:
                assert name not in td.vel_output_names

    def test_bezier_coeffs_shape(self, running_manager):
        m = running_manager
        td = m.traj_data
        assert m.bezier_coeffs_pos.shape == (m.num_domains, td.num_pos_outputs, td.spline_order + 1)
        assert m.bezier_coeffs_vel.shape == (m.num_domains, td.num_vel_outputs, td.spline_order + 1)


# ---------------------------------------------------------------------------
# Group B: Phasing Variable
# ---------------------------------------------------------------------------

class TestPhasingVar:
    def test_phasing_at_zero_periodic(self, standing_manager):
        phi = standing_manager.get_phasing_var(torch.tensor([0.0]))
        assert phi.item() == pytest.approx(0.0, abs=1e-7)

    def test_phasing_at_zero_half_periodic(self, running_manager):
        phi = running_manager.get_phasing_var(torch.tensor([0.0]))
        assert phi.item() == pytest.approx(0.0, abs=1e-7)

    def test_periodic_wraps(self, standing_manager):
        """phi(total_time) should wrap back to 0 for periodic."""
        total = standing_manager.traj_data.total_time
        phi = standing_manager.get_phasing_var(torch.tensor([total]))
        assert phi.item() == pytest.approx(0.0, abs=1e-6)

    def test_half_periodic_midpoint(self, running_manager):
        """phi(total_time) == 0.5 for half-periodic (halfway through full period)."""
        total = running_manager.traj_data.total_time
        phi = running_manager.get_phasing_var(torch.tensor([total]))
        assert phi.item() == pytest.approx(0.5, abs=1e-6)

    def test_half_periodic_full_wrap(self, running_manager):
        """phi(2*total_time) should wrap back to 0."""
        total = running_manager.traj_data.total_time
        phi = running_manager.get_phasing_var(torch.tensor([2 * total]))
        assert phi.item() == pytest.approx(0.0, abs=1e-6)

    def test_phasing_range(self, running_manager):
        """Phasing variable should always be in [0, 1]."""
        times = torch.rand(100) * 10.0
        phi = running_manager.get_phasing_var(times)
        assert (phi >= 0).all()
        assert (phi <= 1).all()

    def test_phasing_monotonic_within_period(self, standing_manager):
        """Phasing should be non-decreasing within a single period."""
        total = standing_manager.traj_data.total_time
        times = torch.linspace(0.0, total - 1e-6, 50)
        phi = standing_manager.get_phasing_var(times)
        diffs = phi[1:] - phi[:-1]
        assert (diffs >= -1e-7).all()

    def test_phasing_batch_shape(self, running_manager):
        N = 32
        times = torch.rand(N)
        phi = running_manager.get_phasing_var(times)
        assert phi.shape == (N,)


# ---------------------------------------------------------------------------
# Group C: Domain Computation
# ---------------------------------------------------------------------------

class TestDomains:
    def test_domain_at_zero(self, standing_manager):
        domains = standing_manager.get_current_domains(torch.tensor([0.0]))
        assert domains.item() == 0

    def test_standing_always_domain_zero(self, standing_manager):
        times = torch.linspace(0.0, 2.0, 50)
        domains = standing_manager.get_current_domains(times)
        assert (domains == 0).all()

    def test_walking_first_half(self, walking_manager):
        """Walking (1 domain, half-periodic): domain 0 in first half."""
        t = torch.tensor([0.1])
        domains = walking_manager.get_current_domains(t)
        assert domains.item() == 0

    def test_walking_second_half(self, walking_manager):
        """Walking (1 domain, half-periodic): domain 1 (reflected) in second half."""
        total = walking_manager.traj_data.total_time
        t = torch.tensor([total + 0.1])
        domains = walking_manager.get_current_domains(t)
        assert domains.item() == 1

    def test_running_first_domain(self, running_manager):
        """t=0.05 should be in domain 0 (single_support, T=0.133)."""
        domains = running_manager.get_current_domains(torch.tensor([0.05]))
        assert domains.item() == 0

    def test_running_second_domain(self, running_manager):
        """t=0.15 should be in domain 1 (flight_phase, after T=0.133)."""
        domains = running_manager.get_current_domains(torch.tensor([0.15]))
        assert domains.item() == 1

    def test_running_reflected_first_domain(self, running_manager):
        """t=total+0.05 should be in domain 2 (reflected single_support)."""
        total = running_manager.traj_data.total_time
        domains = running_manager.get_current_domains(torch.tensor([total + 0.05]))
        assert domains.item() == 2

    def test_running_reflected_second_domain(self, running_manager):
        """t=total+0.15 should be in domain 3 (reflected flight_phase)."""
        total = running_manager.traj_data.total_time
        domains = running_manager.get_current_domains(torch.tensor([total + 0.15]))
        assert domains.item() == 3

    def test_periodic_wrapping(self, standing_manager):
        total = standing_manager.traj_data.total_time
        t = torch.tensor([0.1, 0.2, 0.3])
        d1 = standing_manager.get_current_domains(t)
        d2 = standing_manager.get_current_domains(t + total)
        assert torch.equal(d1, d2)

    def test_half_periodic_wrapping(self, running_manager):
        total = running_manager.traj_data.total_time
        t = torch.tensor([0.05, 0.15])
        d1 = running_manager.get_current_domains(t)
        d2 = running_manager.get_current_domains(t + 2 * total)
        assert torch.equal(d1, d2)

    def test_domain_range(self, running_manager):
        times = torch.rand(100) * 5.0
        domains = running_manager.get_current_domains(times)
        assert (domains >= 0).all()
        assert (domains < running_manager.expanded_num_domains).all()

    def test_domain_batch_shape(self, running_manager):
        N = 32
        domains = running_manager.get_current_domains(torch.rand(N))
        assert domains.shape == (N,)


# ---------------------------------------------------------------------------
# Group F: Contact State
# ---------------------------------------------------------------------------

class TestContactState:
    def test_standing_both_feet_in_contact(self, standing_manager):
        """Standing: both feet should be in contact at all times."""
        contact_frames = ["right_ankle_roll_link", "left_ankle_roll_link"]
        times = torch.linspace(0.0, 0.4, 10)
        contacts = standing_manager.get_contact_state(times, contact_frames)
        assert contacts.shape == (10, 2)
        assert (contacts == 1.0).all()

    def test_running_single_support(self, running_manager):
        """In single_support domain: right=1, left=0."""
        contact_frames = ["right_ankle_roll_link", "left_ankle_roll_link"]
        contacts = running_manager.get_contact_state(torch.tensor([0.05]), contact_frames)
        assert contacts[0, 0].item() == 1.0  # right
        assert contacts[0, 1].item() == 0.0  # left

    def test_running_flight(self, running_manager):
        """In flight_phase: both feet off ground."""
        contact_frames = ["right_ankle_roll_link", "left_ankle_roll_link"]
        contacts = running_manager.get_contact_state(torch.tensor([0.15]), contact_frames)
        assert contacts[0, 0].item() == 0.0
        assert contacts[0, 1].item() == 0.0

    def test_running_reflected_swaps(self, running_manager):
        """In reflected single_support: left=1, right=0 (swapped)."""
        total = running_manager.traj_data.total_time
        contact_frames = ["right_ankle_roll_link", "left_ankle_roll_link"]
        contacts = running_manager.get_contact_state(torch.tensor([total + 0.05]), contact_frames)
        assert contacts[0, 0].item() == 0.0  # right
        assert contacts[0, 1].item() == 1.0  # left

    def test_contact_binary(self, running_manager):
        contact_frames = ["right_ankle_roll_link", "left_ankle_roll_link"]
        times = torch.rand(50) * 2.0
        contacts = running_manager.get_contact_state(times, contact_frames)
        assert ((contacts == 0.0) | (contacts == 1.0)).all()

    def test_contact_shape(self, running_manager):
        contact_frames = ["right_ankle_roll_link", "left_ankle_roll_link"]
        N = 16
        contacts = running_manager.get_contact_state(torch.rand(N), contact_frames)
        assert contacts.shape == (N, len(contact_frames))


# ---------------------------------------------------------------------------
# Group G: Reference Frames
# ---------------------------------------------------------------------------

class TestRefFrames:
    @staticmethod
    def _get_full_ref_frames(manager):
        """Get ref frames including reflected versions for half-periodic trajectories.

        In production, TrajectoryCommand._parse_ref_frames handles this expansion.
        For testing, we replicate it by adding left/right mirrors.
        """
        raw = manager.get_reference_frames()
        seen = set()
        full = []
        for f in raw:
            if f not in seen:
                seen.add(f)
                full.append(f)
        if manager.traj_data.trajectory_type == TrajectoryType.HALF_PERIODIC:
            for f in list(full):
                mirror = f.replace("right", "TEMP").replace("left", "right").replace("TEMP", "left")
                if mirror not in seen:
                    seen.add(mirror)
                    full.append(mirror)
        return full

    def test_ref_frame_shape(self, running_manager):
        ref_frames = self._get_full_ref_frames(running_manager)
        N = 16
        result = running_manager.get_ref_frames_in_use(torch.rand(N) * 0.1, ref_frames)
        assert result.shape == (N,)
        assert result.dtype == torch.long

    def test_standing_consistent_ref_frame(self, standing_manager):
        """Standing has a single domain, so ref frame should be constant."""
        ref_frames = self._get_full_ref_frames(standing_manager)
        times = torch.linspace(0.0, 0.4, 10)
        result = standing_manager.get_ref_frames_in_use(times, ref_frames)
        assert (result == result[0]).all()

    def test_running_reflected_swaps_ref_frame(self, running_manager):
        """Reflected half should swap left/right ref frames."""
        ref_frames = self._get_full_ref_frames(running_manager)
        t_first = torch.tensor([0.05])
        t_reflected = torch.tensor([running_manager.traj_data.total_time + 0.05])
        idx_first = running_manager.get_ref_frames_in_use(t_first, ref_frames)
        idx_reflected = running_manager.get_ref_frames_in_use(t_reflected, ref_frames)
        # They should be different (swapped)
        assert idx_first.item() != idx_reflected.item()


# ---------------------------------------------------------------------------
# Group H: Domain Times & Total Time
# ---------------------------------------------------------------------------

class TestDomainTimes:
    def test_standing_domain_time(self, standing_manager):
        times = torch.linspace(0.0, 0.39, 10)
        dt = standing_manager.get_domain_times(times)
        assert torch.allclose(dt, torch.full_like(dt, 0.4))

    def test_running_single_support_time(self, running_manager):
        dt = running_manager.get_domain_times(torch.tensor([0.05]))
        assert dt.item() == pytest.approx(0.133)

    def test_running_flight_time(self, running_manager):
        dt = running_manager.get_domain_times(torch.tensor([0.15]))
        assert dt.item() == pytest.approx(0.166)

    def test_total_time_periodic(self, standing_manager):
        total = standing_manager.get_total_time()
        assert total.item() == pytest.approx(0.4)

    def test_total_time_half_periodic(self, running_manager):
        total = running_manager.get_total_time()
        assert total.item() == pytest.approx(0.598)  # 2 * 0.299


# ---------------------------------------------------------------------------
# Group D: Output Computation
# ---------------------------------------------------------------------------

class TestOutput:
    def test_output_shapes(self, running_manager):
        td = running_manager.traj_data
        N = 16
        pos, vel = running_manager.get_output(torch.rand(N) * 0.2)
        assert pos.shape == (N, td.num_pos_outputs)
        assert vel.shape == (N, td.num_vel_outputs)

    def test_output_deterministic(self, running_manager):
        t = torch.tensor([0.05, 0.15, 0.25])
        pos1, vel1 = running_manager.get_output(t)
        pos2, vel2 = running_manager.get_output(t)
        assert torch.equal(pos1, pos2)
        assert torch.equal(vel1, vel2)

    def test_output_periodic_wraps(self, standing_manager):
        total = standing_manager.traj_data.total_time
        t = torch.tensor([0.1, 0.2, 0.3])
        pos1, vel1 = standing_manager.get_output(t)
        pos2, vel2 = standing_manager.get_output(t + total)
        assert torch.allclose(pos1, pos2, atol=1e-5)
        assert torch.allclose(vel1, vel2, atol=1e-5)

    def test_output_half_periodic_full_wrap(self, running_manager):
        total = running_manager.traj_data.total_time
        t = torch.tensor([0.05, 0.15])
        pos1, vel1 = running_manager.get_output(t)
        pos2, vel2 = running_manager.get_output(t + 2 * total)
        assert torch.allclose(pos1, pos2, atol=1e-5)
        assert torch.allclose(vel1, vel2, atol=1e-5)

    def test_output_batch_matches_individual(self, running_manager):
        """Each row of batched output should match calling with a single time."""
        times = torch.tensor([0.0, 0.05, 0.15, 0.25])
        pos_batch, vel_batch = running_manager.get_output(times)
        for i, t in enumerate(times):
            pos_i, vel_i = running_manager.get_output(t.unsqueeze(0))
            assert torch.allclose(pos_batch[i], pos_i[0], atol=1e-6)
            assert torch.allclose(vel_batch[i], vel_i[0], atol=1e-6)


# ---------------------------------------------------------------------------
# Group E: Acceleration
# ---------------------------------------------------------------------------

class TestAcceleration:
    def test_acceleration_shape(self, running_manager):
        td = running_manager.traj_data
        N = 16
        accel = running_manager.get_acceleration(torch.rand(N) * 0.2)
        assert accel.shape == (N, td.num_vel_outputs)

    def test_acceleration_finite_difference(self, standing_manager):
        """Acceleration should approximately match finite differences of velocity."""
        dt = 1e-4
        t = torch.tensor([0.15])
        _, vel_before = standing_manager.get_output(t - dt)
        _, vel_after = standing_manager.get_output(t + dt)
        fd_accel = (vel_after - vel_before) / (2 * dt)
        accel = standing_manager.get_acceleration(t)
        assert torch.allclose(accel, fd_accel, atol=1e-2)


# ---------------------------------------------------------------------------
# Group I: Order Outputs
# ---------------------------------------------------------------------------

class TestOrderOutputs:
    def test_order_outputs_reorders_names(self, standing_manager):
        """After reordering, output names match the provided order."""
        # Make a fresh manager to avoid mutating the fixture
        m = TrajectoryManager(STANDING_YAML, None, DEVICE)
        original_pos = list(m.traj_data.pos_output_names)
        original_vel = list(m.traj_data.vel_output_names)

        # Reverse the order
        reversed_pos = list(reversed(original_pos))
        reversed_vel = list(reversed(original_vel))
        m.order_outputs(reversed_pos, reversed_vel)

        assert m.traj_data.pos_output_names == reversed_pos
        assert m.traj_data.vel_output_names == reversed_vel

    def test_order_outputs_reorders_values(self, standing_manager):
        """After reordering, get_output values are the same but reordered."""
        m = TrajectoryManager(STANDING_YAML, None, DEVICE)
        t = torch.tensor([0.1])

        original_pos_names = list(m.traj_data.pos_output_names)
        original_vel_names = list(m.traj_data.vel_output_names)
        pos_orig, vel_orig = m.get_output(t)

        # Reverse order
        reversed_pos = list(reversed(original_pos_names))
        reversed_vel = list(reversed(original_vel_names))
        m.order_outputs(reversed_pos, reversed_vel)
        pos_reordered, vel_reordered = m.get_output(t)

        # Values should be the same but in reversed order
        assert torch.allclose(pos_orig[0], pos_reordered[0].flip(0), atol=1e-5)
        assert torch.allclose(vel_orig[0], vel_reordered[0].flip(0), atol=1e-5)


# ---------------------------------------------------------------------------
# Group J: Relabeling Matrix
# ---------------------------------------------------------------------------

class TestRelabeling:
    def test_relabel_matrix_shape(self, running_manager):
        R = running_manager.relable_ee_stance_coeffs(running_manager.traj_data.pos_output_names)
        n = len(running_manager.traj_data.pos_output_names)
        assert R.shape == (n, n)

    def test_relabel_is_signed_permutation(self, running_manager):
        """Each row should have exactly one nonzero entry with value +1 or -1."""
        R = running_manager.relable_ee_stance_coeffs(running_manager.traj_data.pos_output_names)
        for i in range(R.shape[0]):
            nonzero = R[i][R[i] != 0]
            assert len(nonzero) == 1, f"Row {i} has {len(nonzero)} nonzero entries"
            assert abs(nonzero[0]) == 1.0

    def test_relabel_double_application_is_identity(self, running_manager):
        """R @ R should be the identity matrix (double reflection)."""
        import numpy as np
        R = running_manager.relable_ee_stance_coeffs(running_manager.traj_data.pos_output_names)
        R2 = R @ R
        assert np.allclose(R2, np.eye(R.shape[0]), atol=1e-10)

    def test_relabel_swaps_left_right(self, running_manager):
        """Left ankle pos_x should map to right ankle pos_x."""
        names = running_manager.traj_data.pos_output_names
        R = running_manager.relable_ee_stance_coeffs(names)

        if "left_ankle_roll_link:pos_x" in names and "right_ankle_roll_link:pos_x" in names:
            left_idx = names.index("left_ankle_roll_link:pos_x")
            right_idx = names.index("right_ankle_roll_link:pos_x")
            # Row for left should have nonzero at right column
            assert R[left_idx, right_idx] != 0


# ---------------------------------------------------------------------------
# Group K: Bezier Math
# ---------------------------------------------------------------------------

class TestBezierMath:
    def test_bezier_at_tau_0(self, standing_manager):
        """Position at tau=0 should equal the first control point."""
        m = standing_manager
        ctrl_pts = m._all_coeffs_pos[0:1]  # [1, num_outputs, degree+1]
        T = m._T_all[0:1]
        tau = torch.tensor([0.0])
        result = m._compute_bezier_batched(tau, ctrl_pts, T, derivative=False)
        assert torch.allclose(result[0], ctrl_pts[0, :, 0], atol=1e-6)

    def test_bezier_at_tau_1(self, standing_manager):
        """Position at tau=1 should equal the last control point."""
        m = standing_manager
        ctrl_pts = m._all_coeffs_pos[0:1]
        T = m._T_all[0:1]
        tau = torch.tensor([1.0])
        result = m._compute_bezier_batched(tau, ctrl_pts, T, derivative=False)
        assert torch.allclose(result[0], ctrl_pts[0, :, -1], atol=1e-6)

    def test_bezier_linear_interpolation(self, standing_manager):
        """Degree-1 Bezier with [a, b] at tau=0.5 should give (a+b)/2."""
        m = standing_manager
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([5.0, 6.0, 7.0])
        ctrl_pts = torch.stack([a, b], dim=1).unsqueeze(0)  # [1, 3, 2]
        T = torch.tensor([1.0])
        tau = torch.tensor([0.5])
        result = m._compute_bezier_batched(tau, ctrl_pts, T, derivative=False)
        expected = (a + b) / 2
        assert torch.allclose(result[0], expected, atol=1e-6)

    def test_bezier_linear_constant_velocity(self, standing_manager):
        """Degree-1 Bezier derivative should be constant = (b-a)/T."""
        m = standing_manager
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([5.0, 8.0])
        ctrl_pts = torch.stack([a, b], dim=1).unsqueeze(0)  # [1, 2, 2]
        T = torch.tensor([2.0])
        # Velocity at any tau should be (b-a)/T
        for tau_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            tau = torch.tensor([tau_val])
            result = m._compute_bezier_batched(tau, ctrl_pts, T, derivative=True)
            expected = (b - a) / 2.0
            assert torch.allclose(result[0], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Group L: Static Validation
# ---------------------------------------------------------------------------

class TestStaticValidation:
    def test_empty_domains(self):
        result = TrajectoryManager._verify_consistent_outputs_and_get_info({}, [])
        assert result == (0, 0, [], [], 0)

    def test_mismatched_spline_order_raises(self):
        """Two domains with different spline orders should raise."""
        # Create minimal domain data with different spline orders
        bezier_a = {
            "frames": {"f": {"pos_x": [1, 2, 3]}},
            "joints": {"j": [1, 2, 3]},
            "frame_vels": {"f": {"pos_x": [1, 2, 3]}},
            "joint_vels": {"j": [1, 2, 3]},
        }
        bezier_b = {
            "frames": {"f": {"pos_x": [1, 2, 3]}},
            "joints": {"j": [1, 2, 3]},
            "frame_vels": {"f": {"pos_x": [1, 2, 3]}},
            "joint_vels": {"j": [1, 2, 3]},
        }
        domain_data = {
            "d1": (bezier_a, 5),
            "d2": (bezier_b, 3),
        }
        with pytest.raises(ValueError, match="spline order"):
            TrajectoryManager._verify_consistent_outputs_and_get_info(domain_data, ["d1", "d2"])


# ---------------------------------------------------------------------------
# Group D+: Golden Value Tests
# ---------------------------------------------------------------------------

# Golden values captured from current working code (pre-refactor baseline).
# Tolerance: atol=1e-5, rtol=1e-5

GOLDEN_STANDING_POS_T0 = [0.014758230186998844, 0.12731121480464935, 0.7409535050392151, 0.9996742010116577, 0.0017366540851071477, 0.025006316602230072, 0.004951960872858763, 4.897770850220695e-05, 0.2580300271511078, -1.1447884389781393e-05, 1.0000003576278687, 0.0004635334189515561, 0.0005002105026505888, 0.0004852410056628287, 2.352987347982937e-21, 1.7803805758413752e-20, 2.3426511042280406e-20, 1.0000003576278687, -0.0004995302297174931, -0.0004066589754074812, 0.0004998957738280296, -0.013783931732177734, -0.10504299402236938, 0.6944158673286438, 0.7870019674301147, -0.09661892801523209, 0.6051199436187744, -0.07158707082271576, -0.023741023615002632, 0.35922661423683167, 0.696864128112793, 0.7813379168510437, 0.09294693917036057, 0.6116047501564026, 0.0825461596250534, 4.830511372369843e-16, 0.47957730293273926, 0.23999999463558197, -5.681678899197929e-16, 0.800000011920929, 0.4645775258541107, -0.23999999463558197, -4.3824494721180396e-16, 0.800000011920929, 0.01079954020678997, -0.2427060455083847, -0.013653435744345188, 0.4376544654369354, -0.24412155151367188, -0.01777624897658825, -0.02096151001751423, -0.24458959698677063, -0.0013465717202052474, 0.45039692521095276, -0.2565550208091736, 0.014341109432280064]
GOLDEN_STANDING_VEL_T0 = [-0.007428682874888182, -0.005731888115406036, -0.0002569565549492836, 0.008028090000152588, -0.010000007227063179, -0.00012525450438261032, -0.001028080121614039, -0.0009969010716304183, 0.0009651188738644123, 0.0009999772300943732, -0.0009981096955016255, -0.0010019161272794008, -0.0010281485738232732, -0.001023131306283176, -0.001034824177622795, 0.0010012186830863357, -0.000999263022094965, -0.00025363615714013577, -0.008794588968157768, -0.00533429766073823, -0.0021801802795380354, 0.007910984568297863, -0.005189606919884682, -0.0014840624062344432, -0.008756439201533794, -0.005438126157969236, 0.0014730744296684861, 0.008050858974456787, -0.0051984768360853195, 0.0012693055905401707, -2.3528505126201374e-14, 0.005000001285225153, 5.6259802995754435e-14, 2.3924327010585097e-14, -3.4730078784638713e-15, 0.004999999422580004, 6.026991107794047e-14, 1.9821185911670335e-14, -5.846306952276854e-15, -0.0022737167309969664, -0.0008505237055942416, -0.0002636371355038136, 0.004872040823101997, 0.00500000873580575, -0.005000005941838026, -0.0018780333921313286, -0.000989118474535644, 0.0007625590660609305, 0.005000008270144463, 0.00500000873580575, -0.005000009201467037]
GOLDEN_STANDING_POS_T02 = [0.0140244634822011, 0.1267857849597931, 0.7409535050392151, 0.9996893405914307, 0.002041053958237171, 0.024338802322745323, 0.00494585745036602, 5.133072772878222e-05, 0.2579418420791626, 6.11591458437033e-05, 0.9999996423721313, 0.0004997873911634088, 0.0004467286926228553, 0.00047468699631281197, -5.510926712304354e-06, -6.6640868681133725e-06, -8.153091766871512e-05, 0.9999996423721313, -0.0004997234209440649, -0.000500247988384217, 0.0004997536307200789, -0.01475086621940136, -0.10553562641143799, 0.6942711472511292, 0.7871918082237244, -0.09628915786743164, 0.6049304008483887, -0.0715312659740448, -0.024696698412299156, 0.3587189018726349, 0.6969959139823914, 0.7814733386039734, 0.09310593456029892, 0.6113662123680115, 0.08283684402704239, -6.382349317727014e-16, 0.48037728667259216, 0.23999997973442078, -1.0591575258175629e-16, 0.7999999523162842, 0.465377539396286, -0.23999997973442078, -2.545746670363163e-16, 0.7999999523162842, 0.010771040804684162, -0.24237409234046936, -0.013653285801410675, 0.4377487599849701, -0.24332155287265778, -0.018287820741534233, -0.020845849066972733, -0.2442837655544281, -0.0013467203825712204, 0.450552761554718, -0.2558716833591461, 0.013621899299323559]
GOLDEN_STANDING_VEL_T02 = [0.00032494711922481656, 0.001546406652778387, 2.440419848426245e-07, -0.002544648479670286, 0.006658683065325022, 0.00012133937707403675, 3.0102550226729363e-05, -2.1895964891882613e-05, -0.0006348625756800175, -1.0633229976519942e-06, 0.0009993166895583272, 0.0006269008154049516, -8.453207556158304e-08, 5.484337089001201e-05, -1.8014543456956744e-07, 0.0009579815668985248, 8.417664503213018e-07, 3.897839633282274e-07, 0.001844525570049882, 0.0014034091727808118, 0.0005564778693951666, -0.002429500687867403, 0.0018476202385500073, 0.0014772885479032993, 0.0018018601695075631, 0.0014953638892620802, -0.0005842342507094145, -0.0025657005608081818, 0.0018581947078928351, -0.001276147086173296, 7.275957614183426e-12, -0.0049998098984360695, 1.4551915228366852e-11, 0.0, 0.0, -0.004999817349016666, -1.4551915228366852e-11, -4.3655745685100555e-11, 1.4551915228366852e-11, 0.0006464659818448126, -0.004999905358999968, 0.00033104477915912867, -0.004999920260161161, 0.004339311271905899, 0.002010224387049675, 0.000502840441185981, -0.004999976139515638, -0.00033107027411460876, -0.004741134122014046, 0.0030805335845798254, 0.0028634443879127502]

GOLDEN_RUNNING_POS_T0 = [-0.021803217008709908, 0.1267450898885727, 0.676128089427948, 0.9965722560882568, -0.020452966913580894, 0.0799519270658493, 0.004063386935740709, -0.27626413106918335, 0.2556286156177521, 0.40198183059692383, 0.638390064239502, 0.013859896920621395, 0.7697277665138245, -0.0060809883289039135, -6.0568425396922976e-05, 2.5164190446957946e-05, -0.00020649092039093375, 0.9988004565238953, -0.0023643469903618097, 0.04917267709970474, -0.0021811083424836397, -0.03700116276741028, -0.16611434519290924, 0.705219030380249, 0.8300191164016724, -0.19022125005722046, 0.4800250232219696, -0.210150808095932, 0.08761750161647797, 0.32762107253074646, 0.6741110682487488, 0.9336835741996765, 0.11294484883546829, 0.3350014388561249, -0.055158376693725586, -0.1427295058965683, 0.4519466161727905, 0.2528020441532135, -0.029758943244814873, 0.09536871314048767, 0.5309388637542725, -0.37966763973236084, -0.20570145547389984, 0.3587113916873932, 0.024157032370567322, -0.25274598598480225, -0.07705606520175934, 2.241199493408203, -0.39145535230636597, -0.028089165687561035, 0.015385380946099758, -0.6830016374588013, -0.032896578311920166, 0.9693815112113953, -0.3481932282447815, 0.0021701727528125048]
GOLDEN_RUNNING_VEL_T0 = [2.1832423210144043, -0.18901187181472778, -0.5913735032081604, 0.13882005214691162, -0.549920916557312, -1.4703792333602905, 4.263879299163818, -0.018928296864032745, 1.4082894325256348, -0.14958494901657104, 6.008566379547119, -0.8767206072807312, -0.20722459256649017, -0.008228241465985775, -0.6635470390319824, -0.01437951810657978, -4.8286027908325195, 0.02764500118792057, 5.110552787780762, 0.24928104877471924, -1.24001145362854, 2.120601177215576, -9.796330451965332, 3.584521770477295, -0.26291605830192566, 0.9633484482765198, -1.7340891361236572, 4.064469337463379, 12.275162696838379, 4.349303722381592, 2.1200368404388428, 4.393519878387451, 0.3587726652622223, 1.7055938243865967, 9.224748611450195, -7.739412784576416, 2.4151248931884766, 1.1537513732910156, -2.0534369945526123, -0.767716109752655, -1.3246920108795166, 0.2771087884902954, 9.086758613586426, -1.172654151916504, -0.4460000693798065, 1.2122613191604614, 4.300004959106445, 1.040116310119629, 0.7880983352661133, -9.418892860412598, -0.33441972732543945]
GOLDEN_RUNNING_POS_T005 = [0.07452640682458878, 0.11944958567619324, 0.6721779108047485, 0.9958463907241821, -0.022095471620559692, 0.08052027225494385, -0.03669123724102974, -0.08476430177688599, 0.2520591914653778, 0.4280339181423187, 0.6541369557380676, 0.04574405774474144, 0.7544825673103333, -0.02591179683804512, -0.0032400349155068398, 0.0009142759954556823, -0.011878974735736847, 0.9999819993972778, -0.0007998992223292589, 0.0012871037470176816, -0.002527298405766487, 0.2082008421421051, -0.1372186243534088, 0.7179726958274841, 0.9475208520889282, -0.18110528588294983, 0.24133571982383728, -0.10497740656137466, 0.06688032299280167, 0.3602805733680725, 0.6499826908111572, 0.8113336563110352, 0.1349841058254242, 0.5686498284339905, 0.010901377536356449, -0.03770224004983902, 0.6640084385871887, 0.28052645921707153, 0.0176504235714674, 0.41633138060569763, 0.16559812426567078, -0.3311822712421417, -0.07696066051721573, 0.16541694104671478, 0.04582876339554787, -0.4789689779281616, -0.06345238536596298, 2.532252073287964, -0.4997800290584564, -0.0007661214331164956, 0.08416207134723663, -0.4894232451915741, 0.027296939864754677, 0.8994010090827942, -0.5671827793121338, -0.012224867008626461]
GOLDEN_RUNNING_VEL_T005 = [2.1051175594329834, -0.025546472519636154, 0.4872666001319885, 0.04188169911503792, 0.022130463272333145, -0.5694392919540405, 3.6838037967681885, -0.029978789389133453, -0.36720845103263855, -0.4296266734600067, -8.595102310180664, -1.2647439241409302, -0.005430588498711586, -0.004971419461071491, -0.04389280453324318, 0.0037350410129874945, -0.28222545981407166, -0.0014391363365575671, 4.948452472686768, 1.193326473236084, 1.5134305953979492, 1.0223335027694702, -9.970918655395508, 8.123308181762695, -0.20775634050369263, 0.43184515833854675, 0.72764652967453, 1.7233638763427734, 8.464555740356445, 1.2739579677581787, 2.1008358001708984, 3.826317548751831, 1.6360437870025635, -1.3700765371322632, 4.381688117980957, -6.413280963897705, 0.6301435828208923, 2.97745943069458, -5.38602352142334, 0.10602444410324097, -8.627277374267578, -0.39531993865966797, 1.7951219081878662, -1.8237156867980957, 0.8708479404449463, 0.10323403030633926, 5.87821102142334, 0.9997065663337708, -5.17753267288208, -0.9720156788825989, -0.010532687418162823]


class TestGoldenValues:
    """Regression tests using golden values from the pre-refactor code."""

    def test_standing_pos_at_t0(self, standing_manager):
        pos, _ = standing_manager.get_output(torch.tensor([0.0]))
        expected = torch.tensor(GOLDEN_STANDING_POS_T0)
        assert torch.allclose(pos[0], expected, atol=1e-5, rtol=1e-5)

    def test_standing_vel_at_t0(self, standing_manager):
        _, vel = standing_manager.get_output(torch.tensor([0.0]))
        expected = torch.tensor(GOLDEN_STANDING_VEL_T0)
        assert torch.allclose(vel[0], expected, atol=1e-5, rtol=1e-5)

    def test_standing_pos_at_t02(self, standing_manager):
        pos, _ = standing_manager.get_output(torch.tensor([0.2]))
        expected = torch.tensor(GOLDEN_STANDING_POS_T02)
        assert torch.allclose(pos[0], expected, atol=1e-5, rtol=1e-5)

    def test_standing_vel_at_t02(self, standing_manager):
        _, vel = standing_manager.get_output(torch.tensor([0.2]))
        expected = torch.tensor(GOLDEN_STANDING_VEL_T02)
        assert torch.allclose(vel[0], expected, atol=1e-5, rtol=1e-5)

    def test_running_pos_at_t0(self, running_manager):
        pos, _ = running_manager.get_output(torch.tensor([0.0]))
        expected = torch.tensor(GOLDEN_RUNNING_POS_T0)
        assert torch.allclose(pos[0], expected, atol=1e-5, rtol=1e-5)

    def test_running_vel_at_t0(self, running_manager):
        _, vel = running_manager.get_output(torch.tensor([0.0]))
        expected = torch.tensor(GOLDEN_RUNNING_VEL_T0)
        assert torch.allclose(vel[0], expected, atol=1e-5, rtol=1e-5)

    def test_running_pos_at_t005(self, running_manager):
        pos, _ = running_manager.get_output(torch.tensor([0.05]))
        expected = torch.tensor(GOLDEN_RUNNING_POS_T005)
        assert torch.allclose(pos[0], expected, atol=1e-5, rtol=1e-5)

    def test_running_vel_at_t005(self, running_manager):
        _, vel = running_manager.get_output(torch.tensor([0.05]))
        expected = torch.tensor(GOLDEN_RUNNING_VEL_T005)
        assert torch.allclose(vel[0], expected, atol=1e-5, rtol=1e-5)
