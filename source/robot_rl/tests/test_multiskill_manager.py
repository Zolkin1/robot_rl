"""Tests for MultiSkillManager.

Cross-validates batched results against individual TrajectoryManager outputs
to ensure correctness of the batched tensor implementation.
"""

import pytest
import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.multiskill_manager import (
    MultiSkillManager,
    ConditionerData,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_manager import (
    TrajectoryManager,
    TrajectoryType,
)
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.library_manager import (
    LibraryManager,
)
from conftest import MERGED_LIBRARY_DIR, STANDING_YAML, WALKING_YAML, RUNNING_YAML, DEVICE, TEST_DATA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def multiskill_single_skill() -> MultiSkillManager:
    """MultiSkillManager with all trajectories as a single 'default' skill."""
    return MultiSkillManager(
        path=MERGED_LIBRARY_DIR,
        device=DEVICE,
    )


@pytest.fixture(scope="module")
def reference_managers() -> list[TrajectoryManager]:
    """Load each trajectory YAML individually for cross-validation."""
    from pathlib import Path
    yamls = sorted(Path(MERGED_LIBRARY_DIR).glob("*.yaml"))
    managers = []
    for y in yamls:
        mgr = TrajectoryManager(str(y), None, DEVICE)
        managers.append(mgr)
    # Sort by first conditioner value (same as MultiSkillManager/LibraryManager)
    managers.sort(key=lambda m: m.traj_data.conditioner[0])
    return managers


@pytest.fixture(scope="module")
def reference_library() -> LibraryManager:
    """LibraryManager for comparison."""
    return LibraryManager(MERGED_LIBRARY_DIR, None, DEVICE)


# ---------------------------------------------------------------------------
# Construction & Loading
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_loads_all_trajectories(self, multiskill_single_skill, reference_managers):
        """Should load the same number of trajectories as individual loading."""
        assert multiskill_single_skill.num_trajectories == len(reference_managers)

    def test_single_skill_created(self, multiskill_single_skill):
        """With one folder, should have exactly one skill."""
        assert len(multiskill_single_skill.skills) == 1
        assert multiskill_single_skill.skills[0].name == "default"

    def test_tensor_shapes(self, multiskill_single_skill):
        """Global tensors should have correct shapes."""
        msm = multiskill_single_skill
        T = msm.num_trajectories
        D = msm.max_expanded_domains
        P = msm.num_pos_outputs
        V = msm.num_vel_outputs
        K = msm.spline_order

        assert msm.data["coeffs_pos"].shape == (T, D, P, K + 1)
        assert msm.data["coeffs_vel"].shape == (T, D, V, K + 1)
        assert msm.data["domain_times"].shape == (T, D)
        assert msm.data["domain_boundaries"].shape == (T, D + 1)
        assert msm.data["total_time"].shape == (T,)
        assert msm.data["expanded_domains"].shape == (T,)
        assert msm.data["traj_type"].shape == (T,)
        assert msm.data["skill_idx"].shape == (T,)

    def test_output_counts_match_reference(self, multiskill_single_skill, reference_managers):
        """Pos/vel output counts should match reference trajectories."""
        ref = reference_managers[0]
        assert multiskill_single_skill.num_pos_outputs == ref.traj_data.num_pos_outputs
        assert multiskill_single_skill.num_vel_outputs == ref.traj_data.num_vel_outputs

    def test_output_names_match_reference(self, multiskill_single_skill, reference_managers):
        """Output names should match reference trajectories."""
        ref = reference_managers[0]
        assert multiskill_single_skill.pos_output_names == ref.traj_data.pos_output_names
        assert multiskill_single_skill.vel_output_names == ref.traj_data.vel_output_names

    def test_domain_boundaries_padded(self, multiskill_single_skill):
        """Padded boundary slots should equal the last real boundary."""
        msm = multiskill_single_skill
        for ti in range(msm.num_trajectories):
            ed = msm.data["expanded_domains"][ti].item()
            last_boundary = msm.data["domain_boundaries"][ti, ed].item()
            for di in range(ed + 1, msm.max_expanded_domains + 1):
                assert msm.data["domain_boundaries"][ti, di].item() == last_boundary

    def test_total_time_correct(self, multiskill_single_skill, reference_managers):
        """Total time should match reference managers."""
        msm = multiskill_single_skill
        for ti, ref in enumerate(reference_managers):
            if ref.traj_data.trajectory_type == TrajectoryType.HALF_PERIODIC:
                expected = ref.traj_data.total_time * 2
            else:
                expected = ref.traj_data.total_time
            actual = msm.data["total_time"][ti].item()
            assert actual == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# Conditioner Parsing
# ---------------------------------------------------------------------------

class TestConditionerParsing:
    def test_old_format_list(self):
        """Old-format [min, max] should parse to midpoint vel_x."""
        cond = MultiSkillManager._parse_conditioner([0.8, 1.2], "test")
        assert cond.vel_x == pytest.approx(1.0)
        assert cond.vel_y == 0.0
        assert cond.terrain == "flat"

    def test_new_format_dict(self):
        """New-format dict should parse all fields."""
        raw = {
            "vel_x": 1.5, "vel_y": 0.1, "vel_yaw": 0.3,
            "terrain": "stairs", "terrain_height": 0.15,
        }
        cond = MultiSkillManager._parse_conditioner(raw, "test")
        assert cond.vel_x == 1.5
        assert cond.vel_y == 0.1
        assert cond.vel_yaw == 0.3
        assert cond.terrain == "stairs"
        assert cond.terrain_height == 0.15
        assert cond.terrain_width == 0.0  # default

    def test_to_continuous_tensor(self):
        """to_continuous_tensor should return 6 floats."""
        cond = ConditionerData(vel_x=1.0, vel_y=2.0, vel_yaw=3.0)
        vals = cond.to_continuous_tensor()
        assert len(vals) == 6
        assert vals[0] == 1.0
        assert vals[1] == 2.0
        assert vals[2] == 3.0


# ---------------------------------------------------------------------------
# Trajectory Selection
# ---------------------------------------------------------------------------

class TestTrajectorySelection:
    def test_nearest_neighbor_exact(self, multiskill_single_skill):
        """Exact match on conditioning should select that trajectory."""
        msm = multiskill_single_skill
        skill = msm.skills[0]

        # Use the first trajectory's conditioning vector (pad to 6 dims)
        cond_vec = skill.conditioning_tensor[0:1]  # [1, C]
        global_idx = msm._select_trajectories(cond_vec)
        assert global_idx.item() == skill.traj_indices[0]

    def test_nearest_neighbor_batch(self, multiskill_single_skill):
        """Batch selection should return valid indices."""
        msm = multiskill_single_skill

        N = 10
        cond = torch.randn(N, 6)
        global_idx = msm._select_trajectories(cond)
        assert global_idx.shape == (N,)
        assert (global_idx >= 0).all()
        assert (global_idx < msm.num_trajectories).all()


# ---------------------------------------------------------------------------
# Cross-Validation: Output
# ---------------------------------------------------------------------------

class TestBatchedOutput:
    """Cross-validate batched MultiSkillManager output against individual TrajectoryManagers."""

    @pytest.mark.parametrize("traj_idx", [0, 5, 10, 18])
    def test_get_output_matches_reference(
        self, multiskill_single_skill, reference_managers, traj_idx
    ):
        """Batched output should match per-trajectory TrajectoryManager output."""
        if traj_idx >= len(reference_managers):
            pytest.skip("Not enough trajectories")

        msm = multiskill_single_skill
        ref = reference_managers[traj_idx]
        N = 8

        # Test times spanning the trajectory
        total_t = msm.data["total_time"][traj_idx].item()
        t = torch.linspace(0.001, total_t * 0.99, N)

        # Set all envs to this trajectory
        msm.set_trajectory_indices(torch.full((N,), traj_idx, dtype=torch.long))

        pos_msm, vel_msm = msm.get_output(t)
        pos_ref, vel_ref = ref.get_output(t)

        assert torch.allclose(pos_msm, pos_ref, atol=1e-4, rtol=1e-4), (
            f"Traj {traj_idx} pos mismatch: max diff = {(pos_msm - pos_ref).abs().max()}"
        )
        assert torch.allclose(vel_msm, vel_ref, atol=1e-4, rtol=1e-4), (
            f"Traj {traj_idx} vel mismatch: max diff = {(vel_msm - vel_ref).abs().max()}"
        )

    def test_multi_traj_dispatch(self, multiskill_single_skill, reference_managers):
        """With mixed trajectory assignments, each env should match its reference."""
        msm = multiskill_single_skill
        num_traj = len(reference_managers)

        # 4 envs: first 2 use traj 0, last 2 use last traj
        idx_a, idx_b = 0, num_traj - 1
        indices = torch.tensor([idx_a, idx_a, idx_b, idx_b], dtype=torch.long)
        msm.set_trajectory_indices(indices)

        t = torch.tensor([0.05, 0.1, 0.05, 0.1])

        pos_msm, vel_msm = msm.get_output(t)

        pos_a, vel_a = reference_managers[idx_a].get_output(t[:2])
        pos_b, vel_b = reference_managers[idx_b].get_output(t[2:])

        assert torch.allclose(pos_msm[:2], pos_a, atol=1e-4)
        assert torch.allclose(vel_msm[:2], vel_a, atol=1e-4)
        assert torch.allclose(pos_msm[2:], pos_b, atol=1e-4)
        assert torch.allclose(vel_msm[2:], vel_b, atol=1e-4)

    def test_output_at_t_zero(self, multiskill_single_skill, reference_managers):
        """Output at t=0 should match reference for all trajectories."""
        msm = multiskill_single_skill
        T = msm.num_trajectories
        t = torch.full((T,), 0.001)  # Slightly off zero to avoid boundary
        indices = torch.arange(T, dtype=torch.long)
        msm.set_trajectory_indices(indices)

        pos_msm, vel_msm = msm.get_output(t)

        for ti in range(T):
            pos_ref, vel_ref = reference_managers[ti].get_output(t[ti:ti+1])
            assert torch.allclose(pos_msm[ti:ti+1], pos_ref, atol=1e-4), (
                f"Traj {ti} t=0 pos mismatch"
            )
            assert torch.allclose(vel_msm[ti:ti+1], vel_ref, atol=1e-4), (
                f"Traj {ti} t=0 vel mismatch"
            )


# ---------------------------------------------------------------------------
# Cross-Validation: Phasing Variable
# ---------------------------------------------------------------------------

class TestPhasingVar:
    @pytest.mark.parametrize("traj_idx", [0, 5, 10, 18])
    def test_phasing_matches_reference(
        self, multiskill_single_skill, reference_managers, traj_idx
    ):
        """Phasing variable should match per-trajectory result."""
        if traj_idx >= len(reference_managers):
            pytest.skip("Not enough trajectories")

        msm = multiskill_single_skill
        ref = reference_managers[traj_idx]
        N = 8

        total_t = msm.data["total_time"][traj_idx].item()
        t = torch.linspace(0.001, total_t * 0.99, N)

        msm.set_trajectory_indices(torch.full((N,), traj_idx, dtype=torch.long))
        phi_msm = msm.get_phasing_var(t)
        phi_ref = ref.get_phasing_var(t)

        assert torch.allclose(phi_msm, phi_ref, atol=1e-5), (
            f"Traj {traj_idx} phasing mismatch: max diff = {(phi_msm - phi_ref).abs().max()}"
        )

    def test_phasing_range(self, multiskill_single_skill):
        """Phasing should be in [0, 1]."""
        msm = multiskill_single_skill
        N = 20
        t = torch.rand(N) * 2.0
        msm.set_trajectory_indices(torch.zeros(N, dtype=torch.long))
        phi = msm.get_phasing_var(t)
        assert (phi >= 0).all()
        assert (phi <= 1).all()


# ---------------------------------------------------------------------------
# Cross-Validation: Domains
# ---------------------------------------------------------------------------

class TestDomains:
    @pytest.mark.parametrize("traj_idx", [0, 5, 10, 18])
    def test_domains_match_reference(
        self, multiskill_single_skill, reference_managers, traj_idx
    ):
        """Domain indices should match per-trajectory result."""
        if traj_idx >= len(reference_managers):
            pytest.skip("Not enough trajectories")

        msm = multiskill_single_skill
        ref = reference_managers[traj_idx]
        N = 8

        total_t = msm.data["total_time"][traj_idx].item()
        t = torch.linspace(0.001, total_t * 0.99, N)

        msm.set_trajectory_indices(torch.full((N,), traj_idx, dtype=torch.long))
        dom_msm = msm.get_current_domains(t)
        dom_ref = ref.get_current_domains(t)

        assert torch.equal(dom_msm, dom_ref), (
            f"Traj {traj_idx} domain mismatch:\nMSM: {dom_msm}\nRef: {dom_ref}"
        )

    @pytest.mark.parametrize("traj_idx", [0, 5, 10, 18])
    def test_domain_times_match_reference(
        self, multiskill_single_skill, reference_managers, traj_idx
    ):
        """Domain durations should match per-trajectory result."""
        if traj_idx >= len(reference_managers):
            pytest.skip("Not enough trajectories")

        msm = multiskill_single_skill
        ref = reference_managers[traj_idx]
        N = 8

        total_t = msm.data["total_time"][traj_idx].item()
        t = torch.linspace(0.001, total_t * 0.99, N)

        msm.set_trajectory_indices(torch.full((N,), traj_idx, dtype=torch.long))
        dt_msm = msm.get_domain_times(t)
        dt_ref = ref.get_domain_times(t)

        assert torch.allclose(dt_msm, dt_ref, atol=1e-6), (
            f"Traj {traj_idx} domain time mismatch"
        )

    def test_padded_domains_never_selected(self, multiskill_single_skill):
        """searchsorted should never land in a padded domain slot."""
        msm = multiskill_single_skill
        T = msm.num_trajectories
        N = T
        t = torch.rand(N) * 2.0
        indices = torch.arange(T, dtype=torch.long)
        msm.set_trajectory_indices(indices)

        dom = msm.get_current_domains(t)
        max_valid = msm.data["expanded_domains"][indices] - 1
        assert (dom <= max_valid).all()


# ---------------------------------------------------------------------------
# Cross-Validation: Acceleration
# ---------------------------------------------------------------------------

class TestAcceleration:
    @pytest.mark.parametrize("traj_idx", [0, 5, 10])
    def test_acceleration_matches_reference(
        self, multiskill_single_skill, reference_managers, traj_idx
    ):
        """Acceleration should match per-trajectory result."""
        if traj_idx >= len(reference_managers):
            pytest.skip("Not enough trajectories")

        msm = multiskill_single_skill
        ref = reference_managers[traj_idx]
        N = 8

        total_t = msm.data["total_time"][traj_idx].item()
        t = torch.linspace(0.01, total_t * 0.9, N)

        msm.set_trajectory_indices(torch.full((N,), traj_idx, dtype=torch.long))
        acc_msm = msm.get_acceleration(t)
        acc_ref = ref.get_acceleration(t)

        assert torch.allclose(acc_msm, acc_ref, atol=1e-3, rtol=1e-3), (
            f"Traj {traj_idx} accel mismatch: max diff = {(acc_msm - acc_ref).abs().max()}"
        )


# ---------------------------------------------------------------------------
# Cross-Validation: Contact State
# ---------------------------------------------------------------------------

class TestContactState:
    @pytest.mark.parametrize("traj_idx", [0, 5, 10])
    def test_contact_state_matches_reference(
        self, multiskill_single_skill, reference_managers, traj_idx
    ):
        """Contact states should match per-trajectory result."""
        if traj_idx >= len(reference_managers):
            pytest.skip("Not enough trajectories")

        msm = multiskill_single_skill
        ref = reference_managers[traj_idx]
        N = 8
        contact_frames = ["right_ankle_roll_link", "left_ankle_roll_link"]

        total_t = msm.data["total_time"][traj_idx].item()
        t = torch.linspace(0.001, total_t * 0.99, N)

        msm.set_trajectory_indices(torch.full((N,), traj_idx, dtype=torch.long))
        cs_msm = msm.get_contact_state(t, contact_frames)
        cs_ref = ref.get_contact_state(t, contact_frames)

        assert torch.equal(cs_msm, cs_ref), (
            f"Traj {traj_idx} contact mismatch:\nMSM: {cs_msm}\nRef: {cs_ref}"
        )


# ---------------------------------------------------------------------------
# Output Ordering
# ---------------------------------------------------------------------------

class TestOrderOutputs:
    def test_order_outputs_reversed(self, reference_managers):
        """Reversing output order should still produce correct results."""
        msm = MultiSkillManager(
            path=MERGED_LIBRARY_DIR,
            device=DEVICE,
        )
        ref = reference_managers[0]

        # Get output in original order
        N = 4
        t = torch.tensor([0.01, 0.05, 0.1, 0.15])
        msm.set_trajectory_indices(torch.zeros(N, dtype=torch.long))
        pos_orig, vel_orig = msm.get_output(t)

        # Reverse output names
        rev_pos = list(reversed(msm.pos_output_names))
        rev_vel = list(reversed(msm.vel_output_names))
        msm.order_outputs(rev_pos, rev_vel)

        pos_rev, vel_rev = msm.get_output(t)

        # Reversed output should be the original flipped
        assert torch.allclose(pos_rev, pos_orig.flip(1), atol=1e-6)
        assert torch.allclose(vel_rev, vel_orig.flip(1), atol=1e-6)


# ---------------------------------------------------------------------------
# CLF Logging
# ---------------------------------------------------------------------------

class TestCLFLogging:
    def test_log_and_retrieve(self, multiskill_single_skill):
        """Should be able to log V values and retrieve them."""
        msm = multiskill_single_skill
        N = 10
        msm.set_trajectory_indices(torch.zeros(N, dtype=torch.long))

        phi = torch.linspace(0.05, 0.95, N)
        v = torch.ones(N) * 2.0

        msm.log_v_on_phasing_var(phi, v)

        v_log, phi_keys = msm.get_v_log()
        assert v_log.shape == phi_keys.shape
        assert phi_keys.shape[0] == 10

    def test_per_skill_logs(self, multiskill_single_skill):
        """Per-skill V logs should exist for each skill."""
        msm = multiskill_single_skill
        per_skill = msm.get_v_log_per_skill()
        assert "default" in per_skill
        assert per_skill["default"].shape[0] == 10

    def test_v_log_avg(self, multiskill_single_skill):
        """Average V should return one value per skill."""
        msm = multiskill_single_skill
        avg = msm.get_v_log_avg()
        assert avg.shape == (len(msm.skills),)


# ---------------------------------------------------------------------------
# Placeholder Stubs
# ---------------------------------------------------------------------------

class TestPlaceholders:
    def test_terrain_relative_identity(self, multiskill_single_skill):
        """transform_terrain_relative should return input unchanged."""
        msm = multiskill_single_skill
        positions = torch.randn(4, 3)
        result = msm.transform_terrain_relative(positions)
        assert torch.equal(result, positions)

    def test_skill_transition_identity(self, multiskill_single_skill):
        """interpolate_skill_transition should return first argument."""
        msm = multiskill_single_skill
        a = torch.randn(4, 10)
        b = torch.randn(4, 10)
        alpha = torch.tensor([0.5, 0.5, 0.5, 0.5])
        result = msm.interpolate_skill_transition(a, b, alpha)
        assert torch.equal(result, a)


# ---------------------------------------------------------------------------
# Properties & Interface
# ---------------------------------------------------------------------------

class TestInterface:
    def test_get_num_outputs(self, multiskill_single_skill):
        """get_num_outputs should return pos output count."""
        assert multiskill_single_skill.get_num_outputs() == multiskill_single_skill.num_pos_outputs

    def test_get_num_pos_outputs(self, multiskill_single_skill):
        assert multiskill_single_skill.get_num_pos_outputs() == multiskill_single_skill.num_pos_outputs

    def test_get_num_vel_outputs(self, multiskill_single_skill):
        assert multiskill_single_skill.get_num_vel_outputs() == multiskill_single_skill.num_vel_outputs

    def test_get_output_names(self, multiskill_single_skill):
        assert multiskill_single_skill.get_output_names == multiskill_single_skill.pos_output_names

    def test_get_reference_frames(self, multiskill_single_skill):
        assert isinstance(multiskill_single_skill.get_reference_frames(), list)

    def test_invalidate_cache(self, multiskill_single_skill):
        """Cache invalidation should clear the valid flag."""
        msm = multiskill_single_skill
        msm.set_trajectory_indices(torch.zeros(4, dtype=torch.long))
        msm.invalidate_cache()
        assert msm._cache_valid is False


# ---------------------------------------------------------------------------
# Multi-Skill Loading (using separate folders)
# ---------------------------------------------------------------------------

class TestMultiSkill:
    @pytest.fixture()
    def multi_skill_dir(self, tmp_path):
        """Create a directory with two skill subfolders, each symlinking the merged library."""
        import shutil
        from pathlib import Path

        skill_a = tmp_path / "skill_a"
        skill_b = tmp_path / "skill_b"

        # Copy YAMLs into two separate subfolders
        shutil.copytree(MERGED_LIBRARY_DIR, str(skill_a))
        shutil.copytree(MERGED_LIBRARY_DIR, str(skill_b))

        return tmp_path

    def test_two_skills_discovered(self, multi_skill_dir):
        """Subdirectories should be discovered as separate skills."""
        msm = MultiSkillManager(path=str(multi_skill_dir), device=DEVICE)
        # Each subfolder has 19 trajectories
        assert msm.num_trajectories == 38
        assert len(msm.skills) == 2
        assert msm.skills[0].num_trajectories == 19
        assert msm.skills[1].num_trajectories == 19

    def test_skill_indices_distinct(self, multi_skill_dir):
        """Different skills should have non-overlapping trajectory indices."""
        msm = MultiSkillManager(path=str(multi_skill_dir), device=DEVICE)
        indices_a = set(msm.skills[0].traj_indices)
        indices_b = set(msm.skills[1].traj_indices)
        assert len(indices_a & indices_b) == 0

    def test_skill_selection_routes_by_velocity(self, multi_skill_dir):
        """Global selection should pick the trajectory closest to the commanded velocity."""
        msm = MultiSkillManager(path=str(multi_skill_dir), device=DEVICE)

        # Zero velocity should select the trajectory with lowest vel_x
        cond_zero = torch.zeros(5, 6)
        idx_zero = msm._select_trajectories(cond_zero)
        assert idx_zero.shape == (5,)
        # All envs should get the same trajectory (all have same conditioning)
        assert (idx_zero == idx_zero[0]).all()

        # High velocity should select a different trajectory than zero
        cond_high = torch.zeros(5, 6)
        cond_high[:, 0] = 100.0  # very high vel_x
        idx_high = msm._select_trajectories(cond_high)
        assert (idx_high == idx_high[0]).all()
        # The high-velocity trajectory should differ from the zero-velocity one
        # (assuming skills have different velocity ranges)
        assert idx_high[0].item() != idx_zero[0].item()

    def test_single_folder_fallback(self):
        """A flat folder with YAMLs (no subfolders) should create a 'default' skill."""
        msm = MultiSkillManager(path=MERGED_LIBRARY_DIR, device=DEVICE)
        assert len(msm.skills) == 1
        assert msm.skills[0].name == "default"
        assert msm.num_trajectories == 19


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_empty_folder_raises(self, tmp_path):
        """Empty folder with no YAMLs and no subfolders should raise."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No skill subfolders"):
            MultiSkillManager(path=str(empty), device=DEVICE)

    def test_missing_folder_raises(self):
        """Non-existent folder should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            MultiSkillManager(path="/nonexistent/path", device=DEVICE)

    def test_no_env_cache_raises(self, multiskill_single_skill):
        """_ensure_cache without env or set_trajectory_indices should raise."""
        msm = multiskill_single_skill
        msm.invalidate_cache()
        with pytest.raises(RuntimeError, match="Cannot auto-populate"):
            msm._ensure_cache()
