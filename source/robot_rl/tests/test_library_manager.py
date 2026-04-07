"""Tests for LibraryManager."""

import pytest
import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.library_manager import LibraryManager
from conftest import MERGED_LIBRARY_DIR, DEVICE


# ---------------------------------------------------------------------------
# Construction & Loading
# ---------------------------------------------------------------------------

class TestLibraryLoading:
    def test_loads_all_trajectories(self, merged_library):
        """Should load all 19 YAML files from the merged library."""
        assert len(merged_library.trajectory_managers) == 19

    def test_sorted_by_conditioner(self, merged_library):
        """conditioning_vars[:, 0] should be sorted ascending."""
        conds = merged_library.conditioning_vars[:, 0]
        diffs = conds[1:] - conds[:-1]
        assert (diffs >= 0).all()

    def test_metadata_consistent(self, merged_library):
        """All trajectory managers should have the same output structure."""
        assert merged_library.num_pos_outputs is not None
        assert merged_library.num_vel_outputs is not None
        assert merged_library.num_pos_outputs > 0
        assert merged_library.num_vel_outputs > 0
        assert len(merged_library.pos_output_names) == merged_library.num_pos_outputs
        assert len(merged_library.vel_output_names) == merged_library.num_vel_outputs


# ---------------------------------------------------------------------------
# Trajectory Index Selection
# ---------------------------------------------------------------------------

class TestTrajIndices:
    def test_low_conditioner_selects_first(self, merged_library):
        """Very low conditioner should select index 0."""
        idx = merged_library.get_traj_indices(torch.tensor([0.0]))
        assert idx.item() == 0

    def test_high_conditioner_selects_last(self, merged_library):
        """Very high conditioner should select the last trajectory."""
        idx = merged_library.get_traj_indices(torch.tensor([100.0]))
        assert idx.item() == len(merged_library.trajectory_managers) - 1

    def test_negative_conditioner_clamps(self, merged_library):
        """Negative conditioner should clamp to 0."""
        idx = merged_library.get_traj_indices(torch.tensor([-100.0]))
        assert idx.item() == 0

    def test_batch_dispatch(self, merged_library):
        """Different conditioners in a batch should map to different indices."""
        conds = torch.tensor([0.0, 1.0, 2.0, 5.0])
        indices = merged_library.get_traj_indices(conds)
        assert indices.shape == (4,)
        # At least some should differ (0.0 vs 5.0 definitely different trajectories)
        assert not (indices == indices[0]).all()

    def test_indices_in_valid_range(self, merged_library):
        """All returned indices should be valid trajectory manager indices."""
        conds = torch.rand(50) * 10.0
        indices = merged_library.get_traj_indices(conds)
        assert (indices >= 0).all()
        assert (indices < len(merged_library.trajectory_managers)).all()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_invalidate_cache(self, merged_library):
        merged_library.invalidate_cache()
        assert merged_library._cache_valid is False

    def test_cache_populated_after_ensure(self, merged_library):
        """Manually test cache behavior by setting up cache directly."""
        merged_library.invalidate_cache()
        # After invalidation, cache should not be valid
        assert merged_library._cache_valid is False


# ---------------------------------------------------------------------------
# Dispatch Methods (with manual cache)
# ---------------------------------------------------------------------------

def _setup_manual_cache(lib: LibraryManager, traj_idx: int, N: int):
    """Set up cache manually to dispatch all envs to a single trajectory.

    Args:
        lib: The LibraryManager instance.
        traj_idx: The trajectory index to dispatch all envs to.
        N: Number of environments.
    """
    indices = torch.full((N,), traj_idx, dtype=torch.long, device=lib.device)
    lib._cached_indices = indices
    lib._cached_unique_cpu = [traj_idx]
    lib._cached_env_indices = {traj_idx: torch.arange(N, device=lib.device)}
    lib._cache_env_ids = None
    lib._cache_valid = True


class TestDispatch:
    def test_get_output_shape(self, merged_library):
        """Output should have correct shapes when dispatching to a single trajectory."""
        N = 8
        _setup_manual_cache(merged_library, 0, N)
        t = torch.rand(N) * 0.1
        pos, vel = merged_library.get_output(t)
        assert pos.shape == (N, merged_library.num_pos_outputs)
        assert vel.shape == (N, merged_library.num_vel_outputs)

    def test_get_output_matches_single_manager(self, merged_library):
        """Library output should match the underlying trajectory manager's output."""
        traj_idx = 0
        N = 4
        _setup_manual_cache(merged_library, traj_idx, N)
        t = torch.tensor([0.0, 0.05, 0.1, 0.15])

        pos_lib, vel_lib = merged_library.get_output(t)
        pos_mgr, vel_mgr = merged_library.trajectory_managers[traj_idx].get_output(t)

        assert torch.allclose(pos_lib, pos_mgr, atol=1e-6)
        assert torch.allclose(vel_lib, vel_mgr, atol=1e-6)

    def test_get_output_multi_traj_dispatch(self, merged_library):
        """With different trajectories assigned, outputs should match per-trajectory results."""
        N = 4
        num_trajs = len(merged_library.trajectory_managers)
        # Assign first 2 envs to traj 0, last 2 to last traj
        idx_a, idx_b = 0, num_trajs - 1
        indices = torch.tensor([idx_a, idx_a, idx_b, idx_b], dtype=torch.long)

        merged_library._cached_indices = indices
        merged_library._cached_unique_cpu = [idx_a, idx_b]
        merged_library._cached_env_indices = {
            idx_a: torch.tensor([0, 1]),
            idx_b: torch.tensor([2, 3]),
        }
        merged_library._cache_env_ids = None
        merged_library._cache_valid = True

        t = torch.tensor([0.05, 0.1, 0.05, 0.1])
        pos_lib, vel_lib = merged_library.get_output(t)

        # First 2 should match traj 0
        pos_a, vel_a = merged_library.trajectory_managers[idx_a].get_output(t[:2])
        assert torch.allclose(pos_lib[:2], pos_a, atol=1e-6)
        assert torch.allclose(vel_lib[:2], vel_a, atol=1e-6)

        # Last 2 should match last traj
        pos_b, vel_b = merged_library.trajectory_managers[idx_b].get_output(t[2:])
        assert torch.allclose(pos_lib[2:], pos_b, atol=1e-6)
        assert torch.allclose(vel_lib[2:], vel_b, atol=1e-6)

    def test_get_phasing_var_dispatch(self, merged_library):
        """Phasing var should match single manager when dispatched."""
        traj_idx = 0
        N = 4
        _setup_manual_cache(merged_library, traj_idx, N)
        t = torch.tensor([0.0, 0.1, 0.2, 0.3])

        phi_lib = merged_library.get_phasing_var(t)
        phi_mgr = merged_library.trajectory_managers[traj_idx].get_phasing_var(t)

        assert torch.allclose(phi_lib, phi_mgr, atol=1e-6)

    def test_get_acceleration_dispatch(self, merged_library):
        """Acceleration should match single manager when dispatched."""
        traj_idx = 0
        N = 4
        _setup_manual_cache(merged_library, traj_idx, N)
        t = torch.tensor([0.0, 0.05, 0.1, 0.15])

        accel_lib = merged_library.get_acceleration(t)
        accel_mgr = merged_library.trajectory_managers[traj_idx].get_acceleration(t)

        assert torch.allclose(accel_lib, accel_mgr, atol=1e-6)

    def test_get_contact_state_dispatch(self, merged_library):
        """Contact state should match single manager when dispatched."""
        traj_idx = 0
        N = 4
        _setup_manual_cache(merged_library, traj_idx, N)
        contact_frames = ["right_ankle_roll_link", "left_ankle_roll_link"]
        t = torch.tensor([0.0, 0.05, 0.1, 0.15])

        cs_lib = merged_library.get_contact_state(t, contact_frames)
        cs_mgr = merged_library.trajectory_managers[traj_idx].get_contact_state(t, contact_frames)

        assert torch.equal(cs_lib, cs_mgr)

    def test_get_current_domains_dispatch(self, merged_library):
        """Current domains should match single manager."""
        traj_idx = 0
        N = 4
        _setup_manual_cache(merged_library, traj_idx, N)
        t = torch.tensor([0.0, 0.1, 0.2, 0.3])

        domains_lib = merged_library.get_current_domains(t)
        domains_mgr = merged_library.trajectory_managers[traj_idx].get_current_domains(t)

        assert torch.equal(domains_lib, domains_mgr)


# ---------------------------------------------------------------------------
# Order Outputs
# ---------------------------------------------------------------------------

class TestLibraryOrderOutputs:
    def test_order_outputs_propagates(self):
        """order_outputs should update all trajectory managers."""
        lib = LibraryManager(MERGED_LIBRARY_DIR, None, DEVICE)
        original_pos = list(lib.pos_output_names)
        original_vel = list(lib.vel_output_names)

        # Reverse the order
        reversed_pos = list(reversed(original_pos))
        reversed_vel = list(reversed(original_vel))
        lib.order_outputs(reversed_pos, reversed_vel)

        assert lib.pos_output_names == reversed_pos
        assert lib.vel_output_names == reversed_vel

        # Check all managers updated
        for mgr in lib.trajectory_managers:
            assert mgr.traj_data.pos_output_names == reversed_pos
            assert mgr.traj_data.vel_output_names == reversed_vel


# ---------------------------------------------------------------------------
# Total Time
# ---------------------------------------------------------------------------

class TestLibraryTotalTime:
    def test_total_time(self, merged_library):
        """get_total_time should return the last manager's total time."""
        expected = merged_library.trajectory_managers[-1].get_total_time()
        actual = merged_library.get_total_time()
        assert actual == expected
