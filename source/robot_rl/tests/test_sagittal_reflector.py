"""Tests for the sagittal reflector utility."""

import numpy as np
import pytest
import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.sagittal_reflector import (
    NamedReflector,
    SagittalReflectionConfig,
    swap_left_right,
    _should_negate,
)

DEVICE = "cpu"


# ---------------------------------------------------------------------------
# TestSwapLeftRight
# ---------------------------------------------------------------------------

class TestSwapLeftRight:
    def test_basic_swap_left_to_right(self):
        assert swap_left_right("left_ankle") == "right_ankle"

    def test_basic_swap_right_to_left(self):
        assert swap_left_right("right_hip") == "left_hip"

    def test_no_swap(self):
        assert swap_left_right("com:pos_x") == "com:pos_x"

    def test_both_tokens(self):
        """Names containing both tokens should still swap correctly."""
        result = swap_left_right("left_right_thing")
        assert result == "right_left_thing"

    def test_custom_config_tokens(self):
        cfg = SagittalReflectionConfig(left_token="L", right_token="R")
        assert swap_left_right("L_arm", cfg) == "R_arm"
        assert swap_left_right("R_leg", cfg) == "L_leg"
        assert swap_left_right("center", cfg) == "center"


# ---------------------------------------------------------------------------
# TestShouldNegate
# ---------------------------------------------------------------------------

class TestShouldNegate:
    def test_pos_y_negated(self):
        cfg = SagittalReflectionConfig()
        assert _should_negate("com:pos_y", cfg) is True

    def test_pos_x_not_negated(self):
        cfg = SagittalReflectionConfig()
        assert _should_negate("com:pos_x", cfg) is False

    def test_ori_x_negated(self):
        cfg = SagittalReflectionConfig()
        assert _should_negate("left_ankle_roll_link:ori_x", cfg) is True

    def test_ori_w_not_negated(self):
        cfg = SagittalReflectionConfig()
        assert _should_negate("left_ankle_roll_link:ori_w", cfg) is False

    def test_roll_joint_negated(self):
        cfg = SagittalReflectionConfig()
        assert _should_negate("joint:left_hip_roll_joint", cfg) is True

    def test_yaw_joint_negated(self):
        cfg = SagittalReflectionConfig()
        assert _should_negate("joint:waist_yaw_joint", cfg) is True

    def test_pitch_joint_not_negated(self):
        cfg = SagittalReflectionConfig()
        assert _should_negate("joint:left_hip_pitch_joint", cfg) is False

    def test_knee_joint_not_negated(self):
        cfg = SagittalReflectionConfig()
        assert _should_negate("joint:left_knee_joint", cfg) is False

    def test_elbow_joint_not_negated(self):
        cfg = SagittalReflectionConfig()
        assert _should_negate("joint:left_elbow_joint", cfg) is False


# ---------------------------------------------------------------------------
# TestNamedReflector
# ---------------------------------------------------------------------------

class TestNamedReflector:
    """Unit tests for NamedReflector on synthetic name lists."""

    @pytest.fixture
    def simple_names(self):
        return [
            "left_a:pos_x",
            "right_a:pos_x",
            "left_a:pos_y",
            "right_a:pos_y",
        ]

    @pytest.fixture
    def simple_reflector(self, simple_names):
        return NamedReflector(SagittalReflectionConfig(), simple_names, DEVICE)

    def test_perm_swaps_pairs(self, simple_reflector):
        perm = simple_reflector.perm_indices.tolist()
        # left_a:pos_x (0) <-> right_a:pos_x (1)
        assert perm[0] == 1
        assert perm[1] == 0
        # left_a:pos_y (2) <-> right_a:pos_y (3)
        assert perm[2] == 3
        assert perm[3] == 2

    def test_sign_negates_pos_y(self, simple_reflector):
        sign = simple_reflector.sign_vector.tolist()
        assert sign[0] == 1.0   # pos_x
        assert sign[1] == 1.0   # pos_x
        assert sign[2] == -1.0  # pos_y
        assert sign[3] == -1.0  # pos_y

    def test_reflect_matches_manual(self, simple_reflector):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        result = simple_reflector.reflect(x)
        # perm: [1, 0, 3, 2], sign: [1, 1, -1, -1]
        expected = torch.tensor([[2.0, 1.0, -4.0, -3.0]])
        assert torch.allclose(result, expected)

    def test_double_reflect_is_identity(self, simple_reflector):
        x = torch.randn(5, 4)
        result = simple_reflector.reflect(simple_reflector.reflect(x))
        assert torch.allclose(result, x, atol=1e-6)

    def test_build_relabel_matrix_shape(self, simple_reflector):
        R = simple_reflector.build_relabel_matrix()
        assert R.shape == (4, 4)

    def test_relabel_matrix_is_signed_permutation(self, simple_reflector):
        R = simple_reflector.build_relabel_matrix()
        for i in range(R.shape[0]):
            nonzero = R[i][R[i] != 0]
            assert len(nonzero) == 1, f"Row {i} has {len(nonzero)} nonzero entries"
            assert abs(nonzero.item()) == 1.0

    def test_relabel_matrix_double_is_identity(self, simple_reflector):
        R = simple_reflector.build_relabel_matrix()
        R2 = R @ R
        assert torch.allclose(R2, torch.eye(4), atol=1e-6)

    def test_reflect_equals_matrix_multiply(self, simple_reflector):
        x = torch.randn(5, 4)
        reflected = simple_reflector.reflect(x)
        R = simple_reflector.build_relabel_matrix()
        mat_result = (R @ x.T).T
        assert torch.allclose(reflected, mat_result, atol=1e-6)

    def test_center_name_no_swap(self):
        """Names without left/right stay at their own index."""
        names = ["com:pos_x", "com:pos_y", "com:pos_z"]
        reflector = NamedReflector(SagittalReflectionConfig(), names, DEVICE)
        perm = reflector.perm_indices.tolist()
        assert perm == [0, 1, 2]
        sign = reflector.sign_vector.tolist()
        assert sign == [1.0, -1.0, 1.0]

    def test_reflect_with_history(self):
        """History-stacked tensor should reflect each timestep independently."""
        names = ["left_a:pos_x", "right_a:pos_x"]
        reflector = NamedReflector(SagittalReflectionConfig(), names, DEVICE)
        # history_length=3, stacked as [t0_left, t0_right, t1_left, t1_right, t2_left, t2_right]
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        result = reflector.reflect_with_history(x, single_dim=2)
        expected = torch.tensor([[2.0, 1.0, 4.0, 3.0, 6.0, 5.0]])
        assert torch.allclose(result, expected)


# ---------------------------------------------------------------------------
# TestNamedReflectorG1Regression
# ---------------------------------------------------------------------------

# G1 joint names in IsaacSim ordering (from the old _switch_g1_joints docstring)
G1_JOINT_NAMES = [
    "left_hip_pitch_joint",     # 0
    "right_hip_pitch_joint",    # 1
    "waist_yaw_joint",          # 2
    "left_hip_roll_joint",      # 3
    "right_hip_roll_joint",     # 4
    "left_shoulder_pitch_joint",  # 5
    "right_shoulder_pitch_joint", # 6
    "left_hip_yaw_joint",       # 7
    "right_hip_yaw_joint",      # 8
    "left_shoulder_roll_joint",  # 9
    "right_shoulder_roll_joint", # 10
    "left_knee_joint",          # 11
    "right_knee_joint",         # 12
    "left_shoulder_yaw_joint",  # 13
    "right_shoulder_yaw_joint", # 14
    "left_ankle_pitch_joint",   # 15
    "right_ankle_pitch_joint",  # 16
    "left_elbow_joint",         # 17
    "right_elbow_joint",        # 18
    "left_ankle_roll_joint",    # 19
    "right_ankle_roll_joint",   # 20
]


def _old_switch_g1_joints(joints: torch.Tensor) -> torch.Tensor:
    """Reference implementation of the old hard-coded _switch_g1_joints."""
    joints_switched = torch.zeros_like(joints)

    left_leg = [0, 3, 7, 11, 15, 19]
    right_leg = [1, 4, 8, 12, 16, 20]
    left_arm = [5, 9, 13, 17]
    right_arm = [6, 10, 14, 18]
    waist_yaw = [2]

    joints_switched[:, left_leg] = joints[:, right_leg]
    joints_switched[:, left_arm] = joints[:, right_arm]
    joints_switched[:, right_leg] = joints[:, left_leg]
    joints_switched[:, right_arm] = joints[:, left_arm]
    joints_switched[:, waist_yaw] = joints[:, waist_yaw]

    joints_switched[:, [2, 3, 4, 7, 8, 9, 10, 13, 14, 19, 20]] *= -1.0

    return joints_switched


class TestNamedReflectorG1Regression:
    """Verify that the NamedReflector produces identical results to the old code."""

    @pytest.fixture
    def g1_reflector(self):
        return NamedReflector(SagittalReflectionConfig(), G1_JOINT_NAMES, DEVICE)

    def test_matches_old_switch_g1_joints(self, g1_reflector):
        """Output of NamedReflector.reflect must match _old_switch_g1_joints."""
        torch.manual_seed(42)
        for _ in range(10):
            x = torch.randn(8, 21)
            old_result = _old_switch_g1_joints(x)
            new_result = g1_reflector.reflect(x)
            assert torch.allclose(old_result, new_result, atol=1e-6), (
                f"Mismatch!\nOld: {old_result}\nNew: {new_result}"
            )

    def test_perm_indices_match_old_swap(self, g1_reflector):
        """Verify the permutation matches the old hard-coded index arrays."""
        perm = g1_reflector.perm_indices.tolist()
        # left_leg ↔ right_leg
        assert perm[0] == 1 and perm[1] == 0    # hip_pitch
        assert perm[3] == 4 and perm[4] == 3    # hip_roll
        assert perm[7] == 8 and perm[8] == 7    # hip_yaw
        assert perm[11] == 12 and perm[12] == 11  # knee
        assert perm[15] == 16 and perm[16] == 15  # ankle_pitch
        assert perm[19] == 20 and perm[20] == 19  # ankle_roll
        # left_arm ↔ right_arm
        assert perm[5] == 6 and perm[6] == 5    # shoulder_pitch
        assert perm[9] == 10 and perm[10] == 9  # shoulder_roll
        assert perm[13] == 14 and perm[14] == 13  # shoulder_yaw
        assert perm[17] == 18 and perm[18] == 17  # elbow
        # waist stays
        assert perm[2] == 2

    def test_sign_vector_matches_old_negation(self, g1_reflector):
        """Verify the sign vector matches the old hard-coded negation indices."""
        sign = g1_reflector.sign_vector.tolist()
        old_negated = {2, 3, 4, 7, 8, 9, 10, 13, 14, 19, 20}
        for i in range(21):
            expected = -1.0 if i in old_negated else 1.0
            assert sign[i] == expected, f"Sign mismatch at index {i}: expected {expected}, got {sign[i]}"

    def test_double_reflect_is_identity(self, g1_reflector):
        x = torch.randn(5, 21)
        assert torch.allclose(g1_reflector.reflect(g1_reflector.reflect(x)), x, atol=1e-6)


# ---------------------------------------------------------------------------
# TestNamedReflectorTrajectoryRegression
# ---------------------------------------------------------------------------

class TestNamedReflectorTrajectoryRegression:
    """Verify trajectory reflection matches the old string-based get_symmetric_traj."""

    TRAJ_NAMES = [
        "com:pos_x",
        "com:pos_y",
        "com:pos_z",
        "left_ankle_roll_link:pos_x",
        "left_ankle_roll_link:pos_y",
        "right_ankle_roll_link:pos_x",
        "right_ankle_roll_link:pos_y",
        "joint:left_hip_roll_joint",
        "joint:right_hip_roll_joint",
        "joint:left_hip_pitch_joint",
        "joint:right_hip_pitch_joint",
        "joint:waist_yaw_joint",
    ]

    @staticmethod
    def _old_get_symmetric_traj(traj: torch.Tensor, output_names: list[str]) -> torch.Tensor:
        """Reference implementation of the old get_symmetric_traj."""
        symmetric_traj = traj.clone()
        for i, output_name in enumerate(output_names):
            if "left" in output_name:
                symmetric_name = output_name.replace("left", "right")
            elif "right" in output_name:
                symmetric_name = output_name.replace("right", "left")
            else:
                if any(axis in output_name for axis in ["pos_y", "ori_x", "ori_z", "roll_joint", "yaw_joint"]):
                    symmetric_traj[:, i] = -traj[:, i]
                continue
            if symmetric_name in output_names:
                j = output_names.index(symmetric_name)
                symmetric_traj[:, i] = traj[:, j]
                if any(axis in output_name for axis in ["pos_y", "ori_x", "ori_z", "roll_joint", "yaw_joint"]):
                    symmetric_traj[:, i] = -symmetric_traj[:, i]
        return symmetric_traj

    def test_matches_old_get_symmetric_traj(self):
        reflector = NamedReflector(SagittalReflectionConfig(), self.TRAJ_NAMES, DEVICE)
        torch.manual_seed(123)
        for _ in range(10):
            x = torch.randn(4, len(self.TRAJ_NAMES))
            old = self._old_get_symmetric_traj(x, self.TRAJ_NAMES)
            new = reflector.reflect(x)
            assert torch.allclose(old, new, atol=1e-6), (
                f"Mismatch!\nOld: {old}\nNew: {new}"
            )


# ---------------------------------------------------------------------------
# TestRelabelMatrixRegression
# ---------------------------------------------------------------------------

class TestRelabelMatrixRegression:
    """Verify the relabel matrix matches the old relable_ee_stance_coeffs."""

    def test_matches_old_relabel_matrix(self, running_manager):
        """Compare NamedReflector matrix against the TrajectoryManager's output.

        This test exercises the actual trajectory output names from a loaded YAML.
        """
        names = running_manager.traj_data.pos_output_names
        R_old = running_manager.relable_ee_stance_coeffs(names)

        reflector = NamedReflector(SagittalReflectionConfig(), names, DEVICE)
        R_new = reflector.build_relabel_matrix_numpy()

        assert np.allclose(R_old, R_new, atol=1e-10), (
            f"Relabel matrix mismatch!\nOld:\n{R_old}\nNew:\n{R_new}"
        )

    def test_vel_matches_old_relabel_matrix(self, running_manager):
        names = running_manager.traj_data.vel_output_names
        R_old = running_manager.relable_ee_stance_coeffs(names)

        reflector = NamedReflector(SagittalReflectionConfig(), names, DEVICE)
        R_new = reflector.build_relabel_matrix_numpy()

        assert np.allclose(R_old, R_new, atol=1e-10)
