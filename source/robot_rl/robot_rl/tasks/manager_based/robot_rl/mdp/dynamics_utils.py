# dynamics_utils.py
import torch
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene
from isaaclab.assets import Articulation
from isaaclab.utils.math import matrix_from_quat, quat_inv

def get_mass_and_gravity(robot: Articulation, joint_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      mass_matrix: (N_envs, n_dof, n_dof)
      gravity:     (N_envs, n_dof)
    """
    # PhysX‐computed joint‐space inertia
    M_full = robot.root_physx_view.get_generalized_mass_matrices()      # (N_envs, all_dof, all_dof)
    # extract only the actuated joints
    M = M_full[:, joint_ids, :][:, :, joint_ids]                       # (N_envs, n_dof, n_dof)

    # PhysX‐computed gravity torques
    G_full = robot.root_physx_view.get_gravity_compensation_forces()   # (N_envs, all_dof)
    G = G_full[:, joint_ids]                                           # (N_envs, n_dof)

    return M, G

def get_body_jacobian(robot: Articulation, ee_frame_idx: int, joint_ids: list[int]) -> torch.Tensor:
    """
    Returns the 6×n_dof end-effector Jacobian in the robot's base frame.
    Args:
      ee_frame_idx:  index of the body in robot.data.body_* arrays
    """
    # PhysX returns world-frame Jacobian: shape (N_envs, 6, all_dof)
    J_w_full = robot.root_physx_view.get_jacobians()                  # (N_envs, 6, all_dof)
    # pick the row corresponding to your ee_frame (PhysX packs bodies in a flat list)
    # often ee_frame_idx–1 corresponds to the correct row, but you can adjust if needed
    J_w = J_w_full[:, ee_frame_idx-1, :, joint_ids]                   # (N_envs, 6, n_dof)

    # rotate into the base/body frame
    # get the root→world quaternion
    q_root_w = robot.data.root_quat_w                                 # (N_envs, 4)
    R_root_w = matrix_from_quat(q_root_w)                             # (N_envs, 3, 3)
    R_w_root = R_root_w.transpose(1,2)

    J_b = J_w.clone()
    J_b[:, :3, :] = torch.bmm(R_w_root, J_w[:, :3, :])
    J_b[:, 3:, :] = torch.bmm(R_w_root, J_w[:, 3:, :])

    return J_b

def get_joint_states(robot: Articulation, joint_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      q (N_envs, n_dof), dq (N_envs, n_dof)
    """
    q_full  = robot.data.joint_pos      # (N_envs, all_dof)
    dq_full = robot.data.joint_vel      # (N_envs, all_dof)
    return q_full[:, joint_ids], dq_full[:, joint_ids]
