# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
import math
import re
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv

KAPPA = 0.5

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

def clf_reward(env: ManagerBasedRLEnv, command_name: str, max_eta_err: float = 0.15, eps: float = 1e-6) -> torch.Tensor:
    """CLF-based reward: r = exp(-V(η) / V_max), clipped to [0, 1]."""

    ref_term = env.command_manager.get_term(command_name)
    v = ref_term.v  # [B] scalar CLF value per env
    max_clf = ref_term.clf.lambda_max * max_eta_err ** 2 + eps # principled normalization; lambda_max(P) * eta**2

    reward = torch.exp(-v / max_clf)
    return reward

def clf_decreasing_condition(
    env: ManagerBasedRLEnv,
    command_name: str,
    alpha: float = 1.0,
    eta_max: float = 0.15,
    eta_dot_max: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Penalty for violating CLF decrease condition: 𝑟 = clip((ΔV + αV) / max_violation, [0, 1])
    where:
        max_violation ≈ 2‖P‖ η_max η̇_max + α λ_max(P) η_max²
    """

    ref_term = env.command_manager.get_term(command_name)
    v = ref_term.v        # [B]
    vdot = ref_term.vdot  # [B]

    lambda_max = ref_term.clf.lambda_max
    norm_P = ref_term.clf.norm_P

    # Theoretical upper bound on violation
    dt = 0.02
    max_violation = (
        dt * eta_dot_max * norm_P + alpha * lambda_max * eta_max ** 2 + eps
    )
    # Only penalize when violation is positive
    # violation = torch.clamp(vdot + alpha * v, min=0.0)
    violation = vdot + alpha * v
    penalty = violation / max_violation
    penalty = torch.clamp(penalty, min=0.0, max=1.0)
    return penalty

def base_pos_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_base_pos = cmd_term.clf.v_subgroups["pelvis_pos"]

    return torch.exp(-KAPPA * v_base_pos / sigma)

def base_lin_vel_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_base_lin_vel = cmd_term.clf.v_subgroups["pelvis_lin_vel"]

    return torch.exp(-KAPPA * v_base_lin_vel / sigma)

def base_ori_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_base_ori = cmd_term.clf.v_subgroups["pelvis_ori"]

    return torch.exp(-KAPPA * v_base_ori / sigma)

def base_ang_vel_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_base_ang_vel = cmd_term.clf.v_subgroups["pelvis_ang_vel"]

    return torch.exp(-KAPPA * v_base_ang_vel / sigma)

def joint_pos_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_joint_pos = cmd_term.clf.v_subgroups["joint_pos"]

    return torch.exp(-KAPPA * v_joint_pos / sigma)

def joint_vel_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_joint_vel = cmd_term.clf.v_subgroups["joint_vel"]

    return torch.exp(-KAPPA * v_joint_vel / sigma)

def body_pos_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_body_pos = cmd_term.clf.v_subgroups["other_body_pos"]

    return torch.exp(-KAPPA * v_body_pos / sigma)

def body_lin_vel_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_body_lin_vel = cmd_term.clf.v_subgroups["other_body_lin_vel"]

    return torch.exp(-KAPPA * v_body_lin_vel / sigma)

def body_ori_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_body_ori = cmd_term.clf.v_subgroups["other_body_ori"]

    return torch.exp(-KAPPA * v_body_ori / sigma)

def body_ang_vel_reward(env: ManagerBasedRLEnv, command_name: str, sigma: float) -> torch.Tensor:
    cmd_term = env.command_manager.get_term(command_name)
    v_body_ang_vel = cmd_term.clf.v_subgroups["other_body_ang_vel"]

    return torch.exp(-KAPPA * v_body_ang_vel / sigma)


def v_dot_penalty(env: ManagerBasedRLEnv, command_name: str,eta_max: float = 0.15,
    eta_dot_max: float = 0.5,eps: float = 1e-6) -> torch.Tensor:
    ref_term = env.command_manager.get_term(command_name)                    # [B]
    vdot = ref_term.vdot # [B]

    norm_P = ref_term.clf.norm_P

    max_violation = (
        2.0 * norm_P * eta_max * eta_dot_max + eps
    )

    vdot_penalty = torch.tanh(torch.clamp(vdot, min=0.0) / max_violation) 
    return vdot_penalty


def contact_no_vel(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward feet contact with zero velocity."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = wp.to_torch(contact_sensor.data.net_forces_w_history)[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = wp.to_torch(asset.data.body_lin_vel_w)[:, asset_cfg.body_ids] * contacts.unsqueeze(-1)
    # shape [B, num_feet, 3]
    penalize = torch.square(body_vel[:,:,:3])
    return torch.sum(penalize, dim=(1,2))


def holonomic_constraint_vel(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma_vel: float = (0.1)**0.5
) -> torch.Tensor:
    """
    Unified holonomic‐velocity constraint reward:
      r = exp( – ‖[v, ω_z]‖² / σ_vel² )
    where v∈R³ is the foot’s linear velocity and ω_z its yaw rate.
    Using σ_vel=√0.1 matches the original bandwidth (denominator=0.1).
    """
    cmd = env.command_manager.get_term(command_name)

    # Get the velocities
    v = cmd.current_contact_vels

    # # linear velocity [B,3] and yaw rate [B,1]
    # v = cmd.stance_foot_vel  # [vx, vy, vz]
    # wz = cmd.stance_foot_ang_vel[:, 2].unsqueeze(-1)  # [ω_z]
    #
    # # stack into [B,4] error vector
    # e_vel = torch.cat([v, wz], dim=-1)
    #
    # not_flight_mask = cmd.get_not_flight_envs()
    # return not_flight_mask * torch.exp(- (e_vel**2).sum(dim=-1) / sigma_vel**2)

    return torch.exp(-(v.sum(dim=-1).sum(dim=-1)**2) / sigma_vel**2)

def holonomic_constraint(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma_pose: float = (5 * 0.01) ** 0.5,
    z_offset: float = 0.036
) -> torch.Tensor:
    """
    Unified holonomic‐pose constraint reward:
        r = exp( – ‖e_pose‖² / σ_pose² )
    where e_pose = [Δx, Δy, Δz, φ, Δψ] and
      • Δx, Δy are planar errors from the recorded foot position,
      • Δz = p_z_cur – z_offset (encourages foot to stay on the floor),
      • φ is roll,
      • Δψ is yaw error wrapped to [–π, π].
    """

    cmd = env.command_manager.get_term(command_name)

    # TODO: Re-write to handle arbitrary contacts

    # Get the current pose
    des_contact_poses = cmd.desired_contact_poses
    contact_poses = cmd.current_contact_poses

    # Compute error
    pose_err = contact_poses - des_contact_poses

    # Wrap yaw error
    pose_err = wrap_to_pi(pose_err[:, -1])

    # # planar position error [B,2]
    # p0_xy = cmd.stance_foot_pos_0[:, :2]
    # p_xy = cmd.stance_foot_pos[:, :2]
    # delta_xy = p_xy - p0_xy
    #
    # # vertical error to the floor plane [B,1]
    # z_cur = cmd.stance_foot_pos[:, 2].unsqueeze(-1)
    # delta_z = z_cur - cmd.stance_foot_pos_0[:, 2].unsqueeze(-1)
    #
    # # roll error [B,1]
    # roll = cmd.stance_foot_ori[:, 0].unsqueeze(-1)
    #
    # # yaw error wrapped to [–π, π] [B,1]
    # psi0 = cmd.stance_foot_ori_0[:, 2]
    # psi = cmd.stance_foot_ori[:, 2]
    # delta_psi = ((psi - psi0 + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)
    #
    # # stack into [B,5] error vector
    # e_pose = torch.cat([delta_xy, delta_z, roll, delta_psi], dim=-1)
    #
    # not_flight_mask = cmd.get_not_flight_envs()
    # return not_flight_mask * torch.exp(- (e_pose ** 2).sum(dim=-1) / sigma_pose ** 2)

    return torch.exp(-(pose_err**2).sum(dim=-1) / sigma_pose ** 2)

def reference_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    term_std: Sequence[float],
    term_weight: Sequence[float],
) -> torch.Tensor:
    """
    Exponential reward per dimension, scaled by weight — ignores zero-weight terms.
    """
    command = env.command_manager.get_term(command_name)
    err = command.y_act - command.y_out  # [B, D]

    weight_vec = torch.as_tensor(term_weight, dtype=err.dtype, device=err.device)  # [D]
    std_vec = torch.as_tensor(term_std, dtype=err.dtype, device=err.device)        # [D]

    # [B, D] scaled squared error per dimension
    err_sq_scaled = (err ** 2) / (std_vec ** 2)

    # Apply element-wise exp(-error²/std²) and weight
    reward_per_dim = weight_vec * torch.exp(-err_sq_scaled)  # [B, D]
    reward = reward_per_dim.sum(dim=1)/torch.sum(weight_vec)  # [B]

    return reward


def reference_vel_tracking(    env: ManagerBasedRLEnv,
    command_name: str,
    term_std: Sequence[float],
    term_weight: Sequence[float],
) -> torch.Tensor:
    """Reference tracking with element-wise term weights."""
    # 1. fetch the command and compute error [B, D]
    command = env.command_manager.get_term(command_name)
    err = command.dy_act - command.dy_out

    weight_vec = torch.as_tensor(term_weight, dtype=err.dtype, device=err.device)  # [D]
    std_vec = torch.as_tensor(term_std, dtype=err.dtype, device=err.device)        # [D]

    # [B, D] scaled squared error per dimension
    err_sq_scaled = (err ** 2) / (std_vec ** 2)

    # Apply element-wise exp(-error²/std²) and weight
    reward_per_dim = weight_vec * torch.exp(-err_sq_scaled)  # [B, D]
    reward = reward_per_dim.sum(dim=1)/torch.sum(weight_vec)  # [B]
    return reward


def foot_clearance(env: ManagerBasedRLEnv,
                   target_height: float,
                   sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
                   height_sensor_cfg: SceneEntityCfg | None = None,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Reward foot clearance."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get contact state
    contacts = wp.to_torch(contact_sensor.data.net_forces_w_history)[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    if height_sensor_cfg is not None:
        sensor: RayCaster = env.scene[height_sensor_cfg.name]
        adjusted_target_height = target_height + torch.mean(wp.to_torch(sensor.data.ray_hits_w)[...,2],dim=1).unsqueeze(-1)
    else:
        adjusted_target_height = target_height

    # Calculate foot heights
    feet_z_err = wp.to_torch(asset.data.body_pos_w)[:, asset_cfg.body_ids, 2] - adjusted_target_height
    pos_error = torch.square(feet_z_err) * ~contacts

    return torch.sum(pos_error, dim=(1))

def phase_contact(
    env: ManagerBasedRLEnv,
        period: float = 0.8,
        command_name: str | None = None,
        Tswing: float =0.4,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward foot contact with regards to phase."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # Get contact state
    res = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    # Contact phase
    tp = (env.sim.current_time % period) / period     # Scaled between 0-1
    phi_c = torch.tensor(math.sin(2*torch.pi*tp)/math.sqrt(math.sin(2*torch.pi*tp)**2 + Tswing), device=env.device)

    stance_i = int(0.5 - 0.5 * torch.sign(phi_c))


     # check if robot needs to be standing
    if command_name is not None:
        command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
        is_small_command = command_norm < 0.005
        for i in range(2):
            is_stance = stance_i == i
            # set is_stance to be true if the command is small
            is_stance = is_stance | is_small_command
            contact = wp.to_torch(contact_sensor.data.net_forces_w_history)[:, :, sensor_cfg.body_ids[i], :].norm(dim=-1).max(dim=1)[0] > 1.0
            res += ~(contact ^ is_stance)
    else:
        for i in range(2):
            is_stance = stance_i == i
            # set is_stance to be true if the command is small
            is_stance = is_stance
            contact = wp.to_torch(contact_sensor.data.net_forces_w_history)[:, :, sensor_cfg.body_ids[i], :].norm(dim=-1).max(dim=1)[0] > 1.0
            res += ~(contact ^ is_stance)
    return res

# TODO: Test
def contact_schedule_penalty(env: ManagerBasedRLEnv, command_name: str,
                           sensor_cfg: SceneEntityCfg, weight_scalar: float) -> torch.Tensor:
    """Penalize contacts while in the flight phase."""
    cmd = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Time into the episode
    t = env.episode_length_buf * env.step_dt

    # Get bodies not in contact for each env
    contact_states = cmd.get_contact_state(t)
    contact_body_names = cmd.contact_bodies

    contact_forces = torch.zeros(t.shape[0], dtype=torch.float, device=env.device)
    for i, body_name in enumerate(contact_body_names):
        contact_mask = contact_states[:, i] == 1
        indices = torch.tensor([i for i, v in enumerate(sensor_cfg.body_names) if v == body_name])
        body_id = sensor_cfg.body_ids[indices]
        contact_forces[contact_mask] += wp.to_torch(contact_sensor.data.net_forces_w)[contact_mask, body_id, :].norm(dim=-1)  # Gets the most recent force only

    penalty = weight_scalar * torch.tanh(contact_forces / 0.5)  # TODO: Think about if this is what I want
    return penalty

def track_lin_vel_y_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error =  torch.square(env.command_manager.get_command(command_name)[:, 1] - wp.to_torch(asset.data.root_lin_vel_b)[:, 1])
    return torch.exp(-lin_vel_error / std**2)


def ankle_roll_zero(
    env: ManagerBasedRLEnv, std: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward keeping both ankle roll joints near zero position using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get ankle roll joint indices - these are typically the last joints in each leg
    # Based on the controller.py joint order:
    # Index 19: left_ankle_roll_joint
    # Index 20: right_ankle_roll_joint
    ankle_roll_indices = [19, 20]  # left and right ankle roll joints
    
    # Get current ankle roll joint positions
    ankle_roll_positions = wp.to_torch(asset.data.joint_pos)[:, ankle_roll_indices]  # [B, 2]
    
    # Compute squared error from zero position
    ankle_roll_error = torch.square(ankle_roll_positions)  # [B, 2]
    
    # Sum errors for both ankle roll joints and apply exponential kernel
    total_error = ankle_roll_error.sum(dim=-1)  # [B]
    reward = torch.exp(-total_error / std**2)
    
    return reward

def torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.
    For implicit actuators, we manually compute the PD controller torques.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Manually compute PD controller torques for implicit actuators
    computed_torque = torch.zeros_like(wp.to_torch(asset.data.joint_pos))

    # Get current joint positions, velocities, and desired positions
    current_pos = wp.to_torch(asset.data.joint_pos)
    current_vel = wp.to_torch(asset.data.joint_vel)
    desired_pos = wp.to_torch(asset.data.joint_pos_target)
    
    # Access actuator configurations from the asset
    actuator_groups = asset.cfg.actuators
    
    for group_name, actuator_cfg in actuator_groups.items():
        # Get joint indices for this actuator group
        joint_indices = asset.find_joints(actuator_cfg.joint_names_expr)[0]
        
        # Get stiffness and damping values for this group
        if isinstance(actuator_cfg.stiffness, dict):
            # Handle per-joint stiffness values
            kp_values = torch.zeros(len(joint_indices), dtype=torch.float32, device=env.device)
            for i, joint_idx in enumerate(joint_indices):
                joint_name = asset.joint_names[joint_idx]
                # Find matching stiffness pattern
                for pattern, value in actuator_cfg.stiffness.items():
                    if re.match(pattern.replace(".*", ".*"), joint_name):
                        kp_values[i] = value
                        break
        else:
            # Single stiffness value for all joints in this group
            kp_values = torch.full((len(joint_indices),), actuator_cfg.stiffness, dtype=torch.float32, device=env.device)
        
        if isinstance(actuator_cfg.damping, dict):
            # Handle per-joint damping values
            kd_values = torch.zeros(len(joint_indices), dtype=torch.float32, device=env.device)
            for i, joint_idx in enumerate(joint_indices):
                joint_name = asset.joint_names[joint_idx]
                # Find matching damping pattern
                for pattern, value in actuator_cfg.damping.items():
                    if re.match(pattern.replace(".*", ".*"), joint_name):
                        kd_values[i] = value
                        break
        else:
            # Single damping value for all joints in this group
            kd_values = torch.full((len(joint_indices),), actuator_cfg.damping, dtype=torch.float32, device=env.device)
        
        # Compute PD torques for this group: tau = kp * (q_des - q) - kd * q_dot
        pos_error = desired_pos[:, joint_indices] - current_pos[:, joint_indices]
        pd_torque = (kp_values[None, :] * pos_error - kd_values[None, :] * current_vel[:, joint_indices])
        
        # Store computed torques
        computed_torque[:, joint_indices] = pd_torque
    
    # Compute torque limit violations
    torque_limits_upper = wp.to_torch(asset.data.joint_effort_limits)[0, asset_cfg.joint_ids]  # Upper limits

    # Get computed torques for the specified joints
    joint_torques = computed_torque[:, asset_cfg.joint_ids]
    
    # Compute violations: how much torques exceed the limits
    violation = torch.clamp(torch.abs(joint_torques) - torque_limits_upper, min=0)

    # Sum all violations
    return torch.sum(violation, dim=1)