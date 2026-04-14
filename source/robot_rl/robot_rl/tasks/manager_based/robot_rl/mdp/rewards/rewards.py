# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp
import math
import re
from typing import TYPE_CHECKING, Sequence

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

def multiple_undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfgs: Sequence[SceneEntityCfg]) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations above a threshold, aggregated across multiple sensors."""
    total = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for sensor_cfg in sensor_cfgs:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        net_contact_forces = wp.to_torch(contact_sensor.data.net_forces_w_history)
        is_contact = (
            torch.max(torch.linalg.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
        )
        # print(f"{sensor_cfg.name} net force: {net_contact_forces.cpu()}")
        total += torch.sum(is_contact, dim=1)
    return total

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

##
# Individual tracking rewards
##
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