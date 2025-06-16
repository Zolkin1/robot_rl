# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def swing_foot_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    penalty_cap: float = 5.0,
) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    swing_foot_indices = cmd.swing_idx.squeeze(-1)  # [num_envs]
    batch_idx = torch.arange(env.num_envs, device=env.device)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    net_forces = contact_sensor.data.net_forces_w_history  # [N, T, B, 3]
    latest_forces = net_forces[:, -1, sensor_cfg.body_ids, :]  # [num_envs,2, 3]
    # get the correct forces
    latest_forces = latest_forces[batch_idx, swing_foot_indices,:]  # [num_envs, 3]
    force_mags = torch.norm(latest_forces, dim=-1)  # [num_envs]
    capped_penalty = torch.clamp(force_mags, max=penalty_cap)  # [num_envs]
    return capped_penalty  # Standard is to return per-env reward/penalty



def track_lin_vel_x_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, :1] - asset.data.root_lin_vel_b[:, :1])
    return torch.exp(-torch.squeeze(lin_vel_error) / std**2)

def clf_reward(env: ManagerBasedRLEnv, command_name: str, max_clf: float = 200.0) -> torch.Tensor:
    """Negative CLF value as a reward (i.e., -V(η)), clipped to [-1, 0]."""
    ref_term = env.command_manager.get_term(command_name)
    v = ref_term.v  # [B] scalar CLF value per env

    # reward = torch.clamp(v, min=0.0, max=max_clf)/max_clf
    reward = torch.exp(-v/max_clf)
    
    return reward


def clf_decreasing_condition(env: ManagerBasedRLEnv, command_name: str, alpha: float = 1.0, max_clf_decreasing: float = 200.0) -> torch.Tensor:
    """Penalty for violating the CLF decrease condition, clipped to [-1, 0]."""
    ref_term = env.command_manager.get_term(command_name)
    v = ref_term.v                     # [B]
    vdot = ref_term.vdot # [B]

    # Compute violation: ΔV + αV
    clf_violation = vdot + alpha * v   # [B]

    #also normalize by max_clf violation
    clf_violation = clf_violation/max_clf_decreasing
    penalty = torch.clamp(clf_violation, min=0.0)  # only penalize violations
  
    reward = torch.clamp(penalty, max=1.0) 
    return reward

def v_dot_penalty(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    ref_term = env.command_manager.get_term(command_name)                    # [B]
    vdot = ref_term.vdot # [B]

    v_dot_penalty = torch.clamp(vdot,min=0.0)
    return v_dot_penalty


def joint_pos_target(
        env, asset_cfg: SceneEntityCfg, joint_des: torch.Tensor, std: float, joint_weight: torch.Tensor
    ) -> torch.Tensor:
    """Reward joints for proximity to a static desired joint position."""
    asset = env.scene[asset_cfg.name]

    q_pos = asset.data.joint_pos.detach().clone()
    q_err = joint_weight * torch.square(q_pos - joint_des)
    return torch.mean(torch.exp(-q_err / std ** 2), dim=-1)

def symmetric_feet_air_time_biped(
        env: ManagerBasedRLEnv,
        command_name: str,
        threshold: float,
        sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps while enforcing symmetric gait patterns for bipeds.

    Ensures balanced stepping by:
    - Tracking air/contact time separately for each foot
    - Penalizing asymmetric gait patterns
    - Maintaining alternating single stance phases
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Split into left and right foot indices
    left_ids = [sensor_cfg.body_ids[0]]
    right_ids = [sensor_cfg.body_ids[1]]

    # Get timing data for each foot
    air_time_left = contact_sensor.data.current_air_time[:, left_ids]
    air_time_right = contact_sensor.data.current_air_time[:, right_ids]
    contact_time_left = contact_sensor.data.current_contact_time[:, left_ids]
    contact_time_right = contact_sensor.data.current_contact_time[:, right_ids]
    last_air_time_left = contact_sensor.data.last_air_time[:, left_ids]
    last_air_time_right = contact_sensor.data.last_air_time[:, right_ids]
    last_contact_time_left = contact_sensor.data.last_contact_time[:, left_ids]
    last_contact_time_right = contact_sensor.data.last_contact_time[:, right_ids]

    # Compute contact states
    in_contact_left = contact_time_left > 0.0
    in_contact_right = contact_time_right > 0.0

    # Calculate mode times for each foot
    left_mode_time = torch.where(in_contact_left, contact_time_left, air_time_left)
    right_mode_time = torch.where(in_contact_right, contact_time_right, air_time_right)
    last_left_mode_time = torch.where(in_contact_left, last_air_time_left, last_contact_time_left)
    last_right_mode_time = torch.where(in_contact_right, last_air_time_right, last_contact_time_right)

    # Check for proper single stance (one foot in contact, one in air)
    left_stance = in_contact_left.any(dim=1) & (~in_contact_right.any(dim=1))
    right_stance = in_contact_right.any(dim=1) & (~in_contact_left.any(dim=1))
    single_stance = left_stance | right_stance

    # Calculate symmetric reward components
    left_reward = torch.min(torch.where(left_stance.unsqueeze(-1), left_mode_time, 0.0), dim=1)[0]
    right_reward = torch.min(torch.where(right_stance.unsqueeze(-1), right_mode_time, 0.0), dim=1)[0]
    last_left_reward = torch.min(torch.where(left_stance.unsqueeze(-1), last_left_mode_time, 0.0), dim=1)[0]
    last_right_reward = torch.min(torch.where(right_stance.unsqueeze(-1), last_right_mode_time, 0.0), dim=1)[0]
    # Combine rewards with symmetry penalty
    base_reward = (left_reward + right_reward) / 2.0
    symmetry_penalty = torch.abs(left_reward - last_right_reward) + torch.abs(right_reward - last_left_reward)
    reward = base_reward - 0.1 * symmetry_penalty

    # Apply threshold and command scaling
    reward = torch.clamp(reward, max=threshold)
    command_scale = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1

    return reward * command_scale

def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    
    reward = torch.sum(torch.sum(body_vel**2, dim=-1) * contacts, dim=1)
    return reward

def contact_no_vel(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward feet contact with zero velocity."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids] * contacts.unsqueeze(-1)
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

    # linear velocity [B,3] and yaw rate [B,1]
    v   = cmd.stance_foot_vel                        # [vx, vy, vz]
    wz  = cmd.stance_foot_ang_vel[:, 2].unsqueeze(-1) # [ω_z]

    # stack into [B,4] error vector
    e_vel = torch.cat([v, wz], dim=-1)

    # unified exponential‐norm reward
    return torch.exp(- (e_vel**2).sum(dim=-1) / sigma_vel**2)

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

    # planar position error [B,2]
    p0_xy = cmd.stance_foot_pos_0[:, :2]
    p_xy  = cmd.stance_foot_pos[:, :2]
    delta_xy = p_xy - p0_xy

    # vertical error to the floor plane [B,1]
    z_cur    = cmd.stance_foot_pos[:, 2].unsqueeze(-1)
    delta_z  = z_cur - cmd.stance_foot_pos_0[:,2].unsqueeze(-1)

    # roll error [B,1]
    roll = cmd.stance_foot_ori[:, 0].unsqueeze(-1)

    # yaw error wrapped to [–π, π] [B,1]
    psi0 = cmd.stance_foot_ori_0[:, 2]
    psi  = cmd.stance_foot_ori[:, 2]
    delta_psi = ((psi - psi0 + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)

    # stack into [B,5] error vector
    e_pose = torch.cat([delta_xy, delta_z, roll, delta_psi], dim=-1)

    # unified Gaussian‐like reward
    return torch.exp(- (e_pose**2).sum(dim=-1) / sigma_pose**2)

def holonomic_constraint_stair(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma_pose: float = (6 * 0.01) ** 0.5,

) -> torch.Tensor:  
    cmd = env.command_manager.get_term(command_name)

    # planar position error [B,2]
    p0_xy = cmd.stance_foot_pos_0[:, :2]
    p_xy  = cmd.stance_foot_pos[:, :2]
    delta_xy = p_xy - p0_xy

    # vertical error to the floor plane [B,1]
    z_cur    = cmd.stance_foot_pos[:, 2].unsqueeze(-1)
    delta_z  = z_cur - cmd.stance_foot_pos_0[:,2].unsqueeze(-1)

    # roll error [B,1]
    roll = cmd.stance_foot_ori[:, 0].unsqueeze(-1)
    pitch = cmd.stance_foot_ori[:, 1].unsqueeze(-1)

    # yaw error wrapped to [–π, π] [B,1]
    psi0 = cmd.stance_foot_ori_0[:, 2]
    psi  = cmd.stance_foot_ori[:, 2]
    delta_psi = ((psi - psi0 + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)

    # stack into [B,5] error vector
    e_pose = torch.cat([delta_xy, delta_z,pitch, roll, delta_psi], dim=-1)

    # unified Gaussian‐like reward
    return torch.exp(- (e_pose**2).sum(dim=-1) / sigma_pose**2)

def lip_gait_tracking(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, period: float, std: float,
                      nom_height: float, Tswing: float, command_name: str, wdes: float,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ) -> torch.Tensor:
    """Reward feet in contact with the ground in the correct phase."""
    # If the feet are in contact at the right time then positive reward, else 0 reward

    # Get the robot asset
    # Contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get the current contacts
    # in_contact = ~contact_sensor.compute_first_air()[:, sensor_cfg.body_ids]  # Checks if the foot recently broke contact - which tells us we are not in contact. Does not reward jitter but use the dt.
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    in_contact = in_contact.float()

    # Contact schedule function
    tp = (env.sim.current_time % period) / period     # Scaled between 0-1
    phi_c = torch.tensor(math.sin(2*torch.pi*tp)/math.sqrt(math.sin(2*torch.pi*tp)**2 + Tswing), device=in_contact.device)

    # Compute reward
    reward = (in_contact[:, 0] - in_contact[:, 1])*phi_c # TODO: Does it help to remove the schedule here? - seemed to get some instability

    return reward

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


def track_heading(env: ManagerBasedRLEnv, command_name: str,
                  std: float,
                  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Reward tracking the heading of the robot."""
    asset = env.scene[asset_cfg.name]
    # command = env.command_manager.get_command(command_name)[:, :2]
    #
    # Get current heading
    # Get the robot's root quaternion in world frame
    robot_quat_w = asset.data.root_quat_w  # Shape: [num_environments, 4]

    # Extract the Yaw angle (Heading)
    heading = euler_xyz_from_quat(robot_quat_w)
    heading = wrap_to_pi(heading[2])
    #
    # # Compute the heading from the commanded velocity
    # # Compute the command in the global frame
    # # TODO: Change where I grab command to grab all 3 entries so I don't need this!
    # command_3 = torch.zeros((command.shape[0], 3), device=command.device)
    # command_3[:, :2] = command
    # command_w = quat_rotate_inverse(robot_quat_w, command_3)
    # # heading_des = torch.atan2(command[:, 1], command[:, 0])
    # heading_des = torch.atan2(command_w[:, 1], command_w[:, 0])
    heading_des = wrap_to_pi(env.command_manager.get_command(command_name)[:, 2])

    # print(f"command: {command}")
    # print(f"heading_des: {heading_des}, heading: {heading}")

    reward = 2.*torch.exp(-torch.abs(wrap_to_pi(heading_des - heading)) / std)

    # print(f"heading: {heading}, heading_des: {heading_des}")
    # print(f"reward: {reward}")
    # print(f"heading error: {wrap_to_pi(heading_des - heading)}")

    return reward

def compute_step_location_local(env: ManagerBasedRLEnv, env_ids: torch.Tensor,
                          nom_height: float, Tswing: float, command_name: str, wdes: float,
                          feet_bodies: SceneEntityCfg,
                          sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                          visualize: bool = True) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    feet = env.scene[feet_bodies.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Commanded velocity in the local frame
    command = env.command_manager.get_command(command_name)

    # COM Position in global frame
    # r = asset.data.root_com_pos_w
    r = asset.data.root_pos_w

    # COM velocity in local frame
    rdot = command
    # rdot = asset.data.root_com_lin_vel_b

    g = 9.81
    omega = math.sqrt(g / nom_height)

    # Instantaneous capture point as a 3-vector
    icp_0 = torch.zeros((r.shape[0], 3), device=env.device)    # For setting the height
    icp_0[:, :2] = rdot[:, :2]/omega


    # Get the stance foot position
    foot_pos = feet.data.body_pos_w[:, feet_bodies.body_ids]
    # Contact schedule function
    tp = (env.sim.current_time % (2*Tswing)) / (2*Tswing)     # Scaled between 0-1
    phi_c = torch.tensor(math.sin(2*torch.pi*tp)/math.sqrt(math.sin(2*torch.pi*tp)**2 + Tswing), device=env.device)

    # Stance foot in global frame
    stance_foot_pos = foot_pos[:, int(0.5 - 0.5*torch.sign(phi_c)), :]
    stance_foot_pos[:, 2] *= 0

    def _transfer_to_global_frame(vec, root_quat):
        return quat_rotate(yaw_quat(root_quat), vec)

    def _transfer_to_local_frame(vec, root_quat):
        return quat_rotate(yaw_quat(quat_inv(root_quat)), vec)

    # Compute final ICP as a 3 vector
    icp_f = (math.exp(omega * Tswing)*icp_0 + (1 - math.exp(omega * Tswing))
             * _transfer_to_local_frame(r - stance_foot_pos, asset.data.root_quat_w))
    icp_f[:, 2] *= 0


    # Compute ICP offsets
    sd = torch.abs(command[:, 0]) * Tswing #TODO: Note this only works if there are no commanded local y velocities
    wd = wdes * torch.ones(r.shape[0], device=env.device)

    bx = sd / (math.exp(omega * Tswing) - 1)
    by = torch.sign(phi_c) * wd / (math.exp(omega * Tswing) + 1)
    b = torch.stack((bx, by, torch.zeros(r.shape[0], device=env.device)), dim=1)

    # Clip the step to be within the kinematic limits
    p_local = icp_f.clone()
    p_local[:, 0] = torch.clip(icp_f[:, 0] - b[:, 0], -0.5, 0.5)    # Clip in the local x direction
    p_local[:, 1] = torch.clip(icp_f[:, 1] - b[:, 1], -0.3, 0.3)    # Clip in the local y direction


    # Compute desired step in the global frame
    p = _transfer_to_global_frame(p_local, asset.data.root_quat_w) + r

    p[:, 2] *= 0

    # print(f"icp_f = {icp_f},\n"
    #       f"icp_0 = {icp_0},\n"
    #       f"b = {b},\n")

    if visualize:
        sw_st_feet = torch.cat((p, foot_pos[:, int(0.5 - 0.5 * torch.sign(phi_c)), :]), dim=0)
        # env.footprint_visualizer.visualize(
        #     # TODO: Visualize both the current stance foot and the desired foot
        #     # translations=foot_pos[:, int(0.5 - 0.5*torch.sign(phi_c)), :], #p,
        #     # translations=foot_pos[:, (env.cfg.control_count % 2), :],
        #     translations=sw_st_feet,
        #     orientations=yaw_quat(asset.data.root_quat_w).repeat_interleave(2, dim=0),
        #     # repeat 0,1 for num_env
        #     # marker_indices=torch.tensor([0,1], device=env.device).repeat(env.num_envs),
        # )

    env.cfg.current_des_step[env_ids, :] = p[env_ids,:]  # This only works if I compute the new location once per step/on a timer

    return p


def foot_clearance(env: ManagerBasedRLEnv,
                   target_height: float,
                   sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Reward foot clearance."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get contact state
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    # Calculate foot heights
    feet_z_err = asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height
    pos_error = torch.square(feet_z_err) * ~contacts
    # print("feet_z:", asset.data.body_pos_w[:, asset_cfg.body_ids, 2]*~contacts)

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
            contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids[i], :].norm(dim=-1).max(dim=1)[0] > 1.0
            res += ~(contact ^ is_stance)
    else:
        for i in range(2):
            is_stance = stance_i == i
            # set is_stance to be true if the command is small
            is_stance = is_stance
            contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids[i], :].norm(dim=-1).max(dim=1)[0] > 1.0
            res += ~(contact ^ is_stance)
    return res

