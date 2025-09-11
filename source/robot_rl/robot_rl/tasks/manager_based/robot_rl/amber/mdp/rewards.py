# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import math
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi
from isaaclab.sensors import ContactSensor
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv
from pxr import Gf, UsdGeom
import omni.usd  
from .observations import desired_foot_targets_obs,current_foot_positions
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def joint_pos_target(
        env, asset_cfg: SceneEntityCfg, joint_des: torch.Tensor, std: float, joint_weight: torch.Tensor
    ) -> torch.Tensor:
    """Reward joints for proximity to a static desired joint position."""
    asset = env.scene[asset_cfg.name]

    q_pos = asset.data.joint_pos.detach().clone()
    q_err = joint_weight * torch.square(q_pos - joint_des)
    return torch.mean(torch.exp(-q_err / std ** 2), dim=-1)

# def symmetric_feet_air_time_biped(
#         env: ManagerBasedRLEnv,
#         command_name: str,
#         threshold: float,
#         sensor_cfg: SceneEntityCfg
# ) -> torch.Tensor:
#     """Reward long steps while enforcing symmetric gait patterns for bipeds.

#     Ensures balanced stepping by:
#     - Tracking air/contact time separately for each foot
#     - Penalizing asymmetric gait patterns
#     - Maintaining alternating single stance phases
#     """
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

#     # Split into left and right foot indices
#     left_ids = [sensor_cfg.body_ids[0]]
#     right_ids = [sensor_cfg.body_ids[1]]

#     # Get timing data for each foot
#     air_time_left = contact_sensor.data.current_air_time[:, left_ids]
#     air_time_right = contact_sensor.data.current_air_time[:, right_ids]
#     contact_time_left = contact_sensor.data.current_contact_time[:, left_ids]
#     contact_time_right = contact_sensor.data.current_contact_time[:, right_ids]
#     last_air_time_left = contact_sensor.data.last_air_time[:, left_ids]
#     last_air_time_right = contact_sensor.data.last_air_time[:, right_ids]
#     last_contact_time_left = contact_sensor.data.last_contact_time[:, left_ids]
#     last_contact_time_right = contact_sensor.data.last_contact_time[:, right_ids]

#     # Compute contact states
#     in_contact_left = contact_time_left > 0.0
#     in_contact_right = contact_time_right > 0.0

#     # Calculate mode times for each foot
#     left_mode_time = torch.where(in_contact_left, contact_time_left, air_time_left)
#     right_mode_time = torch.where(in_contact_right, contact_time_right, air_time_right)
#     last_left_mode_time = torch.where(in_contact_left, last_air_time_left, last_contact_time_left)
#     last_right_mode_time = torch.where(in_contact_right, last_air_time_right, last_contact_time_right)

#     # Check for proper single stance (one foot in contact, one in air)
#     left_stance = in_contact_left.any(dim=1) & (~in_contact_right.any(dim=1))
#     right_stance = in_contact_right.any(dim=1) & (~in_contact_left.any(dim=1))
#     single_stance = left_stance | right_stance

#     # Calculate symmetric reward components
#     left_reward = torch.min(torch.where(left_stance.unsqueeze(-1), left_mode_time, 0.0), dim=1)[0]
#     right_reward = torch.min(torch.where(right_stance.unsqueeze(-1), right_mode_time, 0.0), dim=1)[0]
#     last_left_reward = torch.min(torch.where(left_stance.unsqueeze(-1), last_left_mode_time, 0.0), dim=1)[0]
#     last_right_reward = torch.min(torch.where(right_stance.unsqueeze(-1), last_right_mode_time, 0.0), dim=1)[0]
#     # Combine rewards with symmetry penalty
#     base_reward = (left_reward + right_reward) / 2.0
#     symmetry_penalty = torch.abs(left_reward - last_right_reward) + torch.abs(right_reward - last_left_reward)
#     reward = base_reward - 0.1 * symmetry_penalty

#     # Apply threshold and command scaling
#     reward = torch.clamp(reward, max=threshold)
#     command_scale = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1

#     return reward * command_scale

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


def print_foot_positions(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Event function that prints left & right foot positions for each selected env.
    Returns a zero‐tensor (shape = [len(env_ids)]) so it satisfies the EventTerm API.
    """
    asset = env.scene[asset_cfg.name]
    # body_pos_w: [N_envs, N_bodies, 3]
    pos = asset.data.body_pos_w
    B   = pos.shape[1]
    # convention: B-2 = right foot, B-1 = left foot
    right = pos[env_ids, B-2, :].cpu().numpy()
    left  = pos[env_ids, B-1, :].cpu().numpy()

    # print out each
    for idx, e in enumerate(env_ids.tolist()):
        print(f"[Env {e}] left_foot={left[idx]}, right_foot={right[idx]}")

    # return dummy zeros
    return torch.zeros(env_ids.shape[0], device=pos.device)
#def phase_feet_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, period: float) -> torch.Tensor:
#     """Reward feet in contact with the ground in the correct phase."""
#     # If the feet are in contact at the right time then positive reward, else 0 reward
#
#     # Contact sensor
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#
#     # Get the current contacts
#     in_contact = not contact_sensor.compute_first_air(period/200.)    # Checks if the foot recently broke contact - which tells us we are not in contact. Does not reward jitter but use the dt.
#
#     # Check if the foot should be in contact by comparing to the phase.
#     ground_phase = is_ground_phase(env, period)
#
#     # Compute reward
#     reward = torch.where(in_contact & ground_phase, 1.0, 0.0)
#
#     return reward

def lip_gait_tracking(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, period: float, std: float,
                      nom_height: float, Tswing: float, command_name: str, wdes: float,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ) -> torch.Tensor:
    """Reward feet in contact with the ground in the correct phase."""
    # If the feet are in contact at the right time then positive reward, else 0 reward

    # Get the robot asset
    robot = env.scene[asset_cfg.name]

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

    # Add in the foot tracking
    foot_pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
    swing_foot_pos = foot_pos[:, int(0.5 + 0.5*torch.sign(phi_c))]
    # swing_foot_pos = foot_pos[:, ((env.cfg.control_count + 1) % 2), :]

    # print(f"swing foot index: {((env.cfg.control_count + 1) % 2)}, in contact 0: {in_contact[:, 0]}")
    # print(f"foot index: {int(0.5 + 0.5*torch.sign(phi_c))}")
    # print(f"stance foot pos: {stance_foot_pos}, des pos: {env.cfg.current_des_step[:, :2]}")

    # TODO: Debug and put back!
    # reward = reward * torch.exp(-torch.norm(env.cfg.current_des_step[:, :2] - swing_foot_pos, dim=1) / std)

    return reward

def lip_feet_tracking(env: ManagerBasedRLEnv, period: float, std: float,
                      Tswing: float,
                      feet_bodies: SceneEntityCfg,
                      asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ) -> torch.Tensor:
    """Reward the lip foot step tracking."""
    # Get the robot asset
    robot = env.scene[asset_cfg.name]

    # Contact schedule function
    tp = (env.sim.current_time % period) / period     # Scaled between 0-1
    phi_c = torch.tensor(math.sin(2*torch.pi*tp)/math.sqrt(math.sin(2*torch.pi*tp)**2 + Tswing), device=env.device)

    # Foot tracking
    foot_pos = robot.data.body_pos_w[:, feet_bodies.body_ids, :2]
    swing_foot_pos = foot_pos[:, int(0.5 + 0.5*torch.sign(phi_c))]
    reward = torch.exp(-torch.norm(env.cfg.current_des_step[:, :2] - swing_foot_pos, dim=1) / std)

    # print(f"swing_foot_norm: {torch.norm(swing_foot_pos, dim=1)}")
    # print(f"distance: {torch.norm(env.cfg.current_des_step[:, :2] - swing_foot_pos, dim=1)}")
    # print(f"reward: {reward}")

    # Update the com linear velocity running average
    alpha = 0.25
    env.cfg.com_lin_vel_avg = (1-alpha)*env.cfg.com_lin_vel_avg + alpha*robot.data.root_com_lin_vel_w

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



def compute_step_location(env: ManagerBasedRLEnv, env_ids: torch.Tensor,
                          nom_height: float, Tswing: float, command_name: str, wdes: float,
                          feet_bodies: SceneEntityCfg,
                          sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
                          asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                          visualize: bool = True) -> torch.Tensor:
    """Compute the step location using the LIP model."""
    asset = env.scene[asset_cfg.name]
    feet = env.scene[feet_bodies.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Desired velocity in the world frame
    vwdes = quat_rotate(asset.data.root_quat_w, env.command_manager.get_command(command_name))

    # Extract the relevant quantities
    # Base position
    r = asset.data.root_pos_w #asset.data.root_com_pos_w

    # print(f"r: {r}")

    # Base linear velocity
    # TODO: Try filtering this to make it less sensitive
    # rdot = 0.2 * torch.ones((r.shape[0], 3), device=r.device) #env.cfg.com_lin_vel_avg #asset.data.root_com_lin_vel_w    # TODO: Is this supposed to be world or body?
    # rdot = env.cfg.com_lin_vel_avg #asset.data.root_com_lin_vel_w    # TODO: Is this supposed to be world or body?
    # rdot = asset.data.root_com_lin_vel_w
    rdot = vwdes
    # print(f"rdot: {rdot}")

    # rdot[:, 1] *= 0

    # Compute the natural frequency
    g = 9.81
    # tnom_height = r[:, 2] #nom_height * torch.ones(r.shape[0], device=env.device)
    tnom_height = nom_height * torch.ones(r.shape[0], device=env.device)
    omega = math.sqrt(g / nom_height) #torch.sqrt(g / tnom_height) #nom_height)
    omega_dup = omega #omega.unsqueeze(1).repeat(1, 2)
    # Compute initial ICP
    icp_0 = r[:, :2] + rdot[:, :2]/omega_dup

    # Get current foot position
    foot_pos = feet.data.body_pos_w[:, feet_bodies.body_ids]
    # Determine what is in contact
    # contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    # stance_foot_pos = torch.sum(foot_pos * contacts, dim=2)
    # TODO: I think using this way to compute the stance foot is prone to error. I should do it based on contact.
    #   If both feet are in contact then average their position to get the stance location
    #   If neither foot is in contact then make it directly under the COM.
    # stance_foot_pos = foot_pos[:, (env.cfg.control_count % 2), :2]

    # TODO: Try using the schedule function
    # Contact schedule function
    tp = (env.sim.current_time % (2*Tswing)) / (2*Tswing)     # Scaled between 0-1
    phi_c = torch.tensor(math.sin(2*torch.pi*tp)/math.sqrt(math.sin(2*torch.pi*tp)**2 + Tswing), device=env.device)

    # print(f"foot idx: {int(0.5 - 0.5*torch.sign(phi_c))}, phi: {phi_c}, tp: {tp}, time: {env.sim.current_time}")
    stance_foot_pos = foot_pos[:, int(0.5 - 0.5*torch.sign(phi_c)), :2]


    # in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    # in_contact = in_contact.float()
    # stance_foot_pos = foot_pos * in_contact

    # Average position
    # TODO: Deal with in_contact all 0's
    # stance_foot_pos = torch.sum(stance_foot_pos, dim=1) / torch.sum(in_contact, dim=1)

    # Compute final ICP
    icp_f = math.exp(omega_dup * Tswing)*icp_0 + (1 - math.exp(omega_dup * Tswing)) * stance_foot_pos #env.cfg.current_des_step[:, :2]

    # print(f"icp0 diff: {icp_0 - r[:, :2]}. icpf diff: {icp_f - r[:, :2]}")

    # print(f"icp 0: {icp_0}, icp_f: {icp_f}")

    # Compute desired step length and width
    command = env.command_manager.get_command(command_name)[:, :2]
    # Convert the local velocity command to world frame
    # vdes = torch.norm(command[:, :2], dim=1)
    sd = torch.abs(command[:, 0]) * Tswing #vdes * Tswing
    wd = torch.abs(command[:, 1]) * Tswing #wdes * torch.ones((foot_pos.shape[0]), device=env.device)

    # Compute ICP offsets
    bx = sd / (math.exp(omega * Tswing) - 1)
    by = wd / (math.exp(omega * Tswing) - 1) #(math.exp(omega * Tswing) + 1)

    # Compute desired foot positions
    heading = euler_xyz_from_quat(asset.data.root_quat_w)[2]
    # print(f"heading: {heading}")
    cos_head = torch.cos(heading)
    sin_head = torch.sin(heading)
    row1 = torch.stack([cos_head, -sin_head], dim=1)
    row2 = torch.stack([sin_head, cos_head], dim=1)
    R = torch.stack([row1, row2], dim=1)  # Shape (N, 2, 2)

    # print(f"R shape: {R.shape}")
    # print(f"icp shape: {icp_f.shape}")

    # b = torch.stack([bx, torch.pow(torch.tensor(-1., device=command.device), env.cfg.control_count)*by])
    b = torch.zeros((r.shape[0], 3), device=env.device)
    b[:, :2] = torch.stack((-bx, -by), dim=1) #torch.stack((bx, torch.sign(phi_c)*by), dim=1)

    # Convert offset to the global frame
    b = quat_rotate(asset.data.root_quat_w, b)


    print(f"r: {r}\n"
          f"rdot: {rdot},\n"
          f"omega: {omega},\n"
          f"icp_0: {icp_0},\n"
          f"icp_f: {icp_f},\n"
          f"stance_foot_pos: {stance_foot_pos},\n"
          f"Tswing: {Tswing},\n"
          f"vwdes: {vwdes},\n"
          f"offset: {b}")

    # b = b.repeat(icp_f.shape[0], 1)
    # print(f"b shape: {b.shape}")

    # ph = r[:, :2] + torch.stack((sd, wd), dim=1)
    # The subtraction is weird, I need to get the frames correct
    ph = icp_0 #- b[:, :2] #torch.bmm(R, b.unsqueeze(-1)).squeeze(-1)

    # print(f"sd: {sd}, wd: {wd}, vwdes: {vwdes}")

    # print(f"ph: {ph}")

    # print(f"ph shape: {ph.shape}")

    # print(f"p shape: {p.shape}")    # Need to compute for all the envs
    p = torch.zeros((ph.shape[0], 3), device=command.device)    # For setting the height
    p[:, :2] = ph

    # TODO Remove
    # p[:, 0] *= 0
    # p[:, :2] = r[:, :2]

    # TODO: Clip to be within the kinematic limits (I'm not sure this should really be needed)

    # print(f"des pos: {p}")

    # print(f"r: {r}, by: {torch.pow(torch.tensor(-1., device=command.device), env.cfg.control_count)*by}")
    # print(f"sim time: {env.sim.current_time}, p: {p}")
    # print(f"env.cfg.control_count {env.cfg.control_count}")

    if visualize:
        sw_st_feet = torch.cat((p, foot_pos[:, int(0.5 - 0.5*torch.sign(phi_c)), :]), dim=0)
        env.footprint_visualizer.visualize(
            # TODO: Visualize both the current stance foot and the desired foot
            # translations=foot_pos[:, int(0.5 - 0.5*torch.sign(phi_c)), :], #p,
            # translations=foot_pos[:, (env.cfg.control_count % 2), :],
            translations=sw_st_feet,
            orientations=yaw_quat(asset.data.root_quat_w).repeat_interleave(2, dim=0),
            # repeat 0,1 for num_env
            # marker_indices=torch.tensor([0,1], device=env.device).repeat(env.num_envs),
        )

    # env.cfg.current_des_step = p    # This only works if I compute the new location once per step/on a timer
    env.cfg.current_des_step[env_ids, :] = p[env_ids, :]    # This only works if I compute the new location once per step/on a timer
    # env.cfg.control_count += 1
    # print(f"updated des pos! time: {env.sim.current_time}")
    return p

def contact_no_vel(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward feet contact with zero velocity."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids] * contacts.unsqueeze(-1)
    penalize = torch.square(body_vel[:,:,:3])
    return torch.sum(penalize, dim=(1,2))


def debug_print_joint_order(env: ManagerBasedRLEnv,
                            asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """
    Print the full joint (DOF) ordering for the given articulated asset.
    """
    asset = env.scene[asset_cfg.name]

    # IsaacLab puts the canonical order in `asset.dof_names`; fall back if needed
    try:
        joint_names = asset.dof_names
    except AttributeError:
        try:
            joint_names = asset.joint_names
        except AttributeError:
            joint_names = list(asset.cfg.init_state.joint_pos.keys())

    print("\n===== Joint order for", asset_cfg.name, "=====")
    for i, name in enumerate(joint_names):
        print(f"[{i:02d}]  {name}")
    print("="*40, "\n")

def track_joint_angles_exp(
    env: ManagerBasedRLEnv,
    std: float = 0.05,           # controls steepness of the exponential
    threshold_deg: float = 15.0, # degrees beyond which penalty starts
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Exponentially penalize only q1_left, q2_left, q1_right, q2_right when they
    exceed ±threshold_deg, with penalty = exp((|θ|–threshold_rad)/std) – 1, otherwise zero.
    Returns a [num_envs] tensor of summed penalties.
    """
    # 1) grab the articulation
    asset = env.scene[asset_cfg.name]
    # debug_print_joint_order(env)

    # 2) get the ordering of joint names
    try:
        joint_names = asset.dof_names
    except AttributeError:
        try:
            joint_names = asset.joint_names
        except AttributeError:
            joint_names = list(asset.cfg.init_state.joint_pos.keys())

    # 3) select our four target joints
    target = ["q1_left", "q2_left", "q1_right", "q2_right"]
    idxs   = [joint_names.index(n) for n in target]

    # 4) pull their angles (radians)
    joint_pos = asset.data.joint_pos        # [N, D]
    angles    = joint_pos[:, idxs]          # [N, 4]

    # 5) compute excess beyond ±threshold
    thresh_rad = math.radians(threshold_deg)
    excess     = torch.relu(angles.abs() - thresh_rad)  # [N,4]

    # 6) exponential penalty per joint
    pen = torch.exp(excess / std) - 1.0                  # [N,4]

    # 7) sum across the four joints
    penalty = pen.sum(dim=1)                             # [N]
    penalty = torch.clamp(penalty,0,50)
    return penalty


def foot_phase_cycle_reward(
    env: ManagerBasedRLEnv,
    period: float = 0.8,
    command_name: Optional[str] = "base_velocity",
    left_sensor_name: str  = "contact_forces_left",
    right_sensor_name: str = "contact_forces_right",
    force_thresh: float    = 1.0,
    cmd_thresh: float      = 0.005,
) -> torch.Tensor:
    """
    *Per‐cycle* reward:

        +1  if **every** timestep of the *previous* cycle both shins were
            in-phase with the analytic stance (or the robot was “idle”)
        −2  otherwise.

    The returned value is held constant throughout the *current* cycle so the
    critic sees a stable signal.
    """
    N, dev = env.num_envs, env.device
    t      = env.sim.current_time

    # ------------------------------------------------------------------ #
    # 1)  current cycle index
    # ------------------------------------------------------------------ #
    cycle_idx = int(math.floor(t / period))

    # ------------------------------------------------------------------ #
    # 2)  commanded speed → idle mask
    # ------------------------------------------------------------------ #
    if command_name is not None:
        cmd2D = env.command_manager.get_command(command_name)[:, :2]      # [N,2]
        idle  = torch.norm(cmd2D, dim=1) < cmd_thresh                     # [N]
    else:
        idle = torch.zeros(N, device=dev, dtype=torch.bool)

    # ------------------------------------------------------------------ #
    # 3)  per-step stance correctness
    # ------------------------------------------------------------------ #
    tp       = (env.sim.current_time % period) / period
    stance_i = 1 if math.sin(2 * math.pi * tp) > 0 else 0                # 0=L,1=R

    sensors = [
        env.scene.sensors[left_sensor_name],
        env.scene.sensors[right_sensor_name],
    ]
    correct_flags = torch.ones(N, device=dev, dtype=torch.bool)

    for idx, sensor in enumerate(sensors):
        contact = (
            sensor.data.net_forces_w_history
                  .norm(dim=-1)            # [N,hist,bods]
                  .max(dim=1)[0]           # [N,bods]
                  .max(dim=1)[0]           # [N]
                  > force_thresh
        )
        # if idle this env is automatically “correct” for this foot
        this_ok = idle | (contact ^ (idx != stance_i))
        correct_flags &= this_ok                                              # AND across feet

    # ------------------------------------------------------------------ #
    # 4)  persistent per-cycle state
    # ------------------------------------------------------------------ #
    fn = foot_phase_cycle_reward
    if not hasattr(fn, "_prev_cycle"):
        fn._prev_cycle   = torch.full((N,), -1, device=dev, dtype=torch.long)
        fn._cycle_good   = torch.ones((N,), device=dev, dtype=torch.bool)
        fn._last_reward  = torch.zeros((N,), device=dev)

    prev_cycle  = fn._prev_cycle
    cycle_good  = fn._cycle_good
    last_reward = fn._last_reward

    # accumulate: if any step is bad, mark cycle_good=False
    cycle_good &= correct_flags

    # ------------------------------------------------------------------ #
    # 5)  when new cycle begins, emit reward for the *previous* cycle
    # ------------------------------------------------------------------ #
    new_cycle = cycle_idx != prev_cycle
    finished  = new_cycle & (prev_cycle >= 0)

    if finished.any():
        idx = finished.nonzero(as_tuple=False).flatten()
        last_reward[idx] = torch.where(
            cycle_good[idx],
            torch.tensor(1.0, device=dev),     # perfect cycle
            torch.tensor(-2.0, device=dev),    # had any out-of-phase step
        )
        # reset good-flag for the upcoming cycle
        cycle_good[idx] = True

    # update cycle index for envs that just entered a new one
    prev_cycle[new_cycle] = cycle_idx

    return last_reward


def foot_phase_contact_amber(
    env: ManagerBasedRLEnv,
    period: float = 0.8,
    command_name: Optional[str] = "base_velocity",
    left_sensor_name: str  = "contact_forces_left",
    right_sensor_name: str = "contact_forces_right",
    force_thresh: float    = 1.0,
    cmd_thresh: float      = 0.005,
    debug: bool            = False,
) -> torch.Tensor:
    """
    +1  for every foot that is *in phase* this step
    −5  for every foot that is *out of phase* this step.

    When the planar command speed-norm < `cmd_thresh`, treat both feet as
    “allowed stance”, i.e. always award +1 and never penalise.
    """
    N, dev = env.num_envs, env.device

    # -------------------------------------------------- phase → stance foot --
    tp       = (env.sim.current_time % period) / period
    phi      = math.sin(2 * math.pi * tp)
    stance_i = 1 if phi > 0 else 0          # 0-left stance, 1-right stance

    # ----------------------------------------------------- idle mask per env --
    if command_name is not None:
        cmd2D = env.command_manager.get_command(command_name)[:, :2]      # [N,2]
        idle  = torch.norm(cmd2D, dim=1) < cmd_thresh                     # [N]
    else:
        idle = torch.zeros(N, device=dev, dtype=torch.bool)

    # -------------------------------------------------------- contact flags --
    sensors  = [
        env.scene.sensors[left_sensor_name],
        env.scene.sensors[right_sensor_name],
    ]
    contacts = []
    for sensor in sensors:
        c = (
            sensor.data.net_forces_w_history
                  .norm(dim=-1)            # [N,hist,bods]
                  .max(dim=1)[0]           # [N,bods]
                  .max(dim=1)[0]           # [N]
                  > force_thresh
        )
        contacts.append(c)                 # list index 0=left, 1=right

    # ------------------------------------------------ compute reward per env --
    reward = torch.zeros(N, device=dev)

    for idx, contact in enumerate(contacts):
        in_phase = torch.where(
            idle,
            torch.ones_like(contact, dtype=torch.bool),      # idle ⇒ always ok
            contact ^ (idx != stance_i)                      # bool[N]
        )
        per_foot = torch.where(
            in_phase,
            torch.tensor(1.0,  device=dev),                  # +1  if good
            torch.tensor(-2.0, device=dev)                   # –5  if bad
        )
        reward += per_foot

    # -------------------------------------------- optional debug for env-0 --
    if debug and N > 0:
        print(
            f"[{env.sim.current_time:6.3f}s] stance={'R' if stance_i else 'L'} "
            f"idle={idle[0].item()}  "
            f"Lc={contacts[0][0].item()} Rc={contacts[1][0].item()}  "
            f"reward0={reward[0].item():.1f}"
        )

    return reward



def foot_contact_cycle_reward(
    env: ManagerBasedRLEnv,
    period: float                = 0.8,
    left_sensor_name: str        = "contact_forces_left",
    right_sensor_name: str       = "contact_forces_right",
    contact_thresh: float        = 0.5,
    expo_rate: float             = 2.0,    # growth rate for penalty
) -> torch.Tensor:
    """
    Once per `period`, count rising-edge contacts per foot:
      Ci = # of times force > contact_thresh transitioned 0→1.
    Reward = +1 if C_left==1 and C_right==1,
           = - [ (exp(expo_rate*(C_left-1)) -1)
                 + (exp(expo_rate*(C_right-1)) -1) ]
    Holds that same value for the entire next cycle.
    """
    N, dev = env.num_envs, env.device
    t      = env.sim.current_time

    # 1) current cycle idx
    cycle_idx = int(math.floor(t / period))

    # 2) detect rising-edge contact
    def rising(name):
        fh = env.scene.sensors[name].data.net_forces_w_history
        m  = fh.norm(dim=-1).max(dim=1)[0].max(dim=1)[0]
        c  = m > contact_thresh
        return c

    c_l = rising(left_sensor_name)
    c_r = rising(right_sensor_name)

    # 3) persistent state
    fn = foot_contact_cycle_reward
    if not hasattr(fn, "_prev_cycle"):
        fn._prev_cycle   = torch.full((N,), -1, device=dev, dtype=torch.long)
        fn._prev_l       = torch.zeros((N,), device=dev, dtype=torch.bool)
        fn._prev_r       = torch.zeros((N,), device=dev, dtype=torch.bool)
        fn._count_l      = torch.zeros((N,), device=dev, dtype=torch.long)
        fn._count_r      = torch.zeros((N,), device=dev, dtype=torch.long)
        fn._last_reward  = torch.zeros((N,), device=dev)

    prev_cycle  = fn._prev_cycle
    prev_l      = fn._prev_l
    prev_r      = fn._prev_r
    count_l     = fn._count_l
    count_r     = fn._count_r
    last_reward = fn._last_reward

    # rising edges: now contact & wasn't before
    rise_l = c_l & ~prev_l
    rise_r = c_r & ~prev_r

    # update previous contact flags
    prev_l.copy_(c_l)
    prev_r.copy_(c_r)

    # 4) detect new cycle
    new_cycle = cycle_idx != prev_cycle

    # 5) on finishing a cycle, compute reward & reset counters
    finishing = new_cycle & (prev_cycle >= 0)
    if finishing.any():
        idx = finishing.nonzero(as_tuple=False).flatten()
        Cl  = count_l[idx].to(torch.float)
        Cr  = count_r[idx].to(torch.float)

        # +1 if exactly one each
        perfect = (Cl == 1.0) & (Cr == 1.0)
        reward = torch.where(
            perfect,
            torch.ones_like(Cl, device=dev) * 5,
            - (torch.expm1(expo_rate * (Cl - 1.0)) +
               torch.expm1(expo_rate * (Cr - 1.0)))
        )
        last_reward[idx] = reward

        # reset counts to include any rising now
        count_l[idx] = rise_l[idx].long()
        count_r[idx] = rise_r[idx].long()

    # 6) otherwise accumulate rising-edge counts
    cont = ~new_cycle
    if cont.any():
        idxs = cont.nonzero(as_tuple=False).flatten()
        count_l[idxs] += rise_l[idxs].long()
        count_r[idxs] += rise_r[idxs].long()

    # 7) update cycle index
    prev_cycle[new_cycle] = cycle_idx
    # print(last_reward)
    # 8) return the held reward
    return last_reward



def foot_clearance_amber(
    env: ManagerBasedRLEnv,
    target_height: float,
    left_sensor_name: str = "contact_forces_left",
    right_sensor_name: str = "contact_forces_right",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize squared height‐error of each shin when it's off the ground.
    Returns a [num_envs] tensor = err_left + err_right.
    """
    # grab the robot and the two shin contact sensors
    asset: Articulation = env.scene[asset_cfg.name]
    left_sensor  = env.scene.sensors[left_sensor_name]
    right_sensor = env.scene.sensors[right_sensor_name]

    # compute contact boolean per env (True = in contact)
    def is_contact(sensor):
        fh = sensor.data.net_forces_w_history  # [N, hist, bodies, 3]
        # norm→max over history→max over bodies
        return (
            fh.norm(dim=-1)         # [N, hist, bodies]
              .max(dim=1)[0]        # [N, bodies]
              .max(dim=1)[0] > 1.0  # [N] bool
        )

    contact_l = is_contact(left_sensor)
    contact_r = is_contact(right_sensor)

    # get the body‐indices (each sensor attaches to one shin)
    left_id,  = left_sensor.cfg.body_ids
    right_id, = right_sensor.cfg.body_ids

    # fetch z‐positions
    z_l = asset.data.body_pos_w[:, left_id,  2]  # [N]
    z_r = asset.data.body_pos_w[:, right_id, 2]  # [N]

    # squared error from target, but only when **not** in contact
    err_l = torch.square(z_l - target_height) * (~contact_l).float()
    err_r = torch.square(z_r - target_height) * (~contact_r).float()

    out = err_l + err_r  # [N]
    # _check_nan("foot_clearance_amber", out)
    return out


def foot_air_time_symmetry(
    env: ManagerBasedRLEnv,
    period: float = 0.8,
    left_sensor_name: str  = "contact_forces_left",
    right_sensor_name: str = "contact_forces_right",
    contact_thresh: float   = 0.5,
    diff_threshold: float   = 5.0,   # count‐difference threshold for reward
    reward_good: float       = 2.0,  # reward when |AL−AR| < diff_threshold
) -> torch.Tensor:
    """
    Once per `period`, compute |AL − AR| where
      Ai = count of timesteps force_i < contact_thresh during the last cycle.
    If |AL−AR| < diff_threshold, emit +reward_good, else emit –|AL−AR|.
    Holds that same value for the entire next cycle.
    """
    N, dev = env.num_envs, env.device
    t      = env.sim.current_time

    # 1) which cycle are we in?
    cycle_idx = int(math.floor(t / period))

    # 2) airborne boolean per foot
    def in_air(name):
        fh = env.scene.sensors[name].data.net_forces_w_history
        m  = fh.norm(dim=-1).max(dim=1)[0].max(dim=1)[0]
        return m < contact_thresh

    a_l = in_air(left_sensor_name)
    a_r = in_air(right_sensor_name)

    # 3) persistent state init
    fn = foot_air_time_symmetry
    if not hasattr(fn, "_prev_cycle"):
        fn._prev_cycle   = torch.full((N,), -1, device=dev, dtype=torch.long)
        fn._air_l        = torch.zeros((N,), device=dev, dtype=torch.long)
        fn._air_r        = torch.zeros((N,), device=dev, dtype=torch.long)
        fn._last_reward  = torch.zeros((N,), device=dev)

    prev_cycle  = fn._prev_cycle
    air_l       = fn._air_l
    air_r       = fn._air_r
    last_reward = fn._last_reward

    # 4) detect new cycle
    new_cycle = (cycle_idx != prev_cycle)

    # 5) on finishing a cycle, compute reward & reset counters
    finishing = new_cycle & (prev_cycle >= 0)
    if finishing.any():
        idx  = finishing.nonzero(as_tuple=False).flatten()
        diff = (air_l[idx] - air_r[idx]).abs().to(torch.float)
        # +reward_good if diff under threshold, else –diff
        last_reward[idx] = torch.where(
            diff < diff_threshold,
            torch.full_like(diff, reward_good),
            -diff
        )
        # reset counters for the new cycle, include current step if airborne
        air_l[idx] = a_l[idx].long()
        air_r[idx] = a_r[idx].long()

    # 6) accumulate during the cycle
    cont = ~new_cycle
    if cont.any():
        idx = cont.nonzero(as_tuple=False).flatten()
        air_l[idx] += a_l[idx].long()
        air_r[idx] += a_r[idx].long()

    # 7) update cycle index
    prev_cycle[new_cycle] = cycle_idx

    # 8) return the held reward for this cycle
    return last_reward

def torso_rotation_term(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso"]),
    reward_window_deg: float = 7.0,     # degrees for positive reward
    penalty_threshold_deg: float = 15.0,# degrees beyond which penalty applies
    penalty_cap: float = 20.0,          # max exp(θ_deg)
) -> torch.Tensor:
    """
    +2 reward when |tilt| ≤ reward_window_deg,
    –min(exp(|tilt_deg|), penalty_cap) penalty when |tilt| > penalty_threshold_deg,
    0 otherwise.

    Tilt angle = 2*acos(w) from body_quat_w, converted to degrees.
    """
    # get torso quaternion
    asset = env.scene[asset_cfg.name]
    quat  = asset.data.body_quat_w[:, asset_cfg.body_ids, :].squeeze(1)  # [N,4]
    w     = torch.clamp(quat[:,0], -1.0, 1.0)
    angle_rad = 2.0 * torch.acos(w)               # [N]
    angle_deg = angle_rad * (180.0 / torch.pi)    # [N]

    # positive reward region
    within = angle_deg.abs() <= reward_window_deg
    reward = torch.where(within, torch.tensor(2.0, device=env.device), torch.tensor(0.0, device=env.device))

    # penalty region
    outside = angle_deg.abs() > penalty_threshold_deg
    pen_val = -torch.clamp(torch.exp(angle_deg.abs()), max=penalty_cap)
    penalty = torch.where(outside, pen_val, torch.tensor(0.0, device=env.device))

    # combine
    result = reward + penalty
    # print(result)
    return result


def alternation_contact_reward(
    env: ManagerBasedRLEnv,
    left_sensor_name: str = "contact_forces_left",
    right_sensor_name: str = "contact_forces_right",
) -> torch.Tensor:
    """
    +1 when contact alternates feet (L→R or R→L),
    -1 when the same foot contacts twice in a row,
     0 otherwise (no new contact).
    """
    N, device = env.num_envs, env.device

    # contact flags
    def contact(name):
        fh = env.scene.sensors[name].data.net_forces_w_history
        return (
            fh.norm(dim=-1)
              .max(dim=1)[0]
              .max(dim=1)[0] > 1.0
        )

    c_l = contact(left_sensor_name)
    c_r = contact(right_sensor_name)

    # rising‐edges (new contact events)
    if not hasattr(alternation_contact_reward, "_prev_l"):
        alternation_contact_reward._prev_l = torch.zeros(N, device=device, dtype=torch.bool)
        alternation_contact_reward._prev_r = torch.zeros(N, device=device, dtype=torch.bool)
        # last contact: -1 = none yet, 0 = left, 1 = right
        alternation_contact_reward._last   = torch.full((N,), -1, device=device, dtype=torch.long)
        # stored reward until next event
        alternation_contact_reward._out    = torch.zeros(N, device=device)

    prev_l = alternation_contact_reward._prev_l
    prev_r = alternation_contact_reward._prev_r
    last   = alternation_contact_reward._last
    out    = alternation_contact_reward._out

    event_l = c_l & ~prev_l
    event_r = c_r & ~prev_r

    # LEFT contacts
    idx_l = event_l.nonzero(as_tuple=False).flatten()
    if idx_l.numel():
        # if last was RIGHT (1) → alternation → +1; 
        # if last was LEFT (0) → consecutive → -1;
        # if last==-1 → first contact → 0
        last_vals = last[idx_l]
        # build mask for each case
        mask_alt = last_vals == 1
        mask_same= last_vals == 0
        out[idx_l[mask_alt]]  =  1.0
        out[idx_l[mask_same]] = -1.0
        # record LEFT as last
        last[idx_l] = 0

    # RIGHT contacts
    idx_r = event_r.nonzero(as_tuple=False).flatten()
    if idx_r.numel():
        last_vals = last[idx_r]
        mask_alt = last_vals == 0
        mask_same= last_vals == 1
        out[idx_r[mask_alt]]  =  1.0
        out[idx_r[mask_same]] = -1.0
        last[idx_r] = 1

    # update prev flags
    prev_l.copy_(c_l)
    prev_r.copy_(c_r)
    # print(out)
    out = torch.where(out < 0, out * 5 , out)
    return out


def feet_slide_amber(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_shin", "right_shin"]),
    left_sensor_name: str = "contact_forces_left",
    right_sensor_name: str = "contact_forces_right",
) -> torch.Tensor:
    """
    Penalize foot sliding on Amber shins:
    Returns a [num_envs] tensor = sum of squared horizontal foot velocities
    when each shin is in contact with the ground.
    """
    # 1) fetch the two shin contact sensors
    left_s   = env.scene.sensors[left_sensor_name]
    right_s  = env.scene.sensors[right_sensor_name]

    # 2) build contact masks [N]
    def in_contact(sensor):
        fh = sensor.data.net_forces_w_history    # [N, hist_len, bodies, 3]
        # norm → max over history → max over bodies → bool [N]
        return (
            fh.norm(dim=-1)
              .max(dim=1)[0]
              .max(dim=1)[0]
              > 1.0
        )

    c_l = in_contact(left_s)
    c_r = in_contact(right_s)

    # 3) get shin horizontal velocities
    asset     = env.scene[asset_cfg.name]
    left_id, right_id = asset_cfg.body_ids
    body_vel  = asset.data.body_lin_vel_w        # [N, num_bodies, 3]
    v_l       = body_vel[:, left_id, :2]        # [N, 2]
    v_r       = body_vel[:, right_id, :2]       # [N, 2]

    # 4) squared speed
    speed_sq_l = torch.sum(v_l**2, dim=-1)      # [N]
    speed_sq_r = torch.sum(v_r**2, dim=-1)      # [N]

    # 5) penalty only when in contact
    penalty = speed_sq_l * c_l.float() + speed_sq_r * c_r.float()
    return penalty

def alternative_linear_cycle(
    env: ManagerBasedRLEnv,
    command_name: str,
    left_sensor_name: str  = "contact_forces_left",
    right_sensor_name: str = "contact_forces_right",
    min_cmd_speed: float   = 0.05,
    penalty_bad: float     = -5.0,   # punishment when cycle fails
    max_step: float        = 0.2,    # cap on forward‐step reward (m)
) -> torch.Tensor:
    """
    On the second alternating foot contact:
      – If both the first and second steps progressed forward:
          reward = min((x_second - x_first) * sign(cmd), max_step)
      – Else:
          reward = penalty_bad
    That value fires once per cycle, then holds. Zeroed if |cmd|<min_cmd_speed.
    """
    N, dev = env.num_envs, env.device
    asset  = env.scene["robot"]

    # 1) rising‐edge detection
    def contact(name):
        fh = env.scene.sensors[name].data.net_forces_w_history
        return (
            fh.norm(dim=-1)
              .max(dim=1)[0]
              .max(dim=1)[0]
              > 1.0
        )

    c_l, c_r = contact(left_sensor_name), contact(right_sensor_name)

    # initialize on first call
    if not hasattr(alternative_linear_cycle, "_prev_l"):
        alternative_linear_cycle._prev_l      = torch.zeros(N, device=dev, dtype=torch.bool)
        alternative_linear_cycle._prev_r      = torch.zeros(N, device=dev, dtype=torch.bool)
        alternative_linear_cycle._step        = torch.zeros(N, device=dev, dtype=torch.long)
        alternative_linear_cycle._first_foot  = torch.full((N,), -1, device=dev, dtype=torch.long)
        alternative_linear_cycle._first_ok    = torch.zeros(N, device=dev, dtype=torch.bool)
        alternative_linear_cycle._out         = torch.zeros(N, device=dev)
    prev_l     = alternative_linear_cycle._prev_l
    prev_r     = alternative_linear_cycle._prev_r
    step       = alternative_linear_cycle._step
    first_foot = alternative_linear_cycle._first_foot
    first_ok   = alternative_linear_cycle._first_ok
    out        = alternative_linear_cycle._out

    ev_l = c_l & ~prev_l   # new left contact
    ev_r = c_r & ~prev_r   # new right contact
    prev_l.copy_(c_l)
    prev_r.copy_(c_r)

    # 2) compute relative x positions & command direction
    pos     = asset.data.body_pos_w
    root_x  = pos[:,0,0]
    B       = pos.shape[1]
    # body_pos_w rows: [-2]=right, [-1]=left toe
    rel_l   = pos[:, B-2, 0] - root_x
    rel_r   = pos[:, B-1, 0] - root_x
    # print("right:",rel_r,";left:",rel_l)
    cmd_x    = env.command_manager.get_command(command_name)[:,0]
    dir_sign = torch.sign(cmd_x)
    moving   = (cmd_x.abs() > min_cmd_speed).float()

    # 3) first contact in cycle
    idx = (ev_l & (step == 0)).nonzero(as_tuple=False).flatten()
    if idx.numel():
        first_foot[idx] = 0
        first_ok[idx]   = ((rel_l[idx] - rel_r[idx]) * dir_sign[idx]) > 0
        step[idx]       = 1

    idx = (ev_r & (step == 0)).nonzero(as_tuple=False).flatten()
    if idx.numel():
        first_foot[idx] = 1
        first_ok[idx]   = ((rel_r[idx] - rel_l[idx]) * dir_sign[idx]) > 0
        step[idx]       = 1

    # 4) second contact completes the cycle
    # L→R cycle?
    idx = (ev_r & (step == 1) & (first_foot == 0)).nonzero(as_tuple=False).flatten()
    if idx.numel():
        delta    = (rel_r[idx] - rel_l[idx]) * dir_sign[idx]
        # positive forward → reward = min(Δ, max_step)
        reward_p = torch.clamp(delta, min=0.0, max=max_step)
        # negative backward → penalty = max(Δ, -max_step)  (a negative number)
        penalty_p= torch.clamp(delta, min=-max_step, max=0.0)
        both_ok  = first_ok[idx] & (delta > 0)
        out[idx] = torch.where(both_ok, reward_p, penalty_p)
        step[idx] = 0

    # R→L cycle?
    idx = (ev_l & (step == 1) & (first_foot == 1)).nonzero(as_tuple=False).flatten()
    if idx.numel():
        delta    = (rel_l[idx] - rel_r[idx]) * dir_sign[idx]
        reward_p = torch.clamp(delta, min=0.0, max=max_step)
        penalty_p= torch.clamp(delta, min=-max_step, max=0.0)
        both_ok  = first_ok[idx] & (delta > 0)
        out[idx] = torch.where(both_ok, reward_p, penalty_p)
        step[idx] = 0
    # out = torch.where(out > 0.0, out * 50.0, out)

    # 5) hold & mask by commanded movement
    result = out * moving * 100
    result = torch.where(result < 0.0, result * 10, result)
    # print(result)
    return result


def alternate_feet_cycle(
    env: ManagerBasedRLEnv,
    period: float = 0.8,
    left_sensor_name: str = "contact_forces_left",
    right_sensor_name: str = "contact_forces_right",
) -> torch.Tensor:
    """
    Penalise >1 contacts of the same foot within a full gait cycle (length = 'period').
    Penalty grows exponentially with extra contacts:
        contacts_per_cycle = 1 → penalty 0  (perfect)
                              2 → penalty 1
                              3 → penalty 3
                              4 → penalty 7
                              ...
    Return shape: [num_envs]  (non-negative).
    """
    N, dev = env.num_envs, env.device

    # ---------- helpers -------------------------------------------------
    def contact_flag(sensor_name: str) -> torch.Tensor:           # bool[N]
        fh = env.scene.sensors[sensor_name].data.net_forces_w_history
        return (fh.norm(dim=-1).max(dim=1)[0].max(dim=1)[0] > 1.0)

    # ---------- initialise persistent buffers on first call -------------
    if not hasattr(alternate_feet_cycle, "_cycle"):
        # current cycle index (floor(t/period))
        alternate_feet_cycle._cycle_idx = torch.full((N,), -1, device=dev, dtype=torch.long)
        # contact counts in *current* cycle
        alternate_feet_cycle._cnt_l     = torch.zeros(N, device=dev, dtype=torch.long)
        alternate_feet_cycle._cnt_r     = torch.zeros(N, device=dev, dtype=torch.long)
        # previous step contact flags (for rising-edge detection)
        alternate_feet_cycle._prev_cl   = torch.zeros(N, device=dev, dtype=torch.bool)
        alternate_feet_cycle._prev_cr   = torch.zeros(N, device=dev, dtype=torch.bool)

    cyc      = alternate_feet_cycle._cycle_idx
    cnt_l    = alternate_feet_cycle._cnt_l
    cnt_r    = alternate_feet_cycle._cnt_r
    prev_cl  = alternate_feet_cycle._prev_cl
    prev_cr  = alternate_feet_cycle._prev_cr

    # ---------- compute current cycle index -----------------------------
    t              = env.sim.current_time
    cycle_scalar = math.floor(t / period)                               # plain int
    cycle_now    = torch.full((N,), cycle_scalar, device=dev, dtype=torch.long)
    new_cycle_mask = cycle_now != cyc                           # bool[N]

    # ---------- contact edge detection ----------------------------------
    cl = contact_flag(left_sensor_name)
    cr = contact_flag(right_sensor_name)
    rising_l =  cl & ~prev_cl
    rising_r =  cr & ~prev_cr

    # update counts within *current* cycle
    cnt_l += rising_l.long()
    cnt_r += rising_r.long()

    # ---------- at cycle change: compute penalty & reset counts ----------
    # extra_contacts = max(0, count-1)
    extra_l = torch.clamp(cnt_l - 1, min=0)
    extra_r = torch.clamp(cnt_r - 1, min=0)
    # exponential penalty: 2^extra − 1   (0→0, 1→1, 2→3, 3→7,…)
    pen_l   = (2**extra_l) - 1
    pen_r   = (2**extra_r) - 1
    cycle_penalty      = pen_l + pen_r                 # [N]

    # emit penalty *only* on the first step of the new cycle; else 0
    penalty = torch.where(new_cycle_mask, cycle_penalty.float(), torch.zeros_like(cycle_penalty))

    # reset counts for envs that started a new cycle
    cnt_l[new_cycle_mask]  = 0
    cnt_r[new_cycle_mask]  = 0
    cyc[new_cycle_mask]    = cycle_now[new_cycle_mask]

    # ---------- store current contact flags for next step ---------------
    prev_cl.copy_(cl)
    prev_cr.copy_(cr)

    return penalty


from typing import Optional

def contact_step_com_reward(
    env: ManagerBasedRLEnv,
    command_name: str            = "base_velocity",
    left_sensor_name: str        = "contact_forces_left",
    right_sensor_name: str       = "contact_forces_right",
    force_thresh: float          = 1.0,
    max_reward_dist: float       = 0.30,   # clamp in metres  (±)
    min_cmd_speed: float         = 0.05,   # ignore very small commands
    debug: bool                  = False,
) -> torch.Tensor:
    """
    *Rising-edge* reward:

        r = clamp( (x_foot – x_COM) * sign(cmd_x) ,  –max , +max )

    • Computed only at the instant a shin first makes ground contact.
    • Uses Amber link order:   B-2 = **left**,  B-1 = **right** foot.
    • Holds the most recent reward value until the next contact event.
    """
    N, dev = env.num_envs, env.device

    # ------------------------------------------------------------------ #
    # 1)  detect contact & rising edges
    # ------------------------------------------------------------------ #
    def contact_bool(sensor_name: str) -> torch.Tensor:
        fh = env.scene.sensors[sensor_name].data.net_forces_w_history
        return (
            fh.norm(dim=-1).max(dim=1)[0].max(dim=1)[0] > force_thresh
        )  # bool[N]

    c_L = contact_bool(left_sensor_name)
    c_R = contact_bool(right_sensor_name)

    fn = contact_step_com_reward
    if not hasattr(fn, "_prev_L"):
        fn._prev_L      = torch.zeros(N, device=dev, dtype=torch.bool)
        fn._prev_R      = torch.zeros(N, device=dev, dtype=torch.bool)
        fn._reward_hold = torch.zeros(N, device=dev)                # initial 0

    rise_L = c_L & ~fn._prev_L     # left just landed
    rise_R = c_R & ~fn._prev_R     # right just landed

    fn._prev_L.copy_(c_L)
    fn._prev_R.copy_(c_R)

    # ------------------------------------------------------------------ #
    # 2)  positions & command sign
    # ------------------------------------------------------------------ #
    asset = env.scene["robot"]
    pos_w = asset.data.body_pos_w                    # [N,B,3]
    B     = pos_w.shape[1]
    x_L   = pos_w[:, B-2, 0]                        # left foot x
    x_R   = pos_w[:, B-1, 0]                        # right foot x
    x_COM = pos_w[:, 3,   0]                        # COM x (body index 3)

    cmd_x      = env.command_manager.get_command(command_name)[:, 0]   # [N]
    dir_sign   = torch.sign(cmd_x)                                     # ±1,0
    moving     = cmd_x.abs() > min_cmd_speed                           # bool[N]

    # ------------------------------------------------------------------ #
    # 3)  compute reward on contact events
    # ------------------------------------------------------------------ #
    # clamp helper
    def _clamp(v):
        return torch.clamp(v, min=-max_reward_dist, max=+max_reward_dist)

    # left foot lands
    if rise_L.any():
        idx = rise_L.nonzero(as_tuple=False).flatten()
        delta = (x_L[idx] - x_COM[idx]) * dir_sign[idx]
        fn._reward_hold[idx] = _clamp(delta)

    # right foot lands
    if rise_R.any():
        idx = rise_R.nonzero(as_tuple=False).flatten()
        delta = (x_R[idx] - x_COM[idx]) * dir_sign[idx]
        fn._reward_hold[idx] = _clamp(delta)

    # zero reward when not moving to avoid noise
    fn._reward_hold[~moving] = 0.0

    # ------------------------------------------------------------------ #
    # 4)  optional debug prints
    # ------------------------------------------------------------------ #
    if debug:
        for i in rise_L.nonzero(as_tuple=False).flatten().tolist():
            print(f"[{env.sim.current_time:6.3f}s]  Left contact env {i}: "
                  f"x_L={x_L[i]:.3f}, x_COM={x_COM[i]:.3f}, "
                  f"reward={fn._reward_hold[i]:+.3f}")
        for i in rise_R.nonzero(as_tuple=False).flatten().tolist():
            print(f"[{env.sim.current_time:6.3f}s]  Right contact env {i}: "
                  f"x_R={x_R[i]:.3f}, x_COM={x_COM[i]:.3f}, "
                  f"reward={fn._reward_hold[i]:+.3f}")

    # ------------------------------------------------------------------ #
    return fn._reward_hold



def contact_no_vel_amber(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["left_shin", "right_shin"]
    ),
    left_sensor_name: str = "contact_forces_left",
    right_sensor_name: str = "contact_forces_right",
) -> torch.Tensor:
    """
    Penalise squared foot velocity (x-y-z) **only** when the left/right shin
    is in ground contact.  Returns a [num_envs] penalty (≥0).
    """
    # 1) helper for contact flag ------------------------------------------------
    def in_contact(sensor_name: str) -> torch.Tensor:        # bool [N]
        fh = env.scene.sensors[sensor_name].data.net_forces_w_history
        return (
            fh.norm(dim=-1)           # [N, hist, bodies]
              .max(dim=1)[0]          # [N, bodies]
              .max(dim=1)[0] > 1.0    # bool [N]
        )

    c_left  = in_contact(left_sensor_name).float()           # [N]
    c_right = in_contact(right_sensor_name).float()          # [N]

    # 2) fetch foot world velocities -------------------------------------------
    asset  = env.scene[asset_cfg.name]
    left_id, right_id = asset_cfg.body_ids
    vel    = asset.data.body_lin_vel_w                      # [N, bodies, 3]
    v_l    = vel[:, left_id,  :]                            # [N, 3]
    v_r    = vel[:, right_id, :]                            # [N, 3]

    # 3) squared speed, mask by contact ----------------------------------------
    pen_l = torch.sum(v_l**2, dim=-1) * c_left              # [N]
    pen_r = torch.sum(v_r**2, dim=-1) * c_right             # [N]

    penalty = pen_l + pen_r                                 # [N]
    return penalty


def stride_consistency_penalty(
    env: ManagerBasedRLEnv,
    left_sensor_name:  str = "contact_forces_left",
    right_sensor_name: str = "contact_forces_right",
    force_thresh: float     = 1.0,
    max_penalty:   float    = 0.30,   # clamp (m) so an outlier can’t explode loss
    debug: bool             = False,
) -> torch.Tensor:
    """
    On every *new* foot-landing event:
        stride  = |x_curr − x_prev_same_foot|
        diff    = |stride − stride_prev_same_foot|
        reward  = − clamp(diff , 0 , max_penalty)

    • Returns latest penalty, held constant until the next landing.
    • First two landings of each foot get 0 (no previous stride to compare).
    """
    N, dev = env.num_envs, env.device

    # ------------- detect “new landing” (rising‐edge contact) -------------- #
    def contact_bool(sensor_name):
        fh = env.scene.sensors[sensor_name].data.net_forces_w_history
        return fh.norm(dim=-1).max(dim=1)[0].max(dim=1)[0] > force_thresh   # bool[N]

    c_L = contact_bool(left_sensor_name)
    c_R = contact_bool(right_sensor_name)

    fn = stride_consistency_penalty
    if not hasattr(fn, "_init"):
        fn._prev_L        = torch.zeros(N, device=dev, dtype=torch.bool)
        fn._prev_R        = torch.zeros(N, device=dev, dtype=torch.bool)
        fn._prev_x_L      = torch.zeros(N, device=dev)   # last landing x
        fn._prev_x_R      = torch.zeros(N, device=dev)
        fn._prev_stride_L = torch.zeros(N, device=dev)   # last stride length
        fn._prev_stride_R = torch.zeros(N, device=dev)
        fn._reward_hold   = torch.zeros(N, device=dev)
        fn._have_prev_L   = torch.zeros(N, device=dev, dtype=torch.bool)
        fn._have_prev_R   = torch.zeros(N, device=dev, dtype=torch.bool)
        fn._init          = True

    rise_L = c_L & ~fn._prev_L
    rise_R = c_R & ~fn._prev_R
    fn._prev_L.copy_(c_L)
    fn._prev_R.copy_(c_R)

    # ------------- x positions -------------------------------------------- #
    asset = env.scene["robot"]
    pos_w = asset.data.body_pos_w
    B     = pos_w.shape[1]
    x_L   = pos_w[:, B-2, 0]        # left toe x
    x_R   = pos_w[:, B-1, 0]        # right toe x

    # ------------- process left landings ---------------------------------- #
    if rise_L.any():
        idx = rise_L.nonzero(as_tuple=False).flatten()
        stride = (x_L[idx] - fn._prev_x_L[idx]).abs()
        diff   = (stride - fn._prev_stride_L[idx]).abs()
        penalty = -torch.clamp(diff, 0.0, max_penalty)

        # store for next time
        fn._prev_stride_L[idx] = torch.where(fn._have_prev_L[idx], stride, stride*0)
        fn._prev_x_L[idx]      = x_L[idx]
        fn._have_prev_L[idx]   = True

        fn._reward_hold[idx]   = penalty

        if debug:
            for i,s,d,p in zip(idx.tolist(), stride.tolist(), diff.tolist(), penalty.tolist()):
                print(f"[L land env{i}] stride={s:.3f} diff={d:.3f} penalty={p:.3f}")

    # ------------- process right landings --------------------------------- #
    if rise_R.any():
        idx = rise_R.nonzero(as_tuple=False).flatten()
        stride = (x_R[idx] - fn._prev_x_R[idx]).abs()
        diff   = (stride - fn._prev_stride_R[idx]).abs()
        penalty = -torch.clamp(diff, 0.0, max_penalty)

        fn._prev_stride_R[idx] = torch.where(fn._have_prev_R[idx], stride, stride*0)
        fn._prev_x_R[idx]      = x_R[idx]
        fn._have_prev_R[idx]   = True

        fn._reward_hold[idx]   = penalty

        if debug:
            for i,s,d,p in zip(idx.tolist(), stride.tolist(), diff.tolist(), penalty.tolist()):
                print(f"[R land env{i}] stride={s:.3f} diff={d:.3f} penalty={p:.3f}")

    # hold last computed penalty until next landing
    return fn._reward_hold


def joint_symmetry_cycle(
    env: ManagerBasedRLEnv,
    Ts: float             = 0.4,      # single-step duration  (2 Ts per cycle)
    asset_name: str       = "robot",
    std: float            = 0.15,     # width for exp decay (rad)
    diff_threshold: float = 0.10,     # rad; beyond ⇒ penalty
    reward_good: float    = 1.0,      # base reward if within threshold
    penalty_slope: float  = 2.0,      # slope for excess diff
    debug: bool           = False,
) -> torch.Tensor:
    """
    Same half-cycle definition as `rcs_phase_reward_no_pos`:

        φ  = (t mod 2Ts) / 2Ts
        sin(2πφ) > 0  →  “first” half (right stance in `rcs_phase_reward`)
        sin(2πφ) < 0  →  “second” half (left stance)

    We collect joint averages in each half and, at every transition
    2nd→1st half, compute

        diff = Σ_j | μ_L^j(first) − μ_R^j(second) |
             + Σ_j | μ_R^j(first) − μ_L^j(second) |

    Reward held for the next cycle:

        +reward_good * exp(−diff/std)           if diff ≤ diff_threshold
        −penalty_slope · (diff − diff_threshold) otherwise
    """
    N, dev = env.num_envs, env.device
    t      = env.sim.current_time
    asset  = env.scene[asset_name]

    # ---------------- joint indices --------------------------------------- #
    try:
        jnames = asset.dof_names
    except AttributeError:
        jnames = list(asset.cfg.init_state.joint_pos.keys())

    idx_q1L = jnames.index("q1_left")
    idx_q2L = jnames.index("q2_left")
    idx_q1R = jnames.index("q1_right")
    idx_q2R = jnames.index("q2_right")

    q = asset.data.joint_pos                                # [N,D]
    qL = torch.stack((q[:, idx_q1L], q[:, idx_q2L]), dim=1) # [N,2]
    qR = torch.stack((q[:, idx_q1R], q[:, idx_q2R]), dim=1) # [N,2]

    # ---------------- half-cycle indexing (matched to rcs_phase) ---------- #
    cycle_len = 2 * Ts
    phi       = (t % cycle_len) / cycle_len                 # φ ∈ [0,1)
    half_idx  = 0 if math.sin(2 * math.pi * phi) > 0 else 1 # 0=first,1=second
    cycle_id  = int(math.floor(t / cycle_len))

    # ---------------- persistent buffers ---------------------------------- #
    fn = joint_symmetry_cycle
    if not hasattr(fn, "_prev_half"):
        fn._prev_half   = torch.full((N,), -1, device=dev, dtype=torch.long)
        fn._sum_L_h0    = torch.zeros((N,2), device=dev)
        fn._sum_R_h0    = torch.zeros((N,2), device=dev)
        fn._cnt_h0      = torch.zeros((N,),  device=dev, dtype=torch.long)
        fn._sum_L_h1    = torch.zeros((N,2), device=dev)
        fn._sum_R_h1    = torch.zeros((N,2), device=dev)
        fn._cnt_h1      = torch.zeros((N,),  device=dev, dtype=torch.long)
        fn._reward_hold = torch.zeros((N,),  device=dev)

    # accumulate
    if half_idx == 0:
        fn._sum_L_h0 += qL
        fn._sum_R_h0 += qR
        fn._cnt_h0   += 1
    else:
        fn._sum_L_h1 += qL
        fn._sum_R_h1 += qR
        fn._cnt_h1   += 1

    # detect full-cycle boundary (half 1 → 0)
    prev_half = fn._prev_half
    new_cycle = (prev_half == 1) & (half_idx == 0)

    if new_cycle.any():
        idx = new_cycle.nonzero(as_tuple=False).flatten()

        μL0 = fn._sum_L_h0[idx] / fn._cnt_h0[idx].clamp(min=1).unsqueeze(-1)
        μR0 = fn._sum_R_h0[idx] / fn._cnt_h0[idx].clamp(min=1).unsqueeze(-1)
        μL1 = fn._sum_L_h1[idx] / fn._cnt_h1[idx].clamp(min=1).unsqueeze(-1)
        μR1 = fn._sum_R_h1[idx] / fn._cnt_h1[idx].clamp(min=1).unsqueeze(-1)

        diff = (μL0 - μR1).abs().sum(dim=1) + (μR0 - μL1).abs().sum(dim=1)

        reward_cycle = torch.where(
            diff <= diff_threshold,
            reward_good * torch.exp(-diff / std),
            - penalty_slope * (diff - diff_threshold)
        )
        fn._reward_hold[idx] = reward_cycle

        # reset accumulators
        fn._sum_L_h0[idx].zero_(); fn._sum_R_h0[idx].zero_(); fn._cnt_h0[idx].zero_()
        fn._sum_L_h1[idx].zero_(); fn._sum_R_h1[idx].zero_(); fn._cnt_h1[idx].zero_()

        if debug:
            for i, d, r in zip(idx.tolist(), diff.tolist(), reward_cycle.tolist()):
                print(f"[cycle {cycle_id-1}] env{i} diff={d:.3f} reward={r:+.3f}")

    # update half tracker
    fn._prev_half.fill_(half_idx)

    # optional phase alignment check
    if debug and env.num_envs > 0:
        print(f"[{t:6.3f}s] half_idx={half_idx}  (0=sin>0)  prev_half={prev_half[0]}")

    return fn._reward_hold


def continuous_contact_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),        # kept for API symmetry
    left_sensor_name: str  = "contact_forces_left",
    right_sensor_name: str = "contact_forces_right",
    threshold: float       = 0.10,   # s before penalty starts
    rate: float            = 10.0,   # grows like exp(rate·(t-τ))
    min_cmd_speed: float   = 0.05,   # ignore when basically idle
) -> torch.Tensor:
    """
    Penalty = Σ_f  max(0, exp(rate·(t_f – threshold)) – 1),
    where t_f is the uninterrupted contact duration of foot f.
    Returns a [num_envs] tensor.   **Use a NEGATIVE weight in RewardCfg.**
    """
    N, dev = env.num_envs, env.device
    dt     = env.step_dt

    # ------------------------------------------------------------------ #
    # 1) contact flags for each foot
    # ------------------------------------------------------------------ #
    def in_contact(sensor_name: str) -> torch.Tensor:          # bool [N]
        fh = env.scene.sensors[sensor_name].data.net_forces_w_history
        return (fh
                .norm(dim=-1)
                .max(dim=1)[0]      # over history
                .max(dim=1)[0]      # over bodies
                > 1.0)

    c_l = in_contact(left_sensor_name)
    c_r = in_contact(right_sensor_name)

    # ------------------------------------------------------------------ #
    # 2) keep our own contact timers (seconds)
    # ------------------------------------------------------------------ #
    if not hasattr(continuous_contact_penalty, "_timer_l"):
        continuous_contact_penalty._timer_l = torch.zeros(N, device=dev)
        continuous_contact_penalty._timer_r = torch.zeros(N, device=dev)

    timer_l = continuous_contact_penalty._timer_l
    timer_r = continuous_contact_penalty._timer_r

    # update timers: add dt when in contact, reset to 0 when not
    timer_l.copy_(torch.where(c_l, timer_l + dt, torch.zeros_like(timer_l)))
    timer_r.copy_(torch.where(c_r, timer_r + dt, torch.zeros_like(timer_r)))

    # ------------------------------------------------------------------ #
    # 3) exponential cost beyond threshold
    # ------------------------------------------------------------------ #
    def exp_cost(t: torch.Tensor) -> torch.Tensor:
        excess = t - threshold
        return torch.where(excess > 0,
                           torch.exp(rate * excess) - 1.0,
                           torch.zeros_like(t))

    penalty = exp_cost(timer_l) + exp_cost(timer_r)            # [N]

    # ------------------------------------------------------------------ #
    # 4) zero out when robot is hardly moving
    # ------------------------------------------------------------------ #
    cmd_x = env.command_manager.get_command("base_velocity")[:, 0]
    moving_mask = (cmd_x.abs() > min_cmd_speed).float()
    penalty = penalty * moving_mask
    penalty = torch.clamp(penalty, 0, 50)
    return penalty

def curr_vel_fn(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window_size: int = 10,        # low-pass filter window
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel,
       with a simple moving-average low-pass filter on the measured velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    device = env.device
    N = env.num_envs

    # 1) pull the raw measured velocity for the base link in world frame
    raw_act = asset.data.body_link_lin_vel_w[:, 3, :2]  # [N,2]

    # 2) setup persistent ring buffer for low-pass
    if not hasattr(track_lin_vel_x_amber, "_vel_buf"):
        # buffer shape: [window, N, 2]
        track_lin_vel_x_amber._vel_buf = torch.zeros(window_size, N, 2, device=device)
        track_lin_vel_x_amber._buf_ptr = 0

    buf = track_lin_vel_x_amber._vel_buf
    ptr = track_lin_vel_x_amber._buf_ptr

    # insert the newest measurement
    buf[ptr] = raw_act
    ptr = (ptr + 1) % window_size
    track_lin_vel_x_amber._buf_ptr = ptr

    # compute the filtered velocity as the window-average
    filt_act = buf.mean(dim=0)  # [N,2]
    
    # 3) now compute the command vs. filtered actual error
    # cmd = env.command_manager.get_command(command_name)[:, :2]  # [N,2]
    # lin_vel_error = torch.sum((cmd - filt_act)**2, dim=1)       # [N]
    
    return filt_act[:,0]


def cmd_vel_fn(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window_size: int = 10,        # low-pass filter window
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel,
       with a simple moving-average low-pass filter on the measured velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    device = env.device
    N = env.num_envs

    # 1) pull the raw measured velocity for the base link in world frame
    raw_act = asset.data.body_link_lin_vel_w[:, 3, :2]  # [N,2]

    # 2) setup persistent ring buffer for low-pass
    if not hasattr(track_lin_vel_x_amber, "_vel_buf"):
        # buffer shape: [window, N, 2]
        track_lin_vel_x_amber._vel_buf = torch.zeros(window_size, N, 2, device=device)
        track_lin_vel_x_amber._buf_ptr = 0

    buf = track_lin_vel_x_amber._vel_buf
    ptr = track_lin_vel_x_amber._buf_ptr

    # insert the newest measurement
    buf[ptr] = raw_act
    ptr = (ptr + 1) % window_size
    track_lin_vel_x_amber._buf_ptr = ptr

    # 3) now compute the command vs. filtered actual error
    cmd = env.command_manager.get_command(command_name)[:, :2]  # [N,2]
    # lin_vel_error = torch.sum((cmd - filt_act)**2, dim=1)       # [N]

    return cmd[:,0]

def track_lin_vel_x_amber(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    window_size: int = 10,        # low-pass filter window
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel,
       with a simple moving-average low-pass filter on the measured velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    device = env.device
    N = env.num_envs

    # 1) pull the raw measured velocity for the base link in world frame
    raw_act = asset.data.body_link_lin_vel_w[:, 3, :2]  # [N,2]

    # 2) setup persistent ring buffer for low-pass
    if not hasattr(track_lin_vel_x_amber, "_vel_buf"):
        # buffer shape: [window, N, 2]
        track_lin_vel_x_amber._vel_buf = torch.zeros(window_size, N, 2, device=device)
        track_lin_vel_x_amber._buf_ptr = 0

    buf = track_lin_vel_x_amber._vel_buf
    ptr = track_lin_vel_x_amber._buf_ptr

    # insert the newest measurement
    buf[ptr] = raw_act
    ptr = (ptr + 1) % window_size
    track_lin_vel_x_amber._buf_ptr = ptr

    # compute the filtered velocity as the window-average
    filt_act = buf.mean(dim=0)  # [N,2]
    
    # 3) now compute the command vs. filtered actual error
    cmd = env.command_manager.get_command(command_name)[:, :2]  # [N,2]
    lin_vel_error = torch.sum((cmd - filt_act)**2, dim=1)       # [N]
    # print("---------",filt_act,"/",cmd)
    # 4) exponential kernel
    std = max(std, 1e-4)
    final = -lin_vel_error / (std**2)
    final = torch.clamp(final, -50, 50)
    final = torch.exp(final)

    return final

def _phi_contact_scalar(phi: float) -> float:
    """ φ_contact(φ) = sin(2πφ) / √(sin²(2πφ)+0.04)  — Eq.(18) """
    s = math.sin(2 * math.pi * phi)
    return s / math.sqrt(s * s + 0.04)

def rcs_phase_reward_no_pos(
    env: ManagerBasedRLEnv,
    Ts: float                   = 0.4,      # single-step duration
    left_sensor_name: str       = "contact_forces_left",
    right_sensor_name: str      = "contact_forces_right",
    force_thresh: float         = 1.0,
    debug: bool                 = False,
) -> torch.Tensor:
    """
    r_cs = 9 · (1_R − 1_L) · φ_contact            [ NO foot-placement term ]

    • Link order for Amber:  B-2  =  **left**  foot,  B-1 = **right** foot
    • Outputs [N] reward, updated every step.
    """
    N, dev = env.num_envs, env.device
    t      = env.sim.current_time

    # φ_contact scalar for current moment
    phi_cycle = (t % (2 * Ts)) / (2 * Ts)        # φ ∈ [0,1)
    phi_cont  = _phi_contact_scalar(phi_cycle)   # float

    # contact booleans
    def contact(name: str) -> torch.Tensor:
        fh = env.scene.sensors[name].data.net_forces_w_history
        return (
            fh.norm(dim=-1).max(dim=1)[0].max(dim=1)[0] > force_thresh
        )                                         # bool[N]

    I_L = contact(left_sensor_name).float()      # [N]  (left foot: B-2)
    I_R = contact(right_sensor_name).float()     # [N]  (right foot: B-1)

    reward = 9.0 * (I_R - I_L) * phi_cont        # element-wise [N]

    # optional debug print for env-0
    if debug and N > 0:
        print(f"[t={t:6.3f}s] φc={phi_cont:+.3f}  I_L={I_L[0]}  I_R={I_R[0]}  r0={reward[0]:+.2f}")

    return reward

@torch.no_grad()
def rcs_phase_reward_with_placement(
    env: ManagerBasedRLEnv,
    Ts: float                   = 0.4,
    left_sensor_name: str       = "contact_forces_left",
    right_sensor_name: str      = "contact_forces_right",
    force_thresh: float         = 1.0,
    sigma: float                = 0.1,
    command_name: str           = "base_velocity",
    # note: asset_cfg only needed for contact flags, not for the obs
    asset_cfg: SceneEntityCfg   = SceneEntityCfg("robot"),
    debug: bool                 = False,
    visualise: bool             = False,
    half_cycle: bool            = True,
) -> torch.Tensor:
    """
    Uses the *relative* 4-d ObsTerm [ΔLx,ΔLz,ΔRx,ΔRz] from
    desired_foot_targets_obs.  No more 6-d slicing!
    r_cs = 9·(I_R−I_L)·φ_contact·exp(−‖Δx_z‖²/σ)
    """
    N, dev = env.num_envs, env.device
    t      = env.sim.current_time

    # 1) φ_contact scalar
    phi_cycle = (t % (2*Ts)) / (2*Ts)                 # [0,1)
    phi_cont  = _phi_contact_scalar(phi_cycle)       # float

    # 2) contact flags → I_L, I_R in {0,1}
    def contact(name: str) -> torch.Tensor:
        fh = env.scene.sensors[name].data.net_forces_w_history
        return (
            fh.norm(dim=-1)
              .max(dim=1)[0]
              .max(dim=1)[0]
              > force_thresh
        ).float()

    I_L = contact(left_sensor_name)   # [N]
    I_R = contact(right_sensor_name)  # [N]

    # 3) pull in your *relative* future targets: [ΔLx, ΔLz, ΔRx, ΔRz]
    fut_rel = desired_foot_targets_obs(
        env,
        Ts=Ts,
        nom_height=0.77,      # ignored by this obs‐call (already baked in)
        wdes=0.0,            # ignore
        command_name=command_name,
        asset_cfg=asset_cfg,
        debug=False,
        visualize=visualise,
        half_cycle=half_cycle,
    )  # [N,4]

    # 4) split into left vs right rel-vectors
    rel_L = fut_rel[:, 0:2]   # [ΔLx, ΔLz]
    rel_R = fut_rel[:, 2:4]   # [ΔRx, ΔRz]

    # 5) choose the stance‐foot rel-error per env
    #    if I_L==1 → stance=left → use rel_L, else (I_R==1) use rel_R
    #    (if both/neither, we just pick rel_R by default)
    stance_rel = torch.where(
        I_L.unsqueeze(1) > 0,
        rel_L,
        rel_R,
    )  # [N,2]

    # 6) placement penalty
    sq_err  = (stance_rel.pow(2)).sum(dim=1)         # [N]
    penalty = torch.exp(-sq_err / sigma)            # [N]

    # 7) assemble final reward
    reward = 9.0 * (I_R - I_L) * phi_cont * penalty  # [N]

    # 8) optional debug
    if debug and N > 0:
        print(
            f"[t={t:.3f}] φ={phi_cont:+.3f} "
            f"I_L={I_L[0]:.0f} I_R={I_R[0]:.0f} "
            f"rel_L={rel_L[0].cpu().numpy()} "
            f"rel_R={rel_R[0].cpu().numpy()} "
            f"pen={penalty[0]:.3f} r0={reward[0]:+.3f}"
        )
    # print(reward)
    return reward

## Critic

def base_lin_vel_amber(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.body_link_lin_vel_w[:,3,:]

## Observation

def base_ang_vel_amber(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Amber-specific angular velocity observation.

    Returns a [num_envs, 3] tensor with the **world-frame** angular velocity
    of body-link index 3 (the de-facto “root” for Amber).
    """
    asset = env.scene[asset_cfg.name]                      # articulation / rigid-object
    # body_link_ang_vel_w: [N, n_bodies, 3]  – pick index 3
    # print(asset.data.body_link_ang_vel_w[:, 3, :])
    return asset.data.body_link_ang_vel_w[:, 3, :]

# World gravity as a constant 3-vector (same one Isaac Lab uses)
_G_VEC_W = torch.tensor([0.0, 0.0, -1.0])


def projected_gravity_amber(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Returns the 3-vector of the world gravity direction expressed
    in the *body frame of link 3* (Amber’s de-facto “root”).

    Shape:  [num_envs, 3]
    """
    asset = env.scene[asset_cfg.name]                     # articulation / rigid-object
    quat  = asset.data.body_link_quat_w[:, 3, :]          # [N,4]  (w,x,y,z) of link 3
    # ensure gravity vector on correct device & dtype
    g_vec = _G_VEC_W.to(quat)

    # broadcast g_vec to match envs
    g = g_vec.expand(quat.shape[0], 3)

    # rotate into body frame via inverse quaternion
    g_body = quat_apply_inverse(quat, g)       # [N,3]
    return g_body


## LIP


from pxr import UsdGeom, Gf, Sdf

# def compute_step_location_local_amber(
#     env: ManagerBasedRLEnv,
#     env_ids: torch.Tensor,
#     nom_height: float,
#     Tswing: float,
#     command_name: str,
#     wdes: float,
#     feet_bodies: SceneEntityCfg,                           # API compatibility
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
#     asset_cfg: SceneEntityCfg  = SceneEntityCfg("robot"),
#     visualize: bool = True
# ) -> torch.Tensor:
#     """
#     Amber‐specific ICP step planner:
#       • COM ≈ body_pos_w[:,3]
#       • toes ≈ last two body indices
#       • computes both next‐left and next‐right targets
#       • visualizes red=right, green=left spheres
#       • writes env.current_des_step
#     """
#     N, dev = env.num_envs, env.device
#     asset  = env.scene[asset_cfg.name]

#     # 1) commanded velocity in local frame
#     command = env.command_manager.get_command(command_name)  # [N,?]

#     # 2) COM position in global (use body_pos_w index 3)
#     r = asset.data.body_pos_w[:, 3, :]                      # [N,3]

#     # 3) initial ICP offset
#     g     = 9.81
#     omega = math.sqrt(g / nom_height)
#     icp_0 = torch.zeros((N, 3), device=dev)
#     icp_0[:, :2] = command[:, :2] / omega

#     # 4) toe positions: second-last = right, last = left
#     pos      = asset.data.body_pos_w                       # [N, B, 3]
#     B        = pos.size(1)
#     foot_pos = pos[:, [B-2, B-1], :]                       # [N,2,3]

#     # 5) scalar phase
#     tp_scalar    = (env.sim.current_time % (2*Tswing)) / (2*Tswing)
#     phi_c_scalar = math.sin(2*math.pi*tp_scalar) / math.sqrt(math.sin(2*math.pi*tp_scalar)**2 + Tswing)
#     # broadcast to all envs
#     phi_c        = torch.full((N,), phi_c_scalar, device=dev)
#     phi_sign     = torch.sign(phi_c)                       # tensor [N] of ±1

#     # 6) helper to compute one target given stance index
#     def _compute_target(stance_idx: int) -> torch.Tensor:
#         # stance foot global pos
#         st = foot_pos[:, stance_idx, :].clone()
#         st[:, 2] = 0.0

#         # transform COM difference into local frame
#         local_diff = quat_rotate(
#             yaw_quat(quat_inv(asset.data.root_quat_w)),
#             (r - st)
#         )
#         exp_wT = math.exp(omega * Tswing)
#         icp_f  = exp_wT * icp_0 + (1 - exp_wT) * local_diff
#         icp_f[:, 2] = 0.0

#         # desired offset b
#         sd = torch.abs(command[:, 0]) * Tswing
#         bx = sd / (exp_wT - 1.0)
#         by = phi_sign * wdes / (exp_wT + 1.0)
#         b  = torch.stack((bx, by, torch.zeros(N, device=dev)), dim=1)

#         # clip in local
#         p_loc = icp_f.clone()
#         p_loc[:, 0] = torch.clamp(p_loc[:, 0] - b[:, 0], -0.5, 0.5)
#         p_loc[:, 1] = torch.clamp(p_loc[:, 1] - b[:, 1], -0.3, 0.3)

#         # back to global
#         glob = quat_rotate(yaw_quat(asset.data.root_quat_w), p_loc) + r
#         glob[:, 2] = 0.0
#         return glob

#     # 7) compute both swing targets
#     p_left  = _compute_target(0)   # right stance → left swing
#     p_right = _compute_target(1)   # left stance  → right swing

#     # 8) visualize both
#     if visualize:
#         stage = omni.usd.get_context().get_stage()
#         # grab the *true* COM for Amber
#         com = asset.data.root_com_pos_w      # [N,3]
#         for i in range(N):
#             # -- next-right in red --
#             prim = Sdf.Path(f"/World/debug/next_right_{i}")
#             if not stage.GetPrimAtPath(prim):
#                 sph = UsdGeom.Sphere.Define(stage, prim)
#                 sph.GetRadiusAttr().Set(0.05)
#             else:
#                 sph = UsdGeom.Sphere(stage.GetPrimAtPath(prim))
#             UsdGeom.XformCommonAPI(sph).SetTranslate(
#                 Gf.Vec3d(*p_right[i].tolist())
#             )
#             sph.CreateDisplayColorAttr().Set([Gf.Vec3f(1,0,0)])

#             # -- next-left in green --
#             prim = Sdf.Path(f"/World/debug/next_left_{i}")
#             if not stage.GetPrimAtPath(prim):
#                 sph = UsdGeom.Sphere.Define(stage, prim)
#                 sph.GetRadiusAttr().Set(0.05)
#             else:
#                 sph = UsdGeom.Sphere(stage.GetPrimAtPath(prim))
#             UsdGeom.XformCommonAPI(sph).SetTranslate(
#                 Gf.Vec3d(*p_left[i].tolist())
#             )
#             sph.CreateDisplayColorAttr().Set([Gf.Vec3f(0,1,0)])

#             # -- COM in blue --
#             prim = Sdf.Path(f"/World/debug/com_{i}")
#             if not stage.GetPrimAtPath(prim):
#                 sph = UsdGeom.Sphere.Define(stage, prim)
#                 sph.GetRadiusAttr().Set(0.03)
#             else:
#                 sph = UsdGeom.Sphere(stage.GetPrimAtPath(prim))
#             UsdGeom.XformCommonAPI(sph).SetTranslate(
#                 Gf.Vec3d(*com[i].tolist())
#             )
#             sph.CreateDisplayColorAttr().Set([Gf.Vec3f(0,0,1)])
#     # 9) store and return the “primary” step (here, right foot for example)
#     if not hasattr(env, "current_des_step"):
#         env.current_des_step = torch.zeros((N, 3), device=dev)
#     env.current_des_step[env_ids, :] = p_right[env_ids, :]
#     return p_right


def compute_step_location_local_amber(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    nom_height: float,
    Tswing: float,
    command_name: str,
    wdes: float,
    feet_bodies: SceneEntityCfg,                           # still here for API compatibility
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg  = SceneEntityCfg("robot"),
    visualize: bool = True
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # print("OK")
    # 1) commanded velocity in local frame
    command = env.command_manager.get_command(command_name)

    # 2) COM position in global from body_pos_w index 3
    r = asset.data.body_pos_w[:, 3, :]                      # [N,3]

    # 3) COM “velocity” placeholder
    rdot = command                                         # [N,2] ⟶ [N,3] when needed

    # 4) capture-point base
    g     = 9.81
    omega = math.sqrt(g / nom_height)

    icp_0 = torch.zeros((r.shape[0], 3), device=env.device)
    icp_0[:, :2] = rdot[:, :2] / omega

    # 5) foot positions from the last two body indices (B-1=right, B-2=left)
    pos    = asset.data.body_pos_w                          # [N, bodies, 3]
    B      = pos.shape[1]
    foot_pos = pos[:, [B-1, B-2], :]                        # [N,2,3]

    # 6) phase clock (unchanged)
    tp    = (env.sim.current_time % (2*Tswing)) / (2*Tswing)
    phi_c = torch.tensor(
        math.sin(2*math.pi*tp) / math.sqrt(math.sin(2*math.pi*tp)**2 + Tswing),
        device=env.device
    )

    # 7) stance foot in global
    idx       = (0.5 - 0.5 * torch.sign(phi_c)).long().item()
    stance_foot_pos = foot_pos[:, idx, :].clone()
    stance_foot_pos[:, 2] = 0.0

    # 8) coordinate transforms
    def to_global(v, quat):
        return quat_rotate(yaw_quat(quat), v)
    def to_local (v, quat):
        return quat_rotate(yaw_quat(quat_inv(quat)), v)

    # 9) final ICP in local frame
    exp_omT = math.exp(omega * Tswing)
    icp_f = (
        exp_omT * icp_0
        + (1 - exp_omT) * to_local(r - stance_foot_pos, asset.data.body_quat_w[:,3,:])
    )
    icp_f[:, 2] = 0.0

    # 10) offset b
    sd = torch.abs(command[:, 0]) * Tswing
    wd = wdes * torch.ones(r.shape[0], device=env.device)
    bx = sd / (exp_omT - 1.0)
    by = torch.sign(phi_c) * wd / (exp_omT + 1.0)
    b  = torch.stack((bx, by, torch.zeros_like(bx)), dim=1)

    # 11) clip in local
    p_local = icp_f.clone()
    p_local[:, 0] = torch.clamp(icp_f[:, 0] - b[:, 0], -0.5, 0.5)
    p_local[:, 1] = torch.clamp(icp_f[:, 1] - b[:, 1], -0.3, 0.3)

    # 12) back to global
    p = to_global(p_local, asset.data.body_quat_w[:,3,:]) + r
    p[:, 2] = 0.0

    # 13) optional visualization
    # if visualize:
    #     # visualize desired (p) and actual stance foot
    #     vis_pts = torch.cat([p, foot_pos[:, idx, :]], dim=0)
    #     env.footprint_visualizer.visualize(
    #         translations=vis_pts,
    #         orientations=yaw_quat(asset.data.root_quat_w).repeat_interleave(2, dim=0),
    #     )
    if visualize:
        stage = omni.usd.get_context().get_stage()
        for i in range(p.shape[0]):
            prim_path = f"/World/debug/future_step_{i}"
            if not stage.GetPrimAtPath(prim_path):
                sph = UsdGeom.Sphere.Define(stage, prim_path)
                sph.GetRadiusAttr().Set(0.02)
            else:
                sph = UsdGeom.Sphere(stage.GetPrimAtPath(prim_path))
            # set its translation without re-adding the op each time
            UsdGeom.XformCommonAPI(sph).SetTranslate(
                Gf.Vec3d(p[i,0].item(), p[i,1].item(), p[i,2].item())
            )
    # 14) write out and return
    # ensure we have a place to store it on the env
    if not hasattr(env, "current_des_step"):
        # allocate [num_envs × 3]
        env.current_des_step = torch.zeros((env.num_envs, 3), device=env.device)

    # now write into it
    env.current_des_step[env_ids, :] = p[env_ids, :]
    # print(p)
    return p


## plotting rewards

@torch.no_grad()
def left_foot_target_reward(
    env: ManagerBasedRLEnv,
    Ts: float                  = 0.4,
    nom_height: float          = 0.45,
    wdes: float                = 0.1,
    command_name: str          = "base_velocity",
    asset_cfg: SceneEntityCfg  = SceneEntityCfg("robot", body_names=["left_shin","right_shin"]),
    debug: bool                = False,
    visualize: bool            = False,
    half_cycle: bool           = True,
) -> torch.Tensor:
    """
    Reward = ΔLx, the desired‐minus‐current left‐foot X offset from
    desired_foot_targets_obs (which returns [ΔLx,ΔLz,ΔRx,ΔRz]).
    """
    # [N,4] = [ΔLx,ΔLz,ΔRx,ΔRz]
    fut_rel = desired_foot_targets_obs(
        env,
        Ts=Ts,
        nom_height=nom_height,
        wdes=wdes,
        command_name=command_name,
        asset_cfg=asset_cfg,
        debug=debug,
        visualize=visualize,
        half_cycle=half_cycle,
    )
    # take only the left‐foot X component
    left_x = fut_rel[:, 0]  # shape [N]
    return left_x

@torch.no_grad()
def right_foot_target_reward(
    env: ManagerBasedRLEnv,
    Ts: float                  = 0.4,
    nom_height: float          = 0.45,
    wdes: float                = 0.1,
    command_name: str          = "base_velocity",
    asset_cfg: SceneEntityCfg  = SceneEntityCfg("robot", body_names=["left_shin","right_shin"]),
    debug: bool                = False,
    visualize: bool            = False,
    half_cycle: bool           = True,
) -> torch.Tensor:
    """
    Reward = ΔLx, the desired‐minus‐current left‐foot X offset from
    desired_foot_targets_obs (which returns [ΔLx,ΔLz,ΔRx,ΔRz]).
    """
    # [N,4] = [ΔLx,ΔLz,ΔRx,ΔRz]
    fut_rel = desired_foot_targets_obs(
        env,
        Ts=Ts,
        nom_height=nom_height,
        wdes=wdes,
        command_name=command_name,
        asset_cfg=asset_cfg,
        debug=debug,
        visualize=visualize,
        half_cycle=half_cycle,
    )
    # take only the left‐foot X component
    right_x = fut_rel[:, 2]  # shape [N]
    return right_x

@torch.no_grad()
def current_left_foot_x(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_shin","right_shin"]),
) -> torch.Tensor:
    """
    Returns the current world-frame X position of the left shin link.
    """
    amber = env.scene[asset_cfg.name]
    left_id, _ = asset_cfg.body_ids
    pos0       = amber.data.body_pos_w          # [N, bodies, 3]
    B          = pos0.shape[1]
    # body_pos_w: [N, bodies, 3]
    return amber.data.body_pos_w[:, B-2, 0]  # [N,2,3]


@torch.no_grad()
def current_left_foot_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_shin","right_shin"]),
) -> torch.Tensor:
    """
    Returns the current world-frame X position of the left shin link.
    """
    amber = env.scene[asset_cfg.name]
    left_id, _ = asset_cfg.body_ids
    pos0       = amber.data.body_pos_w          # [N, bodies, 3]
    B          = pos0.shape[1]
    # body_pos_w: [N, bodies, 3]
    return amber.data.body_pos_w[:, B-2, 2]  # [N,2,3]


@torch.no_grad()
def current_right_foot_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_shin","right_shin"]),
) -> torch.Tensor:
    """
    Returns the current world-frame X position of the left shin link.
    """
    amber = env.scene[asset_cfg.name]
    left_id, _ = asset_cfg.body_ids
    pos0       = amber.data.body_pos_w          # [N, bodies, 3]
    B          = pos0.shape[1]
    # body_pos_w: [N, bodies, 3]
    return amber.data.body_pos_w[:, B-1, 2]  # [N,2,3]


@torch.no_grad()
def current_right_foot_x(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["left_shin","right_shin"]),
) -> torch.Tensor:
    """
    Returns the current world-frame X position of the right shin link.
    """
    amber = env.scene[asset_cfg.name]
    left_id, _ = asset_cfg.body_ids
    pos0       = amber.data.body_pos_w          # [N, bodies, 3]
    B          = pos0.shape[1]
    # body_pos_w: [N, bodies, 3]
    return amber.data.body_pos_w[:, B-1, 0]  # [N,2,3]