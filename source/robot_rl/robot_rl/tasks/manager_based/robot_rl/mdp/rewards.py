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
        env.footprint_visualizer.visualize(
            # TODO: Visualize both the current stance foot and the desired foot
            # translations=foot_pos[:, int(0.5 - 0.5*torch.sign(phi_c)), :], #p,
            # translations=foot_pos[:, (env.cfg.control_count % 2), :],
            translations=sw_st_feet,
            orientations=yaw_quat(asset.data.root_quat_w).repeat_interleave(2, dim=0),
            # repeat 0,1 for num_env
            # marker_indices=torch.tensor([0,1], device=env.device).repeat(env.num_envs),
        )

    env.cfg.current_des_step[env_ids, :] = p[env_ids,:]  # This only works if I compute the new location once per step/on a timer

    return p

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
    phi_c = torch.tensor(math.sin(2*torch.pi*tp)/math.sqrt(math.sin(2*torch.pi*tp)**2 + 0.04), device=env.device)

    stance_i = 0
    if phi_c > 0:
        stance_i = 1

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

def contact_no_vel(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward feet contact with zero velocity."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids] * contacts.unsqueeze(-1)
    penalize = torch.square(body_vel[:,:,:3])
    return torch.sum(penalize, dim=(1,2))