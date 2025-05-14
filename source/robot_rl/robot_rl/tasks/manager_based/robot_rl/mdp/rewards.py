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
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse
from warp import quat_rotate_inv

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

def phase_feet_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, period: float, std: float,) -> torch.Tensor:
    """Reward feet in contact with the ground in the correct phase."""
    # If the feet are in contact at the right time then positive reward, else 0 reward

    # Contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get the current contacts
    # in_contact = ~contact_sensor.compute_first_air()[:, sensor_cfg.body_ids]  # Checks if the foot recently broke contact - which tells us we are not in contact. Does not reward jitter but use the dt.
    in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    in_contact = in_contact.float()

    # Contact schedule function
    tp = (env.sim.current_time % period) / period     # Scaled between 0-1
    phi_c = math.sin(2*torch.pi*tp)/math.sqrt(math.sin(2*torch.pi*tp)**2 + 0.04)

    # Compute reward
    reward = (in_contact[:, 0] - in_contact[:, 1])*phi_c

    # Add in the foot tracking
    # foot_pos = # TODO: Get foot position
    # reward = reward * torch.exp(-torch.norm(env.current_des_step - foot_pos, dim=1) / std)

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
    print(f"heading error: {wrap_to_pi(heading_des - heading)}")

    return reward

def compute_step_location(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg,
                          nom_height: float, Tswing: float, command_name: str, wdes: float) -> torch.Tensor:
    """Compute the step location using the LIP model."""
    asset = env.scene[asset_cfg.name]
    # Extract the relevant quantities
    # Base position
    r = asset.data.root_pos_w

    # Base linear velocity
    rdot = asset.data.root_lin_vel_w

    # Compute the natural frequency
    g = 9.81
    omega = math.sqrt(g / nom_height)

    # Compute initial ICP
    icp_0 = r[:2] + rdot[:2]/omega

    # Compute final ICP
    icp_f = math.exp(omega * Tswing)*icp_0 + (1 - math.exp(omega * Tswing)) * env.current_des_step

    # Compute desired step length and width
    command = env.command_manager.get_command(command_name)[:, :2]
    vdes = torch.norm(command[:, :2])
    sd = vdes * Tswing
    wd = wdes * 1

    # Compute ICP offsets
    bx = sd / (math.exp(omega * Tswing) - 1)
    by = wd / (math.exp(omega * Tswing) + 1)

    # Compute desired foot positions
    heading = euler_xyz_from_quat(asset.data.root_quat_w)[2]
    R = torch.tensor([[torch.cos(heading), -torch.sin(heading)], [torch.sin(heading), torch.cos(heading)]])

    p = icp_f - R*torch.stack([bx, torch.pow(torch.tensor(-1.), env.control_count)*by])

    env.current_des_step = p    # This only works if I compute the new location once per step/on a timer

