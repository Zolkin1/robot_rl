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

def phase_feet_contacts(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, period: float) -> torch.Tensor:
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

    return reward