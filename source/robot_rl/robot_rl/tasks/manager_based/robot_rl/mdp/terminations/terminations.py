# Copyright 2026 Zachary Olkin. All rights reserved.

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

def no_progress(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Terminates the episode early if the robot is not making enough progress
    compared to expected distance at current time step.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")

    # Distance traveled from starting point
    root_pos = asset.data.root_pos_w[:, :2]
    origin = env.scene.env_origins[:, :2]
    distance = torch.norm(root_pos - origin, dim=1)

    # Expected distance so far = commanded_speed * time_elapsed
    commanded_speed = torch.norm(command[:, :2], dim=1)
    elapsed_time = env.episode_length_buf * env.step_dt  # [num_envs]
    expected_distance = commanded_speed * elapsed_time

    # Flag for insufficient progress
    behind_schedule = distance < (0.5 * expected_distance)

    # Optional: only trigger after a minimum time has passed (e.g., 30% of episode)
    enough_time_passed = env.episode_length_buf > (0.5 * env.max_episode_length)
    no_progress_flag = behind_schedule & enough_time_passed

    return no_progress_flag

def base_orientation(env, cmd_name: str, roll_limit_deg: float = 30.0, pitch_limit_deg: float = 30.0,
                     base_link: str = "pelvis_link",
                     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Terminates the episode if the robot's base orientation exceeds certain limits.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(cmd_name)
    ref_traj = cmd.y_des
    output_names = cmd.ordered_output_names

    pitch_idx = output_names.index(f"{base_link}:ori_y")
    roll_idx = output_names.index(f"{base_link}:ori_x")

    # Get base orientation in Euler angles
    root_quat = asset.data.root_quat_w  # [num_envs, 4]
    root_euler = euler_xyz_from_quat(root_quat, wrap_to_2pi=False)  # [num_envs, 3]

    roll_error = root_euler[0][:] - ref_traj[:, roll_idx]
    pitch_error = root_euler[1][:] - ref_traj[:, pitch_idx]

    # Define orientation limits (in radians)
    roll_limit = torch.deg2rad(torch.tensor(roll_limit_deg))  # ±30 degrees
    pitch_limit = torch.deg2rad(torch.tensor(pitch_limit_deg))  # ±30 degrees

    # Check if limits are exceeded
    roll_exceeded = (roll_error.abs() > roll_limit)
    pitch_exceeded = (pitch_error.abs() > pitch_limit)

    orientation_flag = roll_exceeded | pitch_exceeded

    return orientation_flag