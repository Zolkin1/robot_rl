from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_conjugate, quat_mul, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

def no_progress(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    Terminates the episode early if the robot is not making enough progress
    compared to expected distance at current time step.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command("base_velocity")

    # Distance traveled from starting point
    root_pos = wp.to_torch(asset.data.root_pos_w)[:, :2]
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
    """Terminate when the base roll/pitch deviates from the reference by more than the limits.

    The reference orientation is stored as a quaternion ``(x, y, z, w)`` inside ``cmd.y_des``.
    Error is taken as the world-frame relative quaternion ``q_act * q_ref^-1`` and decomposed
    into XYZ Euler angles to recover proper roll/pitch.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(cmd_name)
    ref_traj = cmd.y_des
    output_names = cmd.ordered_pos_output_names

    ori_idx = [output_names.index(f"{base_link}:ori_{a}") for a in ("x", "y", "z", "w")]
    ref_quat = ref_traj[:, ori_idx]
    act_quat = wp.to_torch(asset.data.root_quat_w)  # already (x, y, z, w)

    q_err = quat_mul(act_quat, quat_conjugate(ref_quat))
    roll_err, pitch_err, _ = euler_xyz_from_quat(q_err, wrap_to_2pi=False)
    roll_err = wrap_to_pi(roll_err)
    pitch_err = wrap_to_pi(pitch_err)

    roll_limit = torch.deg2rad(torch.tensor(roll_limit_deg, device=q_err.device))
    pitch_limit = torch.deg2rad(torch.tensor(pitch_limit_deg, device=q_err.device))

    return (roll_err.abs() > roll_limit) | (pitch_err.abs() > pitch_limit)


def illegal_terrain_contact(
    env, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate when the peak contact force on any of the sensor's bodies
    exceeds ``threshold``.

    Mirrors :func:`isaaclab.envs.mdp.terminations.illegal_contact` and reads
    unfiltered :attr:`ContactSensor.data.net_forces_w_history`. Backend-level
    filtering of contact partners (e.g. terrain-only) does not work reliably
    against the static terrain mesh in the Newton backend, so instead we set
    ``threshold`` high enough that incidental self-collision forces don't trip
    the termination — only a hard impact (e.g. torso/thigh striking a stair)
    will. The "terrain" in the name reflects the intent.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = wp.to_torch(contact_sensor.data.net_forces_w_history)
    body_slice = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, body_slice], dim=-1), dim=1)[0] > threshold,
        dim=1,
    )