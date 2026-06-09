# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply, quat_inv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# TODO: Break the key parts of this into another function that can be called from the play script easily.
#   Then I can more easily log initial conditions.

def reset_on_reference(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        command_name: str,
        conditioner_command_name: str,
        base_frame_name: str,
        base_z_offset: float = 0.03,
        joint_add_range: tuple[float, float] = (0.0, 0.0),
        rel_envs_on_ref: float = 0.5,
        special_val: float = 1.0,
        rel_envs_on_special: float = 0.4,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Reset the robot to a random point along the reference trajectory.

    This event samples a random time from the trajectory, extracts the base frame pose
    and joint positions at that time, and resets the robot to that state. The time offset
    is stored in the command so it knows the current phase.

    Args:
        env: The environment instance.
        env_ids: The environment IDs to reset.
        command_name: Name of the trajectory command term.
        base_frame_name: Name of the body frame to use for root pose (must exist in trajectory outputs).
        joint_scale_range: Tuple of (min_scale, max_scale) for uniform random scaling of joint positions.
            Default (1.0, 1.0) means no scaling.
        rel_envs_on_ref: Float giving the relative amount of envs to start on the reference trajectory.
        asset_cfg: Configuration for the robot asset.

    Raises:
        ValueError: If base_frame_name is not found in trajectory outputs.
        ValueError: If any robot joint is missing from the trajectory outputs.
    """
    # Get the robot asset and trajectory command
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    env.command_manager.get_term(conditioner_command_name)._resample(env_ids)   #     env.command_manager.get_term(conditioner_command_name).reset(env_ids)
    env.command_manager.get_term(conditioner_command_name)._update_command()
    num_env = len(env_ids)

    if num_env == 0:
        return

    r = torch.empty(len(env_ids), device=env.device)

    ref_env = torch.zeros(num_env, device=env.device)
    ref_env = r.uniform_(0.0, 1.0) <= (rel_envs_on_ref)
    ref_ids = env_ids[ref_env]
    num_ref_envs = len(ref_ids)

    # Temporarily adjust the commands to be in the special range
    r_on_ref = torch.empty(num_ref_envs, device=env.device)
    special_envs = r_on_ref.uniform_(0.0, 1.0) < rel_envs_on_special
    command = env.command_manager.get_term(conditioner_command_name).command
    command_clone = command.clone()
    command[ref_ids[special_envs], 0] = special_val * torch.ones(len(ref_ids[special_envs]), device=env.device)

    nonref_env = ref_env == False
    nonref_ids = env_ids[nonref_env]
    num_nonref_envs = len(nonref_ids)


    # Validate base frame exists in trajectory outputs
    pos_indices = _find_output_indices(cmd.ordered_pos_output_names, base_frame_name, "pos_")
    ori_indices = _find_output_indices(cmd.ordered_pos_output_names, base_frame_name, "ori_")

    if len(pos_indices) != 3:
        raise ValueError(
            f"Base frame '{base_frame_name}' must have pos_x, pos_y, pos_z in trajectory outputs. "
            f"Found {len(pos_indices)} position outputs."
        )
    if len(ori_indices) != 4:
        raise ValueError(
            f"Base frame '{base_frame_name}' must have ori_x, ori_y, ori_z, ori_w in trajectory outputs. "
            f"Found {len(ori_indices)} orientation outputs."
        )

    # TODO: Can get rid of this to speed things up
    # Validate all robot joints are in trajectory outputs
    traj_joint_names = set()
    for name in cmd.ordered_pos_output_names:
        if name.startswith("joint:"):
            traj_joint_names.add(name.split(":", 1)[1])

    missing_joints = []
    for joint_name in asset.joint_names:
        if joint_name not in traj_joint_names:
            missing_joints.append(joint_name)

    if missing_joints:
        raise ValueError(
            f"The following robot joints are missing from the trajectory outputs: {missing_joints}"
        )

    # Sample random times for each environment
    total_time = cmd.manager.get_total_time()
    random_times = torch.rand(num_ref_envs, device=env.device) * total_time

    # Get trajectory outputs at sampled times
    cmd.get_desired_outputs(random_times, env_ids=ref_ids)
    des_outputs = cmd.y_des
    y_sampled = des_outputs[ref_ids]  # Position outputs

    # Extract base frame position and orientation from outputs
    base_pos_rel = y_sampled[:, pos_indices]  # Shape: [num_env, 3]
    base_ori_quat_w = y_sampled[:, ori_indices]  # Shape: [num_env, 3] - roll, pitch, yaw

    # Add the ground->ankle_roll_link offset
    base_pos_rel[:, 2] += base_z_offset

    # Compute world-frame base pose
    base_pos_w = base_pos_rel + env.scene.env_origins[ref_ids]

    # Build pose tensor: [x, y, z, qw, qx, qy, qz]
    base_pose = torch.cat([base_pos_w, base_ori_quat_w], dim=-1)

    # Extract Base Vel
    des_doutputs = cmd.dy_des
    dy_sampled = des_doutputs[ref_ids]  # Velocity outputs (not used for now)

    lin_vel_indices = _find_output_indices(cmd.ordered_vel_output_names, base_frame_name, "pos_")
    ang_vel_indices = _find_output_indices(cmd.ordered_vel_output_names, base_frame_name, "ori_")


    # Set base velocity
    base_vel = torch.cat([dy_sampled[:, lin_vel_indices], dy_sampled[:, ang_vel_indices]], dim=-1) #torch.zeros(num_env, 6, device=env.device)


    # Set the reference frame position
    cmd.ref_poses[ref_ids, :3] = env.scene.env_origins[ref_ids]
    cmd.ref_poses[ref_ids, 3:] *= 0

    # Extract joint positions from trajectory output
    # Build a mapping from robot joint indices to trajectory output indices
    joint_pos = torch.zeros(num_ref_envs, len(asset.joint_names), device=env.device)
    joint_vel = torch.zeros_like(joint_pos)

    for i, joint_name in enumerate(asset.joint_names):
        traj_output_name = f"joint:{joint_name}"
        pos_traj_idx = cmd.ordered_pos_output_names.index(traj_output_name)
        vel_traj_idx = cmd.ordered_vel_output_names.index(traj_output_name)

        joint_pos[:, i] = y_sampled[:, pos_traj_idx]
        joint_vel[:, i] = dy_sampled[:, vel_traj_idx]

    # Apply optional uniform scaling to joint positions
    # TODO: Can put back for the on reference reset
    # min_scale, max_scale = joint_scale_range
    # if min_scale != 1.0 or max_scale != 1.0:
    #     scale_factors = torch.rand(num_ref_envs, 1, device=env.device) * (max_scale - min_scale) + min_scale
    #     joint_pos = joint_pos * scale_factors
    #     joint_vel = joint_vel * scale_factors

    # Store time offset in command so it knows the current phase
    cmd.init_time_offset[ref_ids] = random_times

    # Write states to simulation
    asset.write_root_pose_to_sim(base_pose, env_ids=ref_ids)
    asset.write_root_link_velocity_to_sim(base_vel, env_ids=ref_ids)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=ref_ids)

    # Restore command
    env.command_manager.get_term(conditioner_command_name).command[:] = command_clone

    # if num_ref_envs > 0:
    #     # Compute the measured outputs and print
    #     cmd.get_measured_outputs(random_times, env_ids=ref_ids)
    #
    #     # Get the desired outputs and print
    #     cmd.get_desired_outputs(random_times, env_ids=ref_ids)
    #
    #     # Compute V
    #     vdot, v = cmd.clf.compute_vdot(cmd.y_act, cmd.y_des, cmd.dy_act, cmd.dy_des)
    #
    #     cmd_vel = command_clone #command
    #
    #     # Pretty print the output names, desired values, and measured values
    #     _pretty_print_reset_state(
    #         cmd.ordered_pos_output_names,
    #         cmd.ordered_vel_output_names,
    #         cmd.y_des,
    #         cmd.y_act,
    #         cmd.dy_des,
    #         cmd.dy_act,
    #         random_times,
    #         v,
    #         vdot,
    #         cmd.clf.v_subgroups,
    #         cmd_vel,
    #         cmd.clf,
    #         env_idx=0,
    #     )


    if num_nonref_envs != 0:
        # Non reference resets
        root_states = asset.data.default_root_state[nonref_ids].clone()

        base_pose = root_states[:, 0:7]
        base_pose[:, :3] += env.scene.env_origins[nonref_ids]
        base_vel = root_states[:, 7:]

        # get default joint state
        joint_pos = asset.data.default_joint_pos[nonref_ids, asset_cfg.joint_ids].clone()
        joint_vel = asset.data.default_joint_vel[nonref_ids, asset_cfg.joint_ids].clone()

        # Add noise
        r = torch.empty(num_nonref_envs, len(asset.joint_names), device=env.device)
        r.uniform_(joint_add_range[0], joint_add_range[1])
        joint_pos += r

        asset.write_root_pose_to_sim(base_pose, env_ids=nonref_ids)
        asset.write_root_velocity_to_sim(base_vel, env_ids=nonref_ids)
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=nonref_ids)

        # Reset the time offset
        cmd.init_time_offset[nonref_ids] = 0.0



def _find_output_indices(ordered_names: list[str], frame_name: str, suffix_pattern: str) -> list[int]:
    """
    Find indices of outputs matching frame_name:suffix_pattern.

    Args:
        ordered_names: List of ordered output names.
        frame_name: The frame name to search for.
        suffix_pattern: The suffix pattern to match (e.g., "pos_" or "ori_").

    Returns:
        List of indices where the pattern matches.
    """
    indices = []
    for i, name in enumerate(ordered_names):
        if name.startswith(f"{frame_name}:") and suffix_pattern in name:
            indices.append(i)
    return indices


def _pretty_print_reset_state(
    output_pos_names: list[str],
    output_vel_names: list[str],
    y_des: torch.Tensor,
    y_act: torch.Tensor,
    dy_des: torch.Tensor,
    dy_act: torch.Tensor,
    times: torch.Tensor,
    v: torch.Tensor,
    vdot: torch.Tensor,
    v_subgroups: dict[str, torch.Tensor],
    cmd_vel: torch.Tensor,
    clf,
    env_idx: int = 0,
):
    """
    Pretty print the desired and measured outputs in a table format.

    Args:
        output_names: List of output names.
        y_des: Desired position outputs, shape [num_envs, num_outputs].
        y_act: Measured position outputs, shape [num_envs, num_outputs].
        dy_des: Desired velocity outputs, shape [num_envs, num_outputs].
        dy_act: Measured velocity outputs, shape [num_envs, num_outputs].
        times: Sampled times, shape [num_envs].
        v: Lyapunov function value, shape [num_envs].
        vdot: Lyapunov derivative, shape [num_envs].
        v_subgroups: Dict mapping subgroup names to their V contributions, each shape [num_envs].
        cmd_vel: Commanded velocity, shape [num_envs, vel_dim].
        env_idx: Index of the environment to print (default 0).
    """
    # Header
    print(f"\n{'='*94}")
    print(f"Reset State for Environment {env_idx}")
    print(f"Time: {times[env_idx].item():.4f} s")
    print(f"Commanded Velocity: {cmd_vel[env_idx].tolist()} m/s")
    print(f"{'='*94}")

    # Position table header
    print(f"\n{'--- Positions ---':^94}")
    print(f"{'Output Name':<30} {'Desired':>12} {'Measured':>12} {'Error':>12}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12}")

    # Position table rows
    err_val = clf.compute_y_err(y_act, y_des)[env_idx]
    err_idx = 0
    for i, name in enumerate(output_pos_names):
        des_val = y_des[env_idx, i].item()
        act_val = y_act[env_idx, i].item()
        if ":ori_w" in output_pos_names[i]:
            print(f"{name:<30} {des_val:>12.5f} {act_val:>12.5f}")
        else:
            print(f"{name:<30} {des_val:>12.5f} {act_val:>12.5f} {err_val[err_idx]:>12.5f}")
            err_idx += 1


    # Velocity table header
    print(f"\n{'--- Velocities ---':^94}")
    print(f"{'Output Name':<30} {'Desired':>12} {'Measured':>12} {'Error':>12}")
    print(f"{'-'*30} {'-'*12} {'-'*12} {'-'*12}")

    # Velocity table rows
    for i, name in enumerate(output_vel_names):
        des_val = dy_des[env_idx, i].item()
        act_val = dy_act[env_idx, i].item()
        err_val = act_val - des_val
        print(f"{name:<30} {des_val:>12.5f} {act_val:>12.5f} {err_val:>12.5f}")

    # V subgroups section
    print(f"\n{'V Subgroups':<30} {'Value':>12}")
    print(f"{'-'*30} {'-'*12}")
    for name, values in v_subgroups.items():
        print(f"{name:<30} {values[env_idx].item():>12.6f}")

    # Footer with V and vdot
    print(f"{'-'*94}")
    print(f"V: {v[env_idx].item():.6f}    Vdot: {vdot[env_idx].item():.6f}")
    print(f"{'='*94}\n")