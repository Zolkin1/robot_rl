from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_apply, quat_inv

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv

def reset_on_reference_dual(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        primary_command_name: str,
        v2_command_name: str,
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
    Reset two trajectory commands (one OLD-style time-based, one V2 phase-based)
    to the same sampled state, so per-step traces are directly comparable.

    Identical pose/joint/velocity setup to ``reset_on_reference``. Uses cmd1's
    OLD-style scalar ``get_total_time()`` for random sampling so cmd1 behaves
    bit-equivalently to the standalone reset; cmd2 derives its starting phase
    from the same sampled times via ``(t / total_per_env_traj) % 1.0``.

    Args:
        primary_command_name: OLD-style cmd (writes ``init_time_offset``).
        v2_command_name: V2 phase-based cmd (writes ``manager.phase`` via ``set_phase``).
        Other args: identical to :func:`reset_on_reference`.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd1 = env.command_manager.get_term(primary_command_name)
    cmd2 = env.command_manager.get_term(v2_command_name)

    env.command_manager.get_term(conditioner_command_name)._resample(env_ids)
    env.command_manager.get_term(conditioner_command_name)._update_command()
    num_env = len(env_ids)
    if num_env == 0:
        return

    # Ref / non-ref split (single random draw)
    r = torch.empty(num_env, device=env.device)
    ref_env = r.uniform_(0.0, 1.0) <= rel_envs_on_ref
    ref_ids = env_ids[ref_env]
    num_ref_envs = len(ref_ids)

    # Special-value override on conditioner for a fraction of ref envs
    r_on_ref = torch.empty(num_ref_envs, device=env.device)
    special_envs = r_on_ref.uniform_(0.0, 1.0) < rel_envs_on_special
    command = env.command_manager.get_term(conditioner_command_name).command
    command_clone = command.clone()
    command[ref_ids[special_envs], 0] = special_val * torch.ones(
        len(ref_ids[special_envs]), device=env.device
    )

    nonref_ids = env_ids[~ref_env]
    num_nonref_envs = len(nonref_ids)

    # Validate frame against cmd1's outputs (cmd2 has the same trajectory data).
    pos_indices = _find_output_indices(cmd1.ordered_pos_output_names, base_frame_name, "pos_")
    ori_indices = _find_output_indices(cmd1.ordered_pos_output_names, base_frame_name, "ori_")
    if len(pos_indices) != 3:
        raise ValueError(
            f"Base frame '{base_frame_name}' must have pos_x, pos_y, pos_z. Found {len(pos_indices)}."
        )
    if len(ori_indices) != 4:
        raise ValueError(
            f"Base frame '{base_frame_name}' must have ori_x, ori_y, ori_z, ori_w. Found {len(ori_indices)}."
        )

    traj_joint_names = set()
    for name in cmd1.ordered_pos_output_names:
        if name.startswith("joint:"):
            traj_joint_names.add(name.split(":", 1)[1])
    missing_joints = [jn for jn in asset.joint_names if jn not in traj_joint_names]
    if missing_joints:
        raise ValueError(f"Trajectory missing joints: {missing_joints}")

        # ---- Single random sample, OLD-style scalar-total convention ----
    total_time_scalar = cmd1.manager.get_total_time()
    random_times = torch.rand(num_ref_envs, device=env.device) * total_time_scalar

    # Trajectory eval for the robot pose (cmd1 is fine — both cmds share
    # the same trajectory data, eval result would be identical).
    cmd1.get_desired_outputs(random_times, env_ids=ref_ids)
    des_outputs = cmd1.y_des
    y_sampled = des_outputs[ref_ids]

    base_pos_rel = y_sampled[:, pos_indices]
    base_ori_quat_w = y_sampled[:, ori_indices]
    base_pos_rel[:, 2] += base_z_offset
    base_pos_w = base_pos_rel + env.scene.env_origins[ref_ids]
    base_pose = torch.cat([base_pos_w, base_ori_quat_w], dim=-1)

    des_doutputs = cmd1.dy_des
    dy_sampled = des_doutputs[ref_ids]
    lin_vel_indices = _find_output_indices(cmd1.ordered_vel_output_names, base_frame_name, "pos_")
    ang_vel_indices = _find_output_indices(cmd1.ordered_vel_output_names, base_frame_name, "ori_")
    base_vel = torch.cat(
        [dy_sampled[:, lin_vel_indices], dy_sampled[:, ang_vel_indices]], dim=-1
    )

    # ---- Sync ref_poses on BOTH cmds (env-origin frame, identity quat) ----
    for cmd_ in (cmd1, cmd2):
        cmd_.ref_poses[ref_ids, :3] = env.scene.env_origins[ref_ids]
        cmd_.ref_poses[ref_ids, 3:6] = 0.0
        cmd_.ref_poses[ref_ids, 6] = 1.0

        # Joint pos/vel from cmd1's eval (same trajectory, would match cmd2's)
    joint_pos = torch.zeros(num_ref_envs, len(asset.joint_names), device=env.device)
    joint_vel = torch.zeros_like(joint_pos)
    for i, joint_name in enumerate(asset.joint_names):
        traj_output_name = f"joint:{joint_name}"
        pos_traj_idx = cmd1.ordered_pos_output_names.index(traj_output_name)
        vel_traj_idx = cmd1.ordered_vel_output_names.index(traj_output_name)
        joint_pos[:, i] = y_sampled[:, pos_traj_idx]
        joint_vel[:, i] = dy_sampled[:, vel_traj_idx]

        # ---- Anchor each cmd's clock with the same sampled phase ----
    # OLD-style: time-based offset.
    cmd1.init_time_offset[ref_ids] = random_times

    # V2: convert to per-env phase mod 1.0 (matches OLD's eval-time wrap).
    cur_traj_v2 = cmd2.manager.get_current_trajectory_indices()[ref_ids]
    cmd2_totals = cmd2.manager.data["total_time"][cur_traj_v2]
    cmd2_phase = (random_times / cmd2_totals) % 1.0
    cmd2.manager.set_phase(cmd2_phase, ref_ids)
    # TODO: Can also do a check to make sure the desired trajectories are the same.

    # Write states to sim
    asset.write_root_pose_to_sim_index(root_pose=base_pose, env_ids=ref_ids)
    asset.write_root_link_velocity_to_sim_index(root_velocity=base_vel, env_ids=ref_ids)
    asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=ref_ids)
    asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=ref_ids)

    # Restore conditioner command
    env.command_manager.get_term(conditioner_command_name).command[:] = command_clone

    # ---- Non-ref envs: default pose, anchor both cmds at phase=0 ----
    if num_nonref_envs != 0:
        root_states = wp.to_torch(asset.data.default_root_state)[nonref_ids].clone()
        base_pose = root_states[:, 0:7]
        base_pose[:, :3] += env.scene.env_origins[nonref_ids]
        base_vel = root_states[:, 7:]

        joint_pos = wp.to_torch(asset.data.default_joint_pos)[
            nonref_ids, asset_cfg.joint_ids
        ].clone()
        joint_vel = wp.to_torch(asset.data.default_joint_vel)[
            nonref_ids, asset_cfg.joint_ids
        ].clone()
        r = torch.empty(num_nonref_envs, len(asset.joint_names), device=env.device)
        r.uniform_(joint_add_range[0], joint_add_range[1])
        joint_pos += r

        asset.write_root_pose_to_sim_index(root_pose=base_pose, env_ids=nonref_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=base_vel, env_ids=nonref_ids)
        asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=nonref_ids)
        asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=nonref_ids)

        # OLD-style: zero the offset.
        cmd1.init_time_offset[nonref_ids] = 0.0
        # V2: phase 0 on the NEW manager.
        cmd2.manager.set_phase(
            torch.zeros(num_nonref_envs, device=env.device), nonref_ids
        )

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