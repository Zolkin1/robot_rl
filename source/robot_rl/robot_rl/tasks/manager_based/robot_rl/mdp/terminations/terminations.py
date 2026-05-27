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


def base_orientation_from_upright(
    env,
    roll_limit_deg: float = 70.0,
    pitch_limit_deg: float = 70.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base roll or pitch deviates from upright (world z) by more than the limits.

    Unlike :func:`base_orientation`, the reference here is the world-frame
    upright (identity rotation) — yaw is ignored.  Useful as a face-plant /
    fallen-robot detector that is independent of the trajectory reference,
    and therefore not subject to whatever grace gating the deviation-from-
    reference termination uses.

    Args:
        env: IsaacLab environment.
        roll_limit_deg: Maximum allowed |roll| from upright in degrees.
        pitch_limit_deg: Maximum allowed |pitch| from upright in degrees.
        asset_cfg: Robot scene entity.

    Returns:
        Boolean tensor of shape ``[num_envs]`` — ``True`` for envs that
        should terminate this step.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    act_quat = wp.to_torch(asset.data.root_quat_w)  # (x, y, z, w)
    roll, pitch, _ = euler_xyz_from_quat(act_quat, wrap_to_2pi=False)
    roll = wrap_to_pi(roll)
    pitch = wrap_to_pi(pitch)

    roll_limit = torch.deg2rad(torch.tensor(roll_limit_deg, device=act_quat.device))
    pitch_limit = torch.deg2rad(torch.tensor(pitch_limit_deg, device=act_quat.device))
    return (roll.abs() > roll_limit) | (pitch.abs() > pitch_limit)


def frame_deviation_from_reference(
    env,
    cmd_name: str,
    frame_names: list[str],
    max_frac: float,
    min_dist: float = 0.0,
    grace_period_s: float = 0.0,
    debug: bool = False,
) -> torch.Tensor:
    """Terminate when any listed frame deviates from its reference by more
    than ``max(max_frac * chord, min_dist)``.

    For each frame the deviation is the (ref-frame-local) position-error
    norm ``||y_act_xyz - y_des_xyz||``.  The reference scale is the
    **current domain's** Bezier chord length per frame
    (``||last_cp - first_cp||``) — the same scalar the debug-viz sphere
    uses.  Because stance frames have chord ≈ 0 in their stance domain,
    ``min_dist`` (metres) sets an absolute floor on the threshold so
    those frames don't terminate trivially.  The effective threshold per
    env per frame is ``max(max_frac * chord, min_dist)``.

    Args:
        env: The IsaacLab environment.
        cmd_name: Trajectory command term exposing ``y_act``, ``y_des``,
            ``ordered_pos_output_names``, and a multiskill ``manager``.
        frame_names: Frames to monitor.  Each must have
            ``{frame_name}:pos_{x,y,z}`` in the command's
            ``ordered_pos_output_names``.
        max_frac: Allowed fractional deviation (e.g. ``0.5`` for 50%).
        min_dist: Absolute minimum threshold in metres.  Default ``0.0``
            disables the floor — set this above the noise floor / typical
            stance-foot drift to avoid spurious termination.
        grace_period_s: Seconds after a trajectory swap (within-skill bucket
            swap or full skill change) and after a fresh episode reset
            during which this termination is suppressed.  Reads
            ``cmd.time_since_traj_change_s`` (populated by
            ``BatchedMultiSkillCommand._pre_update_phase``).  Default 0
            disables the grace.  Silently no-ops if the command term does
            not expose ``time_since_traj_change_s`` (e.g. single-skill
            commands).
        debug: If True, print per-frame ``(err_x, err_y, err_z, ||err||,
            threshold)`` for every env that terminates this step.  Off by
            default — enable from the cfg via ``params={"debug": True}``.

    Returns:
        Boolean tensor of shape ``[num_envs]`` — ``True`` for envs that
        should terminate this step.

    Raises:
        ValueError: If any name in ``frame_names`` lacks a complete
            ``pos_{x,y,z}`` triplet in the command's outputs.
    """
    cmd = env.command_manager.get_term(cmd_name)
    output_names = cmd.ordered_pos_output_names

    pos_indices: list[list[int]] = []
    missing: list[str] = []
    for name in frame_names:
        try:
            xyz = [output_names.index(f"{name}:pos_{a}") for a in ("x", "y", "z")]
        except ValueError:
            missing.append(name)
            continue
        pos_indices.append(xyz)
    if missing:
        available = sorted({
            n.split(":", 1)[0] for n in output_names
            if ":pos_" in n and not n.startswith("joint:")
        })
        raise ValueError(
            f"Frames {missing} do not have a full ``pos_{{x,y,z}}`` triplet "
            f"in '{cmd_name}'.ordered_pos_output_names. "
            f"Frames with a pos triplet available: {available}."
        )

    device = cmd.y_des.device
    pos_idx_t = torch.tensor(pos_indices, dtype=torch.long, device=device)    # (F, 3)

    # Position error per frame (ref-frame-local coords).
    err = (cmd.y_act - cmd.y_des)[:, pos_idx_t]                                # (N, F, 3)
    err_norm = torch.linalg.norm(err, dim=-1)                                  # (N, F)

    # Current-target chord per frame.  Prefer the cmd's accessor when it
    # exists — for ``BatchedMultiSkillCommand`` that returns an
    # alpha-blended chord during a cross-fade transition, keeping the
    # scale consistent with the (likewise blended) ``y_des`` the numerator
    # is measured against.  Fallback (single-skill commands without the
    # accessor) computes the raw current-domain chord on the new traj.
    if hasattr(cmd, "current_chord_per_frame"):
        chord_len = cmd.current_chord_per_frame(pos_idx_t)                     # (N, F)
    else:
        manager = cmd.manager
        traj_idx = manager.get_current_trajectory_indices()
        domain_idx = manager._get_domain_indices(manager.phase, traj_idx)
        coeffs_pos = manager.data["coeffs_pos"][traj_idx, domain_idx]
        frame_coeffs = coeffs_pos[:, pos_idx_t, :]
        chord_len = torch.linalg.norm(
            frame_coeffs[..., -1] - frame_coeffs[..., 0], dim=-1
        )                                                                       # (N, F)

    threshold = torch.clamp(max_frac * chord_len, min=min_dist)                # (N, F)
    terminate = (err_norm > threshold).any(dim=-1)                             # (N,)

    if grace_period_s > 0.0:
        time_since = getattr(cmd, "time_since_traj_change_s", None)
        if time_since is not None:
            terminate = terminate & (time_since >= grace_period_s)

    if debug and terminate.any():
        bad_envs = terminate.nonzero(as_tuple=False).flatten().tolist()
        for env_id in bad_envs:
            parts = []
            for f, name in enumerate(frame_names):
                ex, ey, ez = err[env_id, f].tolist()
                n = err_norm[env_id, f].item()
                t = threshold[env_id, f].item()
                marker = "!!" if n > t else "  "
                parts.append(
                    f"{marker} {name}: err=({ex:+.3f}, {ey:+.3f}, {ez:+.3f}) "
                    f"||err||={n:.3f} thresh={t:.3f}"
                )
            print(f"[frame_dev term] env {env_id}: " + " | ".join(parts))

    return terminate


def illegal_terrain_contact(
    env, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate when the peak contact force on any of the sensor's bodies
    exceeds ``threshold``.

    Mirrors :func:`isaaclab.envs.mdp.terminations.illegal_contact` and reads
    unfiltered :attr:`ContactSensor.data.net_forces_w_history` — the total
    force from all contact partners, self-collision included.

    Note: filtering this to terrain-only contact is not viable on the PhysX
    GPU pipeline — it cannot filter against the static terrain mesh (logs
    ``GPU contact filter for collider '/World/ground' is not supported`` and
    zeroes the force matrix), and filtering against the robot's own bodies to
    subtract self-contact does not capture intra-articulation contacts
    reliably.  Use an orientation-based fall detector
    (:func:`base_orientation_from_upright`) instead of trying to isolate
    terrain contact by force.  This function remains for non-terrain hard
    contacts where ``threshold`` can simply be set above self-collision noise.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = wp.to_torch(contact_sensor.data.net_forces_w_history)
    body_slice = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, body_slice], dim=-1), dim=1)[0] > threshold,
        dim=1,
    )