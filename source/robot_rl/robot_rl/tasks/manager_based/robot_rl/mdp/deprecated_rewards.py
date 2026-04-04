raise ImportError("[DEPRECATED REWARDS] This module is deprecated. Use a not-deprecated reward instead.")

# def lip_gait_tracking(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, period: float, std: float,
#                       nom_height: float, Tswing: float, command_name: str, wdes: float,
#                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ) -> torch.Tensor:
#     """Reward feet in contact with the ground in the correct phase."""
#     # If the feet are in contact at the right time then positive reward, else 0 reward
#
#     # Get the robot asset
#     robot = env.scene[asset_cfg.name]
#
#     # Contact sensor
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#
#     # Get the current contacts
#     # in_contact = ~contact_sensor.compute_first_air()[:, sensor_cfg.body_ids]  # Checks if the foot recently broke contact - which tells us we are not in contact. Does not reward jitter but use the dt.
#     in_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
#
#     in_contact = in_contact.float()
#
#     # Contact schedule function
#     tp = (env.sim.current_time % period) / period     # Scaled between 0-1
#     phi_c = torch.tensor(math.sin(2*torch.pi*tp)/math.sqrt(math.sin(2*torch.pi*tp)**2 + Tswing), device=in_contact.device)
#
#     # Compute reward
#     reward = (in_contact[:, 0] - in_contact[:, 1])*phi_c # TODO: Does it help to remove the schedule here? - seemed to get some instability
#
#     # Add in the foot tracking
#     foot_pos = robot.data.body_pos_w[:, asset_cfg.body_ids, :2]
#     swing_foot_pos = foot_pos[:, int(0.5 + 0.5*torch.sign(phi_c))]
#     # swing_foot_pos = foot_pos[:, ((env.cfg.control_count + 1) % 2), :]
#
#     # print(f"swing foot index: {((env.cfg.control_count + 1) % 2)}, in contact 0: {in_contact[:, 0]}")
#     # print(f"foot index: {int(0.5 + 0.5*torch.sign(phi_c))}")
#     # print(f"stance foot pos: {stance_foot_pos}, des pos: {env.cfg.current_des_step[:, :2]}")
#
#     # TODO: Debug and put back!
#     # reward = reward * torch.exp(-torch.norm(env.cfg.current_des_step[:, :2] - swing_foot_pos, dim=1) / std)
#
#     return reward

# def lip_feet_tracking(env: ManagerBasedRLEnv, period: float, std: float,
#                       Tswing: float,
#                       feet_bodies: SceneEntityCfg,
#                       asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ) -> torch.Tensor:
#     """Reward the lip foot step tracking."""
#     # Get the robot asset
#     robot = env.scene[asset_cfg.name]
#
#     # Contact schedule function
#     tp = (env.sim.current_time % period) / period     # Scaled between 0-1
#     phi_c = torch.tensor(math.sin(2*torch.pi*tp)/math.sqrt(math.sin(2*torch.pi*tp)**2 + Tswing), device=env.device)
#
#     # Foot tracking
#     foot_pos = robot.data.body_pos_w[:, feet_bodies.body_ids, :2]
#     swing_foot_pos = foot_pos[:, int(0.5 + 0.5*torch.sign(phi_c))]
#     reward = torch.exp(-torch.norm(env.cfg.current_des_step[:, :2] - swing_foot_pos, dim=1) / std)
#
#     # print(f"swing_foot_norm: {torch.norm(swing_foot_pos, dim=1)}")
#     # print(f"distance: {torch.norm(env.cfg.current_des_step[:, :2] - swing_foot_pos, dim=1)}")
#     # print(f"reward: {reward}")
#
#     # Update the com linear velocity running average
#     alpha = 0.25
#     env.cfg.com_lin_vel_avg = (1-alpha)*env.cfg.com_lin_vel_avg + alpha*robot.data.root_com_lin_vel_w
#
#     return reward

# def compute_step_location_local(env: ManagerBasedRLEnv, env_ids: torch.Tensor,
#                           nom_height: float, Tswing: float, command_name: str, wdes: float,
#                           feet_bodies: SceneEntityCfg,
#                           sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
#                           asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#                           visualize: bool = True) -> torch.Tensor:
#     asset = env.scene[asset_cfg.name]
#     feet = env.scene[feet_bodies.name]
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#
#     # Commanded velocity in the local frame
#     command = env.command_manager.get_command(command_name)
#
#     # COM Position in global frame
#     # r = asset.data.root_com_pos_w
#     r = asset.data.root_pos_w
#
#     # COM velocity in local frame
#     rdot = command
#     # rdot = asset.data.root_com_lin_vel_b
#
#     g = 9.81
#     omega = math.sqrt(g / nom_height)
#
#     # Instantaneous capture point as a 3-vector
#     icp_0 = torch.zeros((r.shape[0], 3), device=env.device)    # For setting the height
#     icp_0[:, :2] = rdot[:, :2]/omega
#
#
#     # Get the stance foot position
#     foot_pos = feet.data.body_pos_w[:, feet_bodies.body_ids]
#     # Contact schedule function
#     tp = (env.sim.current_time % (2*Tswing)) / (2*Tswing)     # Scaled between 0-1
#     phi_c = torch.tensor(math.sin(2*torch.pi*tp)/math.sqrt(math.sin(2*torch.pi*tp)**2 + Tswing), device=env.device)
#
#     # Stance foot in global frame
#     stance_foot_pos = foot_pos[:, int(0.5 - 0.5*torch.sign(phi_c)), :]
#     stance_foot_pos[:, 2] *= 0
#
#     def _transfer_to_global_frame(vec, root_quat):
#         return quat_rotate(yaw_quat(root_quat), vec)
#
#     def _transfer_to_local_frame(vec, root_quat):
#         return quat_rotate(yaw_quat(quat_inv(root_quat)), vec)
#
#     # Compute final ICP as a 3 vector
#     icp_f = (math.exp(omega * Tswing)*icp_0 + (1 - math.exp(omega * Tswing))
#              * _transfer_to_local_frame(r - stance_foot_pos, asset.data.root_quat_w))
#     icp_f[:, 2] *= 0
#
#
#     # Compute ICP offsets
#     sd = torch.abs(command[:, 0]) * Tswing #TODO: Note this only works if there are no commanded local y velocities
#     wd = wdes * torch.ones(r.shape[0], device=env.device)
#
#     bx = sd / (math.exp(omega * Tswing) - 1)
#     by = torch.sign(phi_c) * wd / (math.exp(omega * Tswing) + 1)
#     b = torch.stack((bx, by, torch.zeros(r.shape[0], device=env.device)), dim=1)
#
#     # Clip the step to be within the kinematic limits
#     p_local = icp_f.clone()
#     p_local[:, 0] = torch.clip(icp_f[:, 0] - b[:, 0], -0.5, 0.5)    # Clip in the local x direction
#     p_local[:, 1] = torch.clip(icp_f[:, 1] - b[:, 1], -0.3, 0.3)    # Clip in the local y direction
#
#
#     # Compute desired step in the global frame
#     p = _transfer_to_global_frame(p_local, asset.data.root_quat_w) + r
#
#     p[:, 2] *= 0
#
#     # print(f"icp_f = {icp_f},\n"
#     #       f"icp_0 = {icp_0},\n"
#     #       f"b = {b},\n")
#
#     if visualize:
#         sw_st_feet = torch.cat((p, foot_pos[:, int(0.5 - 0.5 * torch.sign(phi_c)), :]), dim=0)
#         env.footprint_visualizer.visualize(
#             # TODO: Visualize both the current stance foot and the desired foot
#             # translations=foot_pos[:, int(0.5 - 0.5*torch.sign(phi_c)), :], #p,
#             # translations=foot_pos[:, (env.cfg.control_count % 2), :],
#             translations=sw_st_feet,
#             orientations=yaw_quat(asset.data.root_quat_w).repeat_interleave(2, dim=0),
#             # repeat 0,1 for num_env
#             # marker_indices=torch.tensor([0,1], device=env.device).repeat(env.num_envs),
#         )
#
#     env.cfg.current_des_step[env_ids, :] = p[env_ids,:]  # This only works if I compute the new location once per step/on a timer
#
#     return p


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

    # Get the velocities
    v = cmd.current_contact_vels

    # # linear velocity [B,3] and yaw rate [B,1]
    # v = cmd.stance_foot_vel  # [vx, vy, vz]
    # wz = cmd.stance_foot_ang_vel[:, 2].unsqueeze(-1)  # [ω_z]
    #
    # # stack into [B,4] error vector
    # e_vel = torch.cat([v, wz], dim=-1)
    #
    # not_flight_mask = cmd.get_not_flight_envs()
    # return not_flight_mask * torch.exp(- (e_vel**2).sum(dim=-1) / sigma_vel**2)

    return torch.exp(-(v.sum(dim=-1).sum(dim=-1)**2) / sigma_vel**2)

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

    # TODO: Re-write to handle arbitrary contacts

    # Get the current pose
    des_contact_poses = cmd.desired_contact_poses
    contact_poses = cmd.current_contact_poses

    # Compute error
    pose_err = contact_poses - des_contact_poses

    # Wrap yaw error
    pose_err = wrap_to_pi(pose_err[:, -1])

    # # planar position error [B,2]
    # p0_xy = cmd.stance_foot_pos_0[:, :2]
    # p_xy = cmd.stance_foot_pos[:, :2]
    # delta_xy = p_xy - p0_xy
    #
    # # vertical error to the floor plane [B,1]
    # z_cur = cmd.stance_foot_pos[:, 2].unsqueeze(-1)
    # delta_z = z_cur - cmd.stance_foot_pos_0[:, 2].unsqueeze(-1)
    #
    # # roll error [B,1]
    # roll = cmd.stance_foot_ori[:, 0].unsqueeze(-1)
    #
    # # yaw error wrapped to [–π, π] [B,1]
    # psi0 = cmd.stance_foot_ori_0[:, 2]
    # psi = cmd.stance_foot_ori[:, 2]
    # delta_psi = ((psi - psi0 + torch.pi) % (2 * torch.pi) - torch.pi).unsqueeze(-1)
    #
    # # stack into [B,5] error vector
    # e_pose = torch.cat([delta_xy, delta_z, roll, delta_psi], dim=-1)
    #
    # not_flight_mask = cmd.get_not_flight_envs()
    # return not_flight_mask * torch.exp(- (e_pose ** 2).sum(dim=-1) / sigma_pose ** 2)

    return torch.exp(-(pose_err**2).sum(dim=-1) / sigma_pose ** 2)


def foot_clearance(env: ManagerBasedRLEnv,
                   target_height: float,
                   sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor"),
                   height_sensor_cfg: SceneEntityCfg | None = None,
                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """Reward foot clearance."""
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get contact state
    contacts = wp.to_torch(contact_sensor.data.net_forces_w_history)[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    if height_sensor_cfg is not None:
        sensor: RayCaster = env.scene[height_sensor_cfg.name]
        adjusted_target_height = target_height + torch.mean(wp.to_torch(sensor.data.ray_hits_w)[...,2],dim=1).unsqueeze(-1)
    else:
        adjusted_target_height = target_height

    # Calculate foot heights
    feet_z_err = wp.to_torch(asset.data.body_pos_w)[:, asset_cfg.body_ids, 2] - adjusted_target_height
    pos_error = torch.square(feet_z_err) * ~contacts

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
            contact = wp.to_torch(contact_sensor.data.net_forces_w_history)[:, :, sensor_cfg.body_ids[i], :].norm(dim=-1).max(dim=1)[0] > 1.0
            res += ~(contact ^ is_stance)
    else:
        for i in range(2):
            is_stance = stance_i == i
            # set is_stance to be true if the command is small
            is_stance = is_stance
            contact = wp.to_torch(contact_sensor.data.net_forces_w_history)[:, :, sensor_cfg.body_ids[i], :].norm(dim=-1).max(dim=1)[0] > 1.0
            res += ~(contact ^ is_stance)
    return res

# TODO: Test
def contact_schedule_penalty(env: ManagerBasedRLEnv, command_name: str,
                           sensor_cfg: SceneEntityCfg, weight_scalar: float) -> torch.Tensor:
    """Penalize contacts while in the flight phase."""
    cmd = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Time into the episode
    t = env.episode_length_buf * env.step_dt

    # Get bodies not in contact for each env
    contact_states = cmd.get_contact_state(t)
    contact_body_names = cmd.contact_bodies

    contact_forces = torch.zeros(t.shape[0], dtype=torch.float, device=env.device)
    for i, body_name in enumerate(contact_body_names):
        contact_mask = contact_states[:, i] == 1
        indices = torch.tensor([i for i, v in enumerate(sensor_cfg.body_names) if v == body_name])
        body_id = sensor_cfg.body_ids[indices]
        contact_forces[contact_mask] += wp.to_torch(contact_sensor.data.net_forces_w)[contact_mask, body_id, :].norm(dim=-1)  # Gets the most recent force only

    penalty = weight_scalar * torch.tanh(contact_forces / 0.5)  # TODO: Think about if this is what I want
    return penalty


def ankle_roll_zero(
        env: ManagerBasedRLEnv, std: float = 0.1, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward keeping both ankle roll joints near zero position using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Get ankle roll joint indices - these are typically the last joints in each leg
    # Based on the controller.py joint order:
    # Index 19: left_ankle_roll_joint
    # Index 20: right_ankle_roll_joint
    ankle_roll_indices = [19, 20]  # left and right ankle roll joints

    # Get current ankle roll joint positions
    ankle_roll_positions = wp.to_torch(asset.data.joint_pos)[:, ankle_roll_indices]  # [B, 2]

    # Compute squared error from zero position
    ankle_roll_error = torch.square(ankle_roll_positions)  # [B, 2]

    # Sum errors for both ankle roll joints and apply exponential kernel
    total_error = ankle_roll_error.sum(dim=-1)  # [B]
    reward = torch.exp(-total_error / std ** 2)

    return reward

def track_lin_vel_y_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error =  torch.square(env.command_manager.get_command(command_name)[:, 1] - wp.to_torch(asset.data.root_lin_vel_b)[:, 1])
    return torch.exp(-lin_vel_error / std**2)


def contact_no_vel(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward feet contact with zero velocity."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = wp.to_torch(contact_sensor.data.net_forces_w_history)[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = wp.to_torch(asset.data.body_lin_vel_w)[:, asset_cfg.body_ids] * contacts.unsqueeze(-1)
    # shape [B, num_feet, 3]
    penalize = torch.square(body_vel[:,:,:3])
    return torch.sum(penalize, dim=(1,2))


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

def v_dot_penalty(env: ManagerBasedRLEnv, command_name: str,eta_max: float = 0.15,
    eta_dot_max: float = 0.5,eps: float = 1e-6) -> torch.Tensor:
    ref_term = env.command_manager.get_term(command_name)                    # [B]
    vdot = ref_term.vdot # [B]

    norm_P = ref_term.clf.norm_P

    max_violation = (
        2.0 * norm_P * eta_max * eta_dot_max + eps
    )

    vdot_penalty = torch.tanh(torch.clamp(vdot, min=0.0) / max_violation)
    return vdot_penalty