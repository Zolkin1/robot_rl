# Copyright 2026 Zachary Olkin. All rights reserved.

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
