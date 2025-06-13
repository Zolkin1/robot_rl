import torch
import math
from isaaclab.utils import configclass
import numpy as np

from isaaclab.managers import CommandTermCfg,CommandTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_from_euler_xyz,quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv, quat_apply

from .ref_gen import bezier_deg, calculate_cur_swing_foot_pos, HLIP
from .clf import CLF
# from isaaclab.utils.transforms import combine_frame_transforms, quat_from_euler_xyz

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cmd_cfg import HLIPCommandCfg


def euler_rates_to_omega(eul: torch.Tensor,
                         eul_rates: torch.Tensor) -> torch.Tensor:
    """
    Convert Z–Y–X Euler‐angle rates into body‐frame angular velocity.
    
    Args:
        eul:        Tensor of shape (..., 3), Euler angles [φ, θ, ψ]
        eul_rates:  Tensor of shape (..., 3), Euler‐angle rates [φ̇, θ̇, ψ̇]
    Returns:
        omega:      Tensor of shape (..., 3), angular velocity [ωₓ, ωᵧ, ω_z]
    """
    # unpack
    phi, theta, psi = eul.unbind(-1)
    
    # precompute sines/cosines
    c_th = torch.cos(theta)
    s_th = torch.sin(theta)
    c_ps = torch.cos(psi)
    s_ps = torch.sin(psi)
    
    # build the mapping matrix M(...,3,3)
    zeros = torch.zeros_like(theta)
    ones  = torch.ones_like(theta)
    
    M = torch.stack([
        torch.stack([ c_th*c_ps,  s_ps, zeros ], dim=-1),
        torch.stack([-c_th*s_ps,  c_ps, zeros ], dim=-1),
        torch.stack([      s_th,   zeros, ones ], dim=-1),
    ], dim=-2)
    
    # apply to rates: ω = M @ eul_rates
    omega = torch.einsum('...ij,...j->...i', M, eul_rates)
    return omega



def _transfer_to_global_frame(vec, root_quat):
    return quat_rotate(yaw_quat(root_quat), vec)

def _transfer_to_local_frame(vec, root_quat):
    return quat_rotate(yaw_quat(quat_inv(root_quat)), vec)  

class HLIPCommandTerm(CommandTerm):
    def __init__(self, cfg: "HLIPCommandCfg", env):
        super().__init__(cfg, env)
        self.T_ds = cfg.T_ds
        self.z0 = cfg.z0
        self.y_nom = cfg.y_nom
        self.T = cfg.gait_period/2
        self.debug_vis = cfg.debug_vis
        if self.debug_vis:
            self.footprint_visualizer = VisualizationMarkers(cfg.footprint_cfg)
       
        self.env = env
        self.robot = env.scene[cfg.asset_name]
        self.feet_bodies_idx = self.robot.find_bodies(cfg.foot_body_name)[0]
        self.upper_body_joint_idx = self.robot.find_joints(cfg.upper_body_joint_name)[0]

        self.foot_target = torch.zeros((self.num_envs, 2), device=self.device)

        self.metrics = {}
     
        n_output = 12 + len(self.upper_body_joint_idx)
        self.y_out = torch.zeros((self.num_envs, n_output), device=self.device)
        self.dy_out = torch.zeros((self.num_envs, n_output), device=self.device)
        self.y_act = torch.zeros((self.num_envs, n_output), device=self.device)
        self.dy_act = torch.zeros((self.num_envs, n_output), device=self.device)


        self.com_z = torch.ones((self.num_envs), device=self.device)*self.z0

        grav = torch.abs(torch.tensor(self.env.cfg.sim.gravity[2], device=self.device))
        self.hlip_controller = HLIP(grav, self.z0, self.T_ds, self.T, self.y_nom)

        self.mass = sum(self.robot.data.default_mass.T)[0]


        self.clf = CLF(n_output, self.env.cfg.sim.dt*self.env.cfg.sim.render_interval,
            batch_size=self.num_envs,
            Q_weights=np.array(cfg.Q_weights),
            R_weights=np.array(cfg.R_weights),
            device=self.device
        )
        
        self.v = torch.zeros((self.num_envs), device=self.device)
        self.vdot = torch.zeros((self.num_envs), device=self.device)
        self.v_buffer = torch.zeros((self.num_envs, 100), device=self.device)
        self.vdot_buffer = torch.zeros((self.num_envs, 100), device=self.device)
        self.stance_idx = None


    @property
    def command(self):
        return self.foot_target
    

    def _resample_command(self, env_ids):
        self._update_command()
        # Do nothing here
        # device = self.env.command_manager.get_command("base_velocity").device
        
        return
    
    def _update_metrics(self):
        # Foot tracking
        # foot_pos = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :2]  # Only take x,y coordinates
        # # Contact schedule function
        # tp = (self.env.sim.current_time % (2 * self.T)) / (2 * self.T)  # Scaled between 0-1
        # phi_c = torch.tensor(math.sin(2 * torch.pi * tp) / math.sqrt(math.sin(2 * torch.pi * tp)**2 + self.T), device=self.env.device)

        # swing_foot_pos = foot_pos[:, int(0.5 + 0.5 * torch.sign(phi_c))]
        # Only compare x,y coordinates of foot target
        self.metrics["error_sw_z"] = torch.abs(self.y_out[:,8] - self.y_act[:,8])
        self.metrics["error_sw_x"] = torch.abs(self.y_out[:,6] - self.y_act[:,6])
        self.metrics["error_sw_y"] = torch.abs(self.y_out[:,7] - self.y_act[:,7])
        self.metrics["error_sw_roll"] = torch.abs(self.y_out[:,9] - self.y_act[:,9])
        self.metrics["error_sw_pitch"] = torch.abs(self.y_out[:,10] - self.y_act[:,10])
        self.metrics["error_sw_yaw"] = torch.abs(self.y_out[:,11] - self.y_act[:,11])
        

        self.metrics["error_com_x"] = torch.abs(self.y_out[:,0] - self.y_act[:,0])
        self.metrics["error_com_y"] = torch.abs(self.y_out[:,1] - self.y_act[:,1])
        self.metrics["error_com_z"] = torch.abs(self.y_out[:,2] - self.y_act[:,2])
        self.metrics["error_pelvis_roll"] = torch.abs(self.y_out[:,3] - self.y_act[:,3])
        self.metrics["error_pelvis_pitch"] = torch.abs(self.y_out[:,4] - self.y_act[:,4])
        self.metrics["error_pelvis_yaw"] = torch.abs(self.y_out[:,5] - self.y_act[:,5])


        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot
        self.metrics["avg_clf"] =torch.mean(self.v_buffer, dim=-1)
        max_clf = self.env.reward_manager.get_term_cfg("clf_reward").params["max_clf"]
        self.metrics["max_clf"] = torch.ones((self.num_envs), device=self.device) * max_clf
        # return self.foot_target  # Return the foot target tensor for observation


    def update_Stance_Swing_idx(self):
        Tswing = self.T - self.T_ds
        self.tp = (self.env.sim.current_time % (2*Tswing)) / (2*Tswing)  
        phi_c = torch.tensor(math.sin(2 * torch.pi * self.tp) / math.sqrt(math.sin(2 * torch.pi * self.tp)**2 + self.T), device=self.env.device)



        new_stance_idx = int(0.5 - 0.5*torch.sign(phi_c))
        self.swing_idx = 1 - new_stance_idx
        
        if self.stance_idx is None or new_stance_idx != self.stance_idx:
            if self.stance_idx is None:
                self.stance_idx = new_stance_idx
            #update stance foot pos, ori
            foot_pos_w = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]
            foot_ori_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]
            self.stance_foot_pos_0 = foot_pos_w[:, new_stance_idx, :]
            self.stance_foot_ori_quat_0 = foot_ori_w[:,new_stance_idx,:]
            self.stance_foot_ori_0 = self.get_euler_from_quat(foot_ori_w[:,new_stance_idx,:])
            self.swing2stance_foot_pos_0 = _transfer_to_local_frame(
                foot_pos_w[:, self.swing_idx, :]-self.stance_foot_pos_0, self.stance_foot_ori_quat_0
            )
       
        self.stance_idx = new_stance_idx


        if self.tp < 0.5:
            self.phase_var = 2*self.tp
        else:
            self.phase_var = 2*self.tp-1
        self.cur_swing_time = self.phase_var*Tswing



    def generate_orientation_ref(self, base_velocity,N):
        pelvis_euler = torch.zeros((N,3), device=self.device)
        tp_tensor = torch.tensor(self.tp, device=self.device)
        phase_tensor = torch.tensor(self.phase_var, device=self.device)
        
        roll_main_amp = 0.0  # main double bump amplitude
        roll_asym_amp = -0.05  # adds asymmetry

        
        pelvis_euler[:, 0] = (
            roll_main_amp * torch.sin(4 * torch.pi * tp_tensor) +
            roll_asym_amp * torch.sin(2 * torch.pi * tp_tensor)
        )

        #add bias based on lateral velocity
        # lateral bias
        bias_lat = torch.clamp(torch.atan(base_velocity[:,1] / 9.81),-0.15,0.15)

        # turning bias
        bias_yaw = torch.clamp(torch.atan((base_velocity[:,0]*base_velocity[:,2]) / 9.81),-0.2,0.2)

        pelvis_euler[:,0] = pelvis_euler[:,0] + bias_lat + bias_yaw

        pitch_amp = 0.02
        pelvis_euler[:,1] = self.cfg.pelv_pitch_ref + torch.sin(2*torch.pi * tp_tensor) * pitch_amp
    
        yaw_amp = 0.0
        default_yaw = yaw_amp*torch.sin(2* torch.pi * tp_tensor)
        pelvis_euler[:,2] = default_yaw + self.stance_foot_ori_0[:,2] + base_velocity[:,2] * self.cur_swing_time 

        pelvis_eul_dot = torch.zeros((N,3), device=self.device)

        dtp_dt = 1/(2*(self.T-self.T_ds))
        dphase_dt = 1/(self.T-self.T_ds)
        
        pelvis_eul_dot[:, 0] = (
            roll_main_amp * 4 * torch.pi * torch.cos(4 * torch.pi * tp_tensor) * dtp_dt +
            roll_asym_amp * 2 * torch.pi * torch.cos(2 * torch.pi * tp_tensor) * dtp_dt
        )

        pelvis_eul_dot[:,1] = 2*torch.pi * torch.cos(2*torch.pi * tp_tensor) * pitch_amp * dtp_dt
        pelvis_eul_dot[:,2] = base_velocity[:,2] + yaw_amp*2* torch.pi * torch.cos(2* torch.pi * tp_tensor) * dtp_dt


        foot_eul = torch.zeros((N,3), device=self.device)
        #TODO enable foot orientation control
        foot_eul[:,2] = pelvis_euler[:,2]     
        foot_eul_dot = torch.zeros((N,3), device=self.device)
        foot_eul_dot[:,2] = pelvis_eul_dot[:,2]

        return pelvis_euler, pelvis_eul_dot, foot_eul, foot_eul_dot



    def generate_reference_trajectory(self):
        base_velocity = self.env.command_manager.get_command("base_velocity")  # (N,3)
        N = base_velocity.shape[0]

        T = torch.full((N,), self.T, dtype=torch.float32, device=self.device)

        Xdes, Ux, Ydes, Uy = self.hlip_controller.compute_orbit(
            T=T,cmd=base_velocity)
        
        #select init and Xdes, Ux, Ydes, Uy
        x0 = self.hlip_controller.x_init
        y0 = self.hlip_controller.y_init[:,self.stance_idx]
        Uy = Uy[:,self.stance_idx]
    
        com_x, com_xd = self.hlip_controller._compute_desire_com_trajectory(
            cur_time=self.cur_swing_time,
            Xdesire=x0,
        )
        com_y, com_yd = self.hlip_controller._compute_desire_com_trajectory(
            cur_time=self.cur_swing_time,
            Xdesire=y0,
        )
        # Concatenate x and y components
        com_pos_des = torch.stack([com_x, com_y,torch.ones((N,), device=self.device) * self.com_z], dim=-1)  # Shape: (N,2)
        com_vel_des = torch.stack([com_xd, com_yd,torch.zeros((N), device=self.device)], dim=-1)  # Shape: (N,2)

        

        foot_target = torch.stack([Ux,Uy,torch.zeros((N), device=self.device)], dim=-1)
  
        # based on yaw velocity, update com_pos_des, com_vel_des, foot_target,
        delta_psi = base_velocity[:,2] * self.cur_swing_time
        q_delta_yaw = quat_from_euler_xyz(
            torch.zeros_like(delta_psi),               # roll=0
            torch.zeros_like(delta_psi),               # pitch=0
            delta_psi                                  # yaw=Δψ
        ) 

        foot_target_yaw_adjusted = quat_apply(q_delta_yaw, foot_target)  # [B,3]
        com_pos_des_yaw_adjusted = quat_apply(q_delta_yaw, com_pos_des)  # [B,3]
        com_vel_des_yaw_adjusted = quat_apply(q_delta_yaw, com_vel_des)  # [B,3]
        
        foot_target_yaw_adjusted_clipped = foot_target_yaw_adjusted.clone()
        foot_target_yaw_adjusted_clipped[:,1] = torch.clamp(torch.abs(foot_target_yaw_adjusted[:,1]), min=self.cfg.foot_target_range_y[0], max=self.cfg.foot_target_range_y[1]) * torch.sign(Uy)
        # clip foot target based on kinematic range
        self.foot_target = foot_target_yaw_adjusted_clipped[:,0:2]
       
        z_sw_max = self.cfg.z_sw_max
        z_sw_neg = self.cfg.z_sw_min

        # Create horizontal control points with batch dimension
        horizontal_control_points = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0], device=self.device).repeat(N, 1)  # Shape: (N, 5)
        
        # Create tensors with batch dimension N
        phase_var_tensor = torch.full((N,), self.phase_var, device=self.device)
        T_tensor = torch.full((N,), self.T, device=self.device)
        four_tensor = torch.tensor(4, device=self.device)
        
        bht = bezier_deg(
            0, phase_var_tensor, T_tensor, horizontal_control_points, four_tensor
        )
        
        # Convert scalar parameters to tensors with batch dimension N
        z_sw_max_tensor = torch.full((N,), z_sw_max, device=self.device)
        z_sw_neg_tensor = torch.full((N,), z_sw_neg, device=self.device)
        z_init = torch.full((N,), 0.0, device=self.device)
        # Convert bht to tensor if it's not already
        bht_tensor = torch.tensor(bht, device=self.device) if not isinstance(bht, torch.Tensor) else bht
        
        sign = torch.sign(foot_target_yaw_adjusted[:, 1])
        foot_pos, sw_z = calculate_cur_swing_foot_pos(
            bht_tensor, z_init, z_sw_max_tensor, phase_var_tensor,-foot_target_yaw_adjusted[:, 0], sign*self.cfg.y_nom,T_tensor, z_sw_neg_tensor,
            foot_target_yaw_adjusted[:, 0], foot_target_yaw_adjusted[:, 1]
        )

        dbht = bezier_deg(1, phase_var_tensor, T_tensor, horizontal_control_points, four_tensor)
        foot_vel = torch.zeros((N,3), device=self.device)
        foot_vel[:,0] = -dbht * -foot_target_yaw_adjusted[:,0]+ dbht * foot_target_yaw_adjusted[:,0]
        foot_vel[:,1] = -dbht * foot_target_yaw_adjusted[:,1] + dbht * foot_target_yaw_adjusted[:,1]
        foot_vel[:,2] = sw_z.squeeze(-1)  # Remove the last dimension to match foot_vel[:,2] shape

        upper_body_joint_pos, upper_body_joint_vel = self.generate_upper_body_ref()

        pelvis_euler, pelvis_eul_dot, foot_eul, foot_eul_dot = self.generate_orientation_ref(base_velocity,N)
        omega_ref = euler_rates_to_omega(pelvis_euler, pelvis_eul_dot)
        omega_foot_ref = euler_rates_to_omega(foot_eul, foot_eul_dot)  # (N,3)
        #setup up reference trajectory, com pos, pelvis orientation, swing foot pos, ori
        self.y_out = torch.cat([com_pos_des_yaw_adjusted, pelvis_euler, foot_pos, foot_eul,upper_body_joint_pos], dim=-1)
        self.dy_out = torch.cat([com_vel_des_yaw_adjusted, omega_ref, foot_vel, omega_foot_ref,upper_body_joint_vel], dim=-1)

    def generate_upper_body_ref(self):
        # phase: [B]
        forward_vel = self.env.command_manager.get_command("base_velocity")[:, 0]
        N = forward_vel.shape[0]
        phase = 2 * torch.pi * self.tp
        # make it [B,1] so phase+offset broadcasts to [B,9]
        # phase = torch.ones((N,1),device=self.device) * phase

        # fetch forward_vel: [B]
        
        # unpack your cfg scalars
        sh_pitch0, sh_roll0, sh_yaw0 = self.cfg.shoulder_ref
        elb0 = self.cfg.elbow_ref
        waist_yaw0 = self.cfg.waist_yaw_ref

        # build every amp as a [B] tensor
        sh_pitch_amp = sh_pitch0 * forward_vel          # [B]
        sh_roll_amp  = sh_roll0  * torch.ones_like(forward_vel)
        sh_yaw_amp   = sh_yaw0   * torch.ones_like(forward_vel)
        elb_amp      = elb0      * forward_vel
        waist_amp    = waist_yaw0 * torch.ones_like(forward_vel)

        # stack into [B,9]
        amp = torch.stack([
            waist_amp,
            sh_pitch_amp, sh_pitch_amp,
            sh_roll_amp,  sh_roll_amp,
            sh_yaw_amp,   sh_yaw_amp,
            elb_amp,      elb_amp,
        ], dim=1).to(self.device)

        # your sign & offset stay [9] each
        sign = torch.tensor([
            1,         # waist_yaw
            1, -1,     # L/R shoulder_pitch
            1, -1,     # L/R shoulder_roll
            1, -1,     # L/R shoulder_yaw
            1, -1,     # L/R elbow
        ], device=self.device)

        offset = torch.tensor([
            torch.pi,      # waist_yaw
            torch.pi/2,    # L_sh_pitch
            torch.pi/2,    # R_sh_pitch
            torch.pi/2,    # L_sh_roll
            torch.pi/2,    # R_sh_roll
            0,             # L_sh_yaw
            0,             # R_sh_yaw
            torch.pi/2,    # L_elbow
            torch.pi/2,    # R_elbow
        ], device=self.device)

        # joint offsets: [B,9]
        joint_offset = self.robot.data.default_joint_pos[:, self.upper_body_joint_idx]

        # refs: everything now broadcast to [B,9]
        offset = offset.unsqueeze(0).expand(N, -1)
        ref     = amp * sign * torch.sin(phase + offset) + joint_offset

        # velocity
        dphase_dt = 2 * torch.pi / (2*(self.T - self.T_ds))  # scalar
        ref_dot = amp * sign * torch.cos(phase + offset) * dphase_dt

        return ref, ref_dot



    def get_euler_from_quat(self, quat):

        euler_x, euler_y, euler_z = euler_xyz_from_quat(quat)
        euler_x = wrap_to_pi(euler_x)
        euler_y = wrap_to_pi(euler_y)
        euler_z = wrap_to_pi(euler_z)
        return torch.stack([euler_x, euler_y, euler_z], dim=-1)

    def get_actual_state(self):
        """Populate actual state and its time derivative in the robot's local (yaw-aligned) frame."""
        # Convenience
        data = self.robot.data
        root_quat = data.root_quat_w

        # 1. Foot positions and orientations (world frame)
        foot_pos_w = data.body_pos_w[:, self.feet_bodies_idx, :]
        foot_ori_w = data.body_quat_w[:, self.feet_bodies_idx, :]


        # Store raw foot positions
        self.stance_foot_pos = foot_pos_w[:, self.stance_idx, :]
        self.stance_foot_ori = self.get_euler_from_quat(foot_ori_w[:, self.stance_idx, :])

        # Convert foot positions to the robot's yaw-aligned local frame
        # stance_pos_local = _transfer_to_local_frame(
        #     foot_pos_w[:, self.stance_idx, :], root_quat
        # )
        swing2stance_local = _transfer_to_local_frame(
            foot_pos_w[:, self.swing_idx, :]-self.stance_foot_pos_0, self.stance_foot_ori_quat_0
        )

        # Center of mass to stance foot vector in local frame
        com_w = data.root_com_pos_w
        com2stance_local = _transfer_to_local_frame(
            com_w - self.stance_foot_pos_0, self.stance_foot_ori_quat_0
        )


        # Pelvis orientation (Euler XYZ)
        pelvis_ori = self.get_euler_from_quat(root_quat)

        # Foot orientations (Euler XYZ)
        swing_foot_ori = self.get_euler_from_quat(foot_ori_w[:,self.swing_idx,:])

        # 2. Velocities (world frame)
        com_vel_w = data.root_com_vel_w[:,0:3]
        # pelvis_omega_w = data.root_ang_vel_w
        foot_lin_vel_w = data.body_lin_vel_w[:, self.feet_bodies_idx, :]
        foot_ang_vel_w = data.body_ang_vel_w[:, self.feet_bodies_idx, :]

        self.stance_foot_vel = foot_lin_vel_w[:,self.stance_idx,:]
        self.stance_foot_ang_vel = foot_ang_vel_w[:,self.stance_idx,:]
        # Convert velocities to local frame
        # import pdb; pdb.set_trace()
        com_vel_local = _transfer_to_local_frame(com_vel_w, self.stance_foot_ori_quat_0)
      
        pelvis_omega_local = data.root_ang_vel_b
        # foot_lin_vel_local_stance = _transfer_to_local_frame(
        #     foot_lin_vel_w[:,self.stance_idx,:], self.stance_foot_ori_quat_0
        # )
        foot_lin_vel_local_swing = _transfer_to_local_frame(
            foot_lin_vel_w[:,self.swing_idx,:], self.stance_foot_ori_quat_0
        )

        foot_ang_vel_local_swing =quat_apply(quat_inv(foot_ori_w[:,self.swing_idx,:]), foot_ang_vel_w[:,self.swing_idx,:])
        
        # _transfer_to_local_frame(
        #     foot_ang_vel_w[:,self.swing_idx,:], self.stance_foot_ori_quat_0
        # )

        swing2stance_vel = foot_lin_vel_local_swing 
    
        upper_body_joint_pos = self.robot.data.joint_pos[:, self.upper_body_joint_idx]
        upper_body_joint_vel = self.robot.data.joint_vel[:, self.upper_body_joint_idx]
        # 4. Assemble state vectors
        self.y_act = torch.cat([
            com2stance_local,
            pelvis_ori,
            swing2stance_local,
            swing_foot_ori,
            upper_body_joint_pos
        ], dim=-1)

        self.dy_act = torch.cat([
            com_vel_local,
            pelvis_omega_local,
            swing2stance_vel,
            foot_ang_vel_local_swing,
            upper_body_joint_vel
        ], dim=-1)


    def _update_command(self):
        
        self.update_Stance_Swing_idx()
        self.generate_reference_trajectory()
        self.get_actual_state()
        
        #how to handle for the first step?
        #i.e. v is not defined
        vdot,vcur = self.clf.compute_vdot(self.y_act,self.y_out,self.dy_act,self.dy_out)
        self.vdot = vdot
        self.v = vcur
        if torch.sum(self.v_buffer) == 0:
            # (E,) -> (E,1) -> broadcast to (E,100) on assignment
            self.v_buffer[:]    = self.v.unsqueeze(1)
            self.vdot_buffer[:] = self.vdot.unsqueeze(1)

        else:
            self.v_buffer = torch.cat([self.v_buffer[:,1:], self.v.unsqueeze(-1)], dim=-1)
            self.vdot_buffer = torch.cat([self.vdot_buffer[:,1:], self.vdot.unsqueeze(-1)], dim=-1)

       
        if self.debug_vis:
            # Visualize foot target in global frame
            base_velocity = self.env.command_manager.get_command("base_velocity")  # (N,2)
            N = base_velocity.shape[0]
            foot_target = torch.cat([self.foot_target, torch.zeros((N, 1), device=self.device)], dim=-1)
            p_ft_global = _transfer_to_global_frame(foot_target, self.robot.data.root_quat_w) + self.robot.data.root_pos_w
          
            self.footprint_visualizer.visualize(
                translations=p_ft_global,
                orientations=yaw_quat(self.robot.data.root_quat_w).repeat_interleave(2, dim=0),
            )
            
            
            # Print debug info for first environment
            # print(f"Base velocity: {base_velocity[0]}")
            # print(f"y_out reference: {self.y_out}")
            # print(f"dy_out reference: {self.dy_out}")
            # print(f"foot_target: {self.foot_target[0]}")
            # print(f"swing2stance: {self.swing2stance[0]}")
            # print(f"Com2stance: {self.com2stance[0]}")
            # # print(f"Current foot position: {self.robot.data.body_pos_w[0, self.feet_bodies.body_ids[0], :2]}")
            # print("---")

