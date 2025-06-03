import torch
import math
from isaaclab.utils import configclass
import numpy as np

from isaaclab.managers import CommandTermCfg,CommandTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv

from .ref_gen import bezier_deg, calculate_cur_swing_foot_pos, HLIP
from .clf import CLF
# from isaaclab.utils.transforms import combine_frame_transforms, quat_from_euler_xyz

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cmd_cfg import HLIPCommandCfg


def wrap_to_pi(angle):
    """
    Wraps angles in radians to the range [-pi, pi].
    Works with torch tensors or scalars.
    """
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi

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
            self.goal_pose_visualizer = VisualizationMarkers(cfg.goal_pose_visualizer_cfg)
            self.current_pose_visualizer = VisualizationMarkers(cfg.current_pose_visualizer_cfg)
       
        self.env = env
        self.robot = env.scene[cfg.asset_name]
        self.feet_bodies_idx = self.robot.find_bodies(cfg.foot_body_name)[0]

        self.foot_target = torch.zeros((self.num_envs, 2), device=self.device)

        self.metrics = {}
     
        self.y_out = torch.zeros((self.num_envs, 12), device=self.device)
        self.dy_out = torch.zeros((self.num_envs, 12), device=self.device)

        self.com_z = torch.ones((self.num_envs), device=self.device)*self.z0

        grav = torch.abs(torch.tensor(self.env.cfg.sim.gravity[2], device=self.device))
        self.hlip_controller = HLIP(grav, self.z0, self.T_ds, self.T, self.y_nom)

        self.mass = sum(self.robot.data.default_mass.T)[0]
        A_lip = torch.tensor([[0.0, 1.0], [grav / self.z0, 0.0]], device=self.device)
        B_lip = torch.tensor([[0.0], [1.0 / (self.mass * self.z0)]], device=self.device)

        self.clf = CLF(
            A_lip, B_lip, 12, self.env.cfg.sim.dt,
            Q_weights=np.array(cfg.Q_weights),
            R_weights=np.array(cfg.R_weights),
            device=self.device
        )
        
        self.v = torch.zeros((self.num_envs), device=self.device)
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
        
        # return self.foot_target  # Return the foot target tensor for observation


    def update_Stance_Swing_idx(self):
        Tswing = self.T - self.T_ds
        tp = (self.env.sim.current_time % (2*Tswing)) / (2*Tswing)  
        phi_c = torch.tensor(math.sin(2 * torch.pi * tp) / math.sqrt(math.sin(2 * torch.pi * tp)**2 + self.T), device=self.env.device)



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
       
        self.stance_idx = new_stance_idx


        if tp < 0.5:
            self.phase_var = 2*tp
        else:
            self.phase_var = 2*tp-1
        self.cur_swing_time = self.phase_var*Tswing


    def generate_reference_trajectory(self):
        
        base_velocity = self.env.command_manager.get_command("base_velocity")  # (N,2)

        N = base_velocity.shape[0]
        T = torch.full((N,), self.T, dtype=torch.float32, device=base_velocity.device)

        Xdes, Ux, Ydes, Uy = self.hlip_controller.compute_orbit(
            T=T,cmd=base_velocity)
        
        #select init and Xdes, Ux, Ydes, Uy
        com_y_init = self.hlip_controller.y_init[:,self.stance_idx]
        com_x_init = self.hlip_controller.x_init
        Uy_des = Uy[:,self.stance_idx]
    
        com_pos_des_x, com_vel_des_x = self.hlip_controller._compute_desire_com_trajectory(
            cur_time=self.cur_swing_time,
            Xdesire=com_x_init,
        )
        com_pos_des_y, com_vel_des_y = self.hlip_controller._compute_desire_com_trajectory(
            cur_time=self.cur_swing_time,
            Xdesire=com_y_init,
        )
        # Concatenate x and y components
        com_pos_des = torch.stack([com_pos_des_x, com_pos_des_y,self.com_z], dim=-1)  # Shape: (N,2)
        com_vel_des = torch.stack([com_vel_des_x, com_vel_des_y,torch.zeros((N), device=self.device)], dim=-1)  # Shape: (N,2)

        self.foot_target = torch.stack([Ux,Uy_des], dim=-1)

        pelvis_ori = torch.zeros((N,3), device=self.device)
        pelvis_ori[:,1] = self.cfg.pelv_pitch_ref
        #TODO enable heading control
        # heading_target = self.env.command_manager.get_term("base_velocity").heading_target
   
        # pelvis_ori[:,2] = heading_target
        # current_heading = self.robot.data.heading_w
        pelvis_ori[:,2] = self.stance_foot_ori_0[:,2] + base_velocity[:,2] * self.env.cfg.sim.dt

        foot_ori = torch.zeros((N,3), device=self.device)
        #TODO enable foot orientation control
        foot_ori[:,2] = pelvis_ori[:,2]
        foot_vel = torch.zeros((N,3), device=self.device)
        foot_ori_vel = torch.zeros((N,3), device=self.device)
        pelvis_ori_vel = torch.zeros((N,3), device=self.device)

        pelvis_ori_vel[:,2] = base_velocity[:,2]
        foot_ori_vel[:,2] = pelvis_ori_vel[:,2]
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
        
        foot_pos, sw_z = calculate_cur_swing_foot_pos(
            bht_tensor, z_init, z_sw_max_tensor, phase_var_tensor, T_tensor, z_sw_neg_tensor,
            self.foot_target[:, 0], self.foot_target[:, 1]
        )

        
        dbht = bezier_deg(1, phase_var_tensor, T_tensor, horizontal_control_points, four_tensor)

        foot_vel[:,0] = -dbht * -self.foot_target[:,0]+ dbht * self.foot_target[:,0]
        foot_vel[:,1] = -dbht * self.foot_target[:,1] + dbht * self.foot_target[:,1]
        foot_vel[:,2] = sw_z.squeeze(-1)  # Remove the last dimension to match foot_vel[:,2] shape
        #setup up reference trajectory, com pos, pelvis orientation, swing foot pos, ori
        self.y_out = torch.concatenate([com_pos_des, pelvis_ori, foot_pos, foot_ori], dim=-1)
        self.dy_out = torch.concatenate([com_vel_des, pelvis_ori_vel, foot_vel, foot_ori_vel], dim=-1)

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
        pelvis_omega_w = data.root_ang_vel_w
        foot_lin_vel_w = data.body_lin_vel_w[:, self.feet_bodies_idx, :]
        foot_ang_vel_w = data.body_ang_vel_w[:, self.feet_bodies_idx, :]

        self.stance_foot_vel = foot_lin_vel_w[:,self.stance_idx,:]
        self.stance_foot_ang_vel = foot_ang_vel_w[:,self.stance_idx,:]
        # Convert velocities to local frame
        # import pdb; pdb.set_trace()
        com_vel_local = _transfer_to_local_frame(com_vel_w, self.stance_foot_ori_quat_0)
      
        pelvis_omega_local = _transfer_to_local_frame(pelvis_omega_w, self.stance_foot_ori_quat_0)
        foot_lin_vel_local_stance = _transfer_to_local_frame(
            foot_lin_vel_w[:,self.stance_idx,:], self.stance_foot_ori_quat_0
        )
        foot_lin_vel_local_swing = _transfer_to_local_frame(
            foot_lin_vel_w[:,self.swing_idx,:], self.stance_foot_ori_quat_0
        )

        foot_ang_vel_local_swing = _transfer_to_local_frame(
            foot_ang_vel_w[:,self.swing_idx,:], self.stance_foot_ori_quat_0
        )

        swing2stance_vel = foot_lin_vel_local_swing - foot_lin_vel_local_stance
    
        # 4. Assemble state vectors
        self.y_act = torch.cat([
            com2stance_local,
            pelvis_ori,
            swing2stance_local,
            swing_foot_ori
        ], dim=-1)

        self.dy_act = torch.cat([
            com_vel_local,
            pelvis_omega_local,
            swing2stance_vel,
            foot_ang_vel_local_swing
        ], dim=-1)


    def _update_command(self):
        
        self.update_Stance_Swing_idx()
        self.generate_reference_trajectory()
        self.get_actual_state()
        
        #how to handle for the first step?
        #i.e. v is not defined
        vdot,vcur = self.clf.compute_vdot(self.y_act,self.y_out,self.dy_act,self.dy_out,self.v)
        self.vdot = vdot
        self.v = vcur
       
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

