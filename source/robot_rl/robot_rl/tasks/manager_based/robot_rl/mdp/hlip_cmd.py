import torch
import math
from isaaclab.utils import configclass

from isaaclab.managers import CommandTermCfg,CommandTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv

from .ref_gen import precompute_hlip_dynamics, bezier_deg, compute_hlip_orbit_from_dynamics, compute_desire_com_trajectory, calculate_cur_swing_foot_pos
# from isaaclab.utils.transforms import combine_frame_transforms, quat_from_euler_xyz
from isaaclab.managers import SceneEntityCfg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cmd_cfg import HLIPCommandCfg



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
        self.A = None
        self.B = None
        self.sigma1 = None
        self.sigma2 = None
        self.lam = torch.tensor(math.sqrt(9.81 / self.z0), dtype=torch.float32, device=self.device)
        self.env = env
        self.robot = env.scene[cfg.asset_name]
        self.feet_bodies_idx = self.robot.find_bodies(".*_ankle_roll_link")[0]

        self.foot_target = torch.zeros((self.num_envs, 2), device=self.device)
        self.metrics["error_foot_target"] = torch.zeros((self.num_envs), device=self.device)

        self.y_out = torch.zeros((self.num_envs, 12), device=self.device)
        self.dy_out = torch.zeros((self.num_envs, 12), device=self.device)

        self.com_z = torch.ones((self.num_envs), device=self.device)*self.z0

    @property
    def command(self):
        return self.foot_target

    def _resample_command(self, env_ids):
        # Do nothing here
        device = self.env.command_manager.get_command("base_velocity").device
        self.A, self.B, self.sigma1, self.sigma2 = precompute_hlip_dynamics(
            T=self.T,
            T_ds=self.T_ds,
            z0=self.z0,
            device=device
        )
        # self.A_s2s, self.B_s2s = compute_s2s_matrices(
        #     T=self.T,
        #     A_ss=self.A,
        #     A_ds=self.A,
        #     B_usw=self.B
        # )
        return
    
    def _update_metrics(self):
        # Foot tracking
        foot_pos = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :2]  # Only take x,y coordinates
        # Contact schedule function
        tp = (self.env.sim.current_time % (2 * self.T)) / (2 * self.T)  # Scaled between 0-1
        phi_c = torch.tensor(math.sin(2 * torch.pi * tp) / math.sqrt(math.sin(2 * torch.pi * tp)**2 + self.T), device=self.env.device)

        swing_foot_pos = foot_pos[:, int(0.5 + 0.5 * torch.sign(phi_c))]
        # Only compare x,y coordinates of foot target
        self.metrics["error_foot_target"] = torch.norm(swing_foot_pos - self.foot_target, dim=-1)
        # return self.foot_target  # Return the foot target tensor for observation


    def update_Stance_Swing_idx(self):
        Tswing = self.T - self.T_ds
        tp = (self.env.sim.current_time % (2*Tswing)) / (2*Tswing)  
        phi_c = torch.tensor(math.sin(2 * torch.pi * tp) / math.sqrt(math.sin(2 * torch.pi * tp)**2 + self.T), device=self.env.device)

        self.stance_idx = int(0.5 - 0.5*torch.sign(phi_c))
        self.swing_idx = 1 - self.stance_idx

        
        if tp < 0.5:
            self.phase_var = 2*tp
        else:
            self.phase_var = 2*tp-1
        self.cur_swing_time = self.phase_var*Tswing


    def generate_reference_trajectory(self):
        base_velocity = self.env.command_manager.get_command("base_velocity")  # (N,2)
        N = base_velocity.shape[0]
        T = torch.full((N,), self.T, dtype=torch.float32, device=base_velocity.device)

        base_velocity[:,1] = 0.0
        out = compute_hlip_orbit_from_dynamics(
            cmd_vel=base_velocity,
            T=T,
            A=self.A,
            B=self.B,
            y_nom=self.y_nom,
            stance_idx=self.stance_idx
        )

        # Squeeze the last dimension to get (N,2)
        com_x = out["com_x"].squeeze(-1)
        com_y = out["com_y"].squeeze(-1)
        print("cur_swing_time:", self.cur_swing_time)
        print("com_x:", com_x)
        print("com_y:", com_y)


        com_pos_des_x, com_vel_des_x = compute_desire_com_trajectory(
            cur_time=self.cur_swing_time,
            Xdesire=com_x,
            lam=self.lam,
        )
        com_pos_des_y, com_vel_des_y = compute_desire_com_trajectory(
            cur_time=self.cur_swing_time,
            Xdesire=com_y,
            lam=self.lam,
        )

        print("com_pos_des_x:", com_pos_des_x)
        print("com_pos_des_y:", com_pos_des_y)

        # import pdb; pdb.set_trace()

        # Concatenate x and y components
        
        com_pos_des = torch.stack([com_pos_des_x, com_pos_des_y,self.com_z], dim=-1)  # Shape: (N,2)
        com_vel_des = torch.stack([com_vel_des_x, com_vel_des_y,torch.zeros((N), device=self.device)], dim=-1)  # Shape: (N,2)

        self.foot_target = out["foot_placement"]

        pelvis_ori = torch.zeros((N,3), device=self.device)
        #TODO enable heading control
        # heading_target = self.env.command_manager.get_term("base_velocity").heading_target
        # pelvis_ori[:,2] = heading_target
        current_heading = self.robot.data.heading_w
        pelvis_ori[:,2] = current_heading + base_velocity[:,2] * self.env.cfg.sim.dt

        foot_ori = torch.zeros((N,3), device=self.device)
        #TODO enable foot orientation control
        foot_ori[:,2] = pelvis_ori[:,2]
        foot_vel = torch.zeros((N,3), device=self.device)
        foot_ori_vel = torch.zeros((N,3), device=self.device)
        pelvis_ori_vel = torch.zeros((N,3), device=self.device)

        pelvis_ori_vel[:,2] = base_velocity[:,2]
        foot_ori_vel[:,2] = pelvis_ori_vel[:,2]
        z_sw_max = 0.1
        z_sw_neg = 0.0

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
        

    def get_actual_state(self):
         # Get foot positions in global frame
        foot_pos = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]
        self.actual_foot_pos = foot_pos

        # Convert foot positions to robot's yaw frame
        stance_foot_pos_relative = _transfer_to_local_frame(foot_pos[:, self.stance_idx, :], self.robot.data.root_quat_w)
        swing_foot_pos_relative = _transfer_to_local_frame(foot_pos[:, self.swing_idx, :], self.robot.data.root_quat_w)
     
        com2st = self.robot.data.root_com_pos_w - foot_pos[:, self.stance_idx, :]

        self.swing2stance = swing_foot_pos_relative - stance_foot_pos_relative
        self.com2stance = _transfer_to_local_frame(com2st, self.robot.data.root_quat_w)



    def _update_command(self):
        
        self.update_Stance_Swing_idx()
        self.generate_reference_trajectory()
        self.get_actual_state()
        
       
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

