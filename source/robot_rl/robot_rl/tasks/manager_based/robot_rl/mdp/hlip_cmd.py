import torch
import math
from isaaclab.utils import configclass

from isaaclab.managers import CommandTermCfg,CommandTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv


# from isaaclab.utils.transforms import combine_frame_transforms, quat_from_euler_xyz
from isaaclab.managers import SceneEntityCfg
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cmd_cfg import HLIPCommandCfg

def coth(x: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.tanh(x)


def precompute_hlip_dynamics(T: float, T_ds: float, z0: float, device: torch.device):
    lam = math.sqrt(9.81 / z0)
    Ts = T - T_ds
    lamTs = lam * Ts
    lamTs_tensor = torch.tensor(lamTs, dtype=torch.float32, device=device)
    sigma1 = lam * coth(0.5 * lamTs_tensor)
    sigma2 = lam * torch.tanh(0.5 * lamTs_tensor)

    cosh_lTs = math.cosh(lamTs)
    sinh_lTs = math.sinh(lamTs)

    A = torch.tensor([[cosh_lTs, sinh_lTs/lam], [lam*sinh_lTs, cosh_lTs]], dtype=torch.float32, device=device)
    B = torch.tensor([[1.0-cosh_lTs], [-lam*sinh_lTs]], dtype=torch.float32, device=device)
    return A, B, sigma1, sigma2


def compute_hlip_orbit_from_dynamics(
    cmd_vel: torch.Tensor,
    T: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    y_nom: float,
    stance_idx: int
):
    N = cmd_vel.shape[0]
    device = cmd_vel.device

    def Bu(u):
        return B.unsqueeze(0) * u.view(N,1,1)

    u_x = cmd_vel[:,0] * T
    X_des = torch.linalg.solve(
        torch.eye(2, device=device).expand(N,2,2) - A.unsqueeze(0),
        Bu(u_x)
    )

    u_left = cmd_vel[:,1] * T - y_nom
    u_right = cmd_vel[:,1] * T + y_nom
    A2 = A @ A

    Y_left = torch.linalg.solve(
        torch.eye(2, device=device).expand(N,2,2) - A2.unsqueeze(0),
        A.unsqueeze(0) @ Bu(u_left) + Bu(u_right)
    )
    Y_right = torch.linalg.solve(
        torch.eye(2, device=device).expand(N,2,2) - A2.unsqueeze(0),
        A.unsqueeze(0) @ Bu(u_right) + Bu(u_left)
    )

    #TODO need to check left/right order assumption
    #select left or right foot based on contact schedule function
    if stance_idx == 0:
        u_y = u_left
        Y_des = Y_left
    else:
        u_y = u_right
        Y_des = Y_right

    return {
        "com_pos_des": torch.stack([X_des[:,0,0], Y_des[:,0,0]], dim=-1),
        "com_vel_des": torch.stack([X_des[:,1,0], Y_des[:,1,0]], dim=-1),
        "foot_placement": torch.stack([u_x, u_y], dim=-1),
    }


class HLIPCommandTerm(CommandTerm):
    def __init__(self, cfg: "HLIPCommandCfg", env):
        super().__init__(cfg, env)
        self.T_ds = cfg.T_ds
        self.z0 = cfg.z0
        self.y_nom = cfg.y_nom
        self.T = cfg.gait_period
        self.debug_vis = cfg.debug_vis
        if self.debug_vis:
            self.footprint_visualizer = VisualizationMarkers(cfg.footprint_cfg)
            self.goal_pose_visualizer = VisualizationMarkers(cfg.goal_pose_visualizer_cfg)
            self.current_pose_visualizer = VisualizationMarkers(cfg.current_pose_visualizer_cfg)
        self.A = None
        self.B = None
        self.sigma1 = None
        self.sigma2 = None
        self.env = env
        self.robot = env.scene[cfg.asset_name]
        self.feet_bodies = SceneEntityCfg("robot", body_names=".*_ankle_roll_link")
        self.foot_target = torch.zeros((self.num_envs, 2), device=self.device)
        self.metrics["error_foot_target"] = torch.zeros((self.num_envs), device=self.device)

        

    @property
    def command(self):
        return self.foot_target

    def _resample_command(self, env_ids):
        #do nothing here
        device = self.env.command_manager.get_command("base_velocity").device
        self.A, self.B, self.sigma1, self.sigma2 = precompute_hlip_dynamics(
            T=self.T,
            T_ds=self.T_ds,
            z0=self.z0,
            device=device
        )
        return
    
    def _update_metrics(self):
        # Foot tracking
        foot_pos = self.robot.data.body_pos_w[:, self.feet_bodies.body_ids, :2]  # Only take x,y coordinates
        # Contact schedule function
        tp = (self.env.sim.current_time % (2 * self.T)) / (2 * self.T)  # Scaled between 0-1
        phi_c = torch.tensor(math.sin(2 * torch.pi * tp) / math.sqrt(math.sin(2 * torch.pi * tp)**2 + self.T), device=self.env.device)

        swing_foot_pos = foot_pos[:, int(0.5 + 0.5 * torch.sign(phi_c))]
        # Only compare x,y coordinates of foot target
        self.metrics["error_foot_target"] = torch.norm(swing_foot_pos - self.foot_target, dim=-1)
        # return self.foot_target  # Return the foot target tensor for observation

    def _update_command(self):
        base_velocity = self.env.command_manager.get_command("base_velocity")  # (N,2)
        N = base_velocity.shape[0]
        T = torch.full((N,), self.T, dtype=torch.float32, device=base_velocity.device)


        Tswing = self.T - self.T_ds
        tp = (self.env.sim.current_time % (2*Tswing)) / (2*Tswing)     # Scaled between 0-1
        phi_c = torch.tensor(math.sin(2*torch.pi*tp)/math.sqrt(math.sin(2*torch.pi*tp)**2 + Tswing), device=self.device)

        self.stance_idx = int(0.5 - 0.5*torch.sign(phi_c))

        out = compute_hlip_orbit_from_dynamics(
            cmd_vel=base_velocity,
            T=T,
            A=self.A,
            B=self.B,
            y_nom=self.y_nom,
            stance_idx=self.stance_idx
        )

        self.ref_com_pos = out["com_pos_des"]
        self.ref_com_vel = out["com_vel_des"]
        self.foot_target = out["foot_placement"]
        
        foot_pos = self.robot.data.body_pos_w[:, self.feet_bodies.body_ids,:]
       
        self.actual_foot_pos = foot_pos

        stance_foot_pos = foot_pos[:, self.stance_idx, :]
        stance_foot_pos[:, 2] *= 0

        swing_foot_pos = foot_pos[:, int(0.5 + 0.5 * torch.sign(phi_c)), :]
        swing_foot_pos[:, 2] *= 0

        self.swing2stance = swing_foot_pos - stance_foot_pos

        if self.debug_vis:
            # Visualize foot target
            foot_target = torch.cat([self.foot_target, torch.zeros((N,1), device=self.device)], dim=-1)
            self.footprint_visualizer.visualize(
                # TODO: Visualize both the current stance foot and the desired foot
                # translations=foot_pos[:, int(0.5 - 0.5*torch.sign(phi_c)), :], #p,
                # translations=foot_pos[:, (env.cfg.control_count % 2), :],
                translations=foot_target,
                orientations=yaw_quat(self.robot.data.root_quat_w).repeat_interleave(2, dim=0),
                # repeat 0,1 for num_env
                # marker_indices=torch.tensor([0,1], device=env.device).repeat(env.num_envs),
            )
            
            # Print debug info for first environment
            print(f"Base velocity: {base_velocity[0]}")
            print(f"COM position reference: {self.ref_com_pos[0]}")
            print(f"COM velocity reference: {self.ref_com_vel[0]}")
            print(f"Foot target: {self.foot_target[0]}")
            # print(f"Current foot position: {self.robot.data.body_pos_w[0, self.feet_bodies.body_ids[0], :2]}")
            print("---")

            # Visualize goal pose
            # positions = self.foot_target[:, :3]
            # yaw = torch.zeros(N, device=positions.device)
            # quats = torch.cat([
            #     torch.cos(yaw/2).unsqueeze(-1),
            #     torch.zeros((N,2), device=yaw.device),
            #     torch.sin(yaw/2).unsqueeze(-1)
            # ], dim=-1)
            # world_pos, world_quat = combine_frame_transforms(
            #     self.env.robot.data.root_pos_w,
            #     self.env.robot.data.root_quat_w,
            #     positions,
            #     quats
            # )
            # self.goal_pose_visualizer.visualize(world_pos, world_quat)

            # # Visualize current pose
            # current_pos = self.env.sw2st_pos + self.env.robot.data.root_pos_w
            # current_quat = quat_from_euler_xyz(
            #     self.env.sw2st_ori[:,0],
            #     self.env.sw2st_ori[:,1],
            #     self.env.sw2st_ori[:,2]
            # )
            # self.current_pose_visualizer.visualize(current_pos, current_quat)
