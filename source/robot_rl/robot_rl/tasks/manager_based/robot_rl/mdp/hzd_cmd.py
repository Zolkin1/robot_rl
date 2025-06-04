import torch
import math
from isaaclab.utils import configclass
import numpy as np

from isaaclab.managers import CommandTermCfg,CommandTerm
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv

from robot_rl.assets.robots.exo_cfg import JointTrajectoryConfig

from .hlip_cmd import _transfer_to_global_frame, _transfer_to_local_frame, wrap_to_pi
from .ref_gen import _ncr
from .clf import CLF
import re
# from isaaclab.utils.transforms import combine_frame_transforms, quat_from_euler_xyz

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cmd_cfg import HZDCommandCfg


def bezier_deg(
    order: int,
    tau: torch.Tensor,            # [batch], each in [0,1]
    step_dur: torch.Tensor,       # [batch]
    control_points: torch.Tensor, # [n_dim, degree+1]
    degree: int,
) -> torch.Tensor:
    """
    Computes (for each τ in the batch) either
      • the vector‐valued Bezier position B(τ) ∈ R^{n_dim}, or
      • its time derivative B'(τ) ∈ R^{n_dim},

    where `control_points` is shared across the whole batch and has shape [n_dim, degree+1].

    Args:
      order: 0 → position, 1 → time‐derivative.
      tau:       shape [batch], clipped to [0,1].
      step_dur:  shape [batch], positive scalars.
      control_points: shape [n_dim, degree+1].
      degree:    polynomial degree (so there are `degree+1` control points).

    Returns:
      If order==0: a tensor of shape [batch, n_dim], the Bezier‐position at each τ[i].
      If order==1: a tensor of shape [batch, n_dim], the time‐derivative at each τ[i].
    """
    # 1) Clamp tau into [0,1]
    tau = torch.clamp(tau, 0.0, 1.0)         # [batch]
    batch = tau.size(0)

    # 2) Extract n_dim from control_points
    #    control_points: [n_dim, degree+1]
    n_dim = control_points.shape[0]

    if order == 1:
        # ─── DERIVATIVE CASE ────────────────────────────────────────────────────
        # We want:
        #   B'(τ) = degree * sum_{i=0..degree-1} [
        #             (CP_{i+1} - CP_i) * C(degree-1, i)
        #             * (1-τ)^(degree-1-i) * τ^i
        #          ]  / step_dur.

        # 3) Compute CP differences along the "degree+1" axis:
        #    cp_diff: [n_dim, degree], where
        #      cp_diff[:, i] = control_points[:, i+1] - control_points[:, i].
        cp_diff = control_points[:, 1:] - control_points[:, :-1]  # [n_dim, degree]

        # 4) Binomial coefficients for (degree-1 choose i), i=0..degree-1:
        #    coefs_diff: [degree].
        coefs_diff = torch.tensor(
            [_ncr(degree - 1, i) for i in range(degree)],
            dtype=control_points.dtype,
            device=control_points.device
        )  # [degree]

        # 5) Build (τ^i) and ((1-τ)^(degree-1-i)) for i=0..degree-1:
        i_vec = torch.arange(degree, device=control_points.device)  # [degree]

        #    tau_pow:     [batch, degree],  τ^i
        tau_pow = tau.unsqueeze(1).pow(i_vec.unsqueeze(0))

        #    one_minus_pow: [batch, degree], (1-τ)^(degree-1-i)
        one_minus_pow = (1 - tau).unsqueeze(1).pow((degree - 1 - i_vec).unsqueeze(0))

        # 6) Combine into a single “weight matrix” for the derivative:
        #    weight_deriv[b, i] = degree * C(degree-1, i) * (1-τ[b])^(degree-1-i) * (τ[b])^i
        #    → shape [batch, degree]
        weight_deriv = (degree
                        * coefs_diff.unsqueeze(0)        # [1, degree]
                        * one_minus_pow                 # [batch, degree]
                        * tau_pow)                       # [batch, degree]
        # Now weight_deriv: [batch, degree]

        # 7) Multiply these weights by cp_diff to get a [batch, n_dim] result:
        #    For each batch b:  B'_b =  Σ_{i=0..degree-1} weight_deriv[b,i] * cp_diff[:,i],
        #    which is exactly a mat‐mul:  weight_deriv[b,:] @ (cp_diff^T) → [n_dim].
        #
        #    cp_diff^T: [degree, n_dim], so (weight_deriv @ cp_diff^T) → [batch, n_dim].
        Bdot = torch.matmul(weight_deriv, cp_diff.transpose(0, 1))  # [batch, n_dim]

        # 8) Finally divide by step_dur:
        return Bdot / step_dur.unsqueeze(1)  # [batch, n_dim]


    else:
        # ─── POSITION CASE ────────────────────────────────────────────────────────
        # We want:
        #   B(τ) = Σ_{i=0..degree} [
        #            CP_i * C(degree, i) * (1-τ)^(degree-i) * τ^i
        #         ].

        # 3) Binomial coefficients for (degree choose i), i=0..degree:
        #    coefs_pos: [degree+1]
        coefs_pos = torch.tensor(
            [_ncr(degree, i) for i in range(degree + 1)],
            dtype=control_points.dtype,
            device=control_points.device
        )  # [degree+1]

        # 4) Build τ^i and (1-τ)^(degree-i) for i=0..degree:
        i_vec = torch.arange(degree + 1, device=control_points.device)  # [degree+1]

        #    tau_pow:        [batch, degree+1]
        tau_pow = tau.unsqueeze(1).pow(i_vec.unsqueeze(0))

        #    one_minus_pow:  [batch, degree+1]
        one_minus_pow = (1 - tau).unsqueeze(1).pow((degree - i_vec).unsqueeze(0))

        # 5) Combine into a “weight matrix” for position:
        #    weight_pos[b, i] = C(degree, i) * (1-τ[b])^(degree-i) * (τ[b])^i.
        #    → shape [batch, degree+1]
        weight_pos = (coefs_pos.unsqueeze(0)    # [1, degree+1]
                      * one_minus_pow          # [batch, degree+1]
                      * tau_pow)               # [batch, degree+1]
        # Now weight_pos: [batch, degree+1]

        # 6) Multiply by control_points to get [batch, n_dim]:
        #    For each batch b:  B_b = Σ_{i=0..degree} weight_pos[b,i] * control_points[:,i],
        #    i.e.  weight_pos[b,:]  (shape [degree+1]) @ (control_points^T) ([degree+1, n_dim]) → [n_dim].
        #
        #    So:  B = weight_pos @ control_points^T  → [batch, n_dim].
        B = torch.matmul(weight_pos, control_points.transpose(0, 1))  # [batch, n_dim]

        return B



class HZDCommandTerm(CommandTerm):
    def __init__(self, cfg: "HZDCommandCfg", env):
        super().__init__(cfg, env)
       
        self.env = env
        self.robot = env.scene[cfg.asset_name]

        self.debug_vis = cfg.debug_vis

        # load polynomial coefficients
        # propagate remapped coeffcient, step period, etc.
        self.T = 1.0

        self.feet_bodies_idx = self.robot.find_bodies(cfg.foot_body_name)[0]

        self.foot_target = torch.zeros((self.num_envs, 2), device=self.device)

        self.metrics = {}
     
        self.y_out = torch.zeros((self.num_envs, 18), device=self.device)
        self.dy_out = torch.zeros((self.num_envs, 18), device=self.device)

        # self.com_z = torch.ones((self.num_envs), device=self.device)*self.z0

        # load joint trajectory config
        self.jt_config = JointTrajectoryConfig()
        right_jt_coeffs = self.jt_config.joint_trajectories
        right_base_coeffs = self.jt_config.base_trajectories
        left_jt_coeffs = self.jt_config.remap_jt_symmetric()
        left_base_coeffs = self.jt_config.remap_base_symmetric()

        
        left_coeffs = []
        right_coeffs = []
        for key in self.jt_config.base_trajectories.keys():
            right_coeffs.append(right_base_coeffs[key])
            left_coeffs.append(left_base_coeffs[key])

        for key in self.jt_config.joint_trajectories.keys():
           
            right_coeffs.append(right_jt_coeffs[key])
            left_coeffs.append(left_jt_coeffs[key])

        self.right_coeffs = torch.tensor(right_coeffs, device=self.device)
        self.left_coeffs = torch.tensor(left_coeffs, device=self.device)


        
        # env.scene[cfg.asset_name].data.joint_names

        # joint_names = jt_config.get_joint_names()
        # self._left_traj = self.jt_config.get_trajectory_remap(joint_names).to(self.device)
        # self._right_traj = self.jt_config.get_trajectory(joint_names).to(self.device)

        # self.num_joints = num_joints
        # self.joint_ids = joint_indices

        # self.joint_coeffs = torch.stack([torch.tensor(c, device=self.device) if isinstance(c, list) else c.to(self.device) for c in matched_coeffs], dim=0)
        # self.joint_coeffs_remap = torch.stack([torch.tensor(c, device=self.device) if isinstance(c, list) else c.to(self.device) for c in matched_remap_coeffs], dim=0)

        # self._joint_ids, self._joint_names = self._asset.find_joints(
        #     self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        # )
        # # grav = torch.abs(torch.tensor(self.env.cfg.sim.gravity[2], device=self.device))
      

        self.mass = sum(self.robot.data.default_mass.T)[0]
        
        self.clf = CLF(
            [],[], 18, self.env.cfg.sim.dt,
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
        base_offset = 6
        self.metrics["err_left_sagittal_knee"] = torch.abs(self.y_out[:,6+base_offset] - self.y_act[:,6+base_offset])
        self.metrics["err_right_sagittal_knee"] = torch.abs(self.y_out[:,7+base_offset] - self.y_act[:,7+base_offset])
        self.metrics["err_left_sagittal_ankle"] = torch.abs(self.y_out[:,8+base_offset] - self.y_act[:,8+base_offset])
        self.metrics["err_right_sagittal_ankle"] = torch.abs(self.y_out[:,9+base_offset] - self.y_act[:,9+base_offset])
        self.metrics["err_left_henke_ankle"] = torch.abs(self.y_out[:,10+base_offset] - self.y_act[:,10+base_offset])
        self.metrics["err_right_henke_ankle"] = torch.abs(self.y_out[:,11+base_offset] - self.y_act[:,11+base_offset])
        

        self.metrics["err_left_frontal_hip"] = torch.abs(self.y_out[:,0+base_offset] - self.y_act[:,0+base_offset])
        self.metrics["err_right_frontal_hip"] = torch.abs(self.y_out[:,1+base_offset] - self.y_act[:,1+base_offset])
        self.metrics["err_left_transverse_hip"] = torch.abs(self.y_out[:,2+base_offset] - self.y_act[:,2+base_offset])
        self.metrics["err_right_transverse_hip"] = torch.abs(self.y_out[:,3+base_offset] - self.y_act[:,3+base_offset])
        self.metrics["err_left_sagittal_hip"] = torch.abs(self.y_out[:,4+base_offset] - self.y_act[:,4+base_offset])
        self.metrics["err_right_sagittal_hip"] = torch.abs(self.y_out[:,5+base_offset] - self.y_act[:,5+base_offset])

        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot
        
        # return self.foot_target  # Return the foot target tensor for observation


    def update_Stance_Swing_idx(self):
        Tswing = self.T 
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

        if self.stance_idx == 0:
            ctrl_points = self.right_coeffs
        else:
            ctrl_points = self.left_coeffs
     
        phase_var_tensor = torch.full((N,), self.phase_var, device=self.device)
        des_jt_pos = bezier_deg(
            0, phase_var_tensor, T, ctrl_points, torch.tensor(7, device=self.device)
        )
        
        des_jt_vel = bezier_deg(1, phase_var_tensor, T, ctrl_points, 7)

        self.y_out = des_jt_pos
        self.dy_out = des_jt_vel


    def nom_bezier_curve(self,control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluates multiple Bezier curves at given time points using the control points.

        Args:
            control_points (torch.Tensor): Tensor of shape [num_curves, num_control_points, dim]
                                        representing the control points for each curve.
            t (torch.Tensor): Tensor of shape [num_curves] representing the time point at which to evaluate each Bezier curve.
        
        Returns:
            torch.Tensor: Tensor of shape [num_curves, dim] representing the evaluated points on the Bezier curves.
        """
        dim,num_control_points = control_points.shape
        n = num_control_points - 1

        num_curves = t.shape[0]
        # Initialize the result tensor to zeros: shape [num_curves, dim]
        curve_points = torch.zeros((num_curves, dim), dtype=control_points.dtype, device=control_points.device)
        
        for i in range(n + 1):
            binomial_coeff = math.comb(n, i)  # Binomial coefficient
            bernstein_poly = binomial_coeff * (t ** i) * ((1 - t) ** (n - i))  # Bernstein basis polynomial

            # Add the contribution of each control point to each curve
            curve_points += bernstein_poly.unsqueeze(-1) * control_points[:,i]

        return curve_points

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

        data = self.robot.data
        root_quat = data.root_quat_w

        # 1. Foot positions and orientations (world frame)
        foot_pos_w = data.body_pos_w[:, self.feet_bodies_idx, :]
        foot_ori_w = data.body_quat_w[:, self.feet_bodies_idx, :]


        # Store raw foot positions
        self.stance_foot_pos = foot_pos_w[:, self.stance_idx, :]
        self.stance_foot_ori = self.get_euler_from_quat(foot_ori_w[:, self.stance_idx, :])

        foot_lin_vel_w = data.body_lin_vel_w[:, self.feet_bodies_idx, :]
        foot_ang_vel_w = data.body_ang_vel_w[:, self.feet_bodies_idx, :]

        self.stance_foot_vel = foot_lin_vel_w[:,self.stance_idx,:]
        self.stance_foot_ang_vel = foot_ang_vel_w[:,self.stance_idx,:]

        base_pos = data.root_pos_w
        base_vel = data.root_lin_vel_w
        pelvis_ori = self.get_euler_from_quat(data.root_quat_w)
        pelvis_omega_local = _transfer_to_local_frame(data.root_ang_vel_w, self.stance_foot_ori_quat_0)

        #convert to euler rate?

        jt_pos = data.joint_pos
        jt_vel = data.joint_vel
        # 4. Assemble state vectors
        self.y_act = torch.cat([
            base_pos,
            pelvis_ori,
            jt_pos
        ], dim=-1)

        self.dy_act = torch.cat([
            base_vel,
            pelvis_omega_local,
            jt_vel,
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

