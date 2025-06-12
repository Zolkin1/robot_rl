import torch
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_from_euler_xyz,quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv, quat_apply
from .hlip_cmd import HLIPCommandTerm, euler_rates_to_omega, _transfer_to_global_frame, _transfer_to_local_frame
from .ref_gen import bezier_deg, calculate_cur_swing_foot_pos, HLIP
from .clf import CLF
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stair_cfg import StairHLIPCommandCfg

class StairCmd(HLIPCommandTerm):
    def __init__(self, cfg: "StairHLIPCommandCfg", env):
          super().__init__(cfg, env)
          self.z_height = torch.zeros((self.num_envs), device=self.device)
          self.stance_foot_box_z = torch.zeros((self.num_envs), device=self.device)
        
    def find_grid_idx(self, stance_pos_world, terrain_origins):
          H, W, _ = terrain_origins.shape
          B = stance_pos_world.shape[0]

          # compute squared XY-distances: [B, H, W]
          #   broadcast stance_pos_world over the H×W grid
          dist2 = (
               stance_pos_world[:, None, None, :2]  # [B,1,1,2]
               - terrain_origins[None, :, :, :2]    # [1,H,W,2]
          ).pow(2).sum(dim=-1)                    # [B,H,W]

          # flatten H×W → (H*W), find argmin per batch
          dist2_flat = dist2.view(B, -1)          # [B, H*W]
          idx_flat   = dist2_flat.argmin(dim=1)   # [B]

          # unravel flat index to 2D grid coords
          ix = idx_flat // W                      # rows
          iy = idx_flat %  W                      # cols

          return ix, iy
    

    def which_step(
              self,
     x: torch.Tensor,
     y: torch.Tensor,
     origin: torch.Tensor,
     cfg
     ) -> torch.LongTensor:
          """
          Batched version: for each (x, y), returns
               - -1           → outside all steps
               - 0..num_steps-1 → ring index (0 is outermost ring)
               - num_steps    → center platform
          Assumes cfg.holes == False.
          Shapes:
               x, y            → (B,) or broadcastable
               origin          → (2,) or (B, 2)
          """
          # 1) recompute num_steps
          n_x = (cfg.size[0] - 2*cfg.border_width - cfg.platform_width) // (2*cfg.step_width) + 1
          n_y = (cfg.size[1] - 2*cfg.border_width - cfg.platform_width) // (2*cfg.step_width) + 1
          num_steps = int(min(n_x, n_y))

          # 2) extract origin coords (broadcastable)
       
          ox, oy = origin[..., 0], origin[..., 1]

          # 3) compute local offsets
          dx = x - ox
          dy = y - oy
          abs_dx = dx.abs()
          abs_dy = dy.abs()

          # 4) half‐sizes of stepped region
          terrain_w = cfg.size[0] - 2*cfg.border_width
          terrain_h = cfg.size[1] - 2*cfg.border_width
          half_w = terrain_w / 2.0
          half_h = terrain_h / 2.0

          # 5) inward distance from outer edge
          delta_x = half_w - abs_dx
          delta_y = half_h - abs_dy
          delta   = torch.min(delta_x, delta_y)

          # 6) compute raw ring index
          step_w = torch.tensor(cfg.step_width, dtype=delta.dtype, device=delta.device)
          raw_k  = torch.floor(delta / step_w).long()

          #check if it's basically at the center, then return num_steps
          # center_idx = torch.where(delta_x < 0.01, torch.ones_like(raw_k), torch.zeros_like(raw_k))
         
          # 7) clamp into [-1, num_steps]
          return raw_k.clamp(min=0, max=num_steps)


    def box_center(
              self,
     x: torch.Tensor,
     y: torch.Tensor,
     origin: torch.Tensor,
     cfg
     ) -> torch.Tensor:
     """
     For each (x, y), returns the 3D center of the box it lies in:
          • outside → (nan, nan, nan)
          • ring k  → center of that ring’s face
          • center  → center platform
     Outputs (B,3). origin may be (3,) or (B,3).
     """
     # 1) get step_idx
     # we need the 2D origin for which_step
     origin_xy = origin[..., :2]
     step_idx = self.which_step(x, y, origin_xy, cfg)

     # recompute num_steps & heights
     n_x = (cfg.size[0] - 2*cfg.border_width - cfg.platform_width) // (2*cfg.step_width) + 1
     n_y = (cfg.size[1] - 2*cfg.border_width - cfg.platform_width) // (2*cfg.step_width) + 1
     num_steps = int(min(n_x, n_y))


     cx = origin[..., 0]
     cy = origin[..., 1]
     oz = origin[..., 2]

     # derive step_height & total_height
     step_h = -oz / (num_steps + 1)
     total_h = (num_steps + 1) * step_h
     cz = oz + total_h  # should be zero

     # half-dims and width tensors
     half_w = (cfg.size[0] - 2*cfg.border_width) / 2.0
     half_h = (cfg.size[1] - 2*cfg.border_width) / 2.0
     half_w = torch.tensor(half_w, device=x.device, dtype=x.dtype)
     half_h = torch.tensor(half_h, device=y.device, dtype=y.dtype)
     step_w = torch.tensor(cfg.step_width, device=x.device, dtype=x.dtype)

     # offsets
     dx = x - cx
     dy = y - cy
     adx = dx.abs()
     ady = dy.abs()

     # compute offset along face
     offset = (step_idx.float() + 0.5) * step_w

     # masks
     in_ring   = (step_idx >= 0) & (step_idx < num_steps)
     on_middle = (step_idx == num_steps)
     top    = in_ring & (ady >= adx) & (dy >  0)
     bottom = in_ring & (ady >= adx) & (dy <= 0)
     right  = in_ring & (adx >  ady) & (dx >  0)
     left   = in_ring & (adx >  ady) & (dx <= 0)

     # center coordinates
     cx_r = cx.expand_as(x)
     cy_r = cy.expand_as(y)
     center_x = torch.where(right,  cx_r + half_w - offset, cx_r)
     center_x = torch.where(left,   cx_r - half_w + offset, center_x)
     center_y = torch.where(top,    cy_r + half_h - offset, cy_r)
     center_y = torch.where(bottom, cy_r - half_h + offset, center_y)

     # z-centers
     # ring_z   = cz - total_h/2 + (step_idx.float()+1)*step_h/2
     # middle_z = oz + step_h/2
     # center_z = torch.where(in_ring, ring_z, torch.full_like(ring_z, float('nan')))
     # center_z = torch.where(on_middle, middle_z, center_z)

     effective_k = torch.clamp(step_idx+1, min=0, max=num_steps+1)
     surface_z   = - effective_k.float() * step_h
     surface_z   = torch.where(step_idx >= 0, surface_z,
                              torch.full_like(surface_z, float('nan')))

     return torch.stack([center_x, center_y, surface_z], dim=-1)

    def update_z_height(self, Ux: torch.Tensor, Uy: torch.Tensor) -> torch.Tensor:
          """
          Compute and return the stair height under a desired foot target, where Ux and Uy
          are offsets in the stance-foot frame. Analytically evaluates the MeshInvertedPyramid
          stair configuration without raycasts.

          Args:
               Ux (Tensor[N]): local X offsets in stance-foot frame
               Uy (Tensor[N]): local Y offsets in stance-foot frame

          Returns:
               height_under_foot (Tensor[N]): absolute world Z heights at each target
          """
          # 1) Terrain importer & configs
          terrain_importer = self.env.scene.terrain
          env_origins = terrain_importer.env_origins           # (N, 3) world-space origin per env
          terrain_origins = terrain_importer.terrain_origins   # (rows, cols, 3)
          cfg = self.env.cfg.scene.terrain.terrain_generator.sub_terrains['pyramid_stairs_inv']
          cell_x, cell_y = cfg.size

          # 2) Compute world-frame desired foot positions from stance-foot frame offsets
          #    Local offset in stance-foot frame (N,3)
          local_offsets = torch.stack([Ux, Uy, torch.zeros_like(Ux)], dim=-1)
          #    Rotate into world frame using stance-foot orientation
          #    assume self.stance_foot_quat_0 is (N,4) quaternion of stance foot in world
          desired_world = self.stance_foot_pos_0 + local_offsets           # (N,3)

          # 3) Map desired XY into terrain grid-local frame
          # local_xy = desired_world[:, :2] - env_origins[:, :2]       # (N,2)

          # 4) Determine subterrain cell indices
          idx_i, idx_j = self.find_grid_idx(self.stance_foot_pos_0, terrain_origins)

          # 5) Fetch each cell's world origin
          cell_origins = terrain_origins[idx_i, idx_j]               # (N,3)

          box_center = self.box_center(desired_world[:,0], desired_world[:,1], cell_origins, cfg)

          #height change relative to the initial height
          stance_foot_box_center = self.box_center(self.stance_foot_pos_0[:,0], self.stance_foot_pos_0[:,1], cell_origins, cfg)
          self.z_height = box_center[:, 2] - stance_foot_box_center[:, 2]
          self.stance_foot_box_z = stance_foot_box_center[:, 2]
          desired_world[:, 2] = box_center[:, 2]

          
          if torch.any(torch.isnan(self.z_height)):
               import pdb; pdb.set_trace()

          if self.cfg.debug_vis:
               self.footprint_visualizer.visualize(
                    translations=desired_world.detach().cpu().numpy(),
                    orientations=yaw_quat(self.robot.data.root_quat_w).detach().cpu().numpy(),
               )
               print(f"z_height: {self.z_height}, stance_foot_box_center: {self.stance_foot_box_z}, box_center: {box_center[:, 2]}")



    def generate_reference_trajectory(self):
          base_velocity = self.env.command_manager.get_command("base_velocity")  # (N,3)
          N = base_velocity.shape[0]
          T = torch.full((N,), self.T, dtype=torch.float32, device=self.device)

          Xdes, Ux, Ydes, Uy = self.hlip_controller.compute_orbit(
               T=T,cmd=base_velocity)

          Uy = Uy[:,self.stance_idx]
          Uy = torch.clamp(torch.abs(Uy), min=self.cfg.foot_target_range_y[0], max=self.cfg.foot_target_range_y[1]) * torch.sign(Uy)

          # based on the nominal step size, check the stair height
          self.update_z_height(Ux,Uy)

          #adjust the foot target if it's not completely on the step


          #select init and Xdes, Ux, Ydes, Uy
          x0 = self.hlip_controller.x_init
          y0 = self.hlip_controller.y_init[:,self.stance_idx]




          com_x, com_xd = self.hlip_controller._compute_desire_com_trajectory(
               cur_time=self.cur_swing_time,
               Xdesire=x0,
          )
          com_y, com_yd = self.hlip_controller._compute_desire_com_trajectory(
               cur_time=self.cur_swing_time,
               Xdesire=y0,
          )
          # Concatenate x and y components
          com_z = torch.ones((N,), device=self.device) * self.com_z + self.phase_var * self.z_height
          com_zd = torch.ones((N), device=self.device) * self.z_height/T
          com_pos_des = torch.stack([com_x, com_y,com_z], dim=-1)  # Shape: (N,2)
          com_vel_des = torch.stack([com_xd, com_yd,com_zd], dim=-1)  # Shape: (N,2)


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


          # clip foot target based on kinematic range
          self.foot_target = foot_target_yaw_adjusted[:,0:2]

          z_sw_max_tensor = self.cfg.z_sw_max + self.z_height
          z_sw_neg_tensor = self.cfg.z_sw_min + self.z_height

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
          z_init = torch.full((N,), 0.0, device=self.device)
          # Convert bht to tensor if it's not already
          bht_tensor = torch.tensor(bht, device=self.device) if not isinstance(bht, torch.Tensor) else bht

          sign = torch.sign(foot_target_yaw_adjusted[:, 1])
          foot_pos, sw_z = calculate_cur_swing_foot_pos(
               bht_tensor, z_init, z_sw_max_tensor, phase_var_tensor,self.swing2stance_foot_pos_0[:,0], sign*self.cfg.y_nom,T_tensor, z_sw_neg_tensor,
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


#     def _update_command(self):
    
#         super()._update_command()
        
        
        
        