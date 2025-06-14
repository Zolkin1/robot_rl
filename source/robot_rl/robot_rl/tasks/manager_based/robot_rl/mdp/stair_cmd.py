import torch,math
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
          self.tp = torch.zeros((self.num_envs), device=self.device)
          self.z_height = torch.zeros((self.num_envs), device=self.device)
          self.stance_foot_box_z = torch.zeros((self.num_envs), device=self.device)
          self.stance_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
          self.stance_foot_pos_0       = torch.zeros((self.num_envs, 3), device=self.device)
          self.stance_foot_ori_quat_0  = torch.zeros((self.num_envs, 4), device=self.device)
          self.stance_foot_ori_0       = torch.zeros((self.num_envs, 3), device=self.device)
          self.swing2stance_foot_pos_0 = torch.zeros((self.num_envs, 3), device=self.device)

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

     centers = torch.stack([center_x, center_y, surface_z], dim=-1)


     face_half_x = torch.where(
          (top|bottom),
          half_w - offset,              # full ring width in x is 2*(half_w-offset)
          torch.where((right|left),
                         step_w / 2,       # face thickness in x
                         cfg.platform_width / 2)  # center‐platform
          )
     face_half_y = torch.where(
          (top|bottom),
          step_w / 2,                   # face thickness in y
          torch.where((right|left),
                         half_h - offset,  # full ring width in y is 2*(half_h-offset)
                         cfg.platform_width / 2)
          )

     # --- now min/max bounds in world frame ---
     min_x = center_x - face_half_x
     max_x = center_x + face_half_x
     min_y = center_y - face_half_y
     max_y = center_y + face_half_y

     bounds_lo = torch.stack([min_x, min_y],     dim=-1)               # [B,2]
     bounds_hi = torch.stack([max_x, max_y],     dim=-1)               # [B,2]

     return centers, bounds_lo, bounds_hi

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

          box_center, box_bounds_lo, box_bounds_hi = self.box_center(desired_world[:,0], desired_world[:,1], cell_origins, cfg)

          #height change relative to the initial height
          stance_foot_box_center, stance_foot_box_bounds_lo, stance_foot_box_bounds_hi = self.box_center(self.stance_foot_pos_0[:,0], self.stance_foot_pos_0[:,1], cell_origins, cfg)
          self.z_height = box_center[:, 2] - stance_foot_box_center[:, 2]
          self.stance_foot_box_z = stance_foot_box_center[:, 2]
          self.target_foot_box_center = box_center
          self.target_foot_box_bounds_lo = box_bounds_lo
          self.target_foot_box_bounds_hi = box_bounds_hi

          desired_world[:, 2] = box_center[:, 2]

          
          if torch.any(torch.isnan(self.z_height)):
               import pdb; pdb.set_trace()

          if self.cfg.debug_vis:
               # self.footprint_visualizer.visualize(
               #      translations=desired_world.detach().cpu().numpy(),
               #      orientations=yaw_quat(self.robot.data.root_quat_w).detach().cpu().numpy(),
               # )
               print(f"z_height: {self.z_height}, stance_foot_box_center: {self.stance_foot_box_z}, box_center: {box_center[:, 2]}")


    def adjust_foot_target(
              self,
          foot_target: torch.Tensor,     # [B,3]
          toe_offset: float = 0.12,
          heel_offset: float = -0.05
          ) -> torch.Tensor:
          # unpack
          # 1) get per‐env bounds in yaw frame (assumed precomputed via box_center)
          #    target_foot_box_bounds_lo: [B,2] = [min_x, min_y]
          #    target_foot_box_bounds_hi: [B,2] = [max_x, max_y]
          min_x, min_y = self.target_foot_box_bounds_lo.unbind(-1)  # [B], [B]
          max_x, max_y = self.target_foot_box_bounds_hi.unbind(-1)

          foot_global = foot_target + self.stance_foot_pos_0


          toe_offset_tensor =torch.zeros_like(foot_global)
          heel_offset_tensor = torch.zeros_like(foot_global)

          toe_offset_tensor[:,0] = toe_offset
          heel_offset_tensor[:,0] = heel_offset     

          # 4) compute toe & heel positions in that frame
          toe_yaw  = foot_global + _transfer_to_global_frame(toe_offset_tensor, self.stance_foot_ori_quat_0)
          heel_yaw = foot_global + _transfer_to_global_frame(heel_offset_tensor, self.stance_foot_ori_quat_0)

          # 5) build the [low, high] intervals for each corner along X
          low_toe_x,  high_toe_x  = min_x - toe_yaw[:,0],  max_x - toe_yaw[:,0]
          low_heel_x, high_heel_x = min_x - heel_yaw[:,0], max_x - heel_yaw[:,0]

          # 6) intersect them: Δx ∈ [ max(low_toe, low_heel),  min(high_toe, high_heel) ]
          low_x, high_x = torch.max(low_toe_x, low_heel_x), torch.min(high_toe_x, high_heel_x)

          # 7) pick the shift closest to zero within that interval
          #    → if 0∈[low,high], dx=0; if both >0, dx=low; if both <0, dx=high
          zero = torch.zeros_like(low_x)
          dx = torch.where(
               low_x > 0, 
               low_x,
               torch.where(high_x < 0, high_x, zero)
          )

          # 8) same for Y
          low_toe_y,  high_toe_y  = min_y - toe_yaw[:,1],  max_y - toe_yaw[:,1]
          low_heel_y, high_heel_y = min_y - heel_yaw[:,1], max_y - heel_yaw[:,1]
          low_y, high_y = torch.max(low_toe_y, low_heel_y), torch.min(high_toe_y, high_heel_y)

          dy = torch.where(
               low_y > 0,
               low_y,
               torch.where(high_y < 0, high_y, zero)
          )

          
          delta_foot_target = torch.stack([dx, dy, torch.zeros_like(dx)], dim=-1)
          delta_yaw_adjusted = _transfer_to_local_frame(delta_foot_target, self.stance_foot_ori_quat_0)
          # 9) apply the minimal shift and rotate back
          foot_yaw = _transfer_to_local_frame(foot_target, self.stance_foot_ori_quat_0)
          foot_yaw[:,0] += delta_yaw_adjusted[:,0]
          foot_yaw[:,1] += delta_yaw_adjusted[:,1]

          if self.cfg.debug_vis:
               if torch.sum(dx) + torch.sum(dy) > 0.01:
                    print(f"foot_yaw: {foot_yaw}")
                    print(f"delta_foot_target: {delta_foot_target}")
                    print(f"delta_yaw_adjusted: {delta_yaw_adjusted}")

          return foot_yaw


    def update_Stance_Swing_idx(self):
          base_velocity = self.env.command_manager.get_command("base_velocity")  # (N,3)
          #check stair width
          cfg = self.env.cfg.scene.terrain.terrain_generator.sub_terrains['pyramid_stairs_inv']
          stair_width = cfg.step_width

          Tswing = stair_width / (base_velocity[:,0] + 0.001)
          self.T = Tswing
          #    Tswing = self.T - self.T_ds
          self.tp = (self.env.sim.current_time % (2*Tswing)) / (2*Tswing)  
          phi_c = torch.sin(2 * torch.pi * self.tp) / torch.sqrt(torch.sin(2 * torch.pi * self.tp)**2 + self.T)



        # 1) compute new stance & swing indices
          new_stance_idx = (phi_c < 0).long()     # or however you get it
          self.swing_idx = 1 - new_stance_idx

          # 2) grab world-frame feet
          foot_pos_w = self.robot.data.body_pos_w[:, self.feet_bodies_idx, :]   # [B,2,3]
          foot_ori_w = self.robot.data.body_quat_w[:, self.feet_bodies_idx, :]  # [B,2,4]
          batch      = torch.arange(self.num_envs, device=foot_pos_w.device)

          # 3) pick out the “candidate” new pose/quaternion/euler/swing2stance
          pos      = foot_pos_w[batch, new_stance_idx]             # [B,3]
          quat     = foot_ori_w[batch, new_stance_idx]             # [B,4]
          euler    = self.get_euler_from_quat(quat)               # [B,3]
          rel      = foot_pos_w[batch, self.swing_idx] - pos      # [B,3]
          s2s      = _transfer_to_local_frame(rel, quat)          # [B,3]

          # 4) build a mask of “which envs actually flipped stance”
          changed = new_stance_idx != self.stance_idx              # [B] bool

          # 5) stash the new stance_idx
          self.stance_idx = new_stance_idx

          # 6) **only** overwrite those entries that changed
          self.stance_foot_pos_0      [changed] = pos      [changed]
          self.stance_foot_ori_quat_0 [changed] = quat     [changed]
          self.stance_foot_ori_0      [changed] = euler    [changed]
          self.swing2stance_foot_pos_0[changed] = s2s      [changed]


          self.phase_var = torch.where(
               self.tp < 0.5,
               2.0 * self.tp,
               2.0 * self.tp - 1.0
               )
          self.cur_swing_time = self.phase_var*Tswing


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
  
        ref     = amp * sign * torch.sin(phase.unsqueeze(-1) + offset) + joint_offset

        # velocity
        dphase_dt = 2 * torch.pi / (2*(self.T - self.T_ds))  # scalar
        ref_dot = amp * sign * torch.cos(phase.unsqueeze(-1) + offset) * dphase_dt.unsqueeze(-1)

        return ref, ref_dot
    def get_actual_state(self):
        """Populate actual state and its time derivative in the robot's local (yaw-aligned) frame."""
        # Convenience
        data = self.robot.data
        root_quat = data.root_quat_w
        batch_idx = torch.arange(self.num_envs, device=self.device)

        # 1. Foot positions and orientations (world frame)
        foot_pos_w = data.body_pos_w[:, self.feet_bodies_idx, :]
        foot_ori_w = data.body_quat_w[:, self.feet_bodies_idx, :]



        # Store raw foot positions
        self.stance_foot_pos = foot_pos_w[batch_idx, self.stance_idx, :]
        self.stance_foot_ori = self.get_euler_from_quat(foot_ori_w[batch_idx, self.stance_idx, :])

        # Convert foot positions to the robot's yaw-aligned local frame
        # stance_pos_local = _transfer_to_local_frame(
        #     foot_pos_w[:, self.stance_idx, :], root_quat
        # )
        swing2stance_local = _transfer_to_local_frame(
            foot_pos_w[batch_idx, self.swing_idx, :]-self.stance_foot_pos_0, self.stance_foot_ori_quat_0
        )

        # Center of mass to stance foot vector in local frame
        com_w = data.root_com_pos_w
        com2stance_local = _transfer_to_local_frame(
            com_w - self.stance_foot_pos_0, self.stance_foot_ori_quat_0
        )


        # Pelvis orientation (Euler XYZ)
        pelvis_ori = self.get_euler_from_quat(root_quat)

        # Foot orientations (Euler XYZ)
        swing_foot_ori = self.get_euler_from_quat(foot_ori_w[batch_idx,self.swing_idx,:])

        # 2. Velocities (world frame)
        com_vel_w = data.root_com_vel_w[:,0:3]
        # pelvis_omega_w = data.root_ang_vel_w
        foot_lin_vel_w = data.body_lin_vel_w[:, self.feet_bodies_idx, :]
        foot_ang_vel_w = data.body_ang_vel_w[:, self.feet_bodies_idx, :]

        self.stance_foot_vel = foot_lin_vel_w[batch_idx,self.stance_idx,:]
        self.stance_foot_ang_vel = foot_ang_vel_w[batch_idx,self.stance_idx,:]
        # Convert velocities to local frame
        # import pdb; pdb.set_trace()
        com_vel_local = _transfer_to_local_frame(com_vel_w, self.stance_foot_ori_quat_0)
      
        pelvis_omega_local = data.root_ang_vel_b
        # foot_lin_vel_local_stance = _transfer_to_local_frame(
        #     foot_lin_vel_w[:,self.stance_idx,:], self.stance_foot_ori_quat_0
        # )
        foot_lin_vel_local_swing = _transfer_to_local_frame(
            foot_lin_vel_w[batch_idx,self.swing_idx,:], self.stance_foot_ori_quat_0
        )

        foot_ang_vel_local_swing =quat_apply(quat_inv(foot_ori_w[batch_idx,self.swing_idx,:]), foot_ang_vel_w[batch_idx,self.swing_idx,:])
        
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

    def generate_orientation_ref(self, base_velocity,N):
        pelvis_euler = torch.zeros((N,3), device=self.device)
        tp_tensor = self.tp
        phase_tensor = self.phase_var
        
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
          T = self.T
          batch_idx = torch.arange(self.num_envs, device=self.device) 
          Xdes, Ux, Ydes, Uy = self.hlip_controller.compute_orbit(
               T=T,cmd=base_velocity)

          Uy = Uy[batch_idx,self.stance_idx]
          


          foot_target = torch.stack([Ux,Uy,torch.zeros((N), device=self.device)], dim=-1)

          # based on yaw velocity, update com_pos_des, com_vel_des, foot_target,
          delta_psi = base_velocity[:,2] * self.cur_swing_time
          q_delta_yaw = quat_from_euler_xyz(
               torch.zeros_like(delta_psi),               # roll=0
               torch.zeros_like(delta_psi),               # pitch=0
               delta_psi                                  # yaw=Δψ
          ) 

          foot_target_yaw_adjusted = quat_apply(q_delta_yaw, foot_target)  # [B,3]
          self.update_z_height(foot_target_yaw_adjusted[:,0], foot_target_yaw_adjusted[:,1])
          
          #transform it into the global frame
          # foot_target_global_yaw_frame = _transfer_to_global_frame(foot_target_yaw_adjusted, self.stance_foot_ori_quat_0)

          
          # based on the nominal step size, check the stair height
          # self.update_z_height(foot_target_global_yaw_frame[:,0], foot_target_global_yaw_frame[:,1])

         
          # foot_target_yaw_adjusted = self.adjust_foot_target(foot_target_global_yaw_frame)


          #clip based on the kinematics range
          foot_target_yaw_adjusted[:,1] = torch.sign(Uy) * torch.clamp(torch.abs(foot_target_yaw_adjusted[:,1]), min=self.cfg.foot_target_range_y[0], max=self.cfg.foot_target_range_y[1])

          


          #select init and Xdes, Ux, Ydes, Uy
          x0 = self.hlip_controller.x_init
          y0 = self.hlip_controller.y_init[batch_idx,self.stance_idx]




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


          
          com_pos_des_yaw_adjusted = quat_apply(q_delta_yaw, com_pos_des)  # [B,3]
          com_vel_des_yaw_adjusted = quat_apply(q_delta_yaw, com_vel_des)  # [B,3]


          # clip foot target based on kinematic range
          self.foot_target = foot_target_yaw_adjusted[:,0:2]

          z_sw_max_tensor = self.cfg.z_sw_max + self.z_height
          z_sw_neg_tensor = self.cfg.z_sw_min + self.z_height

          # Create horizontal control points with batch dimension
          horizontal_control_points = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=self.device).repeat(N, 1)  # Shape: (N, 5)

          # Create tensors with batch dimension N
          phase_var_tensor = self.phase_var
          T_tensor = self.T
          five_tensor = torch.tensor(5, device=self.device)

          bht = bezier_deg(
               0,phase_var_tensor, T_tensor, horizontal_control_points, five_tensor
          )

          # Convert scalar parameters to tensors with batch dimension N

          z_init = self.swing2stance_foot_pos_0[:,2]
          # Convert bht to tensor if it's not already
          bht_tensor = torch.tensor(bht, device=self.device) if not isinstance(bht, torch.Tensor) else bht

          sign = torch.sign(foot_target_yaw_adjusted[:, 1])
          foot_pos, sw_z = calculate_cur_swing_foot_pos(
               bht_tensor, z_init, z_sw_max_tensor, phase_var_tensor,-foot_target_yaw_adjusted[:, 0], sign*self.cfg.y_nom,T_tensor, z_sw_neg_tensor,
               foot_target_yaw_adjusted[:, 0], foot_target_yaw_adjusted[:, 1]
          )


          dbht = bezier_deg(1, phase_var_tensor, T_tensor, horizontal_control_points, five_tensor)
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
        
        
        
        