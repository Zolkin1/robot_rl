import numpy as np
import re
import torch
import warp as wp
import time
from isaaclab.managers import CommandTerm

from .clf import CLF
from .library_manager import LibraryManager
from .sagittal_reflector import NamedReflector, SagittalReflectionConfig
from .trajectory_manager import TrajectoryManager
from .trajectory_manager import TrajectoryType

from isaaclab.utils.math import wrap_to_pi, quat_apply, quat_mul, quat_from_euler_xyz,euler_xyz_from_quat, wrap_to_pi, yaw_quat, quat_inv

class TrajectoryCommand(CommandTerm):
    """Trajectory command term. This keeps track of the underlying single trajectory or library as well as CLF for tracking."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.env = env
        self.robot = env.scene[cfg.asset_name]

        if cfg.heuristic_func is not None:
            self.set_user_heuristic(cfg.heuristic_func)
        else:
            self.user_heuristic = None

        # Expand wildcards in contact frames
        self.contact_bodies = self._expand_wildcard_frames(cfg.contact_bodies)

        # Extract the index into the robot data bodies for the contact frames
        self.contact_frame_indices = torch.zeros(len(self.contact_bodies), dtype=torch.long, device=self.device)
        for i, frame_name in enumerate(self.contact_bodies):
            if frame_name in self.robot.body_names:
                self.contact_frame_indices[i] = self.robot.body_names.index(frame_name)
            else:
                raise ValueError(f"Contact frame '{frame_name}' not found in robot body names.")

        self.current_contact_poses = torch.zeros(self.num_envs, len(self.contact_bodies), 6, dtype=torch.float, device=self.device)
        self.current_contact_vels = torch.zeros(self.num_envs, len(self.contact_bodies), 6, dtype=torch.float, device=self.device)
        self.desired_contact_poses = torch.zeros(self.num_envs, len(self.contact_bodies), 6, dtype=torch.float, device=self.device)

        self.manager_type = cfg.manager_type
        self.conditioner_generator = cfg.conditioner_generator_name

        # Create trajectory/library manager
        if cfg.manager_type == "trajectory":
            self.manager = TrajectoryManager(cfg.path, cfg.hf_repo, env.device)
            self.trajectory_type = self.manager.traj_data.trajectory_type
        elif cfg.manager_type == "library":
            self.manager = LibraryManager(cfg.path, cfg.hf_repo, env.device,
                                           env=env, conditioner_generator_name=cfg.conditioner_generator_name)
            self.trajectory_type = self.manager.trajectory_type
        else:
            raise NotImplementedError(f"Manager Type {cfg.manager_type} is not implemented!")

        self.verify_contact_frames()

        # Hold a list of current domains for each env
        self.current_domain = -1 * torch.ones(self.num_envs, dtype=torch.long, device=self.device)

        # Create a list of indices to be used
        # Parse outputs using position output names (superset that includes ori_w)
        result = self._parse_outputs(self.manager.get_pos_output_names)
        self.joint_idx, self.body_idx, self.use_com, self.ordered_pos_output_names, self.ordered_vel_output_names, self.body_type = result

        self.y_act = torch.zeros((self.num_envs, len(self.ordered_pos_output_names)), device=self.device)
        self.dy_act = torch.zeros((self.num_envs, len(self.ordered_vel_output_names)), device=self.device)

        self.y_des = torch.zeros((self.num_envs, len(self.ordered_pos_output_names)), device=self.device)
        self.dy_des = torch.zeros((self.num_envs, len(self.ordered_vel_output_names)), device=self.device)

        # Order the manager to match the order here (pass both pos and vel names)
        self.manager.order_outputs(self.ordered_pos_output_names, self.ordered_vel_output_names)

        self.body_type = torch.tensor(self.body_type, dtype=torch.int, device=self.device)

        # --- Precomputed reflectors for sagittal symmetry -----------------
        _rcfg = SagittalReflectionConfig()
        self._traj_pos_reflector = NamedReflector(_rcfg, self.ordered_pos_output_names, self.device)
        self._traj_vel_reflector = NamedReflector(_rcfg, self.ordered_vel_output_names, self.device)
        self._contact_reflector = NamedReflector(_rcfg, self.contact_bodies, self.device)

        print(f"Ordered pos output names: {self.ordered_pos_output_names}")
        print(f"Ordered vel output names: {self.ordered_vel_output_names}")

        self.time_offset = torch.zeros(self.num_envs, device=self.device)

        # For now assuming that all bodies have a yaw tracking
        # Use velocity output names since CLF uses velocity dimensions
        # self.yaw_output_idxs = []
        # for i, name in enumerate(self.ordered_vel_output_names):
        #     if "ori_z" in name:
        #         self.yaw_output_idxs.append(i)

        # Create a mapping from vel output indices to pos output indices
        # This is needed to extract position values that match velocity dimensions
        # (pos includes ori_w, vel excludes it)
        self.vel_to_pos_idx = torch.zeros(len(self.ordered_vel_output_names), dtype=torch.long, device=self.device)
        for i, vel_name in enumerate(self.ordered_vel_output_names):
            if vel_name in self.ordered_pos_output_names:
                self.vel_to_pos_idx[i] = self.ordered_pos_output_names.index(vel_name)
            else:
                raise ValueError(f"Velocity output name '{vel_name}' not found in position output names.")

        # Create a list of indices for the reference frames
        self.ref_frame_indices, self.ref_frames = self._parse_ref_frames(self.manager.get_reference_frames())

        # Verify all reference frames are in contact frames
        for ref_frame in self.ref_frames:
            if ref_frame not in self.contact_bodies:
                raise ValueError(f"Reference frame '{ref_frame}' is not in the contact frames list: {self.contact_bodies}")

        # Create a mapping from ref_frames to contact_frames indices
        # This allows us to map ref_frame_indices to contact_state indices
        self.ref_to_contact_idx = torch.zeros(len(self.ref_frames), dtype=torch.long, device=self.device)
        for i, ref_frame in enumerate(self.ref_frames):
            self.ref_to_contact_idx[i] = self.contact_bodies.index(ref_frame)

        # Current reference frame poses — [N, [pos_x, pos_y, pos_z, qx, qy, qz, qw]]
        # Initialize quaternion to identity so quat_inv doesn't produce NaN before first update.
        self.ref_poses = torch.zeros((self.num_envs, 7), device=self.device)
        self.ref_poses[:, 6] = 1.0  # qw=1 identity quaternion

        # Create CLF using velocity output names (which exclude ori_w)
        self.clf = CLF(
            sim_dt=self.env.cfg.sim.dt,
            batch_size=self.num_envs,
            ordered_vel_output_names=self.ordered_vel_output_names,
            ordered_pos_output_names=self.ordered_pos_output_names,
            Q_weights=self.cfg.Q_weights,
            R_weights=self.cfg.R_weights,
            device=self.device
        )

        self.phasing_var = torch.zeros(self.num_envs, device=self.device)
        self.unmasked_phasing_var = torch.zeros(self.num_envs, device=self.device)
        self.prev_unmasked_phasing_var = torch.zeros(self.num_envs, device=self.device)
        self.hold_envs = torch.ones(self.num_envs, device=self.device)

        # State for phasing variable hold logic (hold at second boundary, not first)
        self.should_hold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.boundaries_crossed = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.hold_phi_value = -1.0 * torch.ones(self.num_envs, device=self.device)  # -1 means not locked

        self.get_measured_output_time = 0.0
        self.get_desired_output_time = 0.0
        self.vdot_time = 0.0

        self.init_time_offset = torch.zeros(self.num_envs, device=self.device)

        print(f"Body names: {self.robot.body_names}")

    def update_phasing_var(self, t: torch.Tensor, env_ids: torch.Tensor = None):
        """Get the phasing variable for the current trajectory.

        Holds phi at the second boundary crossing (0.0 or 0.5) rather than the first,
        allowing a full phase to complete before stopping when velocity is low.

        Args:
            t: Time tensor of shape [N].
            env_ids: Optional environment indices of shape [N]. If provided, only update
                the phasing var for those environments (skips hold logic).

        Returns:
            Phasing variable tensor of shape [N].
        """
        if env_ids is not None:
            # Subset update - just update phasing var, skip hold logic
            new_phi = self.manager.get_phasing_var(t, env_ids)
            self.phasing_var[env_ids] = new_phi
            self.unmasked_phasing_var[env_ids] = new_phi
            return new_phi

        # Full update with hold/boundary tracking
        prev_phi = self.phasing_var
        self.prev_unmasked_phasing_var = self.unmasked_phasing_var
        self.phasing_var = self.manager.get_phasing_var(t)
        self.unmasked_phasing_var = self.phasing_var

        # Determine which envs should hold
        cmd_vel = self.env.command_manager.get_command("base_velocity")
        prev_should_hold = self.should_hold.clone()
        self.should_hold = torch.abs(cmd_vel[:, 0]) < self.cfg.hold_phi_threshold

        # Reset tracking on newly holding envs or episode resets
        newly_holding = self.should_hold & ~prev_should_hold
        reset_mask = newly_holding | (self.env.episode_length_buf == 0)
        self.boundaries_crossed[reset_mask] = 0
        self.hold_phi_value[reset_mask] = -1.0

        # Release hold when no longer should hold
        released = ~self.should_hold
        self.hold_phi_value[released] = -1.0
        self.boundaries_crossed[released] = 0

        # Detect boundary crossings (only for envs that should hold but aren't locked yet)
        active = self.should_hold & (self.hold_phi_value < 0)

        crosses_zero = active & (self.phasing_var < prev_phi) & (prev_phi > 0)
        crosses_half = active & (prev_phi < 0.5) & (self.phasing_var >= 0.5)

        crosses_any = crosses_zero | crosses_half
        self.boundaries_crossed[crosses_any] += 1

        # Lock hold on the second boundary crossing
        lock_at_zero = crosses_zero & (self.boundaries_crossed >= self.cfg.phasing_boundaries)
        lock_at_half = crosses_half & (self.boundaries_crossed >= self.cfg.phasing_boundaries)
        self.hold_phi_value[lock_at_zero] = 0.0
        self.hold_phi_value[lock_at_half] = 0.5

        # Apply hold for all locked envs
        holding = self.hold_phi_value >= 0
        self.phasing_var[holding] = self.hold_phi_value[holding]

        return self.phasing_var

    def get_phasing_var(self) -> torch.Tensor:
        """Get the phasing variable for the current trajectory.

        Args:

        Returns:
            Phasing variable tensor of shape [N].
        """

        return self.phasing_var

    def _expand_wildcard_frames(self, frame_patterns: list[str]) -> list[str]:
        """
        Expand wildcard patterns in contact frame names.

        Args:
            frame_patterns: List of frame names that may contain wildcards (e.g., ".*_ankle_roll_link")

        Returns:
            List of explicit frame names with wildcards expanded
        """
        expanded_frames = []

        # Get all body names from the robot
        body_names = self.robot.body_names

        for pattern in frame_patterns:
            # Check if the pattern contains wildcards (. or *)
            if '*' in pattern or '.*' in pattern:
                # Convert glob-style pattern to regex
                # Replace .* with .* (already regex), and * with .*
                regex_pattern = pattern.replace('*', '.*') if not '.*' in pattern else pattern
                regex_pattern = f'^{regex_pattern}$'

                # Find all matching body names
                matched = False
                for body_name in body_names:
                    if re.match(regex_pattern, body_name):
                        expanded_frames.append(body_name)
                        matched = True

                if not matched:
                    raise ValueError(f"Wildcard pattern '{pattern}' did not match any body names in the robot.")
            else:
                # No wildcard, add as-is
                expanded_frames.append(pattern)

        return expanded_frames

    def get_contact_state(self, t: torch.Tensor, env_ids: torch.Tensor = None):
        """Gets the desired contact state at the given time for the specified contact point.

        Args:
            t: Shape [N] the times in each environment.
            env_ids: Optional environment indices of shape [N]. If provided, only compute
                for those environments.

        Returns:
            Contact states of shape [N, num_contacts]. A tensor of binary values
            with a 1 indicating in contact and 0 otherwise.
        """
        return self.manager.get_contact_state(t, self.contact_bodies, env_ids)

    def get_symmetric_contacts(self, contacts: torch.Tensor) -> torch.Tensor:
        """Return the left-right symmetric reflection of *contacts*.

        Args:
            contacts: ``[N, num_contacts]`` contact states.

        Returns:
            Symmetric contacts of the same shape.
        """
        return self._contact_reflector.reflect(contacts)


    def get_trajectory_type(self):
        """Gets the type of trajectory: periodic or episodic."""
        return self.trajectory_type

    def verify_contact_frames(self):
        traj_frames = []
        if self.manager_type == "trajectory":
            for domain in self.manager.traj_data.domain_data.values():
                traj_frames.append(domain.contact_bodies)
        elif self.manager_type == "library":
            for manager in self.manager.trajectory_managers:
                for domain in manager.traj_data.domain_data.values():
                    traj_frames.append(domain.contact_bodies)
        else:
            raise NotImplementedError(f"Manager Type {self.manager_type} is not implemented!")

        # Verify that every frame in traj_frames appears in self.contact_frames
        for frames in traj_frames:
            for frame in frames:
                if frame not in self.contact_bodies:
                    raise ValueError(f"Contact frame {frame} from a trajectory is not in the contact frames list!")

    def get_ref_frame_poses(self) -> torch.Tensor:
        """
        Get the reference frame poses.

        Returns
            poses is shape [N, num_ref_frames]
        """
        poses = torch.zeros(self.num_envs, len(self.ref_frame_indices), 7, device=self.device)

        poses[:, :, :3] = wp.to_torch(self.robot.data.body_pos_w)[:, self.ref_frame_indices]
        poses[:, :, 3:] = wp.to_torch(self.robot.data.body_quat_w)[:, self.ref_frame_indices]

        return poses

    def get_contact_poses(self, contact_state: torch.Tensor) -> torch.Tensor:
        """
        Determine the pose of each frame that is in contact.

        The idea here will be to always grab the pose of all the frames but then mask it with 0's when out of contact

        Args:
            contact_state: shape [N, num_contacts]

        Returns:
            poses is shape [N, num_contacts, 6]. Everything not in contact is masked to 0
        """

        poses = torch.zeros(self.num_envs, len(self.contact_bodies), 6, device=self.device)

        not_in_contact = contact_state == 0

        # Get the poses of all the possible contact bodies
        poses[:, :, :3] = wp.to_torch(self.robot.data.body_pos_w)[:, self.contact_frame_indices, :]

        # Batch euler conversion for all contact bodies at once
        N = self.num_envs
        C = len(self.contact_bodies)
        all_quats = wp.to_torch(self.robot.data.body_quat_w)[:, self.contact_frame_indices, :]  # [N, C, 4]
        poses[:, :, 3:] = get_euler_from_quat(all_quats.reshape(N * C, 4)).reshape(N, C, 3)

        # Now mask
        poses[not_in_contact, :] *= 0

        return poses

    def get_desired_contact_poses(self, changed: torch.Tensor, current_poses: torch.Tensor) -> torch.Tensor:
        """
        Get the desired contact poses. This is always the pose of the frame when it first makes contact.

        Mask all the current poses by if they are in contact and if the domain just changed (because we want where we just made contact)

        TODO: Need to consider what if we want to rotate but not translate?
        """

        # Check to make sure that there is at least one True in changed
        if torch.any(changed):
            self.desired_contact_poses[changed] = current_poses[changed]

        return self.desired_contact_poses

    def get_contact_vels(self, contact_state: torch.Tensor) -> torch.Tensor:
        """
        Get the velocity of each frame that is in contact.

        Args:
            contact_state: shape [N, num_contacts]

        Returns:
            vels is shape [N, num_contacts, 6]. Everything not in contact is masked to 0
        """
        vels = torch.zeros(self.num_envs, len(self.contact_bodies), 6, device=self.device)

        not_in_contact = contact_state == 0

        vels[:, :, :3] = wp.to_torch(self.robot.data.body_lin_vel_w)[:, self.contact_frame_indices, :]
        vels[:, :, 3:] = wp.to_torch(self.robot.data.body_ang_vel_w)[:, self.contact_frame_indices, :]

        # Now mask
        vels[not_in_contact, :] *= 0

        return vels

    def get_measured_outputs(self, t: torch.Tensor, env_ids: torch.Tensor = None):
        """
        Get the measured state then compute the measured outputs.

        Args:
            t: Time tensor of shape [N].
            env_ids: Optional environment indices of shape [N]. If provided, only update
                state for those environments.
        """
        ref_poses = self.get_ref_frame_poses()        # Get the pose of every frame that should be in contact

        # TODO: For now assume that reference frames are always contact bodies.
        #   Then only updated the reference frame if it changes domain into a domain with contact.
        #   Each env needs its own reference frame, but it only ever needs one at a time.
        #   If there is a half periodic trajectory then assume that it switches to the other (left or right) frame.
        #   The self.ref_poses should be of shape [N, 7] and should just be holding the current in use reference frame.

        # Get the current domains
        new_domains = self.manager.get_current_domains(t, env_ids)

        # Check if the domains changed
        if env_ids is None:
            changed = new_domains != self.current_domain
        else:
            changed = new_domains != self.current_domain[env_ids]

        # Note that domains never change for full periodic single domain trajectories, but I think we still want to update the position.
        # The two options are (1) continually updating the position while in contact, (2) try to check based on the phasing variable

        # So if we are in a standing then update the reference position at the normal stepping cadence
        if env_ids is None:
            single_dom_mask = (self.manager.get_num_domains() == 1) & ((self.prev_unmasked_phasing_var > self.unmasked_phasing_var) |
                                                                       ((self.prev_unmasked_phasing_var < 0.5) & (0.5 < self.unmasked_phasing_var)))
            changed[single_dom_mask] = True

            # Update the list of current domains
            self.current_domain = new_domains
        else:
            single_dom_mask = (self.manager.get_num_domains()[env_ids] == 1) & \
                              ((self.prev_unmasked_phasing_var[env_ids] > self.unmasked_phasing_var[env_ids]) |
                               ((self.prev_unmasked_phasing_var[env_ids] < 0.5) & (0.5 < self.unmasked_phasing_var[env_ids])))
            changed[single_dom_mask] = True

            # Update the list of current domains for the subset
            self.current_domain[env_ids] = new_domains

        # Determine which reference frames/bodies are in contact
        contact_state = self.get_contact_state(t, env_ids)  # Shape: [N, num_contact_frames]
        # print(f"contact_state: {contact_state}, contact_frames: {self.contact_bodies}")

        if env_ids is None:
            self.current_contact_poses = self.get_contact_poses(contact_state)
            self.current_contact_vels = self.get_contact_vels(contact_state)
            self.desired_contact_poses = self.get_desired_contact_poses(changed, self.current_contact_poses)

            # Get the indices into self.ref_frames for the reference frame in use for each env
            ref_frame_indices = self.manager.get_ref_frames_in_use(t, self.ref_frames)  # Shape: [N]

            # Map ref_frame_indices to contact_state indices
            contact_frame_indices = self.ref_to_contact_idx[ref_frame_indices]  # Shape: [N]

            # Check if the reference frames are in contact
            ref_frames_in_contact = torch.gather(contact_state, 1, contact_frame_indices.unsqueeze(1)).squeeze(1)

            # Now index only envs where we are in contact and domains changed
            changed_and_contact = changed & (ref_frames_in_contact > 0)

            # Get the correct reference frame to pass
            if torch.any(changed_and_contact):
                env_indices = torch.where(changed_and_contact)[0]
                self.ref_poses[env_indices, :] = ref_poses[env_indices, ref_frame_indices[env_indices], :]

            # Compute the measured outputs
            self.compute_measured_output(self.ref_poses[:, :3], self.ref_poses[:, 3:])
        else:
            # Subset update - simplified version for reset/debug use case
            # For subset updates, we skip contact pose/vel updates (they require full-sized tensors)
            # and just update ref_poses and compute measured outputs

            # Get the indices into self.ref_frames for the reference frame in use for subset
            ref_frame_indices = self.manager.get_ref_frames_in_use(t, self.ref_frames, env_ids)

            # Map ref_frame_indices to contact_state indices
            contact_frame_indices = self.ref_to_contact_idx[ref_frame_indices]

            # Check if the reference frames are in contact
            ref_frames_in_contact = torch.gather(contact_state, 1, contact_frame_indices.unsqueeze(1)).squeeze(1)

            # Now index only envs where we are in contact and domains changed
            changed_and_contact = changed & (ref_frames_in_contact > 0)

            # Get the correct reference frame to pass
            if torch.any(changed_and_contact):
                subset_indices = torch.where(changed_and_contact)[0]
                global_env_indices = env_ids[subset_indices]
                self.ref_poses[global_env_indices, :] = ref_poses[global_env_indices, ref_frame_indices[subset_indices], :]

            # Compute the measured outputs for all envs (using updated ref_poses)
            self.compute_measured_output(self.ref_poses[:, :3], self.ref_poses[:, 3:])


    def compute_measured_output(self, ref_frame_pos_w, ref_frame_quat) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the measured state."""
        # TODO: Make more general. For now assuming CoM then bodies then joints always

        # TODO: Make sure that the velocities are computed in the LOCAL_WORLD_ALIGNED frame
        # TODO: Make sure positions are in the stance foot frame

        pos_output_idx = 0
        vel_output_idx = 0

        # Get the relevant end effector positions (in global frame)
        if self.use_com:
            # Deal with CoM as a special case
            com_pos_w = wp.to_torch(self.robot.data.root_com_pos_w)
            com_vel_w = wp.to_torch(self.robot.data.root_com_vel_w)[:, :3]

            # Put into the reference frame
            com_pos_local = _align_yaw(com_pos_w - ref_frame_pos_w, ref_frame_quat)
            # import pdb; pdb.set_trace()
            com_vel_local = _align_yaw(com_vel_w, ref_frame_quat)

            self.y_act[:, pos_output_idx:pos_output_idx+3] = com_pos_local
            self.dy_act[:, vel_output_idx:vel_output_idx+3] = com_vel_local
            pos_output_idx += 3
            vel_output_idx += 3

        def _get_pos_ori_vel_relative(ref_frame_pos_w, ref_frame_quat_w, frame_pos_w, frame_quat_w,
                                      frame_lin_vel_w, frame_ang_vel_w):
            """
            Compute the position, orientation, and velocity relative to the reference frame.

            All positions are in the yaw-aligned frame centered at the reference frame.
            All orientations are in the yaw-aligned frame.
            All velocities are in the yaw-aligned frame.

            Args:
                ref_frame_pos_w: tensor of shape [N, 3]
                ref_frame_quat_w: tensor of shape [N, 4]
                frame_pos_w: tensor of shape [N, num_bodies, 3]
                frame_quat_w: tensor of shape [N, num_bodies, 4]
                frame_lin_vel_w: tensor of shape [N, num_bodies, 3]
                frame_ang_vel_w: tensor of shape [N, num_bodies, 3]

            Returns:
                frame_pos_rel_yaw_aligned: tensor of shape [N, num_bodies, 3]
                frame_ori_yaw_aligned: tensor of shape [N, num_bodies, 4]
                frame_vel_yaw_aligned: tensor of shape [N, num_bodies, 3]
                frame_ang_vel_local: tensor of shape [N, num_bodies, 3]
            """
            # Positions: translate then yaw-align (all bodies at once)
            frame_pos_rel_world_aligned = frame_pos_w - ref_frame_pos_w.unsqueeze(1)
            frame_pos_rel_yaw_aligned = _align_yaw_batched(frame_pos_rel_world_aligned, ref_frame_quat_w)

            # Orientations: yaw-align quaternions (all bodies at once)
            frame_ori_yaw_aligned = _align_quat_to_yaw_batched(frame_quat_w, ref_frame_quat_w)

            # Velocities: yaw-align (all bodies at once)
            frame_vel_yaw_aligned = _align_yaw_batched(frame_lin_vel_w, ref_frame_quat)
            frame_ang_vel_local = _align_yaw_batched(frame_ang_vel_w, ref_frame_quat)

            return frame_pos_rel_yaw_aligned, frame_ori_yaw_aligned, frame_vel_yaw_aligned, frame_ang_vel_local

        # Bodies
        if self.body_idx is not None:
            frame_pos = wp.to_torch(self.robot.data.body_pos_w)[:, self.body_idx, :]
            frame_quat = wp.to_torch(self.robot.data.body_quat_w)[:, self.body_idx, :]

            # Get the frame vels, not the COM vels
            body_link_vel = wp.to_torch(self.robot.data.body_link_vel_w)
            frame_lin_vel_w = body_link_vel[:, self.body_idx, :3]
            frame_ang_vel_w = body_link_vel[:, self.body_idx, 3:]

            # These pull the COM velocities in the world frame - want the frame velocities
            # frame_lin_vel_w = self.robot.data.body_lin_vel_w[:, self.body_idx, :]
            # frame_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.body_idx, :]

            body_pos_local, body_ori_local, body_vel_local, body_ang_vel_local = _get_pos_ori_vel_relative(
                ref_frame_pos_w, ref_frame_quat, frame_pos, frame_quat, frame_lin_vel_w, frame_ang_vel_w)

            pos_bodies = (self.body_type == 1) | (self.body_type == 2)
            num_pos_bodies = pos_bodies.sum()
            ori_bodies = (self.body_type == 0) | (self.body_type == 2)
            num_ori_bodies = ori_bodies.sum()

            # Add linear
            self.y_act[:, pos_output_idx:pos_output_idx+(3*num_pos_bodies)] = (
                body_pos_local[:, pos_bodies, :].flatten(1))
            self.dy_act[:, vel_output_idx:vel_output_idx+(3*num_pos_bodies)] = (
                body_vel_local[:, pos_bodies, :].flatten(1))

            pos_output_idx += (3*num_pos_bodies.item())
            vel_output_idx += (3*num_pos_bodies.item())

            # Add angles
            self.y_act[:, pos_output_idx:pos_output_idx+(4*num_ori_bodies)] = (
                body_ori_local[:, ori_bodies, :].flatten(1))
            self.dy_act[:, vel_output_idx:vel_output_idx+(3*num_ori_bodies)] = (
                body_ang_vel_local[:, ori_bodies, :].flatten(1))

            pos_output_idx += (4*num_ori_bodies.item())
            vel_output_idx += (3*num_ori_bodies.item())

        # Get the relevant joint angles
        if self.joint_idx is not None:
            joint_pos = wp.to_torch(self.robot.data.joint_pos)[:, self.joint_idx]
            joint_vel = wp.to_torch(self.robot.data.joint_vel)[:, self.joint_idx]

            self.y_act[:, pos_output_idx:pos_output_idx+(joint_pos.shape[1])] = joint_pos
            self.dy_act[:, vel_output_idx:vel_output_idx+(joint_vel.shape[1])] = joint_vel

            pos_output_idx += joint_pos.shape[1]
            vel_output_idx += joint_vel.shape[1]

    def compute_measured_acceleration(
        self, ref_frame_quat: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute measured accelerations from the robot in the same order as outputs.

        This method extracts accelerations from the robot articulation data
        (COM, body, and joint accelerations) and transforms them to the local
        reference frame.

        Args:
            ref_frame_quat: [N, 4] reference frame quaternions.

        Returns:
            ddy_act: [N, num_outputs] measured accelerations in local frame.
        """
        ddy_act = torch.zeros((self.num_envs, len(self.ordered_vel_output_names)), device=self.device)
        output_idx = 0

        # COM accelerations
        if self.use_com:
            # Get COM acceleration in world frame (linear acceleration only)
            com_acc_w = wp.to_torch(self.robot.data.root_com_acc_w)[:, :3]

            # Transform to local frame
            com_acc_local = _align_yaw(com_acc_w, ref_frame_quat)

            ddy_act[:, output_idx:output_idx + 3] = com_acc_local
            output_idx += 3

        # Body accelerations
        if self.body_idx is not None:
            # Get body accelerations in world frame
            # body_acc_w contains [linear_acc (3), angular_acc (3)] per body
            body_acc = wp.to_torch(self.robot.data.body_acc_w)
            body_lin_acc_w = body_acc[:, self.body_idx, :3]
            body_ang_acc_w = body_acc[:, self.body_idx, 3:]

            # Transform to local frame (batched over all bodies)
            body_lin_acc_local = _align_yaw_batched(body_lin_acc_w, ref_frame_quat)
            body_ang_acc_local = _align_yaw_batched(body_ang_acc_w, ref_frame_quat)

            pos_bodies = (self.body_type == 1) | (self.body_type == 2)
            num_pos_bodies = pos_bodies.sum()
            ori_bodies = (self.body_type == 0) | (self.body_type == 2)
            num_ori_bodies = ori_bodies.sum()

            # Add linear accelerations
            ddy_act[:, output_idx:output_idx + (3 * num_pos_bodies)] = (
                body_lin_acc_local[:, pos_bodies, :].flatten(1)
            )
            output_idx += (3 * num_pos_bodies.item())

            # Add angular accelerations
            ddy_act[:, output_idx:output_idx + (3 * num_ori_bodies)] = (
                body_ang_acc_local[:, ori_bodies, :].flatten(1)
            )
            output_idx += (3 * num_ori_bodies.item())

        # Joint accelerations
        if self.joint_idx is not None:
            joint_acc = wp.to_torch(self.robot.data.joint_acc)[:, self.joint_idx]
            ddy_act[:, output_idx:output_idx + joint_acc.shape[1]] = joint_acc
            output_idx += joint_acc.shape[1]

        return ddy_act

    def get_desired_outputs(self, t: torch.Tensor, env_ids: torch.Tensor = None):
        """Get the desired output to track from the trajectory.

        Args:
            t: Time tensor of shape [N].
            env_ids: Optional environment indices of shape [N]. If provided, only update
                state for those environments.
        """
        phi = self.update_phasing_var(t, env_ids)

        # get_output returns (pos_outputs, vel_outputs) tuple
        # pos_outputs: [N, num_pos_outputs] includes ori_w
        # vel_outputs: [N, num_vel_outputs] excludes ori_w
        y_pos, y_vel = self.manager.get_output(t, env_ids)

        # Apply optional heuristic modification
        # TODO: Put back and fix for quat and global orientation
        if self.user_heuristic is not None:
            # Get current contact state
            contact_states = self.get_contact_state(t, env_ids)

            # Get the phasing variable for these environments
            phi_for_heuristic = phi if env_ids is None else self.phasing_var[env_ids]
            total_time = self.manager.get_total_time()

            # Pass both pos and vel to heuristic - heuristic should return (y_pos, y_vel) tuple
            y_pos, y_vel = self.user_heuristic(self.env, self.ordered_pos_output_names, self.ordered_vel_output_names,
                                               y_pos, y_vel, self.contact_bodies,
                                               contact_states, phi_for_heuristic, total_time, env_ids, self.cfg.hold_phi_threshold)

        # Update state - use slice if env_ids provided
        if env_ids is None:
            self.y_des = y_pos
            self.dy_des = y_vel
        else:
            self.y_des[env_ids] = y_pos
            self.dy_des[env_ids] = y_vel

        if self.manager.get_trajectory_type() == TrajectoryType.EPISODIC:
            if env_ids is None:
                self.dy_des[phi == 1] *= 0
            else:
                episodic_mask = phi == 1
                self.dy_des[env_ids[episodic_mask]] *= 0

    def set_user_heuristic(self, heuristic_func):
        """
        Lets the user set a heuristic function to adjust the trajectories automatically.
        """
        self.user_heuristic = heuristic_func

    def get_symmetric_traj(self, traj: torch.Tensor, traj_type: str) -> torch.Tensor:
        """Return the left-right symmetric reflection of a trajectory.

        Swaps left/right outputs and negates pos_y, ori_x, ori_z, roll, yaw.

        Args:
            traj: ``[N, num_outputs]`` trajectory tensor.
            traj_type: ``"vel"`` or ``"pos"`` to select the output name list.

        Returns:
            Symmetric trajectory of the same shape.
        """
        reflector = self._traj_vel_reflector if traj_type == "vel" else self._traj_pos_reflector
        return reflector.reflect(traj)

    @property
    def command(self):
        return self.y_des

    def _resample_command(self, env_ids):
        """Resample the command."""
        self._update_command()
        return

    def _update_command(self):
        """Update the command values."""
        # Invalidate per-step cache so it refreshes once this step
        if self.manager_type == "library":
            self.manager.invalidate_cache()

        # Time in each env
        t = self.env.episode_length_buf * self.env.step_dt

        # Can add a random start time to offset when the episodic trajectory starts
        # Should only generate at the start of an episode
        if self.cfg.random_start_time_max > 0:
            mask = torch.where(self.env.episode_length_buf == 0)[0]
            self.time_offset[mask] = torch.rand(mask.shape, device=self.device) * self.cfg.random_start_time_max

        t = torch.maximum(t - self.time_offset, torch.zeros_like(t))

        t = t + self.init_time_offset

        if self.cfg.percent_hold_phi > 0:
            mask = torch.where(self.env.episode_length_buf == 0)[0]
            self.hold_envs[mask] = (torch.rand(len(mask), device=self.device) > self.cfg.percent_hold_phi).float()
            t *= self.hold_envs

        # Get conditioning variables (velocity, etc...)
        # cond_vars = self.env.command_manager.get_command(self.conditioner_generator)[:, 0]  # TODO: Allow conditioners to be more than scalars

        # Update the measured outputs
        start = time.perf_counter()
        self.get_measured_outputs(t)
        end = time.perf_counter()
        self.get_measured_output_time = (end - start) * torch.ones(self.num_envs, device=self.device)

        # Get desired output
        start = time.perf_counter()
        self.get_desired_outputs(t)
        end = time.perf_counter()
        self.get_desired_output_time = (end - start) * torch.ones(self.num_envs, device=self.device)

        start = time.perf_counter()
        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_des, self.dy_act, self.dy_des)

        # # TODO: Test
        # ddy_act = self.compute_measured_acceleration(self.ref_poses[:, :-4])
        # ddy_nom = self.manager.get_acceleration(t)
        # vdot, vcur = self.clf.compute_vdot_analytic(self.y_act, self.y_des, self.dy_act, self.dy_des, ddy_act, ddy_nom)
        end = time.perf_counter()
        self.vdot_time = (end - start) * torch.ones(self.num_envs, device=self.device)

        self.vdot = vdot
        self.v = vcur

        self.manager.log_v_on_phasing_var(self.get_phasing_var(), self.v)

    def _update_metrics(self):
        """
        Update the metrics.

        Metrics to update:
        - Position tracking
        - Velocity tracking
        - CLF values

        """
        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot

        # Log position tracking errors (using pos output names)
        for i, output in enumerate(self.ordered_pos_output_names):
            self.metrics[output] = torch.abs(self.y_des[:, i] - self.y_act[:, i])

        # Log velocity tracking errors (using vel output names)
        for i, output in enumerate(self.ordered_vel_output_names):
            self.metrics[output + "_vel"] = torch.abs(self.dy_des[:, i] - self.dy_act[:, i])

        # TODO: Delete these lines
        # # Log times
        # self.metrics["get_measured_output_time"] = self.get_measured_output_time
        # self.metrics["get_desired_output_time"] = self.get_desired_output_time
        # self.metrics["vdot_time"] = self.vdot_time

        # # Log per-reference tracking
        # v_mean = self.manager.get_v_log_avg().squeeze(-1)
        # for i in range(len(v_mean)):
        #     # Use repeat() instead of expand() to create a contiguous tensor that
        #     # can be safely modified in-place by the command manager
        #     self.metrics[f"CLF_EMA_{i}"] = v_mean[i].repeat(self.num_envs)

    def _parse_outputs(self, pos_output_names: list[str]) -> tuple[list[int], list[int], bool, list[str], list[str], list[int]]:
        """
        Parse the output names to indices to be used for getting data from the robot in sim.

        Args:
            pos_output_names: List of position output names in the format "frame:axis" or "joint:joint_name"
                             (includes ori_w for quaternions)

        Returns:
            joint_idx: List of joint indices (or None if no joints)
            body_idx: List of body indices (or None if no bodies)
            use_com: True if CoM is used, False otherwise
            ordered_pos_output_names: List of position output names in the order they appear in compute_measured_output
            ordered_vel_output_names: List of velocity output names (excludes ori_w)
            body_type_list: List indicating type of each body (0=ori only, 1=pos only, 2=both)
        """
        output_names = pos_output_names  # Use pos names as the superset
        joint_indices = []
        joint_names_list = []
        body_indices = []
        body_names_list = []
        use_com = False
        com_axes = []

        body_type = {}
        body_type_list = []

        # Track which frames we've already added to avoid duplicates
        added_bodies = set()

        for output_name in output_names:
            if output_name.startswith('joint:'):
                # Joint output: "joint:joint_name"
                joint_name = output_name.split(':', 1)[1]

                # Get the index of this joint
                if joint_name in self.robot.joint_names:
                    joint_idx = self.robot.joint_names.index(joint_name)
                    if joint_idx not in joint_indices:
                        joint_indices.append(joint_idx)
                        joint_names_list.append(joint_name)
                else:
                    raise ValueError(f"Joint '{joint_name}' not found in robot joint names.")

            elif output_name.startswith('com:'):
                # CoM output: "com:pos_x", "com:pos_y", etc.
                use_com = True
                axis = output_name.split(':', 1)[1]  # e.g., "pos_x", "pos_y", "pos_z"
                if axis not in com_axes:
                    com_axes.append(axis)

            else:
                # Frame output: "frame_name:axis"
                frame_name = output_name.split(':', 1)[0]

                if frame_name not in body_type:
                    body_type[frame_name] = [output_name.split(':', 1)[1]]
                else:
                    body_type[frame_name].append(output_name.split(':', 1)[1])

                # Skip if we've already added this body
                if frame_name in added_bodies:
                    continue

                # Try to get the index of this body
                if frame_name in self.robot.body_names:
                    body_idx = self.robot.body_names.index(frame_name)
                    body_indices.append(body_idx)
                    body_names_list.append(frame_name)
                    added_bodies.add(frame_name)
                else:
                    raise ValueError(f"Body frame '{frame_name}' not found in robot body names.")

        # Build ordered output names in the order: COM, bodies, joints
        # Position output names include ori_w, velocity output names exclude it
        ordered_pos_output_names = []
        ordered_vel_output_names = []

        # Add COM outputs first
        if use_com:
            for axis in com_axes:
                ordered_pos_output_names.append(f"com:{axis}")
                ordered_vel_output_names.append(f"com:{axis}")

        # Add body outputs - start with all positions
        for body_name in body_names_list:
            for btype in body_type[body_name]:
                if "pos" in btype:
                    ordered_pos_output_names.append(f"{body_name}:{btype}")
                    ordered_vel_output_names.append(f"{body_name}:{btype}")

        # Add body outputs - then do orientations
        # Enforce (x, y, z, w) order to match IsaacLab v3.0.0 quaternion convention.
        ORI_ORDER = ["ori_x", "ori_y", "ori_z", "ori_w"]
        for body_name in body_names_list:
            ori_axes = [b for b in body_type[body_name] if "ori" in b]
            ori_axes.sort(key=lambda a: ORI_ORDER.index(a) if a in ORI_ORDER else 0)
            for btype in ori_axes:
                ordered_pos_output_names.append(f"{body_name}:{btype}")
                # Exclude ori_w from velocity output names
                if "ori_w" not in btype:
                    ordered_vel_output_names.append(f"{body_name}:{btype}")

        # Build body_type_list for bodies
        for body_name in body_names_list:
            ori = False
            pos = False
            for btype in body_type[body_name]:
                if "ori" in btype:
                    ori = True
                if "pos" in btype:
                    pos = True
            if pos and ori:
                body_type_list.append(2)
            if pos and not ori:
                body_type_list.append(1)
            if ori and not pos:
                body_type_list.append(0)

        # Add joint outputs
        for joint_name in joint_names_list:
            ordered_pos_output_names.append(f"joint:{joint_name}")
            ordered_vel_output_names.append(f"joint:{joint_name}")

        # Convert to None if empty
        joint_idx_result = joint_indices if len(joint_indices) > 0 else None
        body_idx_result = body_indices if len(body_indices) > 0 else None

        return joint_idx_result, body_idx_result, use_com, ordered_pos_output_names, ordered_vel_output_names, body_type_list

    def _parse_ref_frames(self, reference_frames: list[str]) -> tuple[list[int], list[str]]:
        """
        Parse the reference frame names to body indices.

        Args:
            reference_frames: List of body frame names (e.g., ["left_ankle_roll_link", "right_ankle_roll_link"])

        Returns:
            Tuple of (frame_indices, expanded_frame_names):
                - frame_indices: List of body indices corresponding to the reference frames
                - expanded_frame_names: List of expanded frame names (with left/right pairs)

        Note:
            If any frame starts with "right" or "left", the corresponding opposite side frame
            is also added automatically to ensure bilateral symmetry.
        """
        expanded_frames = []

        for frame_name in reference_frames:
            # Add the original frame
            if frame_name not in expanded_frames:
                expanded_frames.append(frame_name)

            # Check if the frame starts with "right" or "left" and add the opposite side
            if frame_name.startswith("right"):
                # Replace "right" with "left"
                opposite_frame = "left" + frame_name[5:]  # Remove "right" and add "left"
                if opposite_frame not in expanded_frames:
                    expanded_frames.append(opposite_frame)
            elif frame_name.startswith("left"):
                # Replace "left" with "right"
                opposite_frame = "right" + frame_name[4:]  # Remove "left" and add "right"
                if opposite_frame not in expanded_frames:
                    expanded_frames.append(opposite_frame)

        # Convert frame names to body indices
        frame_indices = []
        for frame_name in expanded_frames:
            if frame_name in self.robot.body_names:
                frame_idx = self.robot.body_names.index(frame_name)
                frame_indices.append(frame_idx)
            else:
                raise ValueError(f"Reference frame '{frame_name}' not found in robot body names.")

        return frame_indices, expanded_frames


def _align_yaw(vec, root_quat):
    return quat_apply(yaw_quat(quat_inv(root_quat)), vec)

def _align_yaw_batched(vecs: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    """Align multiple vectors to yaw frame.

    Args:
        vecs: Shape [N, B, 3] vectors to align.
        root_quat: Shape [N, 4] reference quaternions.

    Returns:
        Aligned vectors of shape [N, B, 3].
    """
    N, B, _ = vecs.shape
    yaw_inv = yaw_quat(quat_inv(root_quat))  # [N, 4]
    yaw_inv_expanded = yaw_inv.unsqueeze(1).expand(N, B, 4).reshape(N * B, 4)
    vecs_flat = vecs.reshape(N * B, 3)
    result = quat_apply(yaw_inv_expanded, vecs_flat)
    return result.reshape(N, B, 3)

def _align_quat_to_yaw(quat, root_quat):
    return quat_mul(yaw_quat(quat_inv(root_quat)), quat)

def _align_quat_to_yaw_batched(quats: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    """Align multiple quaternions to yaw frame.

    Args:
        quats: Shape [N, B, 4] quaternions to align.
        root_quat: Shape [N, 4] reference quaternions.

    Returns:
        Aligned quaternions of shape [N, B, 4].
    """
    N, B, _ = quats.shape
    yaw_inv = yaw_quat(quat_inv(root_quat))  # [N, 4]
    yaw_inv_expanded = yaw_inv.unsqueeze(1).expand(N, B, 4).reshape(N * B, 4)
    quats_flat = quats.reshape(N * B, 4)
    result = quat_mul(yaw_inv_expanded, quats_flat)
    return result.reshape(N, B, 4)

def get_euler_from_quat(quat):
    """
    Convert quaternion(s) to Euler angles.

    Args:
        quat: Quaternion tensor of shape [..., 4] (supports both single and batched inputs)

    Returns:
        Euler angles tensor of shape [..., 3] with [roll, pitch, yaw]
    """
    euler_x, euler_y, euler_z = euler_xyz_from_quat(quat)
    euler_x = wrap_to_pi(euler_x)
    euler_y = wrap_to_pi(euler_y)
    euler_z = wrap_to_pi(euler_z)
    return torch.stack([euler_x, euler_y, euler_z], dim=-1)