"""Abstract base class for trajectory command terms.

Extracts all manager-agnostic logic from :class:`TrajectoryCommand` so that
both single-skill (:class:`LibraryCommand`) and multi-skill
(:class:`BatchedMultiSkillCommand`) are thin subclasses.
"""

from __future__ import annotations

import re
from abc import abstractmethod

import torch
import warp as wp
from isaaclab.managers import CommandTerm
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply,
    quat_inv,
    quat_mul,
    wrap_to_pi,
    yaw_quat,
)

from .clf import CLF
from .manager_base import ManagerBase
from .trajectory_manager import TrajectoryType


class BaseTrajectoryCommand(CommandTerm):
    """Trajectory command term base class.

    Handles phasing variables, measured/desired output computation, CLF
    tracking, and metrics logging.  Subclasses only need to implement
    manager creation and contact-frame verification.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_manager(self, cfg, env) -> ManagerBase:
        """Create and return the trajectory manager.

        Args:
            cfg: The command term configuration.
            env: The IsaacLab environment.

        Returns:
            A concrete :class:`ManagerBase` subclass instance.
        """

    @abstractmethod
    def _verify_contact_frames(self) -> None:
        """Verify that every contact frame in the trajectories is present in ``self.contact_bodies``."""

    def _post_init(self) -> None:
        """Hook called at the very end of ``__init__``.

        Override in subclasses that need additional setup after the base
        initialisation is complete (e.g. building reference-frame maps).
        """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.env = env
        self.robot = env.scene[cfg.asset_name]

        # --- Contact bodies (expand wildcards) ----------------------------
        self.contact_bodies: list[str] = self._expand_wildcard_frames(cfg.contact_bodies)

        # --- Manager (subclass-specific) ----------------------------------
        self.manager: ManagerBase = self._create_manager(cfg, env)
        self.trajectory_type = self.manager.get_trajectory_type()

        self._verify_contact_frames()

        # --- Domain tracking ----------------------------------------------
        self.current_domain = -1 * torch.ones(self.num_envs, dtype=torch.long, device=self.device)

        # --- Output parsing -----------------------------------------------
        result = self._parse_outputs(self.manager.get_pos_output_names)
        (self.joint_idx, self.body_idx, self.use_com,
         self.ordered_pos_output_names, self.ordered_vel_output_names,
         self.body_type) = result

        # --- State tensors ------------------------------------------------
        self.y_act = torch.zeros((self.num_envs, len(self.ordered_pos_output_names)), device=self.device)
        self.dy_act = torch.zeros((self.num_envs, len(self.ordered_vel_output_names)), device=self.device)
        self.y_des = torch.zeros((self.num_envs, len(self.ordered_pos_output_names)), device=self.device)
        self.dy_des = torch.zeros((self.num_envs, len(self.ordered_vel_output_names)), device=self.device)

        # --- Order manager outputs to match our ordering ------------------
        self.manager.order_outputs(self.ordered_pos_output_names, self.ordered_vel_output_names)
        self.body_type = torch.tensor(self.body_type, dtype=torch.int, device=self.device)

        # --- vel_to_pos_idx mapping (pos includes ori_w, vel excludes it) -
        self.vel_to_pos_idx = torch.zeros(len(self.ordered_vel_output_names), dtype=torch.long, device=self.device)
        for i, vel_name in enumerate(self.ordered_vel_output_names):
            if vel_name in self.ordered_pos_output_names:
                self.vel_to_pos_idx[i] = self.ordered_pos_output_names.index(vel_name)
            else:
                raise ValueError(f"Velocity output name '{vel_name}' not found in position output names.")

        # --- Reference frames ---------------------------------------------
        self.ref_frame_indices, self.ref_frames = self._parse_ref_frames(
            self.manager.get_reference_frames()
        )
        for ref_frame in self.ref_frames:
            if ref_frame not in self.contact_bodies:
                raise ValueError(
                    f"Reference frame '{ref_frame}' is not in the contact frames list: {self.contact_bodies}"
                )

        # Mapping from ref_frames to contact_bodies indices (needed by get_symmetric_contacts)
        self.ref_to_contact_idx = torch.zeros(len(self.ref_frames), dtype=torch.long, device=self.device)
        for i, ref_frame in enumerate(self.ref_frames):
            self.ref_to_contact_idx[i] = self.contact_bodies.index(ref_frame)

        # Current reference frame poses — [N, 7] = [pos_x, pos_y, pos_z, qx, qy, qz, qw]
        self.ref_poses = torch.zeros((self.num_envs, 7), device=self.device)
        self.ref_poses[:, 6] = 1.0  # identity quaternion

        # --- CLF ----------------------------------------------------------
        self.clf = CLF(
            sim_dt=self.env.cfg.sim.dt,
            batch_size=self.num_envs,
            ordered_vel_output_names=self.ordered_vel_output_names,
            ordered_pos_output_names=self.ordered_pos_output_names,
            Q_weights=self.cfg.Q_weights,
            R_weights=self.cfg.R_weights,
            device=self.device,
        )

        # --- Phasing variable state ---------------------------------------
        self.phasing_var = torch.zeros(self.num_envs, device=self.device)
        self.unmasked_phasing_var = torch.zeros(self.num_envs, device=self.device)
        self.prev_unmasked_phasing_var = torch.zeros(self.num_envs, device=self.device)
        self.hold_envs = torch.ones(self.num_envs, device=self.device)

        # Hold logic state (hold at second boundary, not first)
        self.should_hold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.boundaries_crossed = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.hold_phi_value = -1.0 * torch.ones(self.num_envs, device=self.device)

        # --- Time offsets -------------------------------------------------
        self.time_offset = torch.zeros(self.num_envs, device=self.device)
        self.init_time_offset = torch.zeros(self.num_envs, device=self.device)

        # --- Subclass hook ------------------------------------------------
        self._post_init()

    # ------------------------------------------------------------------
    # Phasing variable
    # ------------------------------------------------------------------

    def update_phasing_var(self, t: torch.Tensor, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Update the phasing variable, with hold logic on full updates.

        Holds phi at the second boundary crossing (0.0 or 0.5) rather than
        the first, allowing a full phase to complete before stopping when
        velocity is low.

        Args:
            t: Time tensor of shape ``[N]``.
            env_ids: Optional environment indices. If provided, only those
                environments are updated (hold logic is skipped).

        Returns:
            Phasing variable tensor of shape ``[N]``.
        """
        if env_ids is not None:
            new_phi = self.manager.get_phasing_var(t, env_ids)
            self.phasing_var[env_ids] = new_phi
            self.unmasked_phasing_var[env_ids] = new_phi
            return new_phi

        # Full update with hold / boundary tracking
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
        """Return the current phasing variable of shape ``[N]``."""
        return self.phasing_var

    # ------------------------------------------------------------------
    # Contact state & symmetry helpers
    # ------------------------------------------------------------------

    def get_contact_state(self, t: torch.Tensor, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Return the desired contact state at time *t*.

        Args:
            t: Shape ``[N]`` times per environment.
            env_ids: Optional subset of environments.

        Returns:
            Binary tensor of shape ``[N, num_contacts]``.
        """
        return self.manager.get_contact_state(t, self.contact_bodies, env_ids)

    def get_symmetric_contacts(self, contacts: torch.Tensor) -> torch.Tensor:
        """Return the left-right symmetric reflection of *contacts*.

        Args:
            contacts: ``[N, num_contacts]`` contact states.

        Returns:
            Symmetric contacts of the same shape.
        """
        symmetric_contacts = contacts.clone()
        for i, frame_name in enumerate(self.contact_bodies):
            if "left" in frame_name:
                symmetric_frame = frame_name.replace("left", "right")
            elif "right" in frame_name:
                symmetric_frame = frame_name.replace("right", "left")
            else:
                continue
            if symmetric_frame in self.contact_bodies:
                j = self.contact_bodies.index(symmetric_frame)
                symmetric_contacts[:, i] = contacts[:, j]
        return symmetric_contacts

    # ------------------------------------------------------------------
    # Trajectory type
    # ------------------------------------------------------------------

    def get_trajectory_type(self):
        """Return the trajectory type (periodic / episodic)."""
        return self.trajectory_type

    # ------------------------------------------------------------------
    # Measured outputs
    # ------------------------------------------------------------------

    def get_measured_outputs(self, t: torch.Tensor, env_ids: torch.Tensor = None) -> None:
        """Compute measured outputs from the robot state.

        Args:
            t: Time tensor of shape ``[N]``.
            env_ids: Optional environment indices.
        """
        # Get poses for all reference-frame bodies: [N, num_ref_frames, 7]
        ref_poses = torch.zeros(self.num_envs, len(self.ref_frame_indices), 7, device=self.device)
        ref_poses[:, :, :3] = wp.to_torch(self.robot.data.body_pos_w)[:, self.ref_frame_indices]
        ref_poses[:, :, 3:] = wp.to_torch(self.robot.data.body_quat_w)[:, self.ref_frame_indices]

        # Detect domain changes
        new_domains = self.manager.get_current_domains(t, env_ids)

        if env_ids is None:
            changed = new_domains != self.current_domain
        else:
            changed = new_domains != self.current_domain[env_ids]

        # For single-domain trajectories, force update at stepping cadence
        if env_ids is None:
            single_dom_mask = (
                (self.manager.get_num_domains() == 1)
                & (
                    (self.prev_unmasked_phasing_var > self.unmasked_phasing_var)
                    | ((self.prev_unmasked_phasing_var < 0.5) & (0.5 < self.unmasked_phasing_var))
                )
            )
            changed[single_dom_mask] = True
            self.current_domain = new_domains
        else:
            single_dom_mask = (
                (self.manager.get_num_domains()[env_ids] == 1)
                & (
                    (self.prev_unmasked_phasing_var[env_ids] > self.unmasked_phasing_var[env_ids])
                    | (
                        (self.prev_unmasked_phasing_var[env_ids] < 0.5)
                        & (0.5 < self.unmasked_phasing_var[env_ids])
                    )
                )
            )
            changed[single_dom_mask] = True
            self.current_domain[env_ids] = new_domains

        # Which reference frame each env should use
        if env_ids is None:
            ref_frame_indices = self.manager.get_ref_frames_in_use(t, self.ref_frames)
        else:
            ref_frame_indices = self.manager.get_ref_frames_in_use(t, self.ref_frames, env_ids)

        # Contact-gating: only update ref_poses when the reference frame body
        # is actually in contact with the ground. This prevents snapping to a
        # foot that is mid-swing (e.g. during a skill transition).
        # TODO: On a skill switch the new trajectory may be out of phase with the
        #   robot (e.g. right foot is currently on the ground but the new skill's
        #   domain expects left foot down). We may want to phase-align the new
        #   trajectory so the stance foot matches, or offset the time into the
        #   new trajectory by half a period.
        contact_state = self.get_contact_state(t, env_ids)
        contact_frame_indices = self.ref_to_contact_idx[ref_frame_indices]
        ref_in_contact = torch.gather(
            contact_state, 1, contact_frame_indices.unsqueeze(1)
        ).squeeze(1)
        changed_and_contact = changed & (ref_in_contact > 0)

        if env_ids is None:
            if torch.any(changed_and_contact):
                env_indices = torch.where(changed_and_contact)[0]
                self.ref_poses[env_indices, :] = ref_poses[env_indices, ref_frame_indices[env_indices], :]
        else:
            if torch.any(changed_and_contact):
                subset_indices = torch.where(changed_and_contact)[0]
                global_env_indices = env_ids[subset_indices]
                self.ref_poses[global_env_indices, :] = ref_poses[
                    global_env_indices, ref_frame_indices[subset_indices], :
                ]

        # Compute measured outputs using current ref_poses
        self.compute_measured_output(self.ref_poses[:, :3], self.ref_poses[:, 3:])

    def compute_measured_output(self, ref_frame_pos_w: torch.Tensor, ref_frame_quat: torch.Tensor) -> None:
        """Extract measured positions and velocities from the robot.

        Args:
            ref_frame_pos_w: ``[N, 3]`` reference frame positions.
            ref_frame_quat: ``[N, 4]`` reference frame quaternions.
        """
        pos_output_idx = 0
        vel_output_idx = 0

        # --- CoM ---
        if self.use_com:
            com_pos_w = wp.to_torch(self.robot.data.root_com_pos_w)
            com_vel_w = wp.to_torch(self.robot.data.root_com_vel_w)[:, :3]

            com_pos_local = _align_yaw(com_pos_w - ref_frame_pos_w, ref_frame_quat)
            com_vel_local = _align_yaw(com_vel_w, ref_frame_quat)

            self.y_act[:, pos_output_idx:pos_output_idx + 3] = com_pos_local
            self.dy_act[:, vel_output_idx:vel_output_idx + 3] = com_vel_local
            pos_output_idx += 3
            vel_output_idx += 3

        def _get_pos_ori_vel_relative(ref_pos_w, ref_quat_w, frame_pos_w, frame_quat_w,
                                      frame_lin_vel_w, frame_ang_vel_w):
            """Compute position, orientation, and velocity relative to the reference frame.

            All outputs are in the yaw-aligned frame centred at the reference.

            Args:
                ref_pos_w: ``[N, 3]``
                ref_quat_w: ``[N, 4]``
                frame_pos_w: ``[N, B, 3]``
                frame_quat_w: ``[N, B, 4]``
                frame_lin_vel_w: ``[N, B, 3]``
                frame_ang_vel_w: ``[N, B, 3]``

            Returns:
                Tuple of (pos, ori, lin_vel, ang_vel), each ``[N, B, ...]``.
            """
            pos_rel = frame_pos_w - ref_pos_w.unsqueeze(1)
            pos_yaw = _align_yaw_batched(pos_rel, ref_quat_w)
            ori_yaw = _align_quat_to_yaw_batched(frame_quat_w, ref_quat_w)
            vel_yaw = _align_yaw_batched(frame_lin_vel_w, ref_quat_w)
            ang_vel_yaw = _align_yaw_batched(frame_ang_vel_w, ref_quat_w)
            return pos_yaw, ori_yaw, vel_yaw, ang_vel_yaw

        # --- Bodies ---
        if self.body_idx is not None:
            frame_pos = wp.to_torch(self.robot.data.body_pos_w)[:, self.body_idx, :]
            frame_quat = wp.to_torch(self.robot.data.body_quat_w)[:, self.body_idx, :]

            body_link_vel = wp.to_torch(self.robot.data.body_link_vel_w)
            frame_lin_vel_w = body_link_vel[:, self.body_idx, :3]
            frame_ang_vel_w = body_link_vel[:, self.body_idx, 3:]

            body_pos_local, body_ori_local, body_vel_local, body_ang_vel_local = _get_pos_ori_vel_relative(
                ref_frame_pos_w, ref_frame_quat, frame_pos, frame_quat, frame_lin_vel_w, frame_ang_vel_w
            )

            pos_bodies = (self.body_type == 1) | (self.body_type == 2)
            num_pos_bodies = pos_bodies.sum()
            ori_bodies = (self.body_type == 0) | (self.body_type == 2)
            num_ori_bodies = ori_bodies.sum()

            # Linear
            self.y_act[:, pos_output_idx:pos_output_idx + (3 * num_pos_bodies)] = (
                body_pos_local[:, pos_bodies, :].flatten(1))
            self.dy_act[:, vel_output_idx:vel_output_idx + (3 * num_pos_bodies)] = (
                body_vel_local[:, pos_bodies, :].flatten(1))
            pos_output_idx += (3 * num_pos_bodies.item())
            vel_output_idx += (3 * num_pos_bodies.item())

            # Orientation
            self.y_act[:, pos_output_idx:pos_output_idx + (4 * num_ori_bodies)] = (
                body_ori_local[:, ori_bodies, :].flatten(1))
            self.dy_act[:, vel_output_idx:vel_output_idx + (3 * num_ori_bodies)] = (
                body_ang_vel_local[:, ori_bodies, :].flatten(1))
            pos_output_idx += (4 * num_ori_bodies.item())
            vel_output_idx += (3 * num_ori_bodies.item())

        # --- Joints ---
        if self.joint_idx is not None:
            joint_pos = wp.to_torch(self.robot.data.joint_pos)[:, self.joint_idx]
            joint_vel = wp.to_torch(self.robot.data.joint_vel)[:, self.joint_idx]

            self.y_act[:, pos_output_idx:pos_output_idx + joint_pos.shape[1]] = joint_pos
            self.dy_act[:, vel_output_idx:vel_output_idx + joint_vel.shape[1]] = joint_vel

    # TODO: Can remove as we don't use accelerations anymore
    def compute_measured_acceleration(self, ref_frame_quat: torch.Tensor) -> torch.Tensor:
        """Compute measured accelerations from the robot.

        Args:
            ref_frame_quat: ``[N, 4]`` reference frame quaternions.

        Returns:
            ``[N, num_vel_outputs]`` measured accelerations in the local frame.
        """
        ddy_act = torch.zeros((self.num_envs, len(self.ordered_vel_output_names)), device=self.device)
        output_idx = 0

        if self.use_com:
            com_acc_w = wp.to_torch(self.robot.data.root_com_acc_w)[:, :3]
            com_acc_local = _align_yaw(com_acc_w, ref_frame_quat)
            ddy_act[:, output_idx:output_idx + 3] = com_acc_local
            output_idx += 3

        if self.body_idx is not None:
            body_acc = wp.to_torch(self.robot.data.body_acc_w)
            body_lin_acc_w = body_acc[:, self.body_idx, :3]
            body_ang_acc_w = body_acc[:, self.body_idx, 3:]

            body_lin_acc_local = _align_yaw_batched(body_lin_acc_w, ref_frame_quat)
            body_ang_acc_local = _align_yaw_batched(body_ang_acc_w, ref_frame_quat)

            pos_bodies = (self.body_type == 1) | (self.body_type == 2)
            num_pos_bodies = pos_bodies.sum()
            ori_bodies = (self.body_type == 0) | (self.body_type == 2)
            num_ori_bodies = ori_bodies.sum()

            ddy_act[:, output_idx:output_idx + (3 * num_pos_bodies)] = (
                body_lin_acc_local[:, pos_bodies, :].flatten(1))
            output_idx += (3 * num_pos_bodies.item())

            ddy_act[:, output_idx:output_idx + (3 * num_ori_bodies)] = (
                body_ang_acc_local[:, ori_bodies, :].flatten(1))
            output_idx += (3 * num_ori_bodies.item())

        if self.joint_idx is not None:
            joint_acc = wp.to_torch(self.robot.data.joint_acc)[:, self.joint_idx]
            ddy_act[:, output_idx:output_idx + joint_acc.shape[1]] = joint_acc

        return ddy_act

    # ------------------------------------------------------------------
    # Desired outputs
    # ------------------------------------------------------------------

    def get_desired_outputs(self, t: torch.Tensor, env_ids: torch.Tensor = None) -> None:
        """Get the desired output from the trajectory manager.

        Args:
            t: Time tensor of shape ``[N]``.
            env_ids: Optional environment indices.
        """
        phi = self.update_phasing_var(t, env_ids)

        y_pos, y_vel = self.manager.get_output(t, env_ids)

        if env_ids is None:
            self.y_des = y_pos
            self.dy_des = y_vel
        else:
            self.y_des[env_ids] = y_pos
            self.dy_des[env_ids] = y_vel

        # Zero velocities at the end of episodic trajectories
        if self.manager.get_trajectory_type() == TrajectoryType.EPISODIC:
            if env_ids is None:
                self.dy_des[phi == 1] *= 0
            else:
                episodic_mask = phi == 1
                self.dy_des[env_ids[episodic_mask]] *= 0

    # ------------------------------------------------------------------
    # Symmetry helpers
    # ------------------------------------------------------------------

    def get_symmetric_traj(self, traj: torch.Tensor, traj_type: str) -> torch.Tensor:
        """Return the left-right symmetric reflection of a trajectory.

        Swaps left/right outputs and negates pos_y, ori_x, ori_z, roll, yaw.

        Args:
            traj: ``[N, num_outputs]`` trajectory tensor.
            traj_type: ``"vel"`` or ``"pos"`` to select the output name list.

        Returns:
            Symmetric trajectory of the same shape.
        """
        symmetric_traj = traj.clone()
        output_names = self.ordered_vel_output_names if traj_type == "vel" else self.ordered_pos_output_names

        for i, output_name in enumerate(output_names):
            if "left" in output_name:
                symmetric_name = output_name.replace("left", "right")
            elif "right" in output_name:
                symmetric_name = output_name.replace("right", "left")
            else:
                if any(axis in output_name for axis in ["pos_y", "ori_x", "ori_z", "roll_joint", "yaw_joint"]):
                    symmetric_traj[:, i] = -traj[:, i]
                continue

            if symmetric_name in output_names:
                j = output_names.index(symmetric_name)
                symmetric_traj[:, i] = traj[:, j]
                if any(axis in output_name for axis in ["pos_y", "ori_x", "ori_z", "roll_joint", "yaw_joint"]):
                    symmetric_traj[:, i] = -symmetric_traj[:, i]

        return symmetric_traj

    # ------------------------------------------------------------------
    # CommandTerm interface
    # ------------------------------------------------------------------

    @property
    def command(self):
        """Return the current desired position output."""
        return self.y_des

    def _resample_command(self, env_ids):
        """Resample the command (delegates to ``_update_command``)."""
        self._update_command()

    def _update_command(self):
        """Main per-step update: measured outputs, desired outputs, CLF."""
        self.manager.invalidate_cache()

        # Time in each env
        t = self.env.episode_length_buf * self.env.step_dt

        # Random start time offset for episodic trajectories
        if self.cfg.random_start_time_max > 0:
            mask = torch.where(self.env.episode_length_buf == 0)[0]
            self.time_offset[mask] = torch.rand(mask.shape, device=self.device) * self.cfg.random_start_time_max

        t = torch.maximum(t - self.time_offset, torch.zeros_like(t))
        t = t + self.init_time_offset

        if self.cfg.percent_hold_phi > 0:
            mask = torch.where(self.env.episode_length_buf == 0)[0]
            self.hold_envs[mask] = (torch.rand(len(mask), device=self.device) > self.cfg.percent_hold_phi).float()
            t *= self.hold_envs

        self.get_measured_outputs(t)
        self.get_desired_outputs(t)

        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_des, self.dy_act, self.dy_des)

        self.vdot = vdot
        self.v = vcur

        self.manager.log_v_on_phasing_var(self.get_phasing_var(), self.v)

    def _update_metrics(self):
        """Log tracking errors and CLF values."""
        self.metrics["v"] = self.v
        self.metrics["vdot"] = self.vdot

        for i, output in enumerate(self.ordered_pos_output_names):
            self.metrics[output] = torch.abs(self.y_des[:, i] - self.y_act[:, i])

        for i, output in enumerate(self.ordered_vel_output_names):
            self.metrics[output + "_vel"] = torch.abs(self.dy_des[:, i] - self.dy_act[:, i])

    # ------------------------------------------------------------------
    # Init-time parsing helpers
    # ------------------------------------------------------------------

    def _expand_wildcard_frames(self, frame_patterns: list[str]) -> list[str]:
        """Expand wildcard patterns (e.g. ``".*_ankle_roll_link"``) to explicit body names.

        Args:
            frame_patterns: Frame name patterns, possibly containing regex wildcards.

        Returns:
            List of explicit body names.
        """
        expanded_frames: list[str] = []
        body_names = self.robot.body_names

        for pattern in frame_patterns:
            if '*' in pattern or '.*' in pattern:
                regex_pattern = pattern.replace('*', '.*') if '.*' not in pattern else pattern
                regex_pattern = f'^{regex_pattern}$'
                matched = False
                for body_name in body_names:
                    if re.match(regex_pattern, body_name):
                        expanded_frames.append(body_name)
                        matched = True
                if not matched:
                    raise ValueError(f"Wildcard pattern '{pattern}' did not match any body names in the robot.")
            else:
                expanded_frames.append(pattern)

        return expanded_frames

    def _parse_outputs(
        self, pos_output_names: list[str]
    ) -> tuple[list[int] | None, list[int] | None, bool, list[str], list[str], list[int]]:
        """Parse output names into robot-data indices and ordered name lists.

        Args:
            pos_output_names: Position output names (the superset including ``ori_w``).

        Returns:
            ``(joint_idx, body_idx, use_com, ordered_pos_names, ordered_vel_names, body_type_list)``
        """
        output_names = pos_output_names
        joint_indices: list[int] = []
        joint_names_list: list[str] = []
        body_indices: list[int] = []
        body_names_list: list[str] = []
        use_com = False
        com_axes: list[str] = []
        body_type: dict[str, list[str]] = {}
        body_type_list: list[int] = []
        added_bodies: set[str] = set()

        for output_name in output_names:
            if output_name.startswith('joint:'):
                joint_name = output_name.split(':', 1)[1]
                if joint_name in self.robot.joint_names:
                    joint_idx = self.robot.joint_names.index(joint_name)
                    if joint_idx not in joint_indices:
                        joint_indices.append(joint_idx)
                        joint_names_list.append(joint_name)
                else:
                    raise ValueError(f"Joint '{joint_name}' not found in robot joint names.")
            elif output_name.startswith('com:'):
                use_com = True
                axis = output_name.split(':', 1)[1]
                if axis not in com_axes:
                    com_axes.append(axis)
            else:
                frame_name = output_name.split(':', 1)[0]
                if frame_name not in body_type:
                    body_type[frame_name] = [output_name.split(':', 1)[1]]
                else:
                    body_type[frame_name].append(output_name.split(':', 1)[1])

                if frame_name in added_bodies:
                    continue
                if frame_name in self.robot.body_names:
                    body_idx = self.robot.body_names.index(frame_name)
                    body_indices.append(body_idx)
                    body_names_list.append(frame_name)
                    added_bodies.add(frame_name)
                else:
                    raise ValueError(f"Body frame '{frame_name}' not found in robot body names.")

        # Build ordered output names: COM → body positions → body orientations → joints
        ordered_pos_output_names: list[str] = []
        ordered_vel_output_names: list[str] = []

        if use_com:
            for axis in com_axes:
                ordered_pos_output_names.append(f"com:{axis}")
                ordered_vel_output_names.append(f"com:{axis}")

        for body_name in body_names_list:
            for btype in body_type[body_name]:
                if "pos" in btype:
                    ordered_pos_output_names.append(f"{body_name}:{btype}")
                    ordered_vel_output_names.append(f"{body_name}:{btype}")

        ORI_ORDER = ["ori_x", "ori_y", "ori_z", "ori_w"]
        for body_name in body_names_list:
            ori_axes = [b for b in body_type[body_name] if "ori" in b]
            ori_axes.sort(key=lambda a: ORI_ORDER.index(a) if a in ORI_ORDER else 0)
            for btype in ori_axes:
                ordered_pos_output_names.append(f"{body_name}:{btype}")
                if "ori_w" not in btype:
                    ordered_vel_output_names.append(f"{body_name}:{btype}")

        for body_name in body_names_list:
            ori = any("ori" in b for b in body_type[body_name])
            pos = any("pos" in b for b in body_type[body_name])
            if pos and ori:
                body_type_list.append(2)
            elif pos:
                body_type_list.append(1)
            elif ori:
                body_type_list.append(0)

        for joint_name in joint_names_list:
            ordered_pos_output_names.append(f"joint:{joint_name}")
            ordered_vel_output_names.append(f"joint:{joint_name}")

        joint_idx_result = joint_indices if len(joint_indices) > 0 else None
        body_idx_result = body_indices if len(body_indices) > 0 else None

        return joint_idx_result, body_idx_result, use_com, ordered_pos_output_names, ordered_vel_output_names, body_type_list

    def _parse_ref_frames(self, reference_frames: list[str]) -> tuple[list[int], list[str]]:
        """Parse reference frame names to body indices, adding left/right pairs.

        Args:
            reference_frames: Body frame names from the trajectory data.

        Returns:
            ``(frame_indices, expanded_frame_names)``
        """
        expanded_frames: list[str] = []
        for frame_name in reference_frames:
            if frame_name not in expanded_frames:
                expanded_frames.append(frame_name)
            if frame_name.startswith("right"):
                opposite = "left" + frame_name[5:]
                if opposite not in expanded_frames:
                    expanded_frames.append(opposite)
            elif frame_name.startswith("left"):
                opposite = "right" + frame_name[4:]
                if opposite not in expanded_frames:
                    expanded_frames.append(opposite)

        frame_indices: list[int] = []
        for frame_name in expanded_frames:
            if frame_name in self.robot.body_names:
                frame_indices.append(self.robot.body_names.index(frame_name))
            else:
                raise ValueError(f"Reference frame '{frame_name}' not found in robot body names.")

        return frame_indices, expanded_frames


# ======================================================================
# Module-level helpers
# ======================================================================

def _align_yaw(vec: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    """Rotate *vec* into the yaw-aligned frame of *root_quat*."""
    return quat_apply(yaw_quat(quat_inv(root_quat)), vec)


def _align_yaw_batched(vecs: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    """Rotate batched vectors ``[N, B, 3]`` into the yaw-aligned frame.

    Args:
        vecs: ``[N, B, 3]`` vectors.
        root_quat: ``[N, 4]`` reference quaternions.

    Returns:
        Aligned vectors of shape ``[N, B, 3]``.
    """
    N, B, _ = vecs.shape
    yaw_inv = yaw_quat(quat_inv(root_quat))
    yaw_inv_expanded = yaw_inv.unsqueeze(1).expand(N, B, 4).reshape(N * B, 4)
    vecs_flat = vecs.reshape(N * B, 3)
    result = quat_apply(yaw_inv_expanded, vecs_flat)
    return result.reshape(N, B, 3)


def _align_quat_to_yaw(quat: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    """Align a quaternion to the yaw-aligned frame."""
    return quat_mul(yaw_quat(quat_inv(root_quat)), quat)


def _align_quat_to_yaw_batched(quats: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
    """Align batched quaternions ``[N, B, 4]`` to the yaw-aligned frame.

    Args:
        quats: ``[N, B, 4]`` quaternions.
        root_quat: ``[N, 4]`` reference quaternions.

    Returns:
        Aligned quaternions of shape ``[N, B, 4]``.
    """
    N, B, _ = quats.shape
    yaw_inv = yaw_quat(quat_inv(root_quat))
    yaw_inv_expanded = yaw_inv.unsqueeze(1).expand(N, B, 4).reshape(N * B, 4)
    quats_flat = quats.reshape(N * B, 4)
    result = quat_mul(yaw_inv_expanded, quats_flat)
    return result.reshape(N, B, 4)


def get_euler_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion(s) to Euler angles ``[roll, pitch, yaw]``.

    Args:
        quat: Quaternion tensor of shape ``[..., 4]``.

    Returns:
        Euler angles tensor of shape ``[..., 3]``.
    """
    euler_x, euler_y, euler_z = euler_xyz_from_quat(quat)
    euler_x = wrap_to_pi(euler_x)
    euler_y = wrap_to_pi(euler_y)
    euler_z = wrap_to_pi(euler_z)
    return torch.stack([euler_x, euler_y, euler_z], dim=-1)
