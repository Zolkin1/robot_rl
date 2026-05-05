"""Abstract base class for trajectory command terms.

Holds the manager-agnostic, phase-free shared logic.  Concrete commands
extend either this class directly (multi-skill, no phase state) or the
:class:`PhasedTrajectoryCommand` subclass (single-skill, phase variable
exposed as observation with hold-phi logic).
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
from .sagittal_reflector import NamedReflector, SagittalReflectionConfig
from .trajectory_manager import TrajectoryType


class BaseTrajectoryCommand(CommandTerm):
    """Trajectory command term base class.

    Handles measured/desired output computation, CLF tracking, ref-frame
    bookkeeping, and metrics logging.  Trajectory advancement is time-based
    only; no phasing variable is stored on this class.  Subclasses that
    want phase-as-observation or hold-phi logic should extend
    :class:`PhasedTrajectoryCommand` instead.

    Subclasses must implement ``_create_manager`` and
    ``_verify_contact_frames``.
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

        # --- Precomputed reflectors for sagittal symmetry -----------------
        _rcfg = SagittalReflectionConfig()
        self._traj_pos_reflector = NamedReflector(_rcfg, self.ordered_pos_output_names, self.device)
        self._traj_vel_reflector = NamedReflector(_rcfg, self.ordered_vel_output_names, self.device)
        self._contact_reflector = NamedReflector(_rcfg, self.contact_bodies, self.device)

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

        # Index (into self.ref_frames) of the ref frame currently in contact
        # per env.  Updated each step inside get_measured_outputs.  Exposed
        # for logging / debug visualisation.
        self.cur_ref_frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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

        # --- Subclass hook ------------------------------------------------
        self._post_init()

    # ------------------------------------------------------------------
    # Contact state & symmetry helpers
    # ------------------------------------------------------------------

    def get_contact_state(self, phase: torch.Tensor, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Return the desired contact state at time *t*.

        Args:
            phase: Shape ``[N]`` phase in [0, 1] per environment.
            env_ids: Optional subset of environments.

        Returns:
            Binary tensor of shape ``[N, num_contacts]``.
        """
        return self.manager.get_contact_state(phase, self.contact_bodies, env_ids)

    def get_symmetric_contacts(self, contacts: torch.Tensor) -> torch.Tensor:
        """Return the left-right symmetric reflection of *contacts*.

        Args:
            contacts: ``[N, num_contacts]`` contact states.

        Returns:
            Symmetric contacts of the same shape.
        """
        return self._contact_reflector.reflect(contacts)

    # ------------------------------------------------------------------
    # Trajectory type
    # ------------------------------------------------------------------

    def get_trajectory_type(self):
        """Return the trajectory type (periodic / episodic)."""
        return self.trajectory_type

    # ------------------------------------------------------------------
    # Measured outputs
    # ------------------------------------------------------------------

    def get_measured_outputs(self, phase: torch.Tensor, env_ids: torch.Tensor = None) -> None:
        """Compute measured outputs from the robot state.

        Args:
            phase: phase in [0, 1] tensor of shape ``[N]``.
            env_ids: Optional environment indices.
        """
        # Get poses for all reference-frame bodies: [N, num_ref_frames, 7]
        ref_poses = torch.zeros(self.num_envs, len(self.ref_frame_indices), 7, device=self.device)
        ref_poses[:, :, :3] = wp.to_torch(self.robot.data.body_pos_w)[:, self.ref_frame_indices]
        ref_poses[:, :, 3:] = wp.to_torch(self.robot.data.body_quat_w)[:, self.ref_frame_indices]

        # Detect domain changes
        new_domains = self.manager.get_current_domains(phase, env_ids)

        if env_ids is None:
            changed = new_domains != self.current_domain
            self.current_domain = new_domains
        else:
            changed = new_domains != self.current_domain[env_ids]
            self.current_domain[env_ids] = new_domains

        # Which reference frame each env should use
        if env_ids is None:
            ref_frame_indices = self.manager.get_ref_frames_in_use(phase, self.ref_frames)
            self.cur_ref_frame_idx = ref_frame_indices
        else:
            ref_frame_indices = self.manager.get_ref_frames_in_use(phase, self.ref_frames, env_ids)
            self.cur_ref_frame_idx[env_ids] = ref_frame_indices

        contact_state = self.get_contact_state(phase, env_ids)
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

    # ------------------------------------------------------------------
    # Desired outputs
    # ------------------------------------------------------------------

    def _transform_desired_outputs(
        self,
        phase: torch.Tensor,
        y_pos: torch.Tensor,
        y_vel: torch.Tensor,
        env_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optional hook to transform desired outputs before storage.

        Default is identity.  Subclasses override to apply a user heuristic
        or other post-processing.
        """
        return y_pos, y_vel

    def get_desired_outputs(self, phase: torch.Tensor, env_ids: torch.Tensor = None) -> None:
        """Get the desired output from the trajectory manager.

        Args:
            phase: phase in [0, 1] tensor of shape ``[N]``.
            env_ids: Optional environment indices.
        """
        y_pos, y_vel = self.manager.get_output(phase, env_ids)
        y_pos, y_vel = self._transform_desired_outputs(phase, y_pos, y_vel, env_ids)

        if env_ids is None:
            self.y_des = y_pos
            self.dy_des = y_vel
        else:
            self.y_des[env_ids] = y_pos
            self.dy_des[env_ids] = y_vel

        # Zero velocities at the end of episodic trajectories
        if self.manager.get_trajectory_type() == TrajectoryType.EPISODIC:
            if env_ids is None:
                self.dy_des[phase == 1] *= 0
            else:
                episodic_mask = phase == 1
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
        reflector = self._traj_vel_reflector if traj_type == "vel" else self._traj_pos_reflector
        return reflector.reflect(traj)

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

    def _pre_update_phase(self) -> None:
        """Hook called at the top of ``_update_command``.

        Responsible for advancing ``manager.phase``, invalidating the
        trajectory-assignment cache, and running any contact-gate logic
        that mutates phase.  The default just invalidates the cache so
        downstream manager reads see a fresh trajectory assignment.
        Subclasses that own phase advance / gating should re-invalidate
        after their mutations.
        """
        self.manager.invalidate_cache()

    def _update_command(self):
        """Main per-step update: measured outputs, desired outputs, CLF."""
        self._pre_update_phase()

        phi = self.manager.phase

        self.get_measured_outputs(phi)
        self.get_desired_outputs(phi)

        vdot, vcur = self.clf.compute_vdot(self.y_act, self.y_des, self.dy_act, self.dy_des)

        self.vdot = vdot
        self.v = vcur

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
