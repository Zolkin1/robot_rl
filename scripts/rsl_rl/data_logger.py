"""Data logger for play-time trajectory recording.

Collects per-timestep data from the Isaac Lab environment during policy playback.
Metadata (axis names, joint names, dt) is captured once at initialization.
Per-step tensors are accumulated and finalized into numpy arrays.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.observations.observations import (
    multiskill_phase,
    ref_cos_phase,
    ref_sin_phase,
)

# Trajectory command term names to search for (in priority order)
_TRAJ_TERM_NAMES = ("traj_ref", "hlip_ref")

# Attributes to pull from the trajectory command term each step
_TRAJ_ATTRS = (
    "y_des", "y_act", "dy_des", "dy_act",
    "v", "vdot", "current_domain", "phasing_var",
    "ref_poses", "cur_ref_frame_idx",
)


class DataLogger:
    """Collects per-timestep data from the environment during policy playback.

    Usage::

        logger = DataLogger(env)
        for step in range(N):
            ...
            logger.collect_step(env)
        data, metadata = logger.finalize()
    """

    def __init__(self, env) -> None:
        """Initialize the logger and capture one-time metadata from the env.

        Args:
            env: The RslRlVecEnvWrapper-wrapped environment.
        """
        self._data: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._step_count: int = 0
        self._warned_sources: set[str] = set()
        self.metadata: dict[str, Any] = {}

        self._collect_metadata(env)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_steps(self) -> int:
        """Number of timesteps collected so far."""
        return self._step_count

    def collect_distillation_actions(
        self,
        student_actions: torch.Tensor,
        teacher_actions: torch.Tensor,
    ) -> None:
        """Log raw student and teacher policy outputs for one step.

        Args:
            student_actions: Student network output for this step, shape ``[N_envs, n_actions]``.
            teacher_actions: Teacher network output for this step, shape ``[N_envs, n_actions]``.
        """
        self._data["student_actions"].append(student_actions.detach().clone())
        self._data["teacher_actions"].append(teacher_actions.detach().clone())

    def collect_step(self, env) -> None:
        """Capture all loggable data for the current timestep.

        Args:
            env: The RslRlVecEnvWrapper-wrapped environment.
        """
        unwrapped = env.unwrapped
        step: dict[str, torch.Tensor] = {}

        # 1. Trajectory command term attributes
        if self._traj_term_name is not None:
            try:
                ref = unwrapped.command_manager.get_term(self._traj_term_name)
                for attr in _TRAJ_ATTRS:
                    val = getattr(ref, attr, None)
                    if val is not None and isinstance(val, torch.Tensor):
                        step[attr] = val.clone()
                    elif val is None:
                        self._warn(f"traj_attr:{attr}",
                                   f"Trajectory term '{self._traj_term_name}' has no attribute '{attr}'")

                # Fallback: if the command doesn't expose phasing_var (e.g.
                # multiskill, which has no phase state), query phi from the
                # manager using the command's per-env trajectory clock so
                # downstream plots still work.
                if "phasing_var" not in step and hasattr(ref, "traj_time") and hasattr(ref, "manager"):
                    try:
                        step["phasing_var"] = ref.manager.get_phasing_var(ref.traj_time).clone()
                    except Exception as exc:
                        self._warn("phasing_var_fallback",
                                   f"Could not compute fallback phasing_var: {exc}")
            except Exception as exc:
                self._warn("traj_term", f"Failed to read trajectory term: {exc}")

        # 2. Base velocity command
        try:
            step["base_velocity"] = unwrapped.command_manager.get_command("base_velocity").clone()
        except Exception as exc:
            self._warn("base_velocity", f"Could not read base_velocity command: {exc}")

        # 3. Action targets
        try:
            action_term = unwrapped.action_manager.get_term("joint_pos")
            step["action_targets"] = action_term._processed_actions.clone()
        except Exception as exc:
            self._warn("action_targets", f"Could not read action targets: {exc}")

        # 4. Robot articulation data
        try:
            robot = unwrapped.scene.articulations["robot"]
            step["joint_pos"] = _to_torch(robot.data.joint_pos).clone()
            step["applied_torque"] = _to_torch(robot.data.applied_torque).clone()
        except Exception as exc:
            self._warn("robot_data", f"Could not read robot articulation data: {exc}")

        # 5. Phase observations
        if self._phase_term_type == "multiskill":
            try:
                step["phase_obs"] = multiskill_phase(unwrapped, self._phase_frequency_list, self._traj_term_name).clone()
            except Exception as exc:
                self._warn("phase_obs", f"Could not compute multiskill_phase: {exc}")
        elif self._phase_term_type == "sin_cos":
            try:
                s = ref_sin_phase(unwrapped, self._traj_term_name)
                c = ref_cos_phase(unwrapped, self._traj_term_name)
                step["phase_obs"] = torch.cat([s, c], dim=-1).clone()
            except Exception as exc:
                self._warn("phase_obs", f"Could not compute sin/cos phase: {exc}")

        for key, val in step.items():
            self._data[key].append(val)

        self._step_count += 1

    def finalize(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Stack all timestep data into numpy arrays.

        Returns:
            Tuple of ``(data, metadata)`` where data values have shape
            ``[T, ...]`` (e.g. ``[T, N_envs, n_dims]``).
        """
        data: dict[str, np.ndarray] = {}
        for key, tensors in self._data.items():
            stacked = torch.stack(tensors, dim=0)
            data[key] = stacked.cpu().numpy()
        return data, self.metadata

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_metadata(self, env) -> None:
        """One-time capture of axis names, joint names, dt, etc."""
        unwrapped = env.unwrapped
        self.metadata["dt"] = unwrapped.step_dt

        # Discover trajectory command term
        self._traj_term_name: str | None = None
        active = unwrapped.command_manager.active_terms
        for name in _TRAJ_TERM_NAMES:
            if name in active:
                self._traj_term_name = name
                break

        if self._traj_term_name is not None:
            try:
                ref = unwrapped.command_manager.get_term(self._traj_term_name)
                if hasattr(ref, "ordered_pos_output_names"):
                    self.metadata["pos_names"] = list(ref.ordered_pos_output_names)
                if hasattr(ref, "ordered_vel_output_names"):
                    self.metadata["vel_names"] = list(ref.ordered_vel_output_names)
                if hasattr(ref, "ref_frames"):
                    self.metadata["ref_frames"] = list(ref.ref_frames)
            except Exception as exc:
                print(f"[WARN DataLogger] Could not read trajectory metadata: {exc}")
        else:
            print(f"[WARN DataLogger] No trajectory command term found among {active}. "
                  f"Trajectory-related data will not be logged.")

        # Detect phase observation type
        self._phase_term_type: str | None = None
        self._phase_frequency_list: list[float] | None = None
        try:
            obs_mgr = unwrapped.observation_manager
            policy_terms = obs_mgr.active_terms.get("policy", [])
            if "multiskill_phases" in policy_terms:
                self._phase_term_type = "multiskill"
                # Extract frequency_list from the obs term config
                term_idx = list(policy_terms).index("multiskill_phases")
                term_cfg = obs_mgr._group_obs_term_cfgs["policy"][term_idx]
                freq_list = term_cfg.params.get("frequency_list", [])
                self._phase_frequency_list = freq_list
                sin_labels = [f"sin({f:.2f}Hz)" for f in freq_list]
                cos_labels = [f"cos({f:.2f}Hz)" for f in freq_list]
                self.metadata["phase_obs_labels"] = sin_labels + cos_labels
            elif "sin_phase" in policy_terms or "cos_phase" in policy_terms:
                self._phase_term_type = "sin_cos"
                self.metadata["phase_obs_labels"] = ["sin_phase", "cos_phase"]
        except Exception as exc:
            print(f"[WARN DataLogger] Could not detect phase observations: {exc}")

        # Joint names
        try:
            robot = unwrapped.scene.articulations["robot"]
            self.metadata["joint_names"] = list(robot.data.joint_names)
        except Exception as exc:
            print(f"[WARN DataLogger] Could not read joint names: {exc}")

    def _warn(self, source: str, msg: str) -> None:
        """Print a warning once per source key."""
        if source not in self._warned_sources:
            print(f"[WARN DataLogger] {msg}")
            self._warned_sources.add(source)


def _to_torch(data: Any) -> torch.Tensor:
    """Convert data to a torch tensor, handling warp arrays."""
    if isinstance(data, torch.Tensor):
        return data
    try:
        import warp as wp
        return wp.to_torch(data)
    except (ImportError, TypeError):
        return torch.as_tensor(data)
