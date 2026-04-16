import os
import re
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml


class RLPolicy:
    """Rl Policy"""

    def __init__(self, policy_params_path, policy_path):
        self.policy_params_path = policy_params_path
        self.policy_path = policy_path

        self.policy_params = None
        self._load_policy_params()

        self.action_isaac = np.zeros(self.get_num_actions())

        # History buffers for observation terms with history_length > 0
        # Maps term_name -> list of np.ndarray (oldest to newest)
        self.history_buffers: dict[str, list[np.ndarray]] = {}

        self.phi = 0.0
        self.prev_phi = 0.0
        self.last_zero_time = 0.0

        # State for phasing variable hold logic (hold at second boundary, not first)
        self.should_hold = False
        self.boundaries_crossed = 0
        self.hold_phi_value = -1.0  # -1 means not locked

    def load(self):
        """Load RL Policy."""
        # Get the cwd and get the logs dir relative to this.
        # NOTE: Assuming we are running from transfer/sim
        two_up = Path.cwd().parent.parent
        policy_logs = os.path.join(two_up, "logs")
        full_path = os.path.join(policy_logs, self.policy_path)
        print(f"Attempting to load {full_path}")

        self.policy = torch.jit.load(full_path)
        # load to cuda
        if torch.cuda.is_available():
            self.policy = self.policy.cuda()

        # Zero LSTM hidden/cell states (no-op for MLP policies)
        self.reset()

    def reset(self):
        """Reset the policy's recurrent hidden states to zero.

        For LSTM policies this zeros the hidden and cell state buffers.
        For MLP policies this is a no-op.
        """
        self.policy.reset()

    def create_obs(self,
                   qfb: np.ndarray,
                   vfb_ang: np.ndarray,
                   qjoints: np.ndarray,
                   vjoints: np.ndarray,
                   time: float,
                   cmd_vel: np.ndarray,
                   joint_names: list[str],):
        """Create the observation from the policy params."""

        obs_np = np.zeros((self.get_num_obs()), dtype=np.float32)

        obs_terms = self.get_obs_terms()

        # Extract floating base quaternion
        quat = qfb[3:7]

        # Convert joint orders
        qjoints_isaac = self.convert_joint_order(qjoints, joint_names, self.get_joint_names())
        vjoints_isaac = self.convert_joint_order(vjoints, joint_names, self.get_joint_names())

        # Compute raw phi from time
        if np.abs(cmd_vel[0]) < 0.1 and (self.prev_phi == 0.0 or self.prev_phi == 0.5):
            self.last_zero_time = time + (self.get_total_time() / 4)

        self.prev_phi = self.phi
        raw_phi = ((time - self.last_zero_time) % self.get_total_time()) / self.get_total_time()

        # Determine if we should hold
        prev_should_hold = self.should_hold
        self.should_hold = np.abs(cmd_vel[0]) < 0.1

        # Reset tracking when newly holding
        if self.should_hold and not prev_should_hold:
            self.boundaries_crossed = 0
            self.hold_phi_value = -1.0

        # Hold at start: if this is the first call and velocity is low, lock at 0.0
        if self.should_hold and self.prev_phi == 0.0 and self.phi == 0.0:
            self.hold_phi_value = 0.0

        # Release hold when no longer should hold
        if not self.should_hold:
            self.hold_phi_value = -1.0
            self.boundaries_crossed = 0

        # Detect boundary crossings (only if should hold and not locked yet)
        if self.should_hold and self.hold_phi_value < 0:
            crosses_zero = (raw_phi < self.prev_phi) and (self.prev_phi > 0)
            crosses_half = (self.prev_phi < 0.5) and (raw_phi >= 0.5)

            if crosses_zero or crosses_half:
                self.boundaries_crossed += 1

            # Lock hold on second boundary crossing
            if self.boundaries_crossed >= 4:
                if crosses_zero:
                    self.hold_phi_value = 0.0
                elif crosses_half:
                    self.hold_phi_value = 0.5

        # Apply hold or use raw phi
        if self.hold_phi_value >= 0:
            self.phi = self.hold_phi_value
        else:
            self.phi = raw_phi

        # self.phi = raw_phi

        # print(f"phi: {self.phi}")

        # Create the observation
        obs_idx = 0
        for term_info in obs_terms:
            term = term_info["name"]
            shape = term_info["shape"]
            scale = term_info["scale"]
            history_length = term_info["history_length"]

            # Compute the single-step observation for this term
            if term == "base_ang_vel":
                raw_obs = self.create_base_ang_vel_obs(vfb_ang)
            elif term == "projected_gravity":
                raw_obs = self.create_projected_gravity_obs(quat)
            elif term == "velocity_commands":
                raw_obs = self.create_velocity_commands_obs(cmd_vel)
            elif term == "joint_pos":
                raw_obs = self.create_joint_pos_obs(qjoints_isaac)
            elif term == "joint_vel":
                raw_obs = self.create_joint_vel_obs(vjoints_isaac)
            elif term == "actions":
                raw_obs = self.create_action_obs()
            elif term == "sin_phase":
                if self.get_skill_type() == "periodic" or self.get_skill_type() == "half_periodic":
                    raw_obs = self.create_sin_phase_obs(self.phi, 1.0)
                elif self.get_skill_type() == "episodic":
                    phi = (min(self.get_total_time() - 1e-8, time) % self.get_total_time()) / self.get_total_time()
                    raw_obs = self.create_sin_phase_obs(phi, 1.0)
                else:
                    raise NotImplementedError(f"Skill type {self.get_skill_type()} is not implemented yet!")
            elif term == "cos_phase":
                if self.get_skill_type() == "periodic" or self.get_skill_type() == "half_periodic":
                    raw_obs = self.create_cos_phase_obs(self.phi, 1.0)
                elif self.get_skill_type() == "episodic":
                    phi = (min(self.get_total_time() - 1e-8, time) % self.get_total_time()) / self.get_total_time()
                    raw_obs = self.create_cos_phase_obs(phi, 1.0)
                else:
                    raise NotImplementedError(f"Skill type {self.get_skill_type()} is not implemented yet!")
            elif term == "multiskill_phases":
                frequency_list = term_info["frequency_list"]
                if frequency_list is None:
                    raise ValueError("multiskill_phases term requires frequency_list in exported parameters!")
                raw_obs = self.create_multiskill_phases_obs(time, frequency_list)
            else:
                raise NotImplementedError(f"Observation term '{term}' not implemented yet!")

            # Ensure raw_obs is a numpy array
            raw_obs = np.atleast_1d(np.asarray(raw_obs, dtype=np.float32))

            # Apply history buffer if needed
            if history_length > 0:
                # Build per-step scale matching raw_obs dimension, then tile across history
                scale_arr = np.broadcast_to(
                    np.atleast_1d(np.asarray(scale, dtype=np.float32)),
                    raw_obs.shape,
                )
                tiled_scale = np.tile(scale_arr, history_length)
                obs_val = self._update_history(term, raw_obs, history_length) * tiled_scale
            else:
                obs_val = raw_obs * scale

            obs_np[obs_idx:obs_idx + shape] = obs_val
            obs_idx += shape

        return torch.from_numpy(obs_np).unsqueeze(0)

    def get_action(self, obs: torch.Tensor, joint_names_out) -> np.ndarray:
        """Get action from RL Policy"""
        if torch.cuda.is_available():
            obs_cuda = obs.cuda()
            self.action_isaac = self.policy(obs_cuda).detach().cpu().numpy().squeeze()
        else:
            self.action_isaac = self.policy(obs).detach().numpy().squeeze()

        return self.convert_joint_order(self.action_isaac * self.action_scale + self.default_joint_angles,
                                        self.get_joint_names(), joint_names_out)

    ##
    # Observation creation
    ##
    def create_base_ang_vel_obs(self, vfb_ang: np.ndarray) -> np.ndarray:
        """Create the base angular velocity observation."""
        return vfb_ang

    def create_projected_gravity_obs(self, quat: np.ndarray) -> np.ndarray:
        """Create the projected gravity observation."""
        qw, qx, qy, qz = quat
        pg = np.zeros(3)
        pg[0] = 2 * (-qz * qx + qw * qy)
        pg[1] = -2 * (qz * qy + qw * qx)
        pg[2] = 1 - 2 * (qw * qw + qz * qz)

        return pg

    def create_velocity_commands_obs(self, cmd_vel: np.ndarray) -> np.ndarray:
        """Create the velocity commands observation."""
        # Clip commanded velocities at the max/min values from the params file
        vel_ranges = self.get_velocity_command_ranges()

        clipped_cmd = np.zeros(3)
        clipped_cmd[0] = np.clip(cmd_vel[0], vel_ranges['v_x_min'], vel_ranges['v_x_max'])
        clipped_cmd[1] = np.clip(cmd_vel[1], vel_ranges['v_y_min'], vel_ranges['v_y_max'])
        clipped_cmd[2] = np.clip(cmd_vel[2], vel_ranges['w_z_min'], vel_ranges['w_z_max'])

        # print(f"clipped_cmd: {clipped_cmd}")

        return clipped_cmd

    def create_joint_pos_obs(self, qjoints: np.ndarray) -> np.ndarray:
        """Create the joint position observation.
        Assumes qjoints in isaac order.
        """
        return qjoints - self.default_joint_angles

    def create_joint_vel_obs(self, vjoints: np.ndarray) -> np.ndarray:
        """Create the joint velocity observation.
        Assumes vjoints in isaac order.
        """
        return vjoints

    def create_action_obs(self) -> np.ndarray:
        """Create the action observation."""
        return self.action_isaac

    def create_sin_phase_obs(self, time: float, freq: float) -> np.ndarray:
        """Create the sinusoidal phase observation."""
        return np.sin(2 * np.pi * time * freq)

    def create_cos_phase_obs(self, time: float, freq: float) -> np.ndarray:
        """Create the cosine phase observation."""
        return np.cos(2 * np.pi * time * freq)

    def create_multiskill_phases_obs(self, time: float, frequency_list: list[float]) -> np.ndarray:
        """Create the multiskill phases observation.

        Computes sin and cos at each frequency, concatenated as [sin_f0, sin_f1, ..., cos_f0, cos_f1, ...].

        Args:
            time: Elapsed episode time in seconds.
            frequency_list: List of frequencies in Hz.

        Returns:
            Array of shape (2 * len(frequency_list),).
        """
        sin_vals = [np.sin(2 * np.pi * f * time) for f in frequency_list]
        cos_vals = [np.cos(2 * np.pi * f * time) for f in frequency_list]
        return np.array(sin_vals + cos_vals, dtype=np.float32)

    def _update_history(self, term_name: str, new_obs: np.ndarray, history_length: int) -> np.ndarray:
        """Update the history buffer for a term and return the flattened history.

        On first call, fills the entire buffer with the initial observation (matching IsaacLab behavior).
        Returns oldest-to-newest flattened: [obs_t-(H-1), ..., obs_t-1, obs_t].

        Args:
            term_name: Name of the observation term.
            new_obs: The current single-step observation.
            history_length: Number of timesteps to buffer.

        Returns:
            Flattened history array of shape (history_length * base_dim,).
        """
        if term_name not in self.history_buffers:
            # First call: fill entire buffer with the initial observation
            self.history_buffers[term_name] = [new_obs.copy() for _ in range(history_length)]
        else:
            buf = self.history_buffers[term_name]
            buf.append(new_obs.copy())
            if len(buf) > history_length:
                buf.pop(0)

        return np.concatenate(self.history_buffers[term_name])

    ##
    # Joint Conversions
    ##
    def convert_joint_order(self, joint_vals: np.ndarray, joint_names_in: list[str], joint_names_out: list[str]) -> np.ndarray:
        """Convert the joint_vals given in order of joint_names to an order given by the params joint names order.

        Args:
            joint_vals: Array of joint values in the order specified by joint_names
            joint_names_in: List of joint names corresponding to joint_vals order
            joint_names_out: Order of joints for the output

        Returns:
            Array of joint values reordered to match the Isaac Lab joint order from params
        """
        reordered_vals = np.zeros_like(joint_vals)

        # Create a mapping from joint name to value
        joint_dict = {name: val for name, val in zip(joint_names_in, joint_vals)}

        # Reorder according to Isaac Lab joint order
        for i, isaac_name in enumerate(joint_names_out):
            if isaac_name in joint_dict:
                reordered_vals[i] = joint_dict[isaac_name]
            else:
                raise ValueError(f"Joint '{isaac_name}' from Isaac order not found in provided joint_names")

        return reordered_vals

    ##
    # Param Reading
    ##
    def _load_policy_params(self):
        """Load the policy parameters from the YAML file."""
        with open(self.policy_params_path, 'r') as f:
            self.policy_params = yaml.safe_load(f)

        self.action_scale = self.get_action_scale()
        self.default_joint_angles = self.get_default_joint_angles()

        # Print loaded observation terms for verification
        obs_terms = self.get_obs_terms()
        print(f"[INFO] Loaded {len(obs_terms)} observation terms (total obs dim: {self.get_num_obs()}):")
        for term in obs_terms:
            extras = ""
            if term["history_length"] > 0:
                extras += f", history={term['history_length']}"
            if term["frequency_list"] is not None:
                extras += f", freqs={term['frequency_list']}"
            print(f"  {term['name']:30s} shape={term['shape']:<6} scale={term['scale']}{extras}")

    def get_num_obs(self) -> int:
        """Get the number of observations from the policy_params file."""
        return self.policy_params['num_obs']

    def get_num_actions(self) -> int:
        """Get the number of actions from the policy_params file."""
        return self.policy_params['num_actions']

    def get_obs_terms(self) -> list[dict]:
        """Get the observation term info in the correct order from the policy_params file.

        Returns:
            List of dicts with keys: name, shape, scale, history_length, frequency_list.

        Raises:
            ValueError: If observation_terms or the policy group is missing from params.
        """
        if 'observation_terms' not in self.policy_params or 'policy' not in self.policy_params['observation_terms']:
            raise ValueError(
                "Could not find observation_terms.policy in policy parameters. "
                "Ensure the policy was exported with export_parameters.py."
            )

        obs_terms = []
        for term_name, term_info in self.policy_params['observation_terms']['policy'].items():
            obs_terms.append({
                "name": term_name,
                "shape": term_info['shape'],
                "scale": term_info['scale'],
                "history_length": term_info.get('history_length', 0),
                "frequency_list": term_info.get('frequency_list', None),
            })
        return obs_terms

    def get_dt(self) -> float:
        """Get the control dt from the policy_params file."""
        return self.policy_params['dt']

    def get_action_scale(self) -> np.ndarray:
        """Get the action scale from the policy_params file.

        Expands wildcard patterns and orders the action scale according to joint_names_isaac.

        Returns:
            Array of action scale values ordered by joint_names_isaac.
        """
        action_scale_dict = self.policy_params.get('action_scale', {})
        joint_names = self.get_joint_names()

        action_scale = np.zeros(len(joint_names))

        for i, joint_name in enumerate(joint_names):
            # Find matching pattern
            matched = False
            for pattern, scale in action_scale_dict.items():
                if re.fullmatch(pattern, joint_name):
                    action_scale[i] = scale
                    matched = True
                    break

            if not matched:
                raise ValueError(f"No action scale pattern matches joint '{joint_name}'")

        return action_scale


    def get_kp(self) -> list[float]:
        """Get the kp gains from the policy_params file."""
        return self.policy_params['kp']

    def get_kd(self) -> list[float]:
        """Get the kd gains from the policy_params file."""
        return self.policy_params['kd']

    def get_default_joint_angles(self) -> np.ndarray:
        """Get the default joint angles from the policy_params file."""
        return np.array(self.policy_params['default_joint_angles'])

    def get_valid_ic_pos(self) -> np.ndarray | None:
        """Get the valid initial condition positions [base_pos(3), base_quat(4), joint_pos(N)]."""
        ic = self.policy_params.get('valid_ic_pos')
        return np.array(ic) if ic is not None else None

    def get_valid_ic_vel(self) -> np.ndarray | None:
        """Get the valid initial condition velocities [base_lin_vel(3), base_ang_vel(3), joint_vel(N)]."""
        ic = self.policy_params.get('valid_ic_vel')
        return np.array(ic) if ic is not None else None

    def get_joint_names(self) -> list[str]:
        """Get the joint names from the policy_params file."""
        return self.policy_params['joint_names_isaac']

    def get_velocity_command_ranges(self) -> dict:
        """Get the velocity command ranges from the policy_params file."""
        return {
            'v_x_max': self.policy_params.get('v_x_max'),
            'v_x_min': self.policy_params.get('v_x_min'),
            'v_y_max': self.policy_params.get('v_y_max'),
            'v_y_min': self.policy_params.get('v_y_min'),
            'w_z_max': self.policy_params.get('w_z_max'),
            'w_z_min': self.policy_params.get('w_z_min'),
        }

    def get_obs_scale(self, term_name: str):
        """Get the observation scale for a specific term.

        Raises:
            ValueError: If observation_terms or the policy group is missing from params.
        """
        if 'observation_terms' not in self.policy_params or 'policy' not in self.policy_params['observation_terms']:
            raise ValueError(
                "Could not find observation_terms.policy in policy parameters. "
                "Ensure the policy was exported with export_parameters.py."
            )

        term_info = self.policy_params['observation_terms']['policy'].get(term_name, {})
        return term_info.get('scale')

    def get_skill_type(self):
        """Get the skill type: episodic, periodic, half_periodic."""
        skill_type = self.policy_params['skill_type']

        if skill_type is not None:
            return skill_type
        else:
            return None

    def get_total_time(self) -> float:
        """Get the total time from the policy_params file."""
        total_time = self.policy_params['total_time']

        if total_time is not None:
            return total_time
        else:
            return None

    def get_max_vx(self) -> float:
        """Get the max vx from the policy_params file."""
        return self.policy_params.get('v_x_max')

    def get_max_vy(self) -> float:
        """Get the max vx from the policy_params file."""
        return self.policy_params.get('v_y_max')

    def get_max_vyaw(self) -> float:
        """Get the max vx from the policy_params file."""
        return self.policy_params.get('w_z_max')