# Policy class for RL controllers
import os
import numpy as np
import torch
import yaml
import re
from ament_index_python.packages import get_package_share_directory

class RLPolicy:
    def __init__(self, hf_repo_id: str, hf_policy_folder: str):
        self.hf_repo_id = hf_repo_id
        self.hf_policy_folder = hf_policy_folder

        pkg_path = get_package_share_directory("g1_control")

        # Download/load policy files from HuggingFace
        self.policy_path, self.policy_params_path = self._load_policy_from_hf(
            pkg_path, self.hf_repo_id, self.hf_policy_folder
        )

        # Load policy parameters first (needed for action initialization)
        self._load_policy_params()

        # Initialize action buffer
        self.action_isaac = np.zeros(self.get_num_actions())

        # Load the policy
        self.policy = torch.jit.load(self.policy_path)
        self.device = next(self.policy.parameters()).device

        self.phi = 0.0
        self.prev_phi = 0.0
        self.last_zero_time = 0.0

        # State for phasing variable hold logic (hold at second boundary, not first)
        self.should_hold = False
        self.boundaries_crossed = 0
        self.hold_phi_value = -1.0  # -1 means not locked

    def get_phi(self) -> float:
        """Get the current phase variable value."""
        return self.phi

    def _load_policy_from_hf(self, pkg_path: str, hf_repo_id: str, hf_policy_folder: str) -> tuple[str, str]:
        """Load policy from Hugging Face with local caching.

        Args:
            pkg_path: Package path for local storage
            hf_repo_id: Hugging Face repository ID (e.g., 'username/repo-name')
            hf_policy_folder: Name of the policy folder on Hugging Face (e.g., 'env_type/folder_name')
        Returns:
            Tuple of (policy_pt_path, policy_params_path)
        """
        # Validate inputs
        if not hf_policy_folder:
            raise ValueError(
                "hf_policy_folder cannot be empty. "
                "Expected format: 'env_type/folder_name' (e.g., 'bow_forward_clf_sym/T0_bowing')"
            )

        # Create cache directory
        cache_dir = os.path.join(
            pkg_path, "resource/policies/hf_cache",
            hf_repo_id.replace("/", "_"), hf_policy_folder
        )
        os.makedirs(cache_dir, exist_ok=True)

        # Define expected local paths
        local_policy_path = os.path.join(cache_dir, "policy.pt")
        local_params_path = os.path.join(cache_dir, "policy_parameters.yaml")

        # Check if both files already exist (cached)
        if os.path.exists(local_policy_path) and os.path.exists(local_params_path):
            print(f"Using cached policy from {cache_dir}")
            return local_policy_path, local_params_path

        # Download from Hugging Face
        try:
            from huggingface_hub import hf_hub_download

            print(f"Downloading policy from {hf_repo_id}/{hf_policy_folder}...")

            # Download policy.pt
            hf_hub_download(
                repo_id=hf_repo_id,
                filename=f"{hf_policy_folder}/policy.pt",
                local_dir=os.path.join(pkg_path, "resource/policies/hf_cache", hf_repo_id.replace("/", "_")),
            )

            # Download policy_parameters.yaml
            hf_hub_download(
                repo_id=hf_repo_id,
                filename=f"{hf_policy_folder}/policy_parameters.yaml",
                local_dir=os.path.join(pkg_path, "resource/policies/hf_cache", hf_repo_id.replace("/", "_")),
            )

            print(f"Successfully downloaded policy to {cache_dir}")
            return local_policy_path, local_params_path

        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for downloading policies. "
                "Install with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download policy from Hugging Face: {e}")
        
    def get_policy_path(self) -> str:
        """Get the policy .pt file path."""
        return self.policy_path

    def get_policy_params_path(self) -> str:
        """Get the policy parameters YAML file path."""
        return self.policy_params_path

    def get_action(self, obs: torch.Tensor, joint_names_out) -> np.ndarray:
        """Get action from RL Policy"""
        # if torch.cuda.is_available():
        #     obs_cuda = obs.cuda()
        #     self.action_isaac = self.policy(obs_cuda).detach().cpu().numpy().squeeze()
        # else:
        self.action_isaac = self.policy(obs).detach().numpy().squeeze()

        return self.convert_joint_order(self.action_isaac * self.action_scale + self.default_joint_angles,
                                        self.get_joint_names(), joint_names_out)

    def reset_last_action(self):
        """Reset the last action to zeros."""
        self.action_isaac = np.zeros(self.get_num_actions())

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

        # TODO: Fix when I have updated policies
        if self.get_skill_type() == "episodic":
            time2 = max(time - 100000.0, 0)  # Adjust time offset if needed
        else:
            time2 = time


        # Convert joint orders
        qjoints_isaac = self.convert_joint_order(qjoints, joint_names, self.get_joint_names())
        vjoints_isaac = self.convert_joint_order(vjoints, joint_names, self.get_joint_names())

        # Compute raw phi from time
        if np.abs(cmd_vel[0]) < 0.1 and (self.prev_phi == 0.0 or self.prev_phi == 0.5):
            self.last_zero_time = time + (self.get_total_time()/4)
            
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

        # Create the observation
        obs_idx = 0
        for term, shape, scale in obs_terms:
            if term == "base_ang_vel":
                obs_np[obs_idx:obs_idx+shape] = self.create_base_ang_vel_obs(vfb_ang) * scale
                obs_idx += shape
            elif term == "projected_gravity":
                obs_np[obs_idx:obs_idx+shape] = self.create_projected_gravity_obs(quat) * scale
                obs_idx += shape
            elif term == "velocity_commands":
                obs_np[obs_idx:obs_idx+shape] = self.create_velocity_commands_obs(cmd_vel) * scale
                obs_idx += shape
            elif term == "joint_pos":
                obs_np[obs_idx:obs_idx+shape] = self.create_joint_pos_obs(qjoints_isaac) * scale
                obs_idx += shape
            elif term == "joint_vel":
                obs_np[obs_idx:obs_idx+shape] = self.create_joint_vel_obs(vjoints_isaac) * scale
                obs_idx += shape
            elif term == "actions":
                obs_np[obs_idx:obs_idx+shape] = self.create_action_obs() * scale
                obs_idx += shape
            elif term == "sin_phase":
                if self.get_skill_type() == "periodic" or self.get_skill_type() == "half_periodic":
                    # if np.linalg.norm(cmd_vel) > 0.1:
                    obs_np[obs_idx:obs_idx+shape] = self.create_sin_phase_obs(self.phi, 1.0) * scale
                    # else:
                    #     obs_np[obs_idx:obs_idx+shape] = 0 * scale
                    # obs_np[obs_idx:obs_idx+shape] = self.create_sin_phase_obs(time2, 1.0/self.get_total_time()) * scale
                elif self.get_skill_type() == "episodic":
                    phi = (min(self.get_total_time() - 1e-8, time2) % self.get_total_time())/self.get_total_time()
                    # phi = 0
                    obs_np[obs_idx:obs_idx + shape] = self.create_sin_phase_obs(phi, 1.0) * scale
                else:
                    raise NotImplementedError(f"Skill type {self.get_skill_type()} is not implemented yet!")

                obs_idx += shape

            elif term == "cos_phase":
                if self.get_skill_type() == "periodic" or self.get_skill_type() == "half_periodic":
                    # if np.linalg.norm(cmd_vel) > 0.1:
                    obs_np[obs_idx:obs_idx+shape] = self.create_cos_phase_obs(self.phi, 1.0) #time2, 1.0/self.get_total_time()) * scale
                    # else:
                        # obs_np[obs_idx:obs_idx+shape] = 1 * scale
                    # obs_np[obs_idx:obs_idx+shape] = self.create_cos_phase_obs(time2, 1.0/self.get_total_time()) * scale
                elif self.get_skill_type() == "episodic":
                    phi = (min(self.get_total_time() - 1e-8, time2) % self.get_total_time())/self.get_total_time()
                    # phi = 0
                    # print(f"phi: {phi}")
                    obs_np[obs_idx:obs_idx + shape] = self.create_cos_phase_obs(phi, 1.0) * scale
                    # print(f"cos phase: {self.create_cos_phase_obs(phi, 1.0)}")
                else:
                    raise NotImplementedError(f"Skill type {self.get_skill_type()} is not implemented yet!")
                obs_idx += shape
            else:
                raise NotImplementedError("Observation term not implemented yet!")

        return torch.from_numpy(obs_np).unsqueeze(0)

    ##
    # Observation creation
    ##
    def create_base_ang_vel_obs(self, vfb_ang: np.ndarray) -> np.ndarray:
        """Create the base angular velocity observation."""
        return vfb_ang

    def create_projected_gravity_obs(self, quat: np.ndarray) -> np.ndarray:
        """Create the projected gravity observation."""
        qx, qy, qz, qw = quat

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

    def set_last_zero_time(self, time: float) -> None:
        """Set the last zero time of the observation."""
        self.last_zero_time = time

    ##
    # Joint Conversions
    ##
    def convert_joint_order(self, joint_vals: np.ndarray, joint_names_in: list[str], joint_names_out: list[str]) -> np.ndarray:
        """Convert the joint_vals given in order of joint_names to an order given by the params joint names order.

        Args:
            joint_vals: Array of joint values in the order specified by joint_names_in
            joint_names_in: List of joint names corresponding to joint_vals order
            joint_names_out: Order of joints for the output

        Returns:
            Array of joint values reordered to match the Isaac Lab joint order from params
        """
        reordered_vals = np.zeros(len(joint_names_out), dtype=np.float32)

        # Create a mapping from joint name to value
        joint_dict = {name: val for name, val in zip(joint_names_in, joint_vals)}

        # Reorder according to Isaac Lab joint order
        for i, joint_out_name in enumerate(joint_names_out):
            if joint_out_name in joint_dict:
                reordered_vals[i] = joint_dict[joint_out_name]
            else:
                raise ValueError(f"Joint '{joint_out_name}' from joint_names_out order not found in provided joint_names_in")
            
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

    def get_num_obs(self) -> int:
        """Get the number of observations from the policy_params file."""
        return self.policy_params['num_obs']

    def get_num_actions(self) -> int:
        """Get the number of actions from the policy_params file."""
        return self.policy_params['num_actions']

    def get_obs_terms(self) -> list[tuple[str, int, float | list[float]]]:
        """Get the observation term names, shape, and scale in the correct order from the policy_params file.

        Returns:
            List of tuples containing (term_name, shape, scale)
        """
        obs_terms = []
        if 'observation_terms' in self.policy_params and 'policy' in self.policy_params['observation_terms']:
            for term_name, term_info in self.policy_params['observation_terms']['policy'].items():
                obs_terms.append((term_name, term_info['shape'], term_info['scale']))
        return obs_terms

    def get_dt(self) -> float:
        """Get the control dt from the policy_params file."""
        return self.policy_params['dt']

    def get_action_scale(self) -> np.ndarray:
        """Get the action scale from the policy_params file.

        Expands wildcard patterns and orders the action scale according to joint_names_isaac.
        If action_scale is a single scalar, it is applied uniformly to all joints.

        Returns:
            Array of action scale values ordered by joint_names_isaac.
        """
        action_scale_raw = self.policy_params.get('action_scale', {})
        joint_names = self.get_joint_names()

        # Handle scalar action scale (single float applied to all joints)
        if isinstance(action_scale_raw, (int, float)):
            return np.full(len(joint_names), action_scale_raw)

        action_scale_dict = action_scale_raw
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


    def get_kp(self, joint_order: list[str]) -> list[float]:
        """Get the kp gains from the policy_params file."""
        return self.convert_joint_order(self.policy_params['kp'], self.get_joint_names(), joint_order)

    def get_kd(self, joint_order: list[str]) -> list[float]:
        """Get the kd gains from the policy_params file."""
        return self.convert_joint_order(self.policy_params['kd'], self.get_joint_names(), joint_order)
    
    def get_default_joint_angles(self) -> np.ndarray:
        """Get the default joint angles from the policy_params file."""
        return np.array(self.policy_params['default_joint_angles'])

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

    # def get_gait_period_range(self) -> tuple[float, float]:
    #     """Get the gait period range from the policy_params file."""
    #     period_range = self.policy_params.get('total_time')
    #     if period_range:
    #         return tuple(period_range)
    #     return None

    def get_obs_scale(self, term_name: str):
        """Get the observation scale for a specific term."""
        if 'observation_terms' in self.policy_params and 'policy' in self.policy_params['observation_terms']:
            term_info = self.policy_params['observation_terms']['policy'].get(term_name, {})
            return term_info.get('scale')
        return None

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