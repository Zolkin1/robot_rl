# Copyright 2026 Zachary Olkin. All rights reserved.

import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.library_manager import LibraryManager
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.manager_base import ManagerBase
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_manager import TrajectoryManager


# This is a class to manage multiple skills (i.e. multiple libraries and/or trajectories)

# The idea here is that we can hold a list of other managers (trajectory and/or library), referred to as sub-managers
#   then when we access something in this manager we also need an env index.
#   Each sub-manager is associated with a set of env indices. So we can access the correct sub-manager
#   and call the sub-managers function.

class MultiSkillManager(ManagerBase):
    """Manages multiple skills (trajectories and/or libraries) for different environment subsets."""

    def __init__(self,
                 traj_paths: list[str],
                 hf_repo: str,
                 traj_types: list[str],
                 traj_envs: list[int],
                 traj_names: list[str],
                 device: torch.device,
                 env=None,
                 conditioner_generator_names: list[str] = None):
        """Initialize the MultiSkillManager class.

        Args:
            traj_paths: List of paths to trajectories or libraries.
            hf_repo: Hugging Face repository to download from.
            traj_types: List of types for each path ("trajectory" or "library").
            traj_envs: List of environment index ranges [start1, end1, start2, end2, ...].
            device: Torch device.
            env: Environment reference (needed for library managers).
            conditioner_generator_names: List of conditioner generator names (one per traj_path).
        """
        self.device = device
        self.env = env
        self.conditioner_generator_names = conditioner_generator_names
        self.traj_paths = traj_paths
        self.traj_names = traj_names

        self.managers = []
        self.manager_indices = {}
        self.traj_envs = traj_envs
        self.traj_types = traj_types
        self.num_envs = 0

        # Create a list of sub-managers
        for i, traj_path in enumerate(self.traj_paths):
            self.manager_indices[i] = range(traj_envs[2 * i], traj_envs[2 * i + 1])
            self.num_envs += self.manager_indices[i].shape[0]

            # Get conditioner name for this manager (only relevant for libraries)
            cond_name = self.conditioner_generator_names[i] if self.conditioner_generator_names else None

            if traj_types[i] == "trajectory":
                self.managers.append(TrajectoryManager(traj_path, hf_repo, device))
            elif traj_types[i] == "library":
                self.managers.append(LibraryManager(traj_path, hf_repo, device,
                                                     env=self.env,
                                                     conditioner_generator_name=cond_name))
            else:
                raise NotImplementedError(f"traj_type: {traj_types[i]} not implemented!")

        self.num_outputs = self.get_num_outputs()

    # def get_reference_frames(self) -> list[str]:
    #     """Get the reference frames corresponding to the trajectory used by those envs."""

    def get_output(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the bezier values at a given time for all the managers.
        """
        outputs = torch.zeros(t.shape[0], 2, self.num_outputs)

        for i in self.manager_indices.keys():
            outputs[self.manager_indices[i]] = self.managers[i].get_output(t[self.manager_indices[i]])


        return outputs

    def get_num_outputs(self) -> int:
        """Get the total number of outputs."""
        # TODO

    def get_current_domains(self, t: torch.Tensor) -> torch.Tensor:
        """
        Determine which domain each env is in.
        """
        domains = torch.zeros_like(t)

        for i in self.manager_indices.keys():
            domains[self.manager_indices[i]] = self.managers[i].get_current_domains(t[self.manager_indices[i]])

        return domains

    def get_num_domains(self) -> int:
        """Get the total number of domains."""

        num_domains = torch.zeros(self.num_envs, device=self.device)
        for i in self.manager_indices.keys():
            num_domains[self.manager_indices[i]] = self.managers[i].get_num_domains()

        return num_domains

    def get_ref_frames_in_use(self, t: torch.Tensor, ref_frames: list[str]) -> torch.Tensor:
        """
                Determine the reference frame in use.

        Args:
            t: shape [N] where N is the number envs.
            ref_frames: a list of reference frames.

        Returns:
            frame_tensor: a torch tensor of shape [N] where each scalar is the index into ref_frames for the active frame.
        """
        frame_indices = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        for i in self.manager_indices.keys():
            frame_indices[self.manager_indices[i]] = self.managers[i].get_ref_frames_in_use(t, ref_frames)

        return frame_indices

    def get_contact_state(self, t: torch.Tensor, contact_frames: list[str]) -> torch.Tensor:
        contact_states = torch.zeros(self.num_envs, len(contact_frames), device=self.device)

        for i in self.manager_indices.keys():
            contact_states[self.manager_indices[i]] = self.managers[i].get_contact_state(t, contact_frames)

        return contact_states

    def get_domain_times(self, t: torch.Tensor) -> torch.Tensor:
        T_domain = torch.zeros(self.num_envs, device=self.device)
        for i in self.manager_indices.keys():
            T_domain[self.manager_indices[i]] = self.managers[i].get_domain_times(t)

        return T_domain