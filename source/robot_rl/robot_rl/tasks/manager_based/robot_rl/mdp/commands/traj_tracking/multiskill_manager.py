import torch

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.library_manager import LibraryManager
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.manager_base import ManagerBase
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_manager import TrajectoryManager

# Improvements for the multiskill manager:
#   - Want to be able to have many trajectories without slowing down data collection time too much.
#       I think the best way to do this will be to just collect all the data from the trajectory manager into a number
#       of individual tensors so that every computation can be batched across all the trajectories. This way there
#       is no looping over all the trajectories. With enough envs its should be that every trajectory is in use anyway
#       so this should only give a speed up.
#   - Need to be able to condition them on more than a single number. I should be able to take in a number of things
#       from commanded velocity (in 3D) to terrain to be able to select a trajectory. So in addition to terrain type
#       which can be literally a semantic tag (i.e. "stairs") we also need to be able to condition on terrain dimensions
#       like stair height and width. We can grab all the terrain info from env.scene.terrain.
#   - Need to be able to place terrain dependent trajectories at the correct place in the global frame (i.e. relative
#       to the terrain), so there needs to be a way to do this.
#   - Should probably be able to support transitioning between skills via some form of linear interpolation.
#       I don't think I want to do that right away, but maybe there should be an empty function put in so the hook
#       is available in the future.
#   - Want to be able to hold different Lyapunov values for each skill. This way we can write multiskill CLF rewards
#       where the CLF is scaled per skill. So we need to be able to compute the Lyapunov function value and store it,
#       separately for each skill (but not necessarily each trajectory). So this means we need to have a distinction
#       between skills and trajectories: a skill can be composed of a library of trajectories.


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