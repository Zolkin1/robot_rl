"""Single-skill trajectory command backed by a single trajectory or library.

TODO: Delete this file. Single-skill commands have been superseded by
:class:`BatchedMultiSkillCommand`. Consumers (g1 walking/running/bow/bend
configs) need to migrate to the multi-skill path first.
"""

from __future__ import annotations

import torch

from .library_manager import LibraryManager
from .manager_base import ManagerBase
from .phased_trajectory_cmd import PhasedTrajectoryCommand
from .trajectory_manager import TrajectoryManager


class TrajectoryCommand(PhasedTrajectoryCommand):
    """Single-skill trajectory command with phase-as-observation.

    Creates either a :class:`TrajectoryManager` (single trajectory, YAML file)
    or a :class:`LibraryManager` (folder of trajectory YAML files) based on
    ``cfg.manager_type``.  An optional ``cfg.heuristic_func`` can be provided
    to post-process the desired outputs each step.
    """

    def _create_manager(self, cfg, env) -> ManagerBase:
        """Create the underlying manager based on ``cfg.manager_type``."""
        self.manager_type = cfg.manager_type
        if cfg.manager_type == "trajectory":
            return TrajectoryManager(cfg.path, cfg.hf_repo, env.device)
        if cfg.manager_type == "library":
            return LibraryManager(
                cfg.path,
                cfg.hf_repo,
                env.device,
                env=env,
                conditioner_generator_name=cfg.conditioner_generator_name,
            )
        raise NotImplementedError(f"Manager Type {cfg.manager_type} is not implemented!")

    def _verify_contact_frames(self) -> None:
        """Check every trajectory contact frame is in ``self.contact_bodies``."""
        if self.manager_type == "trajectory":
            traj_frames = [
                domain.contact_bodies for domain in self.manager.traj_data.domain_data.values()
            ]
        else:  # "library"
            traj_frames = [
                domain.contact_bodies
                for manager in self.manager.trajectory_managers
                for domain in manager.traj_data.domain_data.values()
            ]

        for frames in traj_frames:
            for frame in frames:
                if frame not in self.contact_bodies:
                    raise ValueError(
                        f"Contact frame '{frame}' from a trajectory is not in "
                        f"the contact frames list: {self.contact_bodies}"
                    )

    def _post_init(self) -> None:
        """Install the optional user heuristic."""
        self.user_heuristic = self.cfg.heuristic_func

    def set_user_heuristic(self, heuristic_func) -> None:
        """Set or replace the heuristic function used to adjust trajectories."""
        self.user_heuristic = heuristic_func

    def _transform_desired_outputs(
        self,
        t: torch.Tensor,
        y_pos: torch.Tensor,
        y_vel: torch.Tensor,
        env_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the user heuristic to the desired outputs if configured."""
        if self.user_heuristic is None:
            return y_pos, y_vel

        contact_states = self.get_contact_state(t, env_ids)
        phi = self.phasing_var if env_ids is None else self.phasing_var[env_ids]
        total_time = self.manager.get_total_time()

        return self.user_heuristic(
            self.env,
            self.ordered_pos_output_names,
            self.ordered_vel_output_names,
            y_pos,
            y_vel,
            self.contact_bodies,
            contact_states,
            phi,
            total_time,
            env_ids,
            self.cfg.hold_phi_threshold,
        )
