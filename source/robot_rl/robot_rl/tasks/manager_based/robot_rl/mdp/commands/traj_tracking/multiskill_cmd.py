"""MultiSkill command term — subclass of TrajectoryCommand using MultiSkillManager."""

from __future__ import annotations

import torch

from .multiskill_manager import MultiSkillManager
from .trajectory_cmd import TrajectoryCommand


class MultiSkillCommand(TrajectoryCommand):
    """Trajectory command that uses :class:`MultiSkillManager` for batched multi-skill evaluation.

    Subclasses :class:`TrajectoryCommand` and overrides only the manager
    creation and the few places that branch on ``manager_type``.
    """

    def __init__(self, cfg, env):
        # The parent __init__ inlines manager creation, so we temporarily
        # patch cfg to use "library" mode with a placeholder path, then
        # replace the manager with MultiSkillManager afterwards.

        # We need a valid library folder for the parent to load from.
        # Discover skill subfolders to find the first one.
        skill_folders = MultiSkillManager._discover_skill_folders(
            cfg.path, cfg.hf_repo
        )
        first_folder = str(next(iter(skill_folders.values())))

        original_manager_type = cfg.manager_type
        original_path = cfg.path

        cfg.manager_type = "library"
        cfg.path = first_folder

        # Call parent __init__ — creates a LibraryManager as a placeholder
        super().__init__(cfg, env)

        # Restore original config values
        cfg.manager_type = original_manager_type
        cfg.path = original_path

        # Now replace the placeholder manager with MultiSkillManager
        self.manager = MultiSkillManager(
            path=cfg.path,
            device=env.device,
            env=env,
            conditioner_generator_name=cfg.conditioner_generator_name,
            hf_repo=cfg.hf_repo,
        )
        self.manager_type = "multiskill"

        # Re-do output ordering on the new manager
        self.manager.order_outputs(self.ordered_pos_output_names, self.ordered_vel_output_names)

        # Build reference frame map
        self.manager.build_ref_frame_map(self.ref_frames)

        # Update trajectory type (use the most common type from the manager)
        self.trajectory_type = self.manager.get_trajectory_type()

    def verify_contact_frames(self):
        """Verify that trajectory contact frames are in the configured contact bodies.

        For multiskill, we check the stored contact body metadata.
        """
        if not hasattr(self, 'manager') or not isinstance(self.manager, MultiSkillManager):
            # During super().__init__, the placeholder library manager is used
            super().verify_contact_frames()
            return

        # Check contact bodies from the multiskill manager's stored metadata
        for traj_bodies_list in self.manager._contact_bodies_per_domain:
            for domain_bodies in traj_bodies_list:
                for frame in domain_bodies:
                    if frame not in self.contact_bodies:
                        raise ValueError(
                            f"Contact frame {frame} from a trajectory is not "
                            f"in the contact frames list!"
                        )

    def _update_command(self):
        """Update the command values — invalidate multiskill cache each step."""
        if isinstance(self.manager, MultiSkillManager):
            self.manager.invalidate_cache()
        super()._update_command()
