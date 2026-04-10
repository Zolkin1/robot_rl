"""Single-skill trajectory command using :class:`LibraryManager`."""

from __future__ import annotations

from .base_trajectory_cmd import BaseTrajectoryCommand
from .library_manager import LibraryManager
from .manager_base import ManagerBase


class LibraryCommand(BaseTrajectoryCommand):
    """Trajectory command backed by a :class:`LibraryManager` (single skill folder)."""

    def _create_manager(self, cfg, env) -> ManagerBase:
        """Create a :class:`LibraryManager` from a folder of trajectory YAMLs.

        Args:
            cfg: Command term configuration.
            env: IsaacLab environment.

        Returns:
            A :class:`LibraryManager` instance.
        """
        return LibraryManager(
            cfg.path,
            cfg.hf_repo,
            env.device,
            env=env,
            conditioner_generator_name=cfg.conditioner_generator_name,
        )

    def _verify_contact_frames(self) -> None:
        """Verify every trajectory contact frame is in ``self.contact_bodies``."""
        for manager in self.manager.trajectory_managers:
            for domain in manager.traj_data.domain_data.values():
                for frame in domain.contact_bodies:
                    if frame not in self.contact_bodies:
                        raise ValueError(
                            f"Contact frame '{frame}' from trajectory "
                            f"'{manager.traj_data.name}' is not in the "
                            f"contact frames list: {self.contact_bodies}"
                        )
