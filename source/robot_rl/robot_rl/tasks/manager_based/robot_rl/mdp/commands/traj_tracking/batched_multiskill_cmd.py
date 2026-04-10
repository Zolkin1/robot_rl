"""Batched multi-skill trajectory command using :class:`MultiSkillManager`."""

from __future__ import annotations

from .base_trajectory_cmd import BaseTrajectoryCommand
from .manager_base import ManagerBase
from .multiskill_manager import MultiSkillManager

# TODO: Should probably change this name
class BatchedMultiSkillCommand(BaseTrajectoryCommand):
    """Trajectory command backed by a :class:`MultiSkillManager`.

    Supports multiple skills (subfolders of trajectories) with fully
    batched tensor evaluation — no per-trajectory Python loops at runtime.
    """

    def _create_manager(self, cfg, env) -> ManagerBase:
        """Create a :class:`MultiSkillManager` from a folder of skill subfolders.

        Args:
            cfg: Command term configuration.
            env: IsaacLab environment.

        Returns:
            A :class:`MultiSkillManager` instance.
        """
        return MultiSkillManager(
            path=cfg.path,
            device=env.device,
            env=env,
            conditioner_generator_name=cfg.conditioner_generator_name,
            hf_repo=cfg.hf_repo,
        )

    def _verify_contact_frames(self) -> None:
        """Verify every trajectory contact frame is in ``self.contact_bodies``."""
        for traj_bodies_list in self.manager._contact_bodies_per_domain:
            for domain_bodies in traj_bodies_list:
                for frame in domain_bodies:
                    if frame not in self.contact_bodies:
                        raise ValueError(
                            f"Contact frame '{frame}' from a trajectory is "
                            f"not in the contact frames list: {self.contact_bodies}"
                        )

    def _post_init(self) -> None:
        """Build the reference frame map that :class:`MultiSkillManager` needs."""
        self.manager.build_ref_frame_map(self.ref_frames)
