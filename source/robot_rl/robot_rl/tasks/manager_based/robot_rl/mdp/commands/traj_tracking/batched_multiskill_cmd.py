"""Batched multi-skill trajectory command using :class:`MultiSkillManager`."""

from __future__ import annotations

import torch

from .base_trajectory_cmd import BaseTrajectoryCommand
from .manager_base import ManagerBase
from .multiskill_manager import MultiSkillManager


# TODO: Should probably change this name
class BatchedMultiSkillCommand(BaseTrajectoryCommand):
    """Trajectory command backed by a :class:`MultiSkillManager`.

    Supports multiple skills (subfolders of trajectories) with fully
    batched tensor evaluation — no per-trajectory Python loops at runtime.

    Maintains an explicit per-env trajectory clock ``self.traj_time`` that
    advances linearly at ``step_dt``.  On a detected skill change the clock
    is re-aligned via :meth:`MultiSkillManager.compute_transition_time` so
    the new skill starts at a phi that matches the current stance foot and
    preserves the fractional position in the period (controlled by
    ``cfg.smooth_transitions``).
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
        """Initialise clock state and the manager's ref-frame lookup."""
        self.manager.build_ref_frame_map(self.ref_frames)

        self.traj_time = torch.zeros(self.num_envs, device=self.device)
        # Accumulated offset from smooth skill transitions, persists through
        # the rest of the episode. Cleared on reset.
        self.skill_time_offset = torch.zeros(self.num_envs, device=self.device)
        self.prev_traj_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_skill_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._initialized = False

    def _compute_time(self) -> torch.Tensor:
        """Compute the per-env trajectory time, aligning on skill changes.

        Must be idempotent across the same env step: IsaacLab's
        ``CommandTerm.compute`` calls ``_update_command`` twice when a
        resample fires (once from ``_resample_command`` and once directly),
        so we compute ``traj_time`` deterministically from
        ``episode_length_buf`` + accumulated offsets rather than with a
        ``+= step_dt`` accumulator.
        """
        reset_mask = self.env.episode_length_buf == 0
        advancing = ~reset_mask

        # Clear skill-transition offset at episode start and re-sample the
        # random-start offset if configured.
        self.skill_time_offset[reset_mask] = 0.0
        if self.cfg.random_start_time_max > 0:
            rand_idx = torch.where(reset_mask)[0]
            if rand_idx.numel() > 0:
                self.time_offset[rand_idx] = (
                    torch.rand(rand_idx.shape, device=self.device)
                    * self.cfg.random_start_time_max
                )

        # Baseline: episode counter × step_dt, plus whatever the reset
        # event wrote into init_time_offset, plus random-start offset,
        # plus accumulated skill-transition offset.
        t = (
            self.env.episode_length_buf * self.env.step_dt
            + self.init_time_offset
            + self.time_offset
            + self.skill_time_offset
        )

        # Resolve current trajectory / skill assignment
        self.manager.invalidate_cache()
        cur_traj = self.manager.get_current_trajectory_indices()
        cur_skill = self.manager.data["skill_idx"][cur_traj]

        # Detect skill changes (skip first-ever step and reset envs)
        if self._initialized and self.cfg.smooth_transitions:
            changed = (cur_skill != self.prev_skill_idx) & advancing
            if changed.any():
                target_t = self.manager.compute_transition_time(
                    self.prev_traj_idx[changed],
                    t[changed],
                    cur_traj[changed],
                )
                # Fold the adjustment into the persistent offset so
                # subsequent steps continue from the aligned phase.
                self.skill_time_offset[changed] += target_t - t[changed]
                t = t.clone()
                t[changed] = target_t

        self.traj_time = t
        self.prev_traj_idx = cur_traj.clone()
        self.prev_skill_idx = cur_skill.clone()
        self._initialized = True
        return t
