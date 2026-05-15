from __future__ import annotations

import logging
import torch
import warp as wp
from typing import Sequence, TYPE_CHECKING

from .velocity_commands import VelocityTrackingCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .multiskill_velocity_commands_cfg import MultiskillVelocityTrackingCommandCfg

logger = logging.getLogger(__name__)


class MultiskillVelocityTrackingCommand(VelocityTrackingCommand):
    """Velocity tracking command that samples per-env skills from terrain metadata.

    At every resample, each env's current world XY is queried via
    ``terrain.skill_probs_at(xy)`` for its per-skill probability vector, and
    one skill is drawn via :func:`torch.multinomial`. The chosen skill indexes
    into the trajectory cmd's ``velocity_buckets`` to give a uniform velocity range.

    Requires the scene's terrain importer to be a meta importer exposing
    ``skill_probs_at``, ``skill_list``, and ``world_xy_to_cell``. The bucket
    dict keys must match the importer's ``skill_list`` exactly.
    """

    cfg: MultiskillVelocityTrackingCommandCfg

    def __init__(self, cfg: MultiskillVelocityTrackingCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # --- Resolve and validate the terrain importer ---
        terrain = env.scene[cfg.terrain_name]
        for required in ("skill_probs_at", "skill_list", "world_xy_to_cell"):
            if not hasattr(terrain, required):
                raise TypeError(
                    f"MultiskillVelocityTrackingCommand requires the scene's "
                    f"'{cfg.terrain_name}' entity to expose '{required}'. "
                    f"Got {type(terrain).__name__}; this command needs a "
                    f"MetaTerrainImporter (or subclass)."
                )
        self._terrain = terrain
        self._skill_list: list[str] = list(terrain.skill_list)

        # The trajectory command's ``ref_poses`` is used as the terrain
        # cell-lookup anchor on mid-episode resamples.  The lookup itself
        # has to be lazy: this ``__init__`` runs from inside
        # ``CommandManager.__init__``, which only assigns
        # ``env.command_manager`` *after* all terms are constructed.
        # ``_resample_command`` resolves and caches the term on first use.
        self._traj_cmd = None

        # The bucket dict lives on the trajectory cmd cfg (source of truth
        # for "what skill is the policy actually executing").  The
        # velocity cmd needs the same ranges to sample from on resample —
        # pull them from the trajectory cmd's cfg directly.  Cfgs are
        # populated independently of term-construction order so this is
        # safe even though the trajectory term itself isn't registered yet.
        traj_cfg = getattr(env.cfg.commands, cfg.trajectory_command_name)
        self._velocity_buckets = traj_cfg.velocity_buckets

        bucket_keys = set(self._velocity_buckets.keys())
        skill_set = set(self._skill_list)
        if bucket_keys != skill_set:
            missing = skill_set - bucket_keys
            extra = bucket_keys - skill_set
            raise ValueError(
                f"velocity_buckets keys (on '{cfg.trajectory_command_name}') "
                f"must equal terrain.skill_list. missing buckets for skills: "
                f"{sorted(missing)}; extra buckets not in skill_list: {sorted(extra)}."
            )

        if cfg.skill_transition_prob is not None and not 0.0 <= cfg.skill_transition_prob <= 1.0:
            raise ValueError(
                f"skill_transition_prob must be in [0, 1], got {cfg.skill_transition_prob}."
            )

        if cfg.max_acc_frac is not None and not 0.0 <= cfg.max_acc_frac <= 1.0:
            raise ValueError(
                f"max_acc_frac must be in [0, 1] or None, got {cfg.max_acc_frac}."
            )

        # --- Pre-build per-skill range tensors aligned to terrain.skill_list order ---
        default_heading = torch.tensor(cfg.ranges.heading, device=self.device, dtype=torch.float)
        self._skill_lin_vel_x = torch.tensor(
            [self._velocity_buckets[s].lin_vel_x for s in self._skill_list],
            device=self.device, dtype=torch.float,
        )
        self._skill_lin_vel_y = torch.tensor(
            [self._velocity_buckets[s].lin_vel_y for s in self._skill_list],
            device=self.device, dtype=torch.float,
        )
        self._skill_ang_vel_z = torch.tensor(
            [self._velocity_buckets[s].ang_vel_z for s in self._skill_list],
            device=self.device, dtype=torch.float,
        )
        self._skill_heading = torch.stack([
            torch.tensor(self._velocity_buckets[s].heading, device=self.device, dtype=torch.float)
            if self._velocity_buckets[s].heading is not None
            else default_heading
            for s in self._skill_list
        ])

        # ``sampled_skill_id`` records which bucket the most recent sample
        # for each env was drawn from.  It's informational only — the
        # trajectory cmd's active ``skill_id`` is derived per-step from
        # the *ramped* ``vel_target_b`` via a bucket lookup.  This field
        # exists for logging / plot diagnostics and for the
        # ``skill_transition_prob`` "don't resample the same bucket"
        # logic in :meth:`_sample_skill`.
        self.sampled_skill_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._warned_no_transition = False

        # Per-env max-acc clamp assignment (persists for the episode, re-rolled on reset).
        if cfg.max_acc_frac is None:
            self.max_acc_per_env.fill_(float("inf"))
        else:
            rand = torch.empty(self.num_envs, device=self.device).uniform_(0.0, 1.0)
            has_max_acc = rand < cfg.max_acc_frac
            self.max_acc_per_env = torch.where(
                has_max_acc,
                torch.full_like(rand, float(cfg.max_acc)),
                torch.full_like(rand, float("inf")),
            )

    def __str__(self) -> str:
        msg = "Multiskill Velocity Tracking Command:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tTerrain entity: {self.cfg.terrain_name}\n"
        msg += f"\tNumber of skills: {len(self._skill_list)}\n"
        for i, name in enumerate(self._skill_list):
            bucket = self._velocity_buckets[name]
            msg += (
                f"\tSkill '{name}': vx={bucket.lin_vel_x}, vy={bucket.lin_vel_y}, "
                f"wz={bucket.ang_vel_z}\n"
            )
        if self.cfg.skill_transition_prob is not None:
            msg += f"\tSkill transition probability: {self.cfg.skill_transition_prob:.1%}\n"
        if self.cfg.max_acc_frac is not None:
            msg += f"\tMax-acc fraction: {self.cfg.max_acc_frac:.1%} (max_acc={self.cfg.max_acc})\n"
        else:
            msg += "\tMax-acc disabled for all envs\n"
        return msg

    # ------------------------------------------------------------------
    # Public read-only views of the bucket tables (the trajectory cmd
    # reads these to do its per-step bucket-of-vel_target_b lookup).
    # ------------------------------------------------------------------

    @property
    def skill_lin_vel_x(self) -> torch.Tensor:
        """``[num_skills, 2]`` per-skill lin-x bucket ``[min, max]``."""
        return self._skill_lin_vel_x

    @property
    def skill_lin_vel_y(self) -> torch.Tensor:
        """``[num_skills, 2]`` per-skill lin-y bucket ``[min, max]``."""
        return self._skill_lin_vel_y

    @property
    def skill_ang_vel_z(self) -> torch.Tensor:
        """``[num_skills, 2]`` per-skill ang-z bucket ``[min, max]``."""
        return self._skill_ang_vel_z

    # ------------------------------------------------------------------
    # Sampling / commit helpers
    # ------------------------------------------------------------------

    def _sample_velocity_for_skill(
        self, new_skill: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Draw ``(vx, vy, wz, heading)`` for each env from its skill bucket."""
        rng_vx = self._skill_lin_vel_x[new_skill]
        rng_vy = self._skill_lin_vel_y[new_skill]
        rng_wz = self._skill_ang_vel_z[new_skill]
        rng_h = self._skill_heading[new_skill]
        u = torch.empty(new_skill.shape[0], device=self.device)
        new_vx = rng_vx[:, 0] + (rng_vx[:, 1] - rng_vx[:, 0]) * u.uniform_(0.0, 1.0)
        new_vy = rng_vy[:, 0] + (rng_vy[:, 1] - rng_vy[:, 0]) * u.uniform_(0.0, 1.0)
        new_wz = rng_wz[:, 0] + (rng_wz[:, 1] - rng_wz[:, 0]) * u.uniform_(0.0, 1.0)
        new_heading = rng_h[:, 0] + (rng_h[:, 1] - rng_h[:, 0]) * u.uniform_(0.0, 1.0)
        return new_vx, new_vy, new_wz, new_heading

    def _commit_live(
        self,
        env_ids: torch.Tensor,
        new_skill: torch.Tensor,
        new_vx: torch.Tensor,
        new_vy: torch.Tensor,
        new_wz: torch.Tensor,
        new_heading: torch.Tensor,
    ) -> None:
        """Write a freshly-sampled bucket + velocity into the live buffers."""
        self.sampled_skill_id[env_ids] = new_skill
        self.vel_target_sampled_b[env_ids, 0] = new_vx
        self.vel_target_sampled_b[env_ids, 1] = new_vy
        self.vel_target_sampled_b[env_ids, 2] = new_wz
        self.heading_target[env_ids] = new_heading

    def _assign_control_modes(self, env_ids: Sequence[int], r: torch.Tensor) -> None:
        """Re-roll the per-env open-loop / closed-loop / standing buckets."""
        self.is_closed_loop_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_closed_loop
        self.is_closed_loop_yaw_env[env_ids] = torch.logical_and(
            r <= self.cfg.rel_closed_loop_yaw + self.cfg.rel_open_loop,
            r >= self.cfg.rel_open_loop,
        )
        self.is_standing_env[env_ids] = torch.logical_and(
            r <= self.cfg.rel_standing_envs + self.cfg.rel_closed_loop_yaw + self.cfg.rel_open_loop,
            r >= self.cfg.rel_closed_loop_yaw + self.cfg.rel_open_loop,
        )

    def _resample_y_and_gains(self, env_ids: Sequence[int], r: torch.Tensor) -> None:
        """Re-roll the y target and y PD gains (skill-independent)."""
        self.y_target[env_ids] = (
            r.uniform_(*self.cfg.ranges.y_pos_offset)
            + wp.to_torch(self.robot.data.root_pos_w)[env_ids, 1]
        )
        self.y_kp[env_ids] = r.uniform_(*self.cfg.ranges.y_kp)
        self.y_kd[env_ids] = r.uniform_(*self.cfg.ranges.y_kd)

    def _reroll_max_acc_clamp_for_reset(self, env_ids: torch.Tensor) -> None:
        """Re-roll the per-env max-acc clamp assignment for reset envs."""
        if self.cfg.max_acc_frac is None:
            return
        rand = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0)
        has_max_acc = rand < self.cfg.max_acc_frac
        self.max_acc_per_env[env_ids] = torch.where(
            has_max_acc,
            torch.full_like(rand, float(self.cfg.max_acc)),
            torch.full_like(rand, float("inf")),
        )

    # ------------------------------------------------------------------
    # Main resample / reset / commit entry points
    # ------------------------------------------------------------------

    def _resample_command(self, env_ids: Sequence[int]):
        """Mid-episode resample: sample a fresh velocity target.

        Conceptually this is the joystick driver telling the robot to
        head for a new velocity.  The bucket is sampled from the env's
        current reference-frame cell only to pick which velocity range
        to draw from; the bucket itself is informational
        (``sampled_skill_id``).  The trajectory cmd derives its active
        skill from where the *ramped* ``vel_target_b`` ends up, not from
        the bucket sampled here — so there's no defer-to-gate path on
        this side anymore.
        """
        env_ids_t = (
            env_ids if isinstance(env_ids, torch.Tensor)
            else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        )
        n = env_ids_t.shape[0]
        r = torch.empty(n, device=self.device)
        self._assign_control_modes(env_ids_t, r)

        # Sample bucket from the env's current reference-frame xy cell.
        # ``ref_poses`` (the latest in-contact foot landing pose, snapped
        # to the stair top for terrain-aware skills) is stable across the
        # pelvis's per-step jitter, which avoids cell-misclassification
        # when a stride briefly carries the pelvis off-cell.
        if self._traj_cmd is None:
            self._traj_cmd = self._env.command_manager.get_term(
                self.cfg.trajectory_command_name
            )
        xy_ref = self._traj_cmd.ref_poses[env_ids_t, :2]
        new_skill = self._sample_skill(env_ids_t, xy_w=xy_ref)
        new_vx, new_vy, new_wz, new_heading = self._sample_velocity_for_skill(new_skill)

        self._commit_live(env_ids_t, new_skill, new_vx, new_vy, new_wz, new_heading)
        self._resample_y_and_gains(env_ids_t, r)

    def reset_for_episode(self, env_ids: torch.Tensor) -> None:
        """Reset velocity state for freshly spawned envs.

        Samples a bucket from each env's *spawn cell* (``env.scene.env_origins``,
        not the stale ``robot.data.root_pos_w``), commits the new
        sampled bucket + velocity + heading immediately to the live
        buffers, snaps the ramped target ``vel_target_b`` past the
        max-acc clamp, and re-rolls the per-episode max-acc bucket.

        Replaces the old ``_resample`` + ``_reset_env_mask`` toggle dance
        from :func:`reset_on_reference`.
        """
        env_ids_t = (
            env_ids if isinstance(env_ids, torch.Tensor)
            else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        )
        n = env_ids_t.shape[0]
        if n == 0:
            return
        r = torch.empty(n, device=self.device)
        self._assign_control_modes(env_ids_t, r)

        # Sample from the *spawn* cell — ``root_pos_w`` is still the
        # end-of-last-episode position at reset-event time, so we pass
        # ``env_origins`` through the ``xy_w`` override.
        xy_spawn = self._env.scene.env_origins[env_ids_t, :2]
        new_skill = self._sample_skill(env_ids_t, xy_w=xy_spawn)
        new_vx, new_vy, new_wz, new_heading = self._sample_velocity_for_skill(new_skill)

        # Immediate live commit.
        self._commit_live(env_ids_t, new_skill, new_vx, new_vy, new_wz, new_heading)

        # Snap the ramped target past the max-acc ramp — the previous
        # episode's ``vel_target_b`` is meaningless across a reset
        # discontinuity.
        self.vel_target_b[env_ids_t, 0] = new_vx
        self.vel_target_b[env_ids_t, 1] = new_vy
        self.vel_target_b[env_ids_t, 2] = new_wz

        self._resample_y_and_gains(env_ids_t, r)
        self._reroll_max_acc_clamp_for_reset(env_ids_t)

        # Refresh the parent's resample timer so the framework's
        # ``CommandTerm.reset`` doesn't immediately re-fire ``_resample``.
        self.time_left[env_ids_t] = self.time_left[env_ids_t].uniform_(
            *self.cfg.resampling_time_range
        )

    def _sample_skill(
        self, env_ids: Sequence[int], xy_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample one skill index per env from the terrain's per-cell distribution.

        Without ``skill_transition_prob``: each env's skill is drawn
        independently from its cell's ``skill_probs`` row each resample.

        With ``skill_transition_prob = p``: each env independently either keeps
        its previous skill (prob ``1 − p``) or samples a new one from its cell's
        ``skill_probs`` renormalised to exclude the current skill (prob ``p``).
        Cells whose distribution puts all mass on the previous skill leave the
        env on its current skill (warning logged once).

        Args:
            env_ids: Env indices to sample for.
            xy_w: Optional override for the world-frame ``(x, y)`` used to
                pick each env's cell.  Default uses the live robot pelvis
                position.  Pass the per-env spawn position (e.g.
                ``env.scene.env_origins[..., :2]``) on the reset path —
                otherwise ``root_pos_w`` is stale and resolves to the
                end-of-last-episode cell.
        """
        # Per-env world XY → per-env skill probability vector. The importer
        # encapsulates the cell (and per-block, for composite importers)
        # lookup behind ``skill_probs_at``.
        if xy_w is None:
            xy_w = wp.to_torch(self.robot.data.root_pos_w)[env_ids, :2]
        per_env_probs = self._terrain.skill_probs_at(xy_w)  # (n, num_skills)

        if self.cfg.skill_transition_prob is None:
            return torch.multinomial(per_env_probs, num_samples=1).squeeze(-1)

        n = len(env_ids)
        prev_skill = self.sampled_skill_id[env_ids]
        new_skill = prev_skill.clone()

        transition_rand = torch.empty(n, device=self.device).uniform_(0.0, 1.0)
        is_transition = transition_rand < self.cfg.skill_transition_prob
        transition_idx = is_transition.nonzero(as_tuple=False).flatten()

        if len(transition_idx) > 0:
            trans_probs = per_env_probs[transition_idx].clone()
            trans_prev = prev_skill[transition_idx]
            n_trans = transition_idx.shape[0]
            trans_probs[torch.arange(n_trans, device=self.device), trans_prev] = 0.0
            prob_sums = trans_probs.sum(dim=1)

            degenerate = prob_sums <= 0.0
            if degenerate.any():
                if not self._warned_no_transition:
                    logger.warning(
                        "skill_transition_prob is set, but some envs are on a cell whose "
                        "skill distribution puts all mass on their current skill — no other "
                        "skill available to transition to. Keeping those envs on the current skill."
                    )
                    self._warned_no_transition = True
                deg_idx = degenerate.nonzero(as_tuple=False).flatten()
                trans_probs[deg_idx] = 0.0
                trans_probs[deg_idx, trans_prev[deg_idx]] = 1.0
                prob_sums = trans_probs.sum(dim=1)

            trans_probs = trans_probs / prob_sums.unsqueeze(1)
            sampled = torch.multinomial(trans_probs, num_samples=1).squeeze(-1)
            new_skill[transition_idx] = sampled

        return new_skill
