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

    At every resample, each env's current world XY is mapped to its terrain
    cell and the per-skill probability vector ``terrain.skill_probs[:, r, c]``
    is used to draw one skill via :func:`torch.multinomial`. The chosen skill
    indexes into ``cfg.velocity_buckets`` to give a uniform velocity range.

    Requires the scene's terrain importer to be a meta importer exposing
    ``skill_probs``, ``skill_list``, and ``world_xy_to_cell``. The bucket dict
    keys must match the importer's ``skill_list`` exactly.
    """

    cfg: MultiskillVelocityTrackingCommandCfg

    def __init__(self, cfg: MultiskillVelocityTrackingCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # --- Resolve and validate the terrain importer ---
        terrain = env.scene[cfg.terrain_name]
        for required in ("skill_probs", "skill_list", "world_xy_to_cell"):
            if not hasattr(terrain, required):
                raise TypeError(
                    f"MultiskillVelocityTrackingCommand requires the scene's "
                    f"'{cfg.terrain_name}' entity to expose '{required}'. "
                    f"Got {type(terrain).__name__}; this command needs a "
                    f"MetaTerrainImporter (or subclass)."
                )
        self._terrain = terrain
        self._skill_list: list[str] = list(terrain.skill_list)

        bucket_keys = set(cfg.velocity_buckets.keys())
        skill_set = set(self._skill_list)
        if bucket_keys != skill_set:
            missing = skill_set - bucket_keys
            extra = bucket_keys - skill_set
            raise ValueError(
                f"velocity_buckets keys must equal terrain.skill_list. "
                f"missing buckets for skills: {sorted(missing)}; "
                f"extra buckets not in skill_list: {sorted(extra)}."
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
            [cfg.velocity_buckets[s].lin_vel_x for s in self._skill_list],
            device=self.device, dtype=torch.float,
        )
        self._skill_lin_vel_y = torch.tensor(
            [cfg.velocity_buckets[s].lin_vel_y for s in self._skill_list],
            device=self.device, dtype=torch.float,
        )
        self._skill_ang_vel_z = torch.tensor(
            [cfg.velocity_buckets[s].ang_vel_z for s in self._skill_list],
            device=self.device, dtype=torch.float,
        )
        self._skill_heading = torch.stack([
            torch.tensor(cfg.velocity_buckets[s].heading, device=self.device, dtype=torch.float)
            if cfg.velocity_buckets[s].heading is not None
            else default_heading
            for s in self._skill_list
        ])

        self.skill_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
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
            bucket = self.cfg.velocity_buckets[name]
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

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands by sampling a skill per env from the terrain."""
        n = len(env_ids)
        r = torch.empty(n, device=self.device)

        # --- Assign control modes (same as parent) ---
        self.is_closed_loop_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_closed_loop
        self.is_closed_loop_yaw_env[env_ids] = torch.logical_and(
            r <= self.cfg.rel_closed_loop_yaw + self.cfg.rel_open_loop,
            r >= self.cfg.rel_open_loop,
        )
        self.is_standing_env[env_ids] = torch.logical_and(
            r <= self.cfg.rel_standing_envs + self.cfg.rel_closed_loop_yaw + self.cfg.rel_open_loop,
            r >= self.cfg.rel_closed_loop_yaw + self.cfg.rel_open_loop,
        )

        # --- Sample skill per env from the terrain's per-cell distribution ---
        new_skill = self._sample_skill(env_ids)
        self.skill_id[env_ids] = new_skill

        # --- Vectorised velocity sampling: gather per-env ranges, one uniform per axis ---
        rng_vx = self._skill_lin_vel_x[new_skill]   # (n, 2)
        rng_vy = self._skill_lin_vel_y[new_skill]
        rng_wz = self._skill_ang_vel_z[new_skill]
        rng_h = self._skill_heading[new_skill]

        u = torch.empty(n, device=self.device)
        self.vel_target_sampled_b[env_ids, 0] = rng_vx[:, 0] + (rng_vx[:, 1] - rng_vx[:, 0]) * u.uniform_(0.0, 1.0)
        self.vel_target_sampled_b[env_ids, 1] = rng_vy[:, 0] + (rng_vy[:, 1] - rng_vy[:, 0]) * u.uniform_(0.0, 1.0)
        self.vel_target_sampled_b[env_ids, 2] = rng_wz[:, 0] + (rng_wz[:, 1] - rng_wz[:, 0]) * u.uniform_(0.0, 1.0)
        self.heading_target[env_ids] = rng_h[:, 0] + (rng_h[:, 1] - rng_h[:, 0]) * u.uniform_(0.0, 1.0)

        # y position target and gains (shared across all skills)
        self.y_target[env_ids] = (
            r.uniform_(*self.cfg.ranges.y_pos_offset)
            + wp.to_torch(self.robot.data.root_pos_w)[env_ids, 1]
        )
        self.y_kp[env_ids] = r.uniform_(*self.cfg.ranges.y_kp)
        self.y_kd[env_ids] = r.uniform_(*self.cfg.ranges.y_kd)

        # Per-env max-acc clamp: re-roll only on episode reset so each env's
        # clamped/unclamped state persists for the duration of the episode.
        if self.cfg.max_acc_frac is not None:
            reset_mask = self._env.episode_length_buf[env_ids] == 0
            if reset_mask.any():
                reset_ids = env_ids[reset_mask]
                rand = torch.empty(len(reset_ids), device=self.device).uniform_(0.0, 1.0)
                has_max_acc = rand < self.cfg.max_acc_frac
                self.max_acc_per_env[reset_ids] = torch.where(
                    has_max_acc,
                    torch.full_like(rand, float(self.cfg.max_acc)),
                    torch.full_like(rand, float("inf")),
                )

    def _sample_skill(self, env_ids: Sequence[int]) -> torch.Tensor:
        """Sample one skill index per env from the terrain's per-cell distribution.

        Without ``skill_transition_prob``: each env's skill is drawn
        independently from its cell's ``skill_probs`` row each resample.

        With ``skill_transition_prob = p``: each env independently either keeps
        its previous skill (prob ``1 − p``) or samples a new one from its cell's
        ``skill_probs`` renormalised to exclude the current skill (prob ``p``).
        Cells whose distribution puts all mass on the previous skill leave the
        env on its current skill (warning logged once).
        """
        # Per-env current world XY → cell (r, c) → per-env skill probability vector.
        xy_w = wp.to_torch(self.robot.data.root_pos_w)[env_ids, :2]
        rows, cols = self._terrain.world_xy_to_cell(xy_w)
        per_env_probs = self._terrain.skill_probs[:, rows, cols].T  # (n, num_skills)

        if self.cfg.skill_transition_prob is None:
            return torch.multinomial(per_env_probs, num_samples=1).squeeze(-1)

        n = len(env_ids)
        prev_skill = self.skill_id[env_ids]
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
