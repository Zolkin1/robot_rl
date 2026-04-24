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
    """Velocity tracking command with per-bucket velocity ranges for multiskill training.

    Extends VelocityTrackingCommand to support assigning different fractions of
    environments to different velocity ranges (buckets). Any unassigned fraction
    samples from the default uniform range.
    """

    cfg: MultiskillVelocityTrackingCommandCfg

    def __init__(self, cfg: MultiskillVelocityTrackingCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        total_pct = sum(b.percentage for b in cfg.velocity_buckets)
        if total_pct > 1.0:
            raise ValueError(
                f"Velocity bucket percentages sum to {total_pct:.3f}, which exceeds 1.0."
            )
        if 0.0 < total_pct < 1.0:
            logger.warning(
                f"Velocity bucket percentages sum to {total_pct:.3f}. "
                f"Remaining {1.0 - total_pct:.3f} will use the default uniform range."
            )

        if cfg.skill_transition_prob is not None and not 0.0 <= cfg.skill_transition_prob <= 1.0:
            raise ValueError(
                f"skill_transition_prob must be in [0, 1], got {cfg.skill_transition_prob}."
            )

        self.bucket_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Full bucket probability vector including the virtual default-uniform bucket at the last
        # index. Used by the transition-aware sampler to renormalize over "other" buckets.
        remainder = max(0.0, 1.0 - total_pct)
        self._all_bucket_probs = torch.tensor(
            [b.percentage for b in cfg.velocity_buckets] + [remainder],
            device=self.device,
            dtype=torch.float,
        )
        self._warned_no_transition = False

    def __str__(self) -> str:
        """Return a string representation of the command."""
        msg = "Multiskill Velocity Tracking Command:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tNumber of velocity buckets: {len(self.cfg.velocity_buckets)}\n"
        for i, bucket in enumerate(self.cfg.velocity_buckets):
            msg += (
                f"\tBucket {i}: {bucket.percentage:.1%} of envs, "
                f"vx={bucket.lin_vel_x}, vy={bucket.lin_vel_y}, wz={bucket.ang_vel_z}\n"
            )
        remainder = 1.0 - sum(b.percentage for b in self.cfg.velocity_buckets)
        if remainder > 0.0:
            msg += f"\tDefault uniform: {remainder:.1%} of envs\n"
        if self.cfg.skill_transition_prob is not None:
            msg += f"\tSkill transition probability: {self.cfg.skill_transition_prob:.1%}\n"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands, assigning envs to buckets."""
        r = torch.empty(len(env_ids), device=self.device)

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

        num_buckets = len(self.cfg.velocity_buckets)

        # --- Phase A: assign new bucket id per env ---
        new_bucket_id = self._assign_buckets(env_ids, num_buckets)
        self.bucket_id[env_ids] = new_bucket_id

        # --- Phase B: sample velocity / heading per bucket ---
        for i, bucket in enumerate(self.cfg.velocity_buckets):
            mask = new_bucket_id == i
            bucket_env_ids = env_ids[mask]

            if len(bucket_env_ids) > 0:
                br = torch.empty(len(bucket_env_ids), device=self.device)
                self.vel_target_sampled_b[bucket_env_ids, 0] = br.uniform_(*bucket.lin_vel_x)
                self.vel_target_sampled_b[bucket_env_ids, 1] = br.uniform_(*bucket.lin_vel_y)
                self.vel_target_sampled_b[bucket_env_ids, 2] = br.uniform_(*bucket.ang_vel_z)

                heading_range = bucket.heading if bucket.heading is not None else self.cfg.ranges.heading
                self.heading_target[bucket_env_ids] = br.uniform_(*heading_range)

        default_mask = new_bucket_id == num_buckets
        default_env_ids = env_ids[default_mask]

        if len(default_env_ids) > 0:
            dr = torch.empty(len(default_env_ids), device=self.device)
            self.vel_target_sampled_b[default_env_ids, 0] = dr.uniform_(*self.cfg.ranges.lin_vel_x)
            self.vel_target_sampled_b[default_env_ids, 1] = dr.uniform_(*self.cfg.ranges.lin_vel_y)
            self.vel_target_sampled_b[default_env_ids, 2] = dr.uniform_(*self.cfg.ranges.ang_vel_z)
            self.heading_target[default_env_ids] = dr.uniform_(*self.cfg.ranges.heading)

        # y position target and gains (shared across all buckets)
        self.y_target[env_ids] = (
            r.uniform_(*self.cfg.ranges.y_pos_offset)
            + wp.to_torch(self.robot.data.root_pos_w)[env_ids, 1]
        )
        self.y_kp[env_ids] = r.uniform_(*self.cfg.ranges.y_kp)
        self.y_kd[env_ids] = r.uniform_(*self.cfg.ranges.y_kd)

    def _assign_buckets(self, env_ids: Sequence[int], num_buckets: int) -> torch.Tensor:
        """Pick a bucket for each env in ``env_ids`` and return the resulting ids tensor.

        Without ``skill_transition_prob``, samples each env's bucket independently from the
        configured percentages. With ``skill_transition_prob = p``, each env independently either
        keeps its previous bucket (prob ``1 - p``) or samples a new bucket from the per-bucket
        probabilities renormalized to exclude the current bucket (prob ``p``).
        """
        n = len(env_ids)

        if self.cfg.skill_transition_prob is None:
            bucket_rand = torch.empty(n, device=self.device).uniform_(0.0, 1.0)
            new_bucket_id = torch.full((n,), num_buckets, dtype=torch.long, device=self.device)
            cumulative = 0.0
            for i, bucket in enumerate(self.cfg.velocity_buckets):
                mask = (bucket_rand >= cumulative) & (bucket_rand < cumulative + bucket.percentage)
                new_bucket_id[mask] = i
                cumulative += bucket.percentage
            return new_bucket_id

        prev_bucket = self.bucket_id[env_ids]
        new_bucket_id = prev_bucket.clone()

        transition_rand = torch.empty(n, device=self.device).uniform_(0.0, 1.0)
        is_transition = transition_rand < self.cfg.skill_transition_prob
        transition_idx = is_transition.nonzero(as_tuple=False).flatten()

        if len(transition_idx) > 0:
            trans_prev = prev_bucket[transition_idx]
            n_trans = transition_idx.shape[0]

            per_env_probs = self._all_bucket_probs.unsqueeze(0).expand(n_trans, -1).clone()
            per_env_probs[torch.arange(n_trans, device=self.device), trans_prev] = 0.0
            prob_sums = per_env_probs.sum(dim=1)

            # Degenerate: prev bucket carries all the mass — no other bucket to transition to.
            degenerate = prob_sums <= 0.0
            if degenerate.any():
                if not self._warned_no_transition:
                    logger.warning(
                        "skill_transition_prob is set, but some envs are in a bucket with "
                        "probability 1.0 -- no other buckets available to transition to. "
                        "Keeping those envs in their current bucket."
                    )
                    self._warned_no_transition = True
                # Give degenerate rows a valid 1-hot on prev bucket so multinomial returns prev.
                deg_idx = degenerate.nonzero(as_tuple=False).flatten()
                per_env_probs[deg_idx] = 0.0
                per_env_probs[deg_idx, trans_prev[deg_idx]] = 1.0
                prob_sums = per_env_probs.sum(dim=1)

            per_env_probs = per_env_probs / prob_sums.unsqueeze(1)
            sampled = torch.multinomial(per_env_probs, num_samples=1).squeeze(-1)
            new_bucket_id[transition_idx] = sampled

        return new_bucket_id
