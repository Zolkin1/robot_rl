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

        self.bucket_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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

        # --- Assign velocity buckets ---
        bucket_rand = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0)
        num_buckets = len(self.cfg.velocity_buckets)

        # Default: all envs get the uniform bucket (index = num_buckets)
        self.bucket_id[env_ids] = num_buckets

        cumulative = 0.0
        for i, bucket in enumerate(self.cfg.velocity_buckets):
            mask = (bucket_rand >= cumulative) & (bucket_rand < cumulative + bucket.percentage)
            bucket_env_ids = env_ids[mask]

            if len(bucket_env_ids) > 0:
                self.bucket_id[bucket_env_ids] = i
                br = torch.empty(len(bucket_env_ids), device=self.device)
                self.vel_target_b[bucket_env_ids, 0] = br.uniform_(*bucket.lin_vel_x)
                self.vel_target_b[bucket_env_ids, 1] = br.uniform_(*bucket.lin_vel_y)
                self.vel_target_b[bucket_env_ids, 2] = br.uniform_(*bucket.ang_vel_z)

                heading_range = bucket.heading if bucket.heading is not None else self.cfg.ranges.heading
                self.heading_target[bucket_env_ids] = br.uniform_(*heading_range)

            cumulative += bucket.percentage

        # Default/uniform bucket: environments not assigned to any specific bucket
        default_mask = self.bucket_id[env_ids] == num_buckets
        default_env_ids = env_ids[default_mask]

        if len(default_env_ids) > 0:
            dr = torch.empty(len(default_env_ids), device=self.device)
            self.vel_target_b[default_env_ids, 0] = dr.uniform_(*self.cfg.ranges.lin_vel_x)
            self.vel_target_b[default_env_ids, 1] = dr.uniform_(*self.cfg.ranges.lin_vel_y)
            self.vel_target_b[default_env_ids, 2] = dr.uniform_(*self.cfg.ranges.ang_vel_z)
            self.heading_target[default_env_ids] = dr.uniform_(*self.cfg.ranges.heading)

        # y position target and gains (shared across all buckets)
        self.y_target[env_ids] = (
            r.uniform_(*self.cfg.ranges.y_pos_offset)
            + wp.to_torch(self.robot.data.root_pos_w)[env_ids, 1]
        )
        self.y_kp[env_ids] = r.uniform_(*self.cfg.ranges.y_kp)
        self.y_kd[env_ids] = r.uniform_(*self.cfg.ranges.y_kd)
