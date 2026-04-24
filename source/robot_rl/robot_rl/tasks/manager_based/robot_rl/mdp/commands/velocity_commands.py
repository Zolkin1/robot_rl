from __future__ import annotations

import torch
import warp as wp
from typing import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .velocity_commands_cfg import VelocityTrackingCommandCfg


# rel_open_loop gives the percentage of the envs that just take open loop velocity commands
# rel_closed_loop gives the percentage of the envs that use both y and yaw tracking controllers
# rel_closed_loop_yaw
# no y only because that is just closed loop (needs yaw control)
class VelocityTrackingCommand(UniformVelocityCommand):
    """Velocity tracking command."""
    cfg: VelocityTrackingCommandCfg

    def __init__(self, cfg: VelocityTrackingCommand, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.y_target = torch.zeros_like(env.scene.env_origins[:, 1])
        self.y_kp = torch.zeros_like(env.scene.env_origins[:, 1])
        self.y_kd = torch.zeros_like(env.scene.env_origins[:, 1])

        self.is_closed_loop_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_closed_loop_yaw_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.command_dt = env.cfg.sim.dt * env.cfg.decimation
        self.vel_target_b = torch.zeros(env.num_envs, 3, device=self.device)
        self.vel_target_sampled_b = torch.zeros(env.num_envs, 3, device=self.device)

        rel_sum = (cfg.rel_open_loop + cfg.rel_closed_loop
                   + cfg.rel_closed_loop_yaw + cfg.rel_standing_envs)

        if rel_sum != 1.0:
            raise ValueError("Relative envs for the velocity tracking command don't sum to 1!")

    def __str__(self) -> str:
        msg = "Velocity Tracking Command:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"

        return msg

    def reset(self, env_ids: Sequence[int] | None = None):
        """
        Resets the command, but still respects the resample time.
        """
        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)

        # add logging metrics
        extras = {}
        for metric_name, metric_value in self.metrics.items():
            # compute the mean metric value
            extras[metric_name] = torch.mean(metric_value[env_ids]).item()
            # reset the metric value
            metric_value[env_ids] = 0.0

        # set the command counter to zero
        self.command_counter[env_ids] = 0


        # resample the command, but only ones that should be resampled
        resample_env_ids = (self.time_left[env_ids] <= 0.0).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._resample(env_ids[resample_env_ids])
            self.vel_target_b[env_ids[resample_env_ids]] = self.vel_target_sampled_b[env_ids[resample_env_ids]]

        return extras

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command."""
        r = torch.empty(len(env_ids), device=self.device)

        # Determine which envs use what controller
        self.is_closed_loop_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_closed_loop
        self.is_closed_loop_yaw_env[env_ids] = torch.logical_and(
            r <= self.cfg.rel_closed_loop_yaw + self.cfg.rel_open_loop,
            r >= self.cfg.rel_open_loop)
        self.is_standing_env[env_ids] = torch.logical_and(
            r <= self.cfg.rel_standing_envs + self.cfg.rel_closed_loop_yaw + self.cfg.rel_open_loop,
            r >= self.cfg.rel_closed_loop_yaw + self.cfg.rel_open_loop)

        # -- linear velocity - x direction
        self.vel_target_sampled_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_target_sampled_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_target_sampled_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # # -- linear velocity - x direction
        # self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # # -- linear velocity - y direction
        # self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # # -- ang vel yaw - rotation around z
        # self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # Heading target
        self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)

        # y position target
        self.y_target[env_ids] = r.uniform_(*self.cfg.ranges.y_pos_offset) + wp.to_torch(self.robot.data.root_pos_w)[env_ids, 1]

        # y gains
        self.y_kp[env_ids] = r.uniform_(*self.cfg.ranges.y_kp)
        self.y_kd[env_ids] = r.uniform_(*self.cfg.ranges.y_kd)

    def _update_command(self):
        """Post-processes the velocity command."""
        yaw_env_ids = self.is_closed_loop_yaw_env.nonzero(as_tuple=False).flatten()
        cl_env_ids = self.is_closed_loop_env.nonzero(as_tuple=False).flatten()
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()

        # Ramp the active target toward the latest sampled target at the configured
        # max acceleration. PD overrides below act on the ramped target.
        self.vel_target_b = torch.clamp(
            self.vel_target_sampled_b,
            min=self.vel_target_b - self.cfg.max_acc * self.command_dt,
            max=self.vel_target_b + self.cfg.max_acc * self.command_dt,
        )

        self.vel_command_b[:] = self.vel_target_b

        # yaw only envs
        heading_error = math_utils.wrap_to_pi(self.heading_target[yaw_env_ids] - wp.to_torch(self.robot.data.heading_w)[yaw_env_ids])
        self.vel_command_b[yaw_env_ids, 2] = torch.clip(
            self.cfg.heading_control_stiffness * heading_error,     # TODO: Consider adding a D term
            min=self.cfg.ranges.ang_vel_z[0],
            max=self.cfg.ranges.ang_vel_z[1],
        )

        # cl envs
        y_error = self.y_target[cl_env_ids] - wp.to_torch(self.robot.data.root_pos_w)[cl_env_ids, 1]
        heading_error = math_utils.wrap_to_pi(self.heading_target[cl_env_ids] - wp.to_torch(self.robot.data.heading_w)[cl_env_ids])
        y_vel_error = -wp.to_torch(self.robot.data.root_vel_w)[cl_env_ids, 1]     #  TODO: Consider moving average filter here
        self.vel_command_b[cl_env_ids, 1] = torch.clip(
            self.y_kp[cl_env_ids] * y_error + self.y_kd[cl_env_ids] * y_vel_error,
            min=self.cfg.ranges.lin_vel_y[0],
            max=self.cfg.ranges.lin_vel_y[1],
        )
        self.vel_command_b[cl_env_ids, 2] = torch.clip(
            self.cfg.heading_control_stiffness * heading_error,     # TODO: Consider adding a D term
            min=self.cfg.ranges.ang_vel_z[0],
            max=self.cfg.ranges.ang_vel_z[1],
        )

        # standing
        self.vel_command_b[standing_env_ids, :] = 0.0

# class TreadmillVelocityCommand(UniformVelocityCommand):
#     """Base velocity command that also does PD control about a y position."""
#     cfg: TreadmillVelocityCommandCfg
#
#     def __init__(self, cfg: TreadmillVelocityCommandCfg, env: ManagerBasedEnv):
#         super().__init__(cfg, env)
#
#         self.y_target = env.scene.env_origins[:, 1]
#
#         self.cfg.ranges.heading = (0.0, 0.0)    # Never want to sample a heading that causes a y change.
#
#         self.is_y_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
#
#     def __str__(self) -> str:
#         """Return a string representation of the command."""
#         msg = "NormalVelocityCommand:\n"
#         msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
#         msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
#         msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
#         return msg
#
#     def _resample_command(self, env_ids: Sequence[int]):
#         """Resample the command."""
#         # sample velocity commands
#         r = torch.empty(len(env_ids), device=self.device)
#         # -- linear velocity - x direction
#         self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
#         # -- linear velocity - y direction
#         self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
#         # -- ang vel yaw - rotation around z
#         self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
#
#         # Determine how many envs are using y PD controllers
#         self.is_y_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_y_envs
#
#         # heading target
#         if self.cfg.heading_command:
#             self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
#
#             # all envs should also be heading envs to prevent conflicts
#             self.is_heading_env[env_ids] = self.is_y_env[env_ids]
#
#             # update heading envs
#             # self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
#
#
#         # update standing envs
#         self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs
#
#     def _update_command(self):
#         """Post-processes the velocity command.
#
#         This function sets velocity command to zero for standing environments and computes angular
#         velocity from heading direction if the heading_command flag is set.
#         """
#         # Compute angular velocity from heading direction
#         if self.cfg.heading_command:
#             # resolve indices of heading envs
#             env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
#             # compute angular velocity
#             heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
#             self.vel_command_b[env_ids, 2] = torch.clip(
#                 self.cfg.heading_control_stiffness * heading_error,
#                 min=self.cfg.ranges.ang_vel_z[0],
#                 max=self.cfg.ranges.ang_vel_z[1],
#             )
#
#         y_env_ids = self.is_y_env.nonzero(as_tuple=False).flatten()
#
#         # Compute Y velocity command
#         y_error = self.y_target[y_env_ids] - self.robot.data.root_pos_w[y_env_ids, 1]
#         y_vel_error = -self.robot.data.root_vel_w[y_env_ids, 1]
#         self.vel_command_b[y_env_ids, 1] = torch.clip(
#                 self.cfg.y_pos_kp * y_error + self.cfg.y_pos_kd * y_vel_error,
#                 min=self.cfg.ranges.lin_vel_y[0],
#                 max=self.cfg.ranges.lin_vel_y[1],
#             )
#
#         # Enforce standing (i.e., zero velocity command) for standing envs
#         # TODO: check if conversion is needed
#         standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
#         self.vel_command_b[standing_env_ids, :] = 0.0