# Copyright 2026 Zachary Olkin. All rights reserved.

from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg

from .velocity_commands import VelocityTrackingCommand


# @configclass
# class TreadmillVelocityCommandCfg(UniformVelocityCommandCfg):
#     class_type: type = TreadmillVelocityCommand
#
#     y_pos_kp: float = 0.0
#
#     y_pos_kd: float = 0.0
#
#     rel_y_envs: float = 0.0

@configclass
class VelocityTrackingCommandCfg(UniformVelocityCommandCfg):
    class_type: type = VelocityTrackingCommand

    rel_closed_loop: float = MISSING

    rel_open_loop: float = MISSING

    rel_closed_loop_yaw: float = MISSING

    rel_standing_envs: float = MISSING

    max_acc: float = 100.0

    @configclass
    class VelRanges(UniformVelocityCommandCfg.Ranges):
        """Uniform distribution ranges for the velocity tracking command."""
        y_pos_offset: tuple[float, float] = MISSING
        """Range for the sampled y offset."""

        y_kp: tuple[float, float] = MISSING
        """Range for the sampled y kp."""

        y_kd: tuple[float, float] = MISSING
        """Range for the sampled y kd."""

    ranges: VelRanges = MISSING
