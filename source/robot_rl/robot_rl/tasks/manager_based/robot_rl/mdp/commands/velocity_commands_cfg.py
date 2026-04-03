from dataclasses import MISSING

from isaaclab.utils import configclass

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg


@configclass
class VelocityTrackingCommandCfg(UniformVelocityCommandCfg):
    class_type: type | str = "{DIR}.velocity_commands:VelocityTrackingCommand"

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
