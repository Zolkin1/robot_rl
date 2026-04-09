from isaaclab.utils import configclass

from .velocity_commands_cfg import VelocityTrackingCommandCfg


@configclass
class VelocityBucketCfg:
    """A velocity bucket assigning a fraction of envs to a specific velocity range."""

    percentage: float = 0.0
    """Fraction of environments assigned to this bucket (0.0 to 1.0)."""

    lin_vel_x: tuple[float, float] = (0.0, 0.0)
    """Range for the linear-x velocity command (m/s)."""

    lin_vel_y: tuple[float, float] = (0.0, 0.0)
    """Range for the linear-y velocity command (m/s)."""

    ang_vel_z: tuple[float, float] = (0.0, 0.0)
    """Range for the angular-z velocity command (rad/s)."""

    heading: tuple[float, float] | None = None
    """Range for the heading command (rad). If None, uses the parent config's heading range."""


@configclass
class MultiskillVelocityTrackingCommandCfg(VelocityTrackingCommandCfg):
    """Velocity tracking command config with per-bucket velocity ranges for multiskill training.

    Each velocity bucket specifies a fraction of environments and the velocity range
    they should sample from. Any remaining fraction (1.0 - sum of bucket percentages)
    uses the default uniform range from ``ranges``.

    Standing behavior is achieved via a zero-velocity bucket rather than ``rel_standing_envs``.
    """

    class_type: type | str = "{DIR}.multiskill_velocity_commands:MultiskillVelocityTrackingCommand"

    velocity_buckets: list[VelocityBucketCfg] = []
    """List of velocity buckets. Percentages should sum to <= 1.0.
    Any remainder is assigned to the default uniform range from ``ranges``."""

    rel_standing_envs: float = 0.0
    """Not used -- standing is a zero-velocity bucket. Kept for parent compatibility."""
