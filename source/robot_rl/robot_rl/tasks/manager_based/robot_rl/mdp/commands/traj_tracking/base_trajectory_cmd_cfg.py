"""Shared configuration for trajectory command terms."""

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass


@configclass
class BaseTrajectoryCommandCfg(CommandTermCfg):
    """Base configuration shared by all trajectory command variants.

    Subclasses set ``class_type`` to point at the concrete command class.
    """

    asset_name: str = "robot"
    contact_bodies: list[str] = None
    conditioner_generator_name: str = ""
    path: str = ""
    hf_repo: str = None
    Q_weights: list[float] = None
    R_weights: list[float] = None
    resampling_time_range: tuple[float, float] = (5.0, 15.0)
