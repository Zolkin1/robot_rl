from dataclasses import MISSING
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass


@configclass
class GaitPeriodCfg(CommandTermCfg):
    """Configure a gait period command."""

    class_type: type | str = "{DIR}.gaitPeriod:GaitPeriodCommand"

    gait_period_range: tuple[float, float] = MISSING


