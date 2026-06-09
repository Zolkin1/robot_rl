# Copyright 2026 Zachary Olkin. All rights reserved.

from dataclasses import MISSING
from isaaclab.managers import CommandTermCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils import configclass

from .gaitPeriod import GaitPeriodCommand


@configclass
class GaitPeriodCfg(CommandTermCfg):
    """Configure a gait period command."""

    class_type: type = GaitPeriodCommand

    gait_period_range: tuple[float, float] = MISSING


