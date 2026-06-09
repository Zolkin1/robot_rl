# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .command_cfg import GaitPeriodCfg


class GaitPeriodCommand(CommandTerm):
    """Command generator that generates gait periods."""

    cfg: GaitPeriodCfg

    def __init__(self, cfg: GaitPeriodCfg, env: ManagerBasedEnv):
        """Initialize the gait period generator."""
        super().__init__(cfg, env)

        if self.cfg.gait_period_range[1] < self.cfg.gait_period_range[0]:
            raise ValueError("The GaitPeriodCommandCfg must have max_period >= min_period!")

        # Create a buffer to store the periods
        self.periods = torch.zeros(self.num_envs, device=self.device)

    def __str__(self):
        """Return a string representation of the command."""
        msg = "GaitPeriodCommand:\n"
        msg += f"\tMax period: {self.cfg.gait_period_range[1]}\n"
        msg += f"\tMin period: {self.cfg.gait_period_range[0]}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The commanded stepping period."""
        return self.periods

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update the metrics based on the current state."""
        # TODO: can put in "tracking" metrics here
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample the command for the specified environments."""
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)

        # Gait periods
        self.periods[env_ids] = r.uniform_(*self.cfg.gait_period_range)

    def _update_command(self):
        """Update the command based on the current state."""
        # TODO: Consider any post processing here
        pass