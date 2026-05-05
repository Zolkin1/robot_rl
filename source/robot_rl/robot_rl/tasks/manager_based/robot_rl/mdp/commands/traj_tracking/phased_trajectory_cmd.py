"""Trajectory command that exposes a phasing variable and hold-phi logic.

TODO: Delete this file. Single-skill phase-as-observation path was
superseded by :class:`BatchedMultiSkillCommand` (multi-skill manager
holds phase directly). The remaining consumers are the single-skill
g1 configs (walking/running/bow/bend) and ``observations._phased_cmd``;
migrate those to the multi-skill path before removing this module.

Single-skill trajectory commands (those that feed the phase into the
observation and tie trajectory advancement to the base-velocity command)
should extend this class.  Multi-skill commands that advance time
without a phase variable should extend :class:`BaseTrajectoryCommand`
directly.
"""

from __future__ import annotations

from abc import abstractmethod

import torch

from .base_trajectory_cmd import BaseTrajectoryCommand
from .manager_base import ManagerBase


class PhasedTrajectoryCommand(BaseTrajectoryCommand):
    """Base class for trajectory commands that expose a phasing variable.

    Adds:

    - ``self.phasing_var`` / ``self.unmasked_phasing_var`` state for use as
      observation inputs and for phase-binned CLF logging.
    - Hold-phi logic that pauses phase at a domain boundary when the
      commanded base velocity drops below ``cfg.hold_phi_threshold``.
    - Per-episode random phase-hold enabled by ``cfg.percent_hold_phi``.
    """

    @abstractmethod
    def _create_manager(self, cfg, env) -> ManagerBase:
        """Create the concrete trajectory manager."""

    @abstractmethod
    def _verify_contact_frames(self) -> None:
        """Verify trajectory contact frames match the configured list."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self.phasing_var = torch.zeros(self.num_envs, device=self.device)
        self.unmasked_phasing_var = torch.zeros(self.num_envs, device=self.device)
        self.prev_unmasked_phasing_var = torch.zeros(self.num_envs, device=self.device)
        self.hold_envs = torch.ones(self.num_envs, device=self.device)

        self.should_hold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.boundaries_crossed = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.hold_phi_value = -1.0 * torch.ones(self.num_envs, device=self.device)

    def update_phasing_var(self, t: torch.Tensor, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Update the phasing variable, with hold logic on full updates.

        Holds phi at the second boundary crossing (0.0 or 0.5) rather than
        the first, allowing a full phase to complete before stopping when
        velocity is low.

        Args:
            t: Time tensor of shape ``[N]``.
            env_ids: Optional environment indices. If provided, only those
                environments are updated (hold logic is skipped).

        Returns:
            Phasing variable tensor of shape ``[N]``.
        """
        if env_ids is not None:
            new_phi = self.manager.get_phasing_var(t, env_ids)
            self.phasing_var[env_ids] = new_phi
            self.unmasked_phasing_var[env_ids] = new_phi
            return new_phi

        prev_phi = self.phasing_var
        self.prev_unmasked_phasing_var = self.unmasked_phasing_var
        self.phasing_var = self.manager.get_phasing_var(t)
        self.unmasked_phasing_var = self.phasing_var

        cmd_vel = self.env.command_manager.get_command("base_velocity")
        prev_should_hold = self.should_hold.clone()
        self.should_hold = torch.abs(cmd_vel[:, 0]) < self.cfg.hold_phi_threshold

        newly_holding = self.should_hold & ~prev_should_hold
        reset_mask = newly_holding | (self.env.episode_length_buf == 0)
        self.boundaries_crossed[reset_mask] = 0
        self.hold_phi_value[reset_mask] = -1.0

        released = ~self.should_hold
        self.hold_phi_value[released] = -1.0
        self.boundaries_crossed[released] = 0

        active = self.should_hold & (self.hold_phi_value < 0)

        crosses_zero = active & (self.phasing_var < prev_phi) & (prev_phi > 0)
        crosses_half = active & (prev_phi < 0.5) & (self.phasing_var >= 0.5)

        crosses_any = crosses_zero | crosses_half
        self.boundaries_crossed[crosses_any] += 1

        lock_at_zero = crosses_zero & (self.boundaries_crossed >= self.cfg.phasing_boundaries)
        lock_at_half = crosses_half & (self.boundaries_crossed >= self.cfg.phasing_boundaries)
        self.hold_phi_value[lock_at_zero] = 0.0
        self.hold_phi_value[lock_at_half] = 0.5

        holding = self.hold_phi_value >= 0
        self.phasing_var[holding] = self.hold_phi_value[holding]

        return self.phasing_var

    def get_phasing_var(self) -> torch.Tensor:
        """Return the current phasing variable of shape ``[N]``."""
        return self.phasing_var

    def _compute_time(self) -> torch.Tensor:
        """Apply ``percent_hold_phi`` on top of the base time computation."""
        t = super()._compute_time()
        if self.cfg.percent_hold_phi > 0:
            mask = torch.where(self.env.episode_length_buf == 0)[0]
            self.hold_envs[mask] = (
                torch.rand(len(mask), device=self.device) > self.cfg.percent_hold_phi
            ).float()
            t = t * self.hold_envs
        return t

    def get_desired_outputs(self, t: torch.Tensor, env_ids: torch.Tensor = None) -> None:
        """Populate the phasing variable, then delegate to the base."""
        self.update_phasing_var(t, env_ids)
        super().get_desired_outputs(t, env_ids)

    def _update_command(self):
        """Run the base update and log CLF binned by phase."""
        super()._update_command()
        self.manager.log_v_on_phasing_var(self.phasing_var, self.v)
