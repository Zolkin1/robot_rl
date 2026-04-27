"""Batched multi-skill trajectory command using :class:`MultiSkillManager`."""

from __future__ import annotations

import torch
import warp as wp

from .base_trajectory_cmd import BaseTrajectoryCommand
from .manager_base import ManagerBase
from .multiskill_manager import MultiSkillManager, _EPISODIC_INT


# TODO: Should probably change this name
class BatchedMultiSkillCommand(BaseTrajectoryCommand):
    """Trajectory command backed by a :class:`MultiSkillManager`.

    Supports multiple skills (subfolders of trajectories) with fully
    batched tensor evaluation — no per-trajectory Python loops at runtime.

    Maintains an explicit per-env trajectory clock ``self.traj_time`` that
    advances linearly at ``step_dt``.  On a detected skill change the clock
    is re-aligned via :meth:`MultiSkillManager.compute_transition_time` so
    the new skill starts at a phi that matches the current stance foot and
    preserves the fractional position in the period (controlled by
    ``cfg.smooth_transitions``).

    When ``cfg.contact_gate_window_frac`` is not ``None``, the clock is
    additionally gated on foot contact at every per-trajectory-type period
    boundary (half-period for half-periodic, full cycle for full-periodic /
    episodic).  Inside the gate's validity window a contact event snaps the
    clock to the boundary; past the boundary without contact, the clock
    holds at the boundary until contact lands.
    """

    def _create_manager(self, cfg, env) -> ManagerBase:
        """Create a :class:`MultiSkillManager` from a folder of skill subfolders.

        Args:
            cfg: Command term configuration.
            env: IsaacLab environment.

        Returns:
            A :class:`MultiSkillManager` instance.
        """
        return MultiSkillManager(
            path=cfg.path,
            device=env.device,
            env=env,
            conditioner_generator_name=cfg.conditioner_generator_name,
            hf_repo=cfg.hf_repo,
        )

    def _verify_contact_frames(self) -> None:
        """Verify every trajectory contact frame is in ``self.contact_bodies``."""
        for traj_bodies_list in self.manager._contact_bodies_per_domain:
            for domain_bodies in traj_bodies_list:
                for frame in domain_bodies:
                    if frame not in self.contact_bodies:
                        raise ValueError(
                            f"Contact frame '{frame}' from a trajectory is "
                            f"not in the contact frames list: {self.contact_bodies}"
                        )

    def _post_init(self) -> None:
        """Initialise clock state, the manager's ref-frame lookup, and the
        contact-gate handle when gating is enabled."""
        self.manager.build_ref_frame_map(self.ref_frames)

        self.traj_time = torch.zeros(self.num_envs, device=self.device)
        # Accumulated offset from smooth skill transitions, persists through
        # the rest of the episode. Cleared on reset.
        self.skill_time_offset = torch.zeros(self.num_envs, device=self.device)
        self.prev_traj_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_skill_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._initialized = False

        # Contact-gate state.
        self._gating_enabled = self.cfg.contact_gate_window_frac is not None
        self.contact_gate_offset = torch.zeros(self.num_envs, device=self.device)
        self.next_gate_idx = -torch.ones(self.num_envs, dtype=torch.long, device=self.device)
        self.prev_period_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        if self._gating_enabled:
            self.manager.set_gate_contact_layout(self.contact_bodies)
            # For each contact body, find the (sensor_idx, body_idx_within_sensor)
            # so reads can pull from the right sensor tensor.
            self._contact_sensors = [
                self.env.scene.sensors[name] for name in self.cfg.contact_sensor_names
            ]
            per_body_sensor_idx: list[int] = []
            per_body_local_idx: list[int] = []
            for name in self.contact_bodies:
                found = False
                for si, sensor in enumerate(self._contact_sensors):
                    sensor_body_names = list(sensor.body_names)
                    if name in sensor_body_names:
                        per_body_sensor_idx.append(si)
                        per_body_local_idx.append(sensor_body_names.index(name))
                        found = True
                        break
                if not found:
                    raise ValueError(
                        f"Contact body '{name}' is not tracked by any of "
                        f"{list(self.cfg.contact_sensor_names)}. "
                        f"Per-sensor bodies: "
                        f"{[(n, list(s.body_names)) for n, s in zip(self.cfg.contact_sensor_names, self._contact_sensors)]}"
                    )
            self._contact_body_sensor_idx = torch.tensor(
                per_body_sensor_idx, dtype=torch.long, device=self.device
            )
            self._contact_body_local_idx = torch.tensor(
                per_body_local_idx, dtype=torch.long, device=self.device
            )

    def _read_contact_now(self) -> torch.Tensor:
        """Return ``[N, B]`` boolean tensor of which contact bodies are
        currently in contact, in ``self.contact_bodies`` order.

        Each contact body is fetched from its owning sensor's
        ``net_forces_w_history`` and combined into a single per-env mask.
        """
        result = torch.zeros(
            self.num_envs, len(self.contact_bodies), dtype=torch.bool, device=self.device
        )
        for si, sensor in enumerate(self._contact_sensors):
            sensor_mask = self._contact_body_sensor_idx == si
            if not sensor_mask.any():
                continue
            local_ids = self._contact_body_local_idx[sensor_mask]
            forces = wp.to_torch(sensor.data.net_forces_w_history)
            forces = forces[:, :, local_ids, :]
            force_mag = forces.norm(dim=-1).max(dim=1).values
            result[:, sensor_mask] = force_mag > self.cfg.contact_force_threshold
        return result

    def _apply_contact_gate(
        self,
        t: torch.Tensor,
        traj_idx: torch.Tensor,
        contact_now: torch.Tensor,
    ) -> torch.Tensor:
        """Adjust ``t`` (and ``self.contact_gate_offset`` / ``self.next_gate_idx``)
        according to the contact-gate fire / hold rules.

        Args:
            t: ``[N]`` per-env trajectory time before gating (already includes
                ``contact_gate_offset`` from prior steps).
            traj_idx: ``[N]`` global trajectory index per env.
            contact_now: ``[N, B]`` runtime contact state in
                ``self.contact_bodies`` order.

        Returns:
            ``[N]`` gated trajectory time.
        """
        total = self.manager.data["total_time"][traj_idx]
        traj_type = self.manager.data["traj_type"][traj_idx]
        num_gates = self.manager._num_gates_per_traj[traj_idx]

        # Detect natural wrap (t crossed a period boundary since last step).
        period_idx = torch.div(t, total, rounding_mode="floor").long()
        wrapped = period_idx > self.prev_period_idx
        has_gates = num_gates > 0
        # Reset on wrap: envs with gates → start of period (idx 0); else stay -1.
        self.next_gate_idx = torch.where(
            wrapped & has_gates,
            torch.zeros_like(self.next_gate_idx),
            self.next_gate_idx,
        )
        self.next_gate_idx = torch.where(
            wrapped & ~has_gates,
            -torch.ones_like(self.next_gate_idx),
            self.next_gate_idx,
        )

        active = self.next_gate_idx >= 0
        if not active.any():
            self.prev_period_idx = period_idx
            return t

        safe_idx = torch.clamp(self.next_gate_idx, min=0)
        target_phi = self.manager._gate_phi_table[traj_idx, safe_idx]
        target_active = self.manager._gate_active_table[traj_idx, safe_idx]
        target_mask = self.manager._gate_contact_mask[traj_idx, safe_idx]

        t_in_period = t - period_idx.to(t.dtype) * total
        phi = t_in_period / total

        window_lo = target_phi * (1.0 - self.cfg.contact_gate_window_frac)
        in_window = phi >= window_lo
        expected_landed = (contact_now & target_mask).any(dim=1)

        usable = active & target_active
        fire_mask = in_window & expected_landed & usable
        hold_mask = (phi >= target_phi) & usable & ~fire_mask

        # Fire snaps t forward to gate boundary inside the current period.
        # For target_phi == 1.0 this lands exactly on the next-period start.
        fire_target_t = period_idx.to(t.dtype) * total + target_phi * total
        fire_delta = torch.where(fire_mask, fire_target_t - t, torch.zeros_like(t))

        # Hold pulls t back to just below the boundary; for target_phi == 1.0
        # we keep one step's worth of phi clearance so phi stays near 1.0
        # rather than wrapping to 0.0.
        eps_phi = self.env.step_dt / total
        hold_phi = torch.where(target_phi >= 1.0, 1.0 - eps_phi, target_phi)
        hold_target_t = period_idx.to(t.dtype) * total + hold_phi * total
        hold_delta = torch.where(hold_mask, hold_target_t - t, torch.zeros_like(t))

        self.contact_gate_offset = self.contact_gate_offset + fire_delta + hold_delta
        t = t + fire_delta + hold_delta

        # Advance next_gate_idx for fired envs.
        new_idx = self.next_gate_idx + 1
        last_fire = fire_mask & (new_idx >= num_gates)
        is_episodic = traj_type == _EPISODIC_INT

        self.next_gate_idx = torch.where(
            fire_mask & ~last_fire,
            new_idx,
            self.next_gate_idx,
        )
        self.next_gate_idx = torch.where(
            last_fire & ~is_episodic,
            torch.zeros_like(self.next_gate_idx),
            self.next_gate_idx,
        )
        self.next_gate_idx = torch.where(
            last_fire & is_episodic,
            -torch.ones_like(self.next_gate_idx),
            self.next_gate_idx,
        )

        self.prev_period_idx = torch.div(t, total, rounding_mode="floor").long()
        return t

    def _compute_time(self) -> torch.Tensor:
        """Compute the per-env trajectory time, aligning on skill changes
        and applying contact-gated phase advancement when enabled.

        Must be idempotent across the same env step: IsaacLab's
        ``CommandTerm.compute`` calls ``_update_command`` twice when a
        resample fires (once from ``_resample_command`` and once directly),
        so we compute ``traj_time`` deterministically from
        ``episode_length_buf`` + accumulated offsets rather than with a
        ``+= step_dt`` accumulator.
        """
        reset_mask = self.env.episode_length_buf == 0
        advancing = ~reset_mask

        # Clear skill-transition + gate offsets at episode start and re-sample
        # the random-start offset if configured.
        self.skill_time_offset[reset_mask] = 0.0
        self.contact_gate_offset[reset_mask] = 0.0
        self.prev_period_idx[reset_mask] = 0
        if self.cfg.random_start_time_max > 0:
            rand_idx = torch.where(reset_mask)[0]
            if rand_idx.numel() > 0:
                self.time_offset[rand_idx] = (
                    torch.rand(rand_idx.shape, device=self.device)
                    * self.cfg.random_start_time_max
                )

        # Baseline: episode counter × step_dt, plus whatever the reset
        # event wrote into init_time_offset, plus random-start offset,
        # plus accumulated skill-transition + contact-gate offsets.
        t = (
            self.env.episode_length_buf * self.env.step_dt
            + self.init_time_offset
            + self.time_offset
            + self.skill_time_offset
            + self.contact_gate_offset
        )

        # Resolve current trajectory / skill assignment
        self.manager.invalidate_cache()
        cur_traj = self.manager.get_current_trajectory_indices()
        cur_skill = self.manager.data["skill_idx"][cur_traj]

        # Re-seed gate state for envs that just reset.  We pick the first
        # gate whose phi is >= the env's initial phi so an env placed past a
        # boundary by ``reset_on_reference`` waits for the *next* gate rather
        # than being yanked back.
        if self._gating_enabled and reset_mask.any():
            init_total = self.manager.data["total_time"][cur_traj]
            init_phi = (t % init_total) / init_total
            gates = self.manager._gate_phi_table[cur_traj]
            gate_active = self.manager._gate_active_table[cur_traj]
            upcoming = gate_active & (gates >= init_phi.unsqueeze(1))
            has_upcoming = upcoming.any(dim=1)
            first_upcoming_idx = upcoming.to(torch.long).argmax(dim=1)
            # No upcoming gate this period → wait for natural wrap then start at idx 0.
            init_idx = torch.where(has_upcoming, first_upcoming_idx, torch.zeros_like(first_upcoming_idx))
            num_gates_reset = self.manager._num_gates_per_traj[cur_traj]
            init_idx = torch.where(num_gates_reset > 0, init_idx, -torch.ones_like(init_idx))
            self.next_gate_idx = torch.where(reset_mask, init_idx, self.next_gate_idx)
            init_period = torch.div(t, init_total, rounding_mode="floor").long()
            self.prev_period_idx = torch.where(reset_mask, init_period, self.prev_period_idx)

        # Detect skill changes (skip first-ever step and reset envs)
        if self._initialized and self.cfg.smooth_transitions:
            changed = (cur_skill != self.prev_skill_idx) & advancing
            if changed.any():
                target_t = self.manager.compute_transition_time(
                    self.prev_traj_idx[changed],
                    t[changed],
                    cur_traj[changed],
                )
                # Fold the adjustment into the persistent offset so
                # subsequent steps continue from the aligned phase.
                self.skill_time_offset[changed] += target_t - t[changed]
                t = t.clone()
                t[changed] = target_t

                if self._gating_enabled:
                    new_num_gates = self.manager._num_gates_per_traj[cur_traj[changed]]
                    new_idx_for_changed = torch.where(
                        new_num_gates > 0,
                        torch.zeros_like(new_num_gates),
                        -torch.ones_like(new_num_gates),
                    )
                    self.next_gate_idx[changed] = new_idx_for_changed
                    new_total = self.manager.data["total_time"][cur_traj[changed]]
                    self.prev_period_idx[changed] = torch.div(
                        target_t, new_total, rounding_mode="floor"
                    ).long()

        # Apply contact gating (acts on advancing envs only — reset envs
        # already have offset cleared so the gate logic is a no-op for them).
        if self._gating_enabled:
            contact_now = self._read_contact_now()
            t = self._apply_contact_gate(t, cur_traj, contact_now)

        self.traj_time = t
        self.prev_traj_idx = cur_traj.clone()
        self.prev_skill_idx = cur_skill.clone()
        self._initialized = True
        return t
