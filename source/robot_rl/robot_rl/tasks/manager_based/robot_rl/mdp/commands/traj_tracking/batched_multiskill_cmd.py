"""Batched multi-skill trajectory command using :class:`MultiSkillManager`."""

from __future__ import annotations

import json
import os
import time
from typing import Optional

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
    clock forward to the boundary; past the boundary without contact the
    clock continues advancing (and may wrap), and the next contact event
    still snaps the clock back to the boundary — a backward jump in phi,
    possibly across the wrap.  The same gate stays armed until contact
    lands or until ``t`` advances one full period past it, at which point
    the gate auto-rearms at its next instance.  Set
    ``cfg.hold_on_late_contact=True`` to restore the legacy hold-at-boundary
    behaviour.
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
        # Legacy path uses prev_period_idx; new path uses armed_period.
        # Both are allocated unconditionally — they're cheap.
        self.prev_period_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.armed_period = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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

        # --- Dual-cmd comparison logger -----------------------------------
        # Writes JSONL records to a file during training so this cmd (V1)
        # and its sibling shadow trajectory cmd can be compared offline.
        # - Set ``_compare_log_path`` to ``None`` to disable.
        # - Throttled to one record every ``_compare_log_step_period`` calls
        #   to ``_update_command`` (env 0 only).
        # - Shadow cmd is auto-detected lazily on first use (any sibling
        #   trajectory cmd that isn't ``self``), so this logger works
        #   whether V1 is primary (V2 is shadow) or V2 is primary (V1 is
        #   shadow — this cmd is then the shadow side).
        # Set to ``f"/tmp/v1v2_train_log_{int(time.time())}.jsonl"`` to
        # re-enable the dual-cmd JSONL comparison logger.  Disabled here
        # so training runs aren't slowed by per-tick file I/O and large
        # vector dumps.
        self._compare_log_path: Optional[str] = None
        self._compare_log_step_period: int = 1
        self._compare_log_step_counter: int = 0
        self._compare_log_file: Optional[object] = None
        self._compare_log_shadow_cache: Optional[object] = None
        self._compare_log_shadow_resolved: bool = False
        # Cached per-step gate state for tick-misalignment detection.
        self._compare_log_prev_v1_gate: Optional[int] = None
        self._compare_log_prev_shadow_gate: Optional[int] = None
        self._compare_log_prev_v1_domain: Optional[int] = None
        self._compare_log_prev_shadow_domain: Optional[int] = None
        if self._compare_log_path is not None:
            os.makedirs(os.path.dirname(self._compare_log_path) or ".", exist_ok=True)
            self._compare_log_file = open(self._compare_log_path, "a", buffering=1)
            # Write a one-time metadata record with output names so vector
            # fields in subsequent records can be mapped back to joint /
            # body names without consulting the cmd at analysis time.
            try:
                self._compare_log_file.write(json.dumps({
                    "_meta": True,
                    "ordered_pos_output_names": list(self.ordered_pos_output_names),
                    "ordered_vel_output_names": list(self.ordered_vel_output_names),
                }) + "\n")
            except Exception:
                pass
            print(f"[V1 dual-cmd logger] writing to {self._compare_log_path}")

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
        """Dispatch to the legacy (hold-on-late) or persistence path.

        Args:
            t: ``[N]`` per-env trajectory time before gating (already includes
                ``contact_gate_offset`` from prior steps).
            traj_idx: ``[N]`` global trajectory index per env.
            contact_now: ``[N, B]`` runtime contact state in
                ``self.contact_bodies`` order.

        Returns:
            ``[N]`` gated trajectory time.
        """
        if self.cfg.hold_on_late_contact:
            return self._apply_contact_gate_legacy(t, traj_idx, contact_now)
        return self._apply_contact_gate_persist(t, traj_idx, contact_now)

    def _apply_contact_gate_persist(
        self,
        t: torch.Tensor,
        traj_idx: torch.Tensor,
        contact_now: torch.Tensor,
    ) -> torch.Tensor:
        """Gate-persistence path (default).

        The armed gate's instance is anchored to ``self.armed_period``,
        which is independent of the env's current ``period_idx``.  Late
        contacts produce a backward snap that lands one ``step_dt`` past
        the boundary (possibly across a natural period wrap), so the
        trajectory eval is in the new domain — matching the post-contact
        reference frame.  If ``t`` advances more than one full period past
        the armed instance without contact, ``armed_period`` auto-advances
        by 1 so the same gate's next instance becomes armed.

        Args:
            t: ``[N]`` per-env trajectory time before gating.
            traj_idx: ``[N]`` global trajectory index per env.
            contact_now: ``[N, B]`` runtime contact state.

        Returns:
            ``[N]`` gated trajectory time.
        """
        total = self.manager.data["total_time"][traj_idx]
        traj_type = self.manager.data["traj_type"][traj_idx]
        num_gates = self.manager._num_gates_per_traj[traj_idx]

        active = self.next_gate_idx >= 0
        if not active.any():
            return t

        safe_idx = torch.clamp(self.next_gate_idx, min=0)
        target_phi = self.manager._gate_phi_table[traj_idx, safe_idx]
        target_active = self.manager._gate_active_table[traj_idx, safe_idx]
        target_mask = self.manager._gate_contact_mask[traj_idx, safe_idx]

        # Snap target sits one step's worth of phi *past* the gate boundary
        # so the trajectory eval lands inside the new domain (matching the
        # post-contact reference frame), not exactly at the boundary where
        # the eval would still be in the old domain.
        eps_t = self.env.step_dt
        early_window_size_t = target_phi * self.cfg.contact_gate_window_frac * total
        boundary_t = self.armed_period.to(t.dtype) * total + target_phi * total
        gate_target_t = boundary_t + eps_t
        delta = boundary_t - t  # signed distance to the boundary (window/expiry check)

        # Auto-rearm at the next instance once t passes the boundary by
        # more than (total - early_window_size_t).  After advancing, delta
        # lands exactly at +early_window_size_t (top of early window).
        expired = active & (delta < -(total - early_window_size_t))
        self.armed_period = torch.where(expired, self.armed_period + 1, self.armed_period)
        boundary_t = self.armed_period.to(t.dtype) * total + target_phi * total
        gate_target_t = boundary_t + eps_t
        delta = boundary_t - t

        # Fire window: late side is unbounded (the expired branch handles
        # overflow), early side capped at early_window_size_t.
        in_window = delta <= early_window_size_t
        expected_landed = (contact_now & target_mask).any(dim=1)

        usable = active & target_active
        fire_mask = in_window & expected_landed & usable

        # Fire snaps t to gate_target_t = boundary + eps_t.  For early
        # contact this is a forward jump; for late contact a backward
        # jump, possibly across a period wrap.  Either way phi lands one
        # step past the boundary, inside the new domain.
        fire_delta = torch.where(fire_mask, gate_target_t - t, torch.zeros_like(t))
        self.contact_gate_offset = self.contact_gate_offset + fire_delta
        t = t + fire_delta

        # Advance next_gate_idx and armed_period for fired envs.
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

        # armed_period for the next gate = floor(t_after_fire / total).
        # For non-last fire this stays in armed_period (target_phi < 1.0);
        # for last fire on target_phi == 1.0 it becomes armed_period + 1.
        period_after = torch.div(t, total, rounding_mode="floor").long()
        self.armed_period = torch.where(fire_mask, period_after, self.armed_period)

        return t

    def _apply_contact_gate_legacy(
        self,
        t: torch.Tensor,
        traj_idx: torch.Tensor,
        contact_now: torch.Tensor,
    ) -> torch.Tensor:
        """Legacy hold-on-late-contact path.

        Pulls the trajectory clock back to just below the boundary and
        holds there until contact lands; period wraps reset the armed gate
        to gate 0 of the new period (silently dropping any unfired gate
        from the previous period).

        Args:
            t: ``[N]`` per-env trajectory time before gating.
            traj_idx: ``[N]`` global trajectory index per env.
            contact_now: ``[N, B]`` runtime contact state.

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

        # DEBUG: V1 gate events
        _e = 0
        if int(self.env.episode_length_buf[_e]) < 0:
            if fire_mask[_e]:
                print(f"  [V1 FIRE phi={phi[_e].item():.5f} target_phi={target_phi[_e].item():.4f} gate_idx={int(self.next_gate_idx[_e])}]")
            elif hold_mask[_e]:
                print(f"  [V1 HOLD phi={phi[_e].item():.5f} target_phi={target_phi[_e].item():.4f} gate_idx={int(self.next_gate_idx[_e])}]")

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
        self.armed_period[reset_mask] = 0
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
            init_period = torch.div(t, init_total, rounding_mode="floor").long()
            init_phi = (t - init_period.to(t.dtype) * init_total) / init_total
            gates = self.manager._gate_phi_table[cur_traj]
            gate_active = self.manager._gate_active_table[cur_traj]
            upcoming = gate_active & (gates >= init_phi.unsqueeze(1))
            has_upcoming = upcoming.any(dim=1)
            first_upcoming_idx = upcoming.to(torch.long).argmax(dim=1)
            # No upcoming gate this period → arm gate 0 of the next period.
            init_idx = torch.where(has_upcoming, first_upcoming_idx, torch.zeros_like(first_upcoming_idx))
            num_gates_reset = self.manager._num_gates_per_traj[cur_traj]
            init_idx = torch.where(num_gates_reset > 0, init_idx, -torch.ones_like(init_idx))
            self.next_gate_idx = torch.where(reset_mask, init_idx, self.next_gate_idx)

            if self.cfg.hold_on_late_contact:
                self.prev_period_idx = torch.where(reset_mask, init_period, self.prev_period_idx)
            else:
                # Armed instance is in the current period if there's an
                # upcoming gate; otherwise it's the next period (gate 0).
                armed_period_init = torch.where(has_upcoming, init_period, init_period + 1)
                self.armed_period = torch.where(reset_mask, armed_period_init, self.armed_period)

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
                    new_total = self.manager.data["total_time"][cur_traj[changed]]
                    new_period = torch.div(
                        target_t, new_total, rounding_mode="floor"
                    ).long()

                    if self.cfg.hold_on_late_contact:
                        new_idx_for_changed = torch.where(
                            new_num_gates > 0,
                            torch.zeros_like(new_num_gates),
                            -torch.ones_like(new_num_gates),
                        )
                        self.next_gate_idx[changed] = new_idx_for_changed
                        self.prev_period_idx[changed] = new_period
                    else:
                        # Mirror the reset logic: arm the first upcoming
                        # gate in the new trajectory's current period;
                        # if none, arm gate 0 of the next period.
                        gates_c = self.manager._gate_phi_table[cur_traj[changed]]
                        gate_active_c = self.manager._gate_active_table[cur_traj[changed]]
                        new_phi = (
                            target_t - new_period.to(target_t.dtype) * new_total
                        ) / new_total
                        upcoming_c = gate_active_c & (gates_c >= new_phi.unsqueeze(1))
                        has_upcoming_c = upcoming_c.any(dim=1)
                        first_upcoming_c = upcoming_c.to(torch.long).argmax(dim=1)
                        idx_c = torch.where(
                            has_upcoming_c,
                            first_upcoming_c,
                            torch.zeros_like(first_upcoming_c),
                        )
                        idx_c = torch.where(
                            new_num_gates > 0,
                            idx_c,
                            -torch.ones_like(idx_c),
                        )
                        self.next_gate_idx[changed] = idx_c
                        armed_c = torch.where(has_upcoming_c, new_period, new_period + 1)
                        self.armed_period[changed] = armed_c

        # Apply contact gating (acts on advancing envs only — reset envs
        # already have offset cleared so the gate logic is a no-op for them).
        if self._gating_enabled:
            contact_now = self._read_contact_now()
            # DEBUG: V1 trace
            _e = 0
            if int(self.env.episode_length_buf[_e]) < 0:
                print(f"[V1 contact step={int(self.env.episode_length_buf[_e])} contact={contact_now[_e].tolist()}]")
            t = self._apply_contact_gate(t, cur_traj, contact_now)

        self.traj_time = t
        self.prev_traj_idx = cur_traj.clone()
        self.prev_skill_idx = cur_skill.clone()
        self._initialized = True

        # DEBUG: V1 end-of-step state
        _e = 0
        if int(self.env.episode_length_buf[_e]) < 0:
            _total_e = self.manager.data["total_time"][cur_traj[_e]]
            _t_in_phase = t[_e] % _total_e
            _phi_e = ((t[_e] / _total_e) % 1.0).item()
            _traj_e = int(cur_traj[_e])
            _skill_e = int(cur_skill[_e])
            _vel_e = self.env.command_manager.get_term(self.cfg.conditioner_generator_name).command[_e].tolist()
            print(f"[V1 step={int(self.env.episode_length_buf[_e])} phi={_phi_e:.6f} t={_t_in_phase.item():.6f} total_t={t[_e].item():.6f} gate={int(self.next_gate_idx[_e])} traj={_traj_e} skill={_skill_e} vel={_vel_e}]")

        # DEBUG: compare phases and ref frames between V1 (this) and V2 (other)
        _e = 0
        if False:  # disabled for training
            # V1 (self)
            v1_traj = cur_traj[_e]
            v1_t = t[_e].item()
            v1_total = float(self.manager.data["total_time"][v1_traj])
            v1_phi = (v1_t / v1_total) % 1.0
            v1_rf = int(self.cur_ref_frame_idx[_e])

            # V2 (other cmd)
            v2 = self.env.command_manager.get_term("traj_ref")
            v2_phi = float(v2.manager.phase[_e])
            v2_rf = int(v2.cur_ref_frame_idx[_e])

            print(
                f"[diff step={int(self.env.episode_length_buf[_e])} "
                f"v1_phi={v1_phi:.5f} v2_phi={v2_phi:.5f} dphi={abs(v1_phi - v2_phi):.5f} "
                f"v1_rf={v1_rf} v2_rf={v2_rf} rf_match={v1_rf == v2_rf}]"
            )

        return t

    def _update_command(self):
        """Override to compare V / y_des / y_act between V1 and V2 after both
        cmds have run their full ``_update_command`` for this step.

        With V1 as the primary cmd (``traj_ref``) and V2 as the shadow at
        ``traj_ref_OLD``, the V2 cmd registers AFTER V1 in the parent
        config, so by the time V1's ``super()._update_command()`` runs and
        we read shadow state, V2 hasn't run yet for this step.  In the
        opposite arrangement (V2 primary, V1 shadow), V2 runs first.
        Either way, both cmds end up with valid state for the previous
        step, so reading them here gives meaningful comparison data.
        """
        super()._update_command()

        # Throttled JSONL logger: write one comparison record every Nth call.
        if self._compare_log_file is not None:
            self._compare_log_step_counter += 1
            if (
                self._compare_log_step_counter % self._compare_log_step_period == 0
            ):
                try:
                    # Lazily resolve the shadow cmd: any sibling trajectory
                    # cmd that isn't self.  Retry every call until found —
                    # the first ``_update_command`` may fire before all sibling
                    # cmds are registered with the command manager.
                    if self._compare_log_shadow_cache is None:
                        cmd_mgr = self.env.command_manager
                        candidates: list = []
                        for _name in ("traj_ref", "traj_ref_OLD",
                                      "traj_ref_V2", "traj_ref_v2"):
                            try:
                                candidates.append(cmd_mgr.get_term(_name))
                            except (KeyError, AttributeError, ValueError):
                                pass
                        for attr in ("_terms", "active_terms", "terms"):
                            terms_obj = getattr(cmd_mgr, attr, None)
                            if isinstance(terms_obj, dict):
                                candidates.extend(terms_obj.values())
                            elif isinstance(terms_obj, (list, tuple)):
                                candidates.extend(terms_obj)
                        for _term in candidates:
                            if (
                                _term is not self
                                and hasattr(_term, "y_des")
                                and hasattr(_term, "v")
                                and hasattr(_term, "ref_poses")
                            ):
                                self._compare_log_shadow_cache = _term
                                print(
                                    f"[V1 dual-cmd logger] shadow cmd resolved: "
                                    f"{type(_term).__name__}"
                                )
                                break
                    shadow = self._compare_log_shadow_cache
                    if shadow is None:
                        return

                    # One-time: dump both cmds' ordered output names to a
                    # sidecar JSON file so the user can verify they match.
                    # If the orderings differ, a per-index L2 norm comparison
                    # is meaningless (it'd be subtracting different physical
                    # quantities at the same index).
                    if not getattr(self, "_compare_names_logged", False):
                        v1_names_pos = list(self.ordered_pos_output_names)
                        v2_names_pos = list(shadow.ordered_pos_output_names)
                        v1_names_vel = list(self.ordered_vel_output_names)
                        v2_names_vel = list(shadow.ordered_vel_output_names)
                        same_pos = v1_names_pos == v2_names_pos
                        same_vel = v1_names_vel == v2_names_vel
                        print(
                            f"[V1 dual-cmd logger] ordered_pos_output_names match: {same_pos} "
                            f"(V1 has {len(v1_names_pos)}, shadow has {len(v2_names_pos)})"
                        )
                        print(
                            f"[V1 dual-cmd logger] ordered_vel_output_names match: {same_vel} "
                            f"(V1 has {len(v1_names_vel)}, shadow has {len(v2_names_vel)})"
                        )
                        if not same_pos:
                            mismatches = [
                                (i, a, b) for i, (a, b)
                                in enumerate(zip(v1_names_pos, v2_names_pos))
                                if a != b
                            ]
                            print(f"  pos mismatched indices (first 10): {mismatches[:10]}")
                        if not same_vel:
                            mismatches = [
                                (i, a, b) for i, (a, b)
                                in enumerate(zip(v1_names_vel, v2_names_vel))
                                if a != b
                            ]
                            print(f"  vel mismatched indices (first 10): {mismatches[:10]}")

                        # Compare other compute_measured_output inputs.
                        for attr in ("body_idx", "joint_idx", "body_type",
                                      "use_com", "ref_frame_indices"):
                            v1_val = getattr(self, attr, "<missing>")
                            v2_val = getattr(shadow, attr, "<missing>")
                            v1_repr = (
                                v1_val.tolist() if hasattr(v1_val, "tolist")
                                else v1_val
                            )
                            v2_repr = (
                                v2_val.tolist() if hasattr(v2_val, "tolist")
                                else v2_val
                            )
                            same = v1_repr == v2_repr
                            print(f"  {attr}: V1={v1_repr} | shadow={v2_repr} | match={same}")
                        # Compare per-trajectory ``expanded_domains`` and
                        # ``domain_boundaries`` between V1 and V2 managers.
                        # If they differ, V1 and V2 will disagree about
                        # ``current_domain`` even at identical phi → spurious
                        # ref_pose latches and dy_des spikes.
                        try:
                            v1_ed = self.manager.data["expanded_domains"]
                            sh_ed = shadow.manager.data["expanded_domains"]
                            v1_db = self.manager.data["domain_boundaries"]
                            sh_db = shadow.manager.data["domain_boundaries"]
                            ed_match = bool(torch.equal(v1_ed, sh_ed))
                            db_match = torch.allclose(v1_db, sh_db, atol=1e-6)
                            print(
                                f"  expanded_domains match: {ed_match} "
                                f"(V1.shape={list(v1_ed.shape)}, "
                                f"shadow.shape={list(sh_ed.shape)})"
                            )
                            if not ed_match:
                                # Show first 12 trajectory's counts side by side.
                                v1_list = v1_ed[:12].tolist()
                                sh_list = sh_ed[:12].tolist()
                                print(f"    V1     ed[:12]: {v1_list}")
                                print(f"    shadow ed[:12]: {sh_list}")
                            print(f"  domain_boundaries match (atol=1e-6): {db_match}")
                            # Always print boundaries for the env-0 trajectory.
                            v1_traj0 = int(self.manager.get_current_trajectory_indices()[0])
                            sh_traj0 = int(shadow.manager.get_current_trajectory_indices()[0])
                            print(
                                f"    V1     traj={v1_traj0} ed={int(v1_ed[v1_traj0])} "
                                f"db={v1_db[v1_traj0].tolist()}"
                            )
                            print(
                                f"    shadow traj={sh_traj0} ed={int(sh_ed[sh_traj0])} "
                                f"db={sh_db[sh_traj0].tolist()}"
                            )
                        except Exception as ex:
                            print(f"  [boundaries probe failed] {type(ex).__name__}: {ex}")

                        # Also compare the y_act and dy_act values directly.
                        _e = 0
                        y_act_diff = (self.y_act[_e] - shadow.y_act[_e]).abs()
                        dy_act_diff = (self.dy_act[_e] - shadow.dy_act[_e]).abs()
                        top_y_vals, top_y_idx = torch.topk(
                            y_act_diff, k=min(5, y_act_diff.shape[0])
                        )
                        top_dy_vals, top_dy_idx = torch.topk(
                            dy_act_diff, k=min(5, dy_act_diff.shape[0])
                        )
                        print(f"  y_act top diffs (first call):")
                        for k, i in enumerate(top_y_idx.tolist()):
                            print(
                                f"    {self.ordered_pos_output_names[i]}: "
                                f"v1={self.y_act[_e, i].item():+.5f} "
                                f"shadow={shadow.y_act[_e, i].item():+.5f} "
                                f"d={top_y_vals[k].item():+.5f}"
                            )
                        print(f"  dy_act top diffs (first call):")
                        for k, i in enumerate(top_dy_idx.tolist()):
                            print(
                                f"    {self.ordered_vel_output_names[i]}: "
                                f"v1={self.dy_act[_e, i].item():+.5f} "
                                f"shadow={shadow.dy_act[_e, i].item():+.5f} "
                                f"d={top_dy_vals[k].item():+.5f}"
                            )
                        self._compare_names_logged = True
                    _e = 0
                    v1_traj = int(self.manager.get_current_trajectory_indices()[_e])
                    v1_skill = int(self.manager.data["skill_idx"][v1_traj])
                    shadow_traj = int(
                        shadow.manager.get_current_trajectory_indices()[_e]
                    )
                    shadow_skill = int(
                        shadow.manager.data["skill_idx"][shadow_traj]
                    )

                    # Gate / domain state — used to detect which cmd fired
                    # the gate this tick (vs the previous record).
                    # V1 owns ``next_gate_idx`` on the cmd; V2 owns it on
                    # the manager.  ``current_domain`` lives on the cmd in
                    # both (inherited from BaseTrajectoryCommand).
                    if hasattr(self, "next_gate_idx"):
                        v1_gate_idx = int(self.next_gate_idx[_e])
                    else:
                        v1_gate_idx = int(self.manager.next_gate_idx[_e])
                    if hasattr(shadow, "next_gate_idx"):
                        shadow_gate_idx = int(shadow.next_gate_idx[_e])
                    else:
                        shadow_gate_idx = int(shadow.manager.next_gate_idx[_e])
                    v1_domain = int(self.current_domain[_e])
                    shadow_domain = int(shadow.current_domain[_e])

                    # Fired this tick = gate index advanced or domain
                    # changed since last record (excluding the very first
                    # record, where prev is None and the diff is nonsense).
                    def _changed(cur, prev):
                        return 1 if (prev is not None and cur != prev) else 0
                    v1_gate_fired = _changed(
                        v1_gate_idx, self._compare_log_prev_v1_gate
                    )
                    shadow_gate_fired = _changed(
                        shadow_gate_idx, self._compare_log_prev_shadow_gate
                    )
                    v1_domain_changed = _changed(
                        v1_domain, self._compare_log_prev_v1_domain
                    )
                    shadow_domain_changed = _changed(
                        shadow_domain, self._compare_log_prev_shadow_domain
                    )
                    self._compare_log_prev_v1_gate = v1_gate_idx
                    self._compare_log_prev_shadow_gate = shadow_gate_idx
                    self._compare_log_prev_v1_domain = v1_domain
                    self._compare_log_prev_shadow_domain = shadow_domain

                    # Per-step wrapped time and the searchsorted boundary
                    # right above each cmd's ``t_wrapped``.  Lets us see if
                    # V1 and V2 land on opposite sides of the same boundary
                    # at "the same phi" (FP precision artefact).
                    try:
                        v1_total = float(self.manager.data["total_time"][v1_traj])
                        sh_total = float(shadow.manager.data["total_time"][shadow_traj])
                        v1_t = float((self.traj_time[_e] % v1_total).item())
                        sh_t = float((shadow.traj_time[_e] % sh_total).item())
                        v1_db = self.manager.data["domain_boundaries"][v1_traj]
                        sh_db = shadow.manager.data["domain_boundaries"][shadow_traj]
                        v1_b_above = float(
                            v1_db[min(v1_domain + 1, v1_db.shape[0] - 1)].item()
                        )
                        sh_b_above = float(
                            sh_db[min(shadow_domain + 1, sh_db.shape[0] - 1)].item()
                        )
                    except Exception:
                        v1_t = float("nan"); sh_t = float("nan")
                        v1_b_above = float("nan"); sh_b_above = float("nan")

                    record = {
                        "global_step": self._compare_log_step_counter,
                        "ep_len": int(self.env.episode_length_buf[_e]),
                        "v1_V": float(self.v[_e]),
                        "shadow_V": float(shadow.v[_e]),
                        "dV": float(self.v[_e] - shadow.v[_e]),
                        "dy_des": float((self.y_des[_e] - shadow.y_des[_e]).norm()),
                        "dy_act": float((self.y_act[_e] - shadow.y_act[_e]).norm()),
                        "ddy_des": float((self.dy_des[_e] - shadow.dy_des[_e]).norm()),
                        "ddy_act": float((self.dy_act[_e] - shadow.dy_act[_e]).norm()),
                        "ref_diff": float(
                            (self.ref_poses[_e] - shadow.ref_poses[_e]).norm()
                        ),
                        # Trajectory / skill indices for both cmds.
                        "v1_traj": v1_traj,
                        "v1_skill": v1_skill,
                        "shadow_traj": shadow_traj,
                        "shadow_skill": shadow_skill,
                        # Gate / domain state.
                        "v1_gate_idx": v1_gate_idx,
                        "shadow_gate_idx": shadow_gate_idx,
                        "v1_domain": v1_domain,
                        "shadow_domain": shadow_domain,
                        "v1_gate_fired": v1_gate_fired,
                        "shadow_gate_fired": shadow_gate_fired,
                        "v1_domain_changed": v1_domain_changed,
                        "shadow_domain_changed": shadow_domain_changed,
                        # Boundary-precision probe.
                        "v1_t_wrapped": v1_t,
                        "shadow_t_wrapped": sh_t,
                        "v1_boundary_above": v1_b_above,
                        "shadow_boundary_above": sh_b_above,
                        # Idempotency / call-count diagnostics for V2.  These
                        # tell us how many times V2's _compute_time was called
                        # between two logger records, and how many of those
                        # were idempotency hits (cached) vs misses (advanced).
                        "shadow_calls_since_log": int(
                            getattr(shadow, "_dbg_call_count", 0)
                        ),
                        "shadow_idem_hits_since_log": int(
                            getattr(shadow, "_dbg_idem_hits", 0)
                        ),
                        "shadow_idem_misses_no_cache": int(
                            getattr(shadow, "_dbg_idem_misses_no_cache", 0)
                        ),
                        "shadow_idem_misses_eplen_diff": int(
                            getattr(shadow, "_dbg_idem_misses_eplen_diff", 0)
                        ),
                        "shadow_advance_count_since_log": int(
                            getattr(shadow, "_dbg_advance_count", 0)
                        ),
                        "shadow_last_compute_ep_len_env0": (
                            None if getattr(shadow, "_dbg_last_ep_len_env0", None) is None
                            else int(shadow._dbg_last_ep_len_env0)
                        ),
                        # Full output vectors (positions and velocities).
                        # Per-row size: ~4 * num_outputs floats per cmd.
                        "v1_y_des": self.y_des[_e].detach().cpu().tolist(),
                        "shadow_y_des": shadow.y_des[_e].detach().cpu().tolist(),
                        "v1_y_act": self.y_act[_e].detach().cpu().tolist(),
                        "shadow_y_act": shadow.y_act[_e].detach().cpu().tolist(),
                        "v1_dy_des": self.dy_des[_e].detach().cpu().tolist(),
                        "shadow_dy_des": shadow.dy_des[_e].detach().cpu().tolist(),
                        "v1_dy_act": self.dy_act[_e].detach().cpu().tolist(),
                        "shadow_dy_act": shadow.dy_act[_e].detach().cpu().tolist(),
                    }
                    # Optional: log phi for both if shadow is phase-based.
                    if hasattr(shadow.manager, "phase") and shadow.manager.phase is not None:
                        v1_total = float(self.manager.data["total_time"][v1_traj])
                        record["v1_phi"] = (
                            float(self.traj_time[_e]) % v1_total
                        ) / v1_total
                        record["shadow_phi"] = float(shadow.manager.phase[_e])
                    self._compare_log_file.write(json.dumps(record) + "\n")
                    # Reset shadow's diagnostic counters so the next record's
                    # values represent "since last log entry".
                    if hasattr(shadow, "_dbg_call_count"):
                        shadow._dbg_call_count = 0
                        shadow._dbg_idem_hits = 0
                        shadow._dbg_idem_misses_no_cache = 0
                        shadow._dbg_idem_misses_eplen_diff = 0
                        shadow._dbg_advance_count = 0
                except Exception as e:  # never crash training due to logging
                    if not getattr(self, "_compare_log_warned", False):
                        print(
                            f"[V1 dual-cmd logger] write failed: {type(e).__name__}: {e}"
                        )
                        self._compare_log_warned = True

        # DEBUG: compare CLF / measured / desired outputs between V1 (self) and V2.
        _e = 0
        if False:  # disabled for training
            v2 = self.env.command_manager.get_term("traj_ref")
            v1_V = self.v[_e].item()
            v2_V = v2.v[_e].item()
            dV = v1_V - v2_V

            # L2 norms of the per-output difference vectors for env 0.
            dy_des = (self.y_des[_e] - v2.y_des[_e]).norm().item()
            dy_act = (self.y_act[_e] - v2.y_act[_e]).norm().item()
            ddy_des = (self.dy_des[_e] - v2.dy_des[_e]).norm().item()
            ddy_act = (self.dy_act[_e] - v2.dy_act[_e]).norm().item()
            ref_diff = (self.ref_poses[_e] - v2.ref_poses[_e]).norm().item()

            # Domain index used by each manager when evaluating y_des at this
            # step.  Domain selection at the half-period boundary (phi ≈ 0.5)
            # is FP-sensitive and can diverge between V1's t-based and V2's
            # phase-based representations, producing a sagittal-flipped y_des.
            v1_traj = self.manager.get_current_trajectory_indices()[_e:_e + 1]
            v1_t = self.traj_time[_e:_e + 1]
            v1_domain = int(self.manager._get_domain_indices(v1_t, v1_traj)[0])

            v2_traj = v2.manager.get_current_trajectory_indices()[_e:_e + 1]
            v2_t = v2.manager.phase[_e:_e + 1] * v2.manager.data["total_time"][v2_traj]
            v2_domain = int(v2.manager._get_domain_indices(v2_t, v2_traj)[0])

            # Tau-fold check: half-periodic eval folds tau into the second
            # half iff phi >= 0.5.  At the gate boundary FP can put V1 and V2
            # on opposite sides of this threshold → different sagittal-swap
            # decisions → fully reflected y_des with the same `domain` index.
            v1_total_e = float(self.manager.data["total_time"][v1_traj[0]])
            v1_phi_full = (float(self.traj_time[_e]) % v1_total_e) / v1_total_e
            v2_phi_full = float(v2.manager.phase[_e])
            v1_fold = v1_phi_full >= 0.5
            v2_fold = v2_phi_full >= 0.5

            print(
                f"[VV step={int(self.env.episode_length_buf[_e])} "
                f"v1_V={v1_V:.4f} v2_V={v2_V:.4f} dV={dV:+.4f} "
                f"dy_des={dy_des:.4f} dy_act={dy_act:.4f} "
                f"ddy_des={ddy_des:.4f} ddy_act={ddy_act:.4f} "
                f"ref_diff={ref_diff:.6f} "
                f"v1_dom={v1_domain} v2_dom={v2_domain} "
                f"dom_match={v1_domain == v2_domain} "
                f"v1_phi={v1_phi_full:.10f} v2_phi={v2_phi_full:.10f} "
                f"v1_fold={v1_fold} v2_fold={v2_fold} "
                f"fold_match={v1_fold == v2_fold}]"
            )

            # When y_des diverges, show top-k mismatched outputs by name.
            if dy_des > 1e-3:
                diff_vec = (self.y_des[_e] - v2.y_des[_e]).abs()
                top_vals, top_idx = torch.topk(diff_vec, k=min(5, diff_vec.shape[0]))
                top_pairs = [
                    f"{self.ordered_pos_output_names[i]}: v1={self.y_des[_e, i].item():+.3f} "
                    f"v2={v2.y_des[_e, i].item():+.3f} d={top_vals[k].item():+.3f}"
                    for k, i in enumerate(top_idx.tolist())
                ]
                print(f"  y_des top diffs:")
                for pair in top_pairs:
                    print(f"    {pair}")
            if ddy_des > 1e-2:
                diff_vec = (self.dy_des[_e] - v2.dy_des[_e]).abs()
                top_vals, top_idx = torch.topk(diff_vec, k=min(5, diff_vec.shape[0]))
                top_pairs = [
                    f"{self.ordered_vel_output_names[i]}: v1={self.dy_des[_e, i].item():+.3f} "
                    f"v2={v2.dy_des[_e, i].item():+.3f} d={top_vals[k].item():+.3f}"
                    for k, i in enumerate(top_idx.tolist())
                ]
                print(f"  dy_des top diffs:")
                for pair in top_pairs:
                    print(f"    {pair}")
