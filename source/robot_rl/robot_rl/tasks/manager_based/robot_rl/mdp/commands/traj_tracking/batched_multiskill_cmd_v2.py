"""Batched multi-skill trajectory command using :class:`MultiSkillManager`."""

from __future__ import annotations

import torch
import warp as wp

from .base_trajectory_cmd import BaseTrajectoryCommand
from .manager_base import ManagerBase
from .multiskill_manager_v2 import MultiSkillManagerV2, _PERPETUAL_INT

# Flip to True to re-enable the throttled diagnostic prints inside
# ``_compute_time`` (idempotency miss reasons, traj-change events,
# skill-change handler trace).  Diagnostic counters used by the JSONL
# logger keep updating regardless; only the ``print`` statements are
# gated.  Disabled by default so training stdout isn't polluted.
_V2_DEBUG_PRINTS = False


class BatchedMultiSkillCommandV2(BaseTrajectoryCommand):
    """Trajectory command backed by a :class:`MultiSkillManager`.

    All per-env phase state lives on the manager (``manager.phase`` and
    ``manager.next_gate_idx``).  This command term is a thin orchestrator:
    it calls :meth:`MultiSkillManager.reset_phase` on episode reset,
    :meth:`MultiSkillManager.update_phase` once per step, and applies
    contact-gate snaps via :meth:`MultiSkillManager.snap_phase_to_new_domain`,
    :meth:`MultiSkillManager.snap_phase_to_start_of_current_domain`, and
    :meth:`MultiSkillManager.snap_phase_to_end_of_previous_domain`.

    Skill changes leave the phase unchanged — the new skill inherits the
    current phase value as-is.  Trajectory eval downstream is fed
    ``t = phase * total_time[traj_idx]`` so the existing time-based
    manager methods (``get_output``, ``get_phasing_var``, ...) work
    without modification.
    """

    def _create_manager(self, cfg, env) -> ManagerBase:
        """Create a :class:`MultiSkillManager` from a folder of skill subfolders.

        Args:
            cfg: Command term configuration.
            env: IsaacLab environment.

        Returns:
            A :class:`MultiSkillManager` instance.
        """
        return MultiSkillManagerV2(
            path=cfg.path,
            device=env.device,
            env=env,
            conditioner_generator_name=cfg.conditioner_generator_name,
            hf_repo=cfg.hf_repo,
            track_traj_stats=getattr(cfg, "track_traj_stats", True),
            traj_stats_alpha=getattr(cfg, "traj_stats_alpha", 0.005),
            traj_stats_reset_warmup=getattr(cfg, "traj_stats_reset_warmup", 2),
            traj_stats_transition_warmup=getattr(cfg, "traj_stats_transition_warmup", 3),
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
        """Initialise the manager's ref-frame lookup and the contact-gate
        wiring when gating is enabled.
        """
        self.manager.build_ref_frame_map(self.ref_frames)
        # Eagerly allocate manager phase state to num_envs.
        self.manager._ensure_phase_state(self.num_envs)

        # Idempotency guard for ``_compute_time`` (called twice per env step
        # by IsaacLab's resample-then-update flow).  When the cached step
        # buffer matches the current one we skip phase advance + gate.
        self._last_compute_step: torch.Tensor | None = None
        self._cached_t: torch.Tensor | None = None
        # End-of-last-compute snapshots, used as ``pre_update_*`` at the
        # start of the next compute.  Maintained independently of
        # ``manager.phase`` and ``manager._cached_global_indices`` so that
        # external mutations (reset events that call ``set_phase`` and/or
        # ``get_desired_outputs``) cannot mask the pre-tick state V2 needs
        # to detect skill changes correctly.
        self._prev_compute_phase: torch.Tensor | None = None
        self._prev_compute_traj: torch.Tensor | None = None

        # --- Contact-gate wiring -----------------------------------------
        self._gating_enabled = self.cfg.contact_gate_window_frac is not None

        if self._gating_enabled:
            self.manager.set_gate_contact_layout(self.contact_bodies)
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

    @property
    def traj_time(self) -> torch.Tensor:
        """Backwards-compat view: ``manager.phase * total_time[traj_idx]``.

        Several debug / logging consumers still read ``cmd.traj_time``.
        Compute on demand from the manager's phase rather than mirroring
        a separate buffer.
        """
        cur_traj = self.manager.get_current_trajectory_indices()
        total = self.manager.data["total_time"][cur_traj]
        return self.manager.phase * total

    def _read_contact_now(self) -> torch.Tensor:
        """Return ``[N, B]`` boolean tensor of which contact bodies are
        currently in contact, in ``self.contact_bodies`` order.
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

    def _apply_contact_gate(self, contact_now: torch.Tensor) -> None:
        """Apply the contact-gate logic by issuing snap calls to the manager.

        Operates on ``manager.phase`` and ``manager.next_gate_idx``.

        Args:
            contact_now: ``[N, B]`` boolean tensor of current per-env
                contact state in ``self.contact_bodies`` order.
        """
        phase = self.manager.phase
        next_gate_idx = self.manager.next_gate_idx

        active = next_gate_idx >= 0
        if not active.any():
            return

        cur_traj = self.manager.get_current_trajectory_indices()
        safe_idx = torch.clamp(next_gate_idx, min=0)
        gate_phi = self.manager._gate_phi_table[cur_traj, safe_idx]
        target_active = self.manager._gate_active_table[cur_traj, safe_idx]
        target_mask = self.manager._gate_contact_mask[cur_traj, safe_idx]

        expected_landed = (contact_now & target_mask).any(dim=1)
        usable = active & target_active

        W = self.cfg.contact_gate_window_frac
        # Signed distance from gate (no wrap): positive past, negative
        # before.  We deliberately do NOT wrap with mod 1.0 here so a
        # post-wrap phase isn't mistaken for "just past" the previous
        # cycle's gate (per the user's "no wrap on late contact"
        # requirement).  Wrap-driven gate advancement is handled inside
        # ``update_phase``.
        signed = phase - gate_phi

        # Match V1's window convention: width scales with gate_phi
        # (so a gate at phi=0.5 has half the window of a gate at phi=1.0).
        # V1: in_window = phi >= target_phi * (1 - W) → signed >= -target_phi * W.
        early_window_size = gate_phi * W
        in_early = (signed < 0.0) & (signed >= -early_window_size)
        in_late = (signed >= 0.0) & (signed <= early_window_size)
        # Past the late window without firing — gate has aged out.
        expired_window = signed > early_window_size

        if not self.cfg.hold_on_late_contact:
            early_fire_mask = usable & in_early & expected_landed
            late_fire_mask = usable & in_late & expected_landed
            # Auto-advance gates that have aged past the late window
            # without firing.
            auto_adv_mask = usable & expired_window

            early_ids = torch.where(early_fire_mask)[0]
            late_ids = torch.where(late_fire_mask)[0]
            adv_ids = torch.where(auto_adv_mask)[0]

            self.manager.snap_phase_to_new_domain(early_ids)
            self.manager.snap_phase_to_start_of_current_domain(late_ids)
            self.manager._advance_gate_for_envs(adv_ids)
        else:
            # Hold-on mode: phase that crossed the gate without contact is
            # pulled back to the end of the old domain each step.  Contact
            # in the early window OR landing while held releases via
            # snap_phase_to_new_domain.
            crossed_no_contact = usable & (in_late | expired_window) & ~expected_landed
            crossed_with_contact = usable & (in_late | expired_window) & expected_landed
            early_fire_mask = usable & in_early & expected_landed

            # DEBUG: V2 gate events
            _e = 0
            if int(self.env.episode_length_buf[_e]) < 0:
                if early_fire_mask[_e]:
                    print(f"  [V2 EARLY_FIRE phase={phase[_e].item():.5f} gate_phi={gate_phi[_e].item():.4f} gate_idx={int(next_gate_idx[_e])}]")
                elif crossed_with_contact[_e]:
                    print(f"  [V2 LATE_FIRE phase={phase[_e].item():.5f} gate_phi={gate_phi[_e].item():.4f} gate_idx={int(next_gate_idx[_e])}]")
                elif crossed_no_contact[_e]:
                    print(f"  [V2 HOLD phase={phase[_e].item():.5f} gate_phi={gate_phi[_e].item():.4f} gate_idx={int(next_gate_idx[_e])}]")

            end_prev_ids = torch.where(crossed_no_contact)[0]
            new_dom_ids = torch.where(early_fire_mask | crossed_with_contact)[0]

            self.manager.snap_phase_to_end_of_previous_domain(end_prev_ids)
            self.manager.snap_phase_to_new_domain(new_dom_ids)

    def _compute_time(self) -> torch.Tensor:
        """Compute the per-env trajectory time from the manager's phase.

        Order per step:
          1. Advance phase by ``step_dt`` for non-reset envs (envs with
             ``episode_length_buf == 0`` keep the phase that was set by
             the reset event — :func:`reset_on_reference` calls
             :meth:`MultiSkillManager.set_phase` directly, so we must not
             clobber that here).
          2. Apply contact gate (if enabled) — issues snap calls to the
             manager.
          3. Return ``t = phase * total_time[traj_idx]`` for the base
             class to feed into ``manager.get_output(t)``.

        Idempotent across the same env step: IsaacLab's
        ``CommandTerm.compute`` calls ``_update_command`` twice when a
        resample fires (once from ``_resample_command`` and once
        directly).  We cache the result keyed by ``episode_length_buf``
        and short-circuit the second call.
        """
        ep_len = self.env.episode_length_buf
        # Diagnostic counters (read by the V1-side logger).  Created lazily
        # so existing instances pick them up after a re-init.
        if not hasattr(self, "_dbg_call_count"):
            self._dbg_call_count = 0
            self._dbg_idem_hits = 0
            self._dbg_idem_misses_no_cache = 0
            self._dbg_idem_misses_eplen_diff = 0
            self._dbg_advance_count = 0
            self._dbg_last_ep_len_env0 = None
        self._dbg_call_count += 1

        # Detect envs whose phase was externally set between our last
        # ``_compute_time`` and now (most commonly by ``dual_resets``
        # calling ``set_phase`` on a reset).  For those envs, the reset
        # event owns the phase value — we must not advance it, snap it,
        # or override it via the skill-change handler.
        if self._prev_compute_phase is not None:
            external_set = (
                (self.manager.phase - self._prev_compute_phase).abs() > 1e-3
            )
        else:
            external_set = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )

        # Per-env idempotency: an env needs phase advance iff its ep_len
        # changed since our last advance for it AND it isn't a freshly-reset
        # env (whose phase is owned by the reset event).
        #
        # Full-tensor torch.equal would fail any time a single env reset
        # between calls — even when env 0 (and most others) hadn't moved.
        # That caused phase to be advanced twice per tick for every
        # non-resetting env when any env reset.
        if self._last_compute_step is None:
            advance_mask = (ep_len > 0) & ~external_set
        else:
            advance_mask = (ep_len > 0) & (ep_len != self._last_compute_step) & ~external_set

        if (
            self._last_compute_step is not None
            and self._cached_t is not None
            and not advance_mask.any()
        ):
            self._dbg_idem_hits += 1
            return self._cached_t

        # Idempotency missed — record why (for env 0 at least).
        if self._last_compute_step is None or self._cached_t is None:
            self._dbg_idem_misses_no_cache += 1
        else:
            self._dbg_idem_misses_eplen_diff += 1
            # Print the interesting cases: when env-0's ep_len did NOT
            # change but the full-tensor check still fails (i.e., some
            # OTHER env's ep_len changed between calls).  This is the
            # "stutter" scenario.  Throttled to first 8 of each kind.
            if _V2_DEBUG_PRINTS:
                if not hasattr(self, "_dbg_miss_print_count"):
                    self._dbg_miss_print_count = 0
                    self._dbg_stutter_print_count = 0
                last = self._last_compute_step
                cur = ep_len
                same_count = int((last == cur).sum())
                diff_mask = last != cur
                diff_idx = torch.where(diff_mask)[0]
                env0_same = int(last[0]) == int(cur[0])
                tag = "STUTTER" if env0_same else "TICK"
                count_attr = (
                    "_dbg_stutter_print_count" if env0_same else "_dbg_miss_print_count"
                )
                count = getattr(self, count_attr)
                if count < 8:
                    print(
                        f"[V2 idem MISS {tag} #{count}] "
                        f"shape last={list(last.shape)} cur={list(cur.shape)} "
                        f"dtype last={last.dtype} cur={cur.dtype} "
                        f"matching={same_count}/{cur.numel()} "
                        f"env0: last={int(last[0])} cur={int(cur[0])} "
                        f"first_diffs(idx,last,cur)="
                        f"{[(int(i), int(last[i]), int(cur[i])) for i in diff_idx[:5]]}"
                    )
                    setattr(self, count_attr, count + 1)
        self._dbg_advance_count += 1
        self._dbg_last_ep_len_env0 = int(ep_len[0])

        # Capture pre-update state so we can restore on skill change
        # (mirrors OLD cmd's ``compute_transition_time`` semantics).
        #
        # We use V2's OWN snapshot from the end of the previous
        # ``_compute_time`` call, NOT ``manager.phase`` /
        # ``manager._cached_global_indices``.  External callers (notably
        # the ``reset_on_reference_dual`` event, which calls ``set_phase``
        # and ``get_desired_outputs``) can mutate the manager's state
        # between ticks; reading from the manager here would pick up the
        # post-mutation values and silently skip the skill-change handler.
        # The snapshot is updated at the end of this method.
        pre_update_phase = (
            self._prev_compute_phase.clone()
            if self._prev_compute_phase is not None
            else (self.manager.phase.clone() if self.manager.phase is not None else None)
        )
        pre_update_traj = (
            self._prev_compute_traj.clone()
            if self._prev_compute_traj is not None
            else (
                self.manager._cached_global_indices.clone()
                if self.manager._cached_global_indices is not None
                else None
            )
        )

        # Use the per-env advance_mask computed at the top: only advance
        # envs whose ep_len progressed since the last compute call.  An
        # env's ep_len being unchanged means we're in the same env tick
        # for that env (second of a resample-driven double-call), so its
        # phase has already been advanced.
        if advance_mask.any():
            adv_ids = torch.where(advance_mask)[0]
            self.manager.update_phase(self.env.step_dt, env_ids=adv_ids)
        # Keep ``advancing_mask`` as an alias for backward compat with the
        # skill-change block below.
        advancing_mask = advance_mask

        # Skill-change handling: read ``_skill_changed`` NOW, before the
        # explicit ``invalidate_cache`` below triggers a second rebuild
        # that would wipe it to all-False (since ``_prev_skill_indices``
        # was just updated by ``update_phase``'s implicit rebuild).
        #
        # Match V1's ``compute_transition_time`` semantics:
        #   - For perpetual prev: target = 0 (V1's _compute_phasing_var
        #     of perpetual returns 0 regardless of t).
        #   - For non-perpetual prev: target = pre_update_phase (preserve phi).
        #
        # Reset envs already have their phase set + gate re-armed via
        # ``manager.set_phase`` in the reset event, so we restrict to
        # advancing envs to avoid double-handling.
        # DEBUG: print every time env-0's trajectory differs from the
        # pre-update value, regardless of _skill_changed.  Helps detect
        # whether _skill_changed is wrongly False at a real transition.
        if _V2_DEBUG_PRINTS and pre_update_traj is not None:
            cur_global_for_dbg = self.manager._cached_global_indices
            if cur_global_for_dbg is not None and int(cur_global_for_dbg[0]) != int(pre_update_traj[0]):
                if not hasattr(self, "_dbg_traj_chg_count"):
                    self._dbg_traj_chg_count = 0
                if self._dbg_traj_chg_count < 8:
                    sk_changed_v = (
                        bool(self.manager._skill_changed[0])
                        if self.manager._skill_changed is not None
                        and self.manager._skill_changed.numel() > 0
                        else "None"
                    )
                    print(
                        f"[V2 TRAJ-CHG env=0 ep_len={int(self.env.episode_length_buf[0])} "
                        f"prev_traj={int(pre_update_traj[0])} cur_traj={int(cur_global_for_dbg[0])} "
                        f"_skill_changed[0]={sk_changed_v} "
                        f"advance_mask[0]={bool(advancing_mask[0])} "
                        f"prev_phase={float(pre_update_phase[0]):.5f} "
                        f"cur_phase_post_update={float(self.manager.phase[0]):.5f}]"
                    )
                    self._dbg_traj_chg_count += 1

        if (
            self.manager._skill_changed is not None
            and pre_update_phase is not None
            and pre_update_traj is not None
        ):
            # Skill-change handler also gated by ``~external_set`` (already
            # baked into ``advance_mask`` via the head of this method).
            changed = self.manager._skill_changed & advancing_mask
            if changed.any():
                changed_ids = torch.where(changed)[0]
                prev_type = self.manager.data["traj_type"][pre_update_traj[changed_ids]]
                was_perpetual = prev_type == _PERPETUAL_INT
                target_phase = torch.where(
                    was_perpetual,
                    torch.zeros_like(pre_update_phase[changed_ids]),
                    pre_update_phase[changed_ids],
                )
                # DEBUG: print first few skill-change events for env 0.
                if _V2_DEBUG_PRINTS and 0 in changed_ids.tolist():
                    if not hasattr(self, "_dbg_skill_print_count"):
                        self._dbg_skill_print_count = 0
                    if self._dbg_skill_print_count < 5:
                        idx = (changed_ids == 0).nonzero(as_tuple=True)[0][0].item()
                        print(
                            f"[V2 SKILL-CHG env=0 ep_len={int(self.env.episode_length_buf[0])} "
                            f"prev_traj={int(pre_update_traj[0])} prev_phase={float(pre_update_phase[0]):.5f} "
                            f"prev_type={int(prev_type[idx])} was_perpetual={bool(was_perpetual[idx])} "
                            f"target_phase={float(target_phase[idx]):.5f} "
                            f"phase_before_set={float(self.manager.phase[0]):.5f}]"
                        )
                        self._dbg_skill_print_count += 1
                self.manager.set_phase(target_phase, changed_ids)
                if (
                    _V2_DEBUG_PRINTS
                    and 0 in changed_ids.tolist()
                    and getattr(self, "_dbg_skill_print_count", 0) <= 5
                ):
                    print(
                        f"[V2 SKILL-CHG env=0 phase_after_set={float(self.manager.phase[0]):.5f} "
                        f"next_gate_idx_after_set={int(self.manager.next_gate_idx[0])}]"
                    )
                # Match V1 hold-mode skill-change semantics: arm gate 0 of
                # the new trajectory unconditionally (regardless of current
                # phi).  ``set_phase`` above called ``_reseed_gate_for_envs``
                # which picks the first upcoming gate based on phase, so we
                # must override afterward.  Without this override, when phi
                # at swap is past gate 0's phi, V2 picks the wrap gate and
                # diverges from V1 by up to half a period.
                if self.cfg.hold_on_late_contact:
                    cur_traj_after = self.manager.get_current_trajectory_indices()[changed_ids]
                    new_num_gates = self.manager._num_gates_per_traj[cur_traj_after]
                    gate_init = torch.where(
                        new_num_gates > 0,
                        torch.zeros_like(new_num_gates),
                        -torch.ones_like(new_num_gates),
                    )
                    self.manager.next_gate_idx[changed_ids] = gate_init

        # Commit the new skill assignment so the next ``_ensure_cache``
        # rebuild correctly reports no further change for the envs whose
        # transition we just processed.  Done unconditionally for envs
        # that advanced this tick — covers the case where ``_skill_changed``
        # was wrongly False because some external caller (e.g. the reset
        # event) already invalidated and rebuilt the cache before we got
        # here, but we still need to keep ``_prev_skill_indices`` in sync.
        if advance_mask.any():
            self.manager.commit_skill_state(torch.where(advance_mask)[0])

        # Resolve current trajectory assignment after any phase mutations.
        self.manager.invalidate_cache()
        cur_traj = self.manager.get_current_trajectory_indices()

        if self._gating_enabled:
            contact_now = self._read_contact_now()
            # DEBUG: V2 trace
            _e = 0
            if int(self.env.episode_length_buf[_e]) < 0:
                print(f"[V2 contact step={int(self.env.episode_length_buf[_e])} contact={contact_now[_e].tolist()}]")
            self._apply_contact_gate(contact_now)

        total = self.manager.data["total_time"][cur_traj]
        t = self.manager.phase * total

        self._last_compute_step = ep_len.clone()
        self._cached_t = t
        # Snapshot the post-compute state so the next call's pre_update_*
        # is read from V2's own state, not from a manager that external
        # callers may have mutated in between.
        self._prev_compute_phase = self.manager.phase.clone()
        self._prev_compute_traj = (
            self.manager._cached_global_indices.clone()
            if self.manager._cached_global_indices is not None
            else None
        )

        # DEBUG: V2 end-of-step state
        _e = 0
        if int(ep_len[_e]) < 0:
            _total_e = self.manager.data["total_time"][cur_traj[_e]]
            _phi_e = float(self.manager.phase[_e])
            _t_e = _phi_e * float(_total_e)
            _traj_e = int(cur_traj[_e])
            _skill_e = int(self.manager.data["skill_idx"][cur_traj[_e]])
            _vel_e = self.env.command_manager.get_term(self.cfg.conditioner_generator_name).command[_e].tolist()
            print(f"[V2 step={int(ep_len[_e])} phi={_phi_e:.6f} t={_t_e:.6f} gate={int(self.manager.next_gate_idx[_e])} traj={_traj_e} skill={_skill_e} vel={_vel_e}]")

        return t



# =====================================================================
# Patches applied to make V2 (24531ad NEW cmd) match V1 (3f278d4 OLD cmd)
# behavior during the regression bisect.  All seven are verified via the
# dual-cmd play comparison (V1 shadow vs V2 primary).
# =====================================================================
#
# 1. Fire-target eps_phi shift fix
#    Where: ``multiskill_manager_v2.py``
#           ``snap_phase_to_new_domain``,
#           ``snap_phase_to_start_of_current_domain``.
#    What:  Snap to ``gate_phi % 1.0`` (no ``+ eps_phi`` shift).
#    Why:   Original NEW snap landed phase one ``eps_phi`` *past* the
#           gate; V1's fire snaps phase to ``target_phi`` exactly.  The
#           offset persisted across every fire, so V2 was permanently
#           ahead of V1 by ``eps_phi`` (~3% of a period) after the first
#           gate fire.
#
# 2. Skill-change gate re-arm
#    Where: ``_compute_time`` above, in the skill-change block.
#    What:  Detect skill change via ``manager._skill_changed`` and
#           call ``manager._reseed_gate_for_envs(changed_ids)`` (now
#           via ``set_phase``, which calls reseed internally).
#    Why:   NEW cmd had no skill-change handler.  When the conditioner
#           swapped skills, ``next_gate_idx`` carried over from the
#           previous trajectory unchanged — for perpetual→walking the
#           gate stayed at -1 and V2 never fired any gates.
#
# 3. Phase preserved across skill change (matches compute_transition_time)
#    Where: ``_compute_time`` above.
#    What:  Capture ``pre_update_phase`` and ``pre_update_traj`` before
#           ``update_phase`` runs.  On skill change, restore phase via
#           ``set_phase``: ``target_phase = 0`` if prev trajectory was
#           perpetual, else ``pre_update_phase``.
#    Why:   ``update_phase`` advances phase by ``step_dt / new_total``
#           on the transition step, so the new trajectory is evaluated
#           one step ahead of where V1's ``compute_transition_time``
#           would place it.  For perpetual prev, V1's
#           ``_compute_phasing_var`` returns 0 regardless of t, so the
#           target is 0; for non-perpetual, phi persists across the swap.
#
# 4. ``_skill_changed`` read order
#    Where: ``_compute_time`` above.
#    What:  Read ``manager._skill_changed`` *before* V2's redundant
#           ``invalidate_cache`` + ``get_current_trajectory_indices``.
#    Why:   The parent class ``BaseTrajectoryCommand._update_command``
#           already invalidated the cache before ``_compute_time`` ran.
#           ``update_phase``'s internal ``_get_global_indices`` triggered
#           the *first* rebuild and populated ``_skill_changed=True``.
#           V2's explicit ``invalidate_cache`` triggered a *second*
#           rebuild that wiped ``_skill_changed`` to False (because
#           ``_prev_skill_indices`` was just updated to the new value).
#           Reading the flag between the two rebuilds preserves it.
#
# 5. Hold-mode skill-change gate override
#    Where: ``_compute_time`` above.
#    What:  After ``set_phase`` (which auto-reseeds gate to first
#           upcoming based on phase), unconditionally override
#           ``next_gate_idx[changed] = 0`` when in hold mode.
#    Why:   V1's ``smooth_transitions`` block in hold mode arms gate 0
#           regardless of current phi.  ``_reseed_gate_for_envs`` picks
#           the first gate ``>= phase``, so when the swap happens at
#           ``phi=0.7`` (past the mid-period gate at 0.5), V2 armed the
#           wrap gate (1.0) while V1 armed gate 0 — a half-period
#           divergence that lasted until the trajectories re-synced.
#
# 6. Window width matches gate_phi * W
#    Where: ``_apply_contact_gate`` above.
#    What:  ``early_window_size = gate_phi * W`` (per-gate width),
#           applied to both ``in_early`` and ``in_late`` bounds.
#    Why:   Original NEW used a fixed window ``W`` regardless of
#           ``gate_phi``.  V1 used ``window_lo = target_phi * (1 - W)``
#           ⇒ window width ``target_phi * W``.  At a mid-period gate
#           (``phi=0.5, W=0.2``), V1's window was 0.1 phase-units wide
#           but V2's was 0.2 — twice as permissive on early fires.
#
# 7. Inclusive ``in_late`` (signed >= 0), strict ``in_early`` (signed < 0)
#    Where: ``_apply_contact_gate`` above.
#    What:  Treat ``signed == 0`` as ``in_late`` (so hold pulls back),
#           and ``signed < 0`` strict for ``in_early``.
#    Why:   V1's ``hold_mask = phi >= target_phi`` is *inclusive*: at
#           ``phi == target_phi`` exactly, V1 holds.  Original NEW used
#           strict ``signed > 0`` for ``in_late``, missing the boundary
#           and letting phase pass through without holding (V2 then
#           outpaced V1 by one step per FP-induced edge case).
#
# Also in ``multiskill_manager_v2.py``:
#   - Removed perpetual phase pinning in ``update_phase`` — phase now
#     advances naturally for perpetual trajectories with ``% 1.0`` wrap,
#     matching V1's behavior (where ``t`` advances regardless of
#     trajectory type).