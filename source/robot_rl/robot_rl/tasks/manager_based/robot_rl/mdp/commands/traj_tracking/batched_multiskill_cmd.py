"""Batched multi-skill trajectory command using :class:`MultiSkillManager`."""

from __future__ import annotations

import torch
import warp as wp

from .base_trajectory_cmd import BaseTrajectoryCommand
from .manager_base import ManagerBase
from .multiskill_manager import MultiSkillManager


class BatchedMultiSkillCommand(BaseTrajectoryCommand):
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
        return MultiSkillManager(
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

        Just-reset envs (``episode_length_buf == 0``) are excluded from
        the gate logic: their contact-sensor reading on this first step
        is the *pre-reset* physics force (no physics step runs between
        the reset event and this command compute), so a standing prev
        episode would leak both-feet-in-contact into ``contact_now`` and
        cause a false ``expected_landed`` for the new episode's first
        gate.  The phase was just set by :meth:`set_phase` in the reset
        event, so no contact-driven snap is needed on step 0 anyway.

        Args:
            contact_now: ``[N, B]`` boolean tensor of current per-env
                contact state in ``self.contact_bodies`` order.
        """
        next_gate_idx = self.manager.next_gate_idx

        active = next_gate_idx >= 0
        # Mask out envs that just reset — their contact reading is stale.
        active = active & (self.env.episode_length_buf > 0)
        if not active.any():
            return

        cur_traj = self.manager.get_current_trajectory_indices()
        safe_idx = torch.clamp(next_gate_idx, min=0)
        target_active = self.manager._gate_active_table[cur_traj, safe_idx]
        target_mask = self.manager._gate_contact_mask[cur_traj, safe_idx]
        # Per-env early window scales with the size of the domain *before*
        # this gate (the phi distance from the previous gate to this one).
        # ``contact_gate_window_frac`` is therefore interpreted as a
        # fraction of the local domain, not raw phi: with W=0.1 a
        # half-period gate gets a 0.05 phi early window and a full-period
        # gate gets 0.10.
        pre_size = self.manager._gate_pre_size_table[cur_traj, safe_idx]

        expected_landed = (contact_now & target_mask).any(dim=1)
        usable = active & target_active

        W = self.cfg.contact_gate_window_frac
        W_early = W * pre_size

        # ``gate_rel_phi`` is the manager's monotonic signed distance from
        # the currently armed gate (positive = past, negative = before).
        # No wrap ambiguity, so a single comparison classifies each env.
        signed = self.manager.gate_rel_phi

        in_early = (signed <= 0.0) & (signed >= -W_early)
        # Late side is fully unbounded — any expected contact past the
        # gate snaps the phase back to the start of the new domain, no
        # matter how late.  The gate stays armed until that contact
        # lands.  For phi=1.0 gates this naturally handles the cycle
        # boundary: after a phase wrap, ``gate_rel_phi`` becomes positive
        # and the next contact event fires a late snap.
        in_late = signed > 0.0

        if not self.cfg.hold_on_late_contact:
            early_fire_mask = usable & in_early & expected_landed
            late_fire_mask = usable & in_late & expected_landed

            early_ids = torch.where(early_fire_mask)[0]
            late_ids = torch.where(late_fire_mask)[0]

            self.manager.snap_phase_to_new_domain(early_ids)
            self.manager.snap_phase_to_start_of_current_domain(late_ids)
        else:
            # Hold-on mode: phase that crossed the gate without contact is
            # pulled back to the end of the old domain each step.  Contact
            # anywhere past the gate releases via snap_phase_to_new_domain.
            crossed_no_contact = usable & in_late & ~expected_landed
            crossed_with_contact = usable & in_late & expected_landed
            early_fire_mask = usable & in_early & expected_landed

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
        # TODO: Should probably check all of this again
        ep_len = self.env.episode_length_buf
        if (
            self._last_compute_step is not None
            and torch.equal(self._last_compute_step, ep_len)
            and self._cached_t is not None
        ):
            return self._cached_t

        advancing_mask = ep_len < self.env.max_episode_length
        if advancing_mask.any():
            adv_ids = torch.where(advancing_mask)[0]
            self.manager.update_phase(self.env.step_dt, env_ids=adv_ids)

        # Resolve current trajectory assignment after any phase mutations.
        self.manager.invalidate_cache()
        cur_traj = self.manager.get_current_trajectory_indices()

        if self._gating_enabled:
            contact_now = self._read_contact_now()
            self._apply_contact_gate(contact_now)

        total = self.manager.data["total_time"][cur_traj]
        t = self.manager.phase * total

        self._last_compute_step = ep_len.clone()
        self._cached_t = t
        return t
