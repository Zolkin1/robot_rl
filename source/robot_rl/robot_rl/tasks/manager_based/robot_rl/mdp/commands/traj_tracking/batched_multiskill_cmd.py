"""Batched multi-skill trajectory command using :class:`MultiSkillManager`."""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.utils.math import quat_apply, yaw_quat

from .base_trajectory_cmd import BaseTrajectoryCommand
from .manager_base import ManagerBase
from .multiskill_manager import MultiSkillManager


# Per-ref-frame color palette for debug viz polylines. Cycled if there are
# more reference frames than colors.
_DEBUG_VIZ_COLORS: tuple[tuple[float, float, float, float], ...] = (
    (1.0, 0.2, 0.2, 0.9),   # red
    (0.2, 0.4, 1.0, 0.9),   # blue
    (0.2, 1.0, 0.2, 0.9),   # green
    (1.0, 0.85, 0.2, 0.9),  # yellow
    (0.2, 1.0, 1.0, 0.9),   # cyan
    (1.0, 0.2, 1.0, 0.9),   # magenta
)
_DEBUG_VIZ_NUM_SAMPLES: int = 32
# Isaac Sim's debug_draw.draw_lines expects ints for line widths (see
# ``isaacsim/util/debug_draw/tests/test_debug_draw.py``).
_DEBUG_VIZ_LINE_THICKNESS: int = 3


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
    current phase value as-is.  All downstream manager evaluations
    (``get_output``, ``get_contact_state``, ...) take the phase directly.
    """

    # Class-level default: the parent ``CommandTerm.__init__`` calls
    # ``set_debug_vis`` (and therefore ``_set_debug_vis_impl``) *before*
    # ``_post_init`` runs, so the per-instance attribute may not exist yet.
    _debug_draw_iface: object | None = None

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
        """Initialise the manager's ref-frame lookup and the contact-gate
        wiring when gating is enabled.
        """
        self.manager.build_ref_frame_map(self.ref_frames)
        # Eagerly allocate manager phase state to num_envs.
        self.manager._ensure_phase_state(self.num_envs)

        # Idempotency guard for ``_pre_update_phase`` (``_update_command`` is
        # called twice per env step by IsaacLab's resample-then-update flow).
        # When the cached step buffer matches the current one we skip phase
        # advance + gate.
        self._last_compute_step: torch.Tensor | None = None

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

        # --- Debug viz state ---------------------------------------------
        # Discover every named frame in ``ordered_pos_output_names`` that has
        # a complete (pos_x, pos_y, pos_z) triplet. This includes both ref
        # frames (the feet) and non-ref frames (pelvis, hands, ...).
        # ``self._debug_frame_names`` keeps the discovery order; the
        # corresponding ``self._debug_frame_pos_idx`` holds the slice
        # tensors. Skips ``joint:*`` entries.
        seen_frames: list[str] = []
        for name in self.ordered_pos_output_names:
            if ":pos_" not in name:
                continue
            frame, _ = name.split(":", 1)
            if frame == "joint":
                continue
            if frame in seen_frames:
                continue
            seen_frames.append(frame)
        # Only keep frames that have all three pos_* components present.
        self._debug_frame_names: list[str] = []
        self._debug_frame_pos_idx: list[torch.Tensor] = []
        for frame in seen_frames:
            try:
                idx = [self.ordered_pos_output_names.index(f"{frame}:pos_{a}") for a in ("x", "y", "z")]
            except ValueError:
                continue
            self._debug_frame_names.append(frame)
            self._debug_frame_pos_idx.append(
                torch.tensor(idx, dtype=torch.long, device=self.device)
            )
        # Per-frame colour, cycled through the palette.
        self._debug_frame_colors: list[tuple[float, float, float, float]] = [
            _DEBUG_VIZ_COLORS[i % len(_DEBUG_VIZ_COLORS)] for i in range(len(self._debug_frame_names))
        ]
        # NOTE: do NOT reset ``self._debug_draw_iface`` here. The parent
        # ``CommandTerm.__init__`` calls ``_set_debug_vis_impl`` *before*
        # this method runs, and that's where the iface is acquired and
        # stored on the instance. Clobbering it here breaks per-frame
        # drawing. The class-level default (see top of class) covers the
        # case where ``_set_debug_vis_impl`` is called with ``False``.

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

    def _pre_update_phase(self) -> None:
        """Advance ``manager.phase`` and run the contact gate.

        Called once at the top of :meth:`BaseTrajectoryCommand._update_command`.

        Order per step:
          1. Advance phase by ``step_dt`` for non-reset envs (envs with
             ``episode_length_buf == 0`` keep the phase that was set by
             the reset event — :func:`reset_on_reference` calls
             :meth:`MultiSkillManager.set_phase` directly, so we must not
             clobber that here).
          2. Invalidate the trajectory-assignment cache and re-resolve so
             the contact gate sees the post-advance trajectory.
          3. Apply contact gate (if enabled) — issues snap calls to the
             manager.

        Idempotent across the same env step: IsaacLab's
        ``CommandTerm.compute`` calls ``_update_command`` twice when a
        resample fires (once from ``_resample_command`` and once
        directly).  Idempotency is per-env: only envs whose
        ``episode_length_buf`` actually advanced since the last call
        get their phase advanced.  Using a full-tensor ``torch.equal``
        guard would fail any time *any* env reset between the two
        calls — re-advancing every non-resetting env's phase a second
        time.  In a multi-env training run with frequent resets that
        compounds to roughly 2× the correct phase rate.
        """
        ep_len = self.env.episode_length_buf

        in_window = (ep_len > 0) & (ep_len < self.env.max_episode_length)
        if self._last_compute_step is None:
            advance_mask = in_window
        else:
            advance_mask = in_window & (ep_len != self._last_compute_step)

        if not advance_mask.any():
            # Nothing to advance — keep the snapshot fresh and skip the
            # gate logic.  No call to ``_ensure_cache`` happens here, so
            # ``_traj_changed`` retains whatever state the previous tick's
            # commit left it in (always all-False after a successful commit).
            self._last_compute_step = ep_len.clone()
            return

        adv_ids = torch.where(advance_mask)[0]
        self.manager.update_phase(self.env.step_dt, env_ids=adv_ids)

        # Resolve current trajectory assignment after any phase mutations.
        # Populates ``manager._traj_changed`` against the previous-tick
        # snapshot.  Crucially this no longer commits the snapshot — see
        # ``MultiSkillManager.commit_traj_state``.
        self.manager.invalidate_cache()
        self.manager.get_current_trajectory_indices()

        if self._gating_enabled:
            # Re-arm the gate for envs that just changed trajectory.  Without
            # this, ``next_gate_idx`` stays bound to the old trajectory's
            # gate slots — possibly inactive in the new trajectory — and the
            # gate logic silently no-ops for the rest of the episode.
            traj_changed = self.manager._traj_changed
            if traj_changed is not None and traj_changed.any():
                changed_ids = torch.where(traj_changed)[0]
                self.manager._reseed_gate_for_envs(changed_ids)

            contact_now = self._read_contact_now()
            self._apply_contact_gate(contact_now)

        # Commit the trajectory snapshot only after gate re-arm has read
        # ``_traj_changed``.  Subsequent intra-step ``_ensure_cache`` calls
        # (reset events, observation fns) will then correctly report no
        # further change for the current step.
        self.manager.commit_traj_state()

        self._last_compute_step = ep_len.clone()

    # ------------------------------------------------------------------
    # Debug visualization
    # ------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        """Acquire / release the Isaac Sim debug-draw interface.

        Uses Isaac Sim's ``isaacsim.util.debug_draw`` extension (only
        importable after AppLauncher has started Kit, hence the deferred
        import). The actual per-step drawing happens in
        :meth:`_debug_vis_callback`.
        """
        if debug_vis:
            if self._debug_draw_iface is None:
                try:
                    from isaacsim.util.debug_draw import _debug_draw  # type: ignore
                    self._debug_draw_iface = _debug_draw.acquire_debug_draw_interface()
                except Exception as exc:
                    print(f"[WARN BatchedMultiSkillCommand] "
                          f"Could not acquire debug_draw interface: {exc}")
                    self._debug_draw_iface = None
        else:
            if self._debug_draw_iface is not None:
                try:
                    self._debug_draw_iface.clear_lines()
                except Exception:
                    pass
            self._debug_draw_iface = None

    def _debug_vis_callback(self, event) -> None:  # noqa: D401 — IsaacLab convention
        """Draw, for each env and reference frame, the bezier curve of the
        currently-active domain in world coordinates.

        Vectorised across envs and samples: one ``draw_lines`` call covers
        ``num_envs × num_ref_frames × (S − 1)`` line segments.
        """
        if self._debug_draw_iface is None:
            return
        if not self.robot.is_initialized:
            return
        # ``_post_init`` may not have run yet on the very first
        # ``set_debug_vis(True)`` triggered from the parent ``__init__``.
        if not hasattr(self, "_debug_frame_pos_idx"):
            return

        self._debug_draw_iface.clear_lines()

        manager = self.manager
        phase = manager.phase                                         # (N,)
        traj_idx = manager.get_current_trajectory_indices()           # (N,)
        domain_idx = manager._get_domain_indices(phase, traj_idx)     # (N,)

        coeffs_pos = manager.data["coeffs_pos"][traj_idx, domain_idx]  # (N, P, K+1)
        N, P_dim, K1 = coeffs_pos.shape
        S = _DEBUG_VIZ_NUM_SAMPLES

        # Bernstein-polynomial evaluation, vectorised across S samples.
        degree = K1 - 1
        coefs = manager._binomial_coeffs[degree]                      # (K+1,)
        i_vec = torch.arange(K1, device=self.device)                  # (K+1,)
        tau = torch.linspace(0.0, 1.0, S, device=self.device)         # (S,)
        tau_pow = tau.unsqueeze(1) ** i_vec                           # (S, K+1)
        one_minus_pow = (1.0 - tau).unsqueeze(1) ** (degree - i_vec)  # (S, K+1)
        weights = coefs * tau_pow * one_minus_pow                     # (S, K+1)
        local_outs = torch.einsum("sd,npd->nsp", weights, coeffs_pos) # (N, S, P)

        # Local → world via the env's current ref-frame anchor.
        anchor = self.ref_poses[:, :3]                                # (N, 3)
        ref_quat = self.ref_poses[:, 3:]                              # (N, 4) xyzw
        yaw_q = yaw_quat(ref_quat)                                    # (N, 4)
        yaw_q_exp = yaw_q.unsqueeze(1).expand(N, S, 4)                # (N, S, 4)
        anchor_exp = anchor.unsqueeze(1)                              # (N, 1, 3)

        # Build a curve per discovered frame (feet, pelvis, hands, ...) and
        # stack into one flat ``draw_lines`` call.
        starts_chunks: list[torch.Tensor] = []
        ends_chunks: list[torch.Tensor] = []
        colors_chunks: list[torch.Tensor] = []
        for frame_i, pos_idx in enumerate(self._debug_frame_pos_idx):
            local_pos = local_outs[:, :, pos_idx]                     # (N, S, 3)
            world_pos = anchor_exp + quat_apply(yaw_q_exp, local_pos) # (N, S, 3)
            starts_chunks.append(world_pos[:, :-1, :].reshape(-1, 3))
            ends_chunks.append(world_pos[:, 1:, :].reshape(-1, 3))
            color = torch.tensor(
                self._debug_frame_colors[frame_i], device=self.device
            )
            colors_chunks.append(
                color.expand(starts_chunks[-1].shape[0], 4)
            )

        # ``draw_lines`` expects list[tuple[float, float, float]] for points,
        # list[tuple[float, float, float, float]] for colors, list[int] for
        # widths. tolist() yields list[list[float]] which most pybind11
        # bindings accept, but make the tuple-ness explicit to be safe.
        starts_raw = torch.cat(starts_chunks, dim=0).cpu().tolist()
        ends_raw = torch.cat(ends_chunks, dim=0).cpu().tolist()
        colors_raw = torch.cat(colors_chunks, dim=0).cpu().tolist()
        starts_t = [tuple(p) for p in starts_raw]
        ends_t = [tuple(p) for p in ends_raw]
        colors_t = [tuple(c) for c in colors_raw]
        thicknesses = [_DEBUG_VIZ_LINE_THICKNESS] * len(starts_t)

        if starts_t:
            self._debug_draw_iface.draw_lines(starts_t, ends_t, colors_t, thicknesses)
