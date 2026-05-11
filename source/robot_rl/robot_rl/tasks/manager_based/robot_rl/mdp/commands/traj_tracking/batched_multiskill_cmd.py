"""Batched multi-skill trajectory command using :class:`MultiSkillManager`."""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.utils.math import quat_apply, quat_from_angle_axis, yaw_quat

from .base_trajectory_cmd import BaseTrajectoryCommand
from .manager_base import ManagerBase
from .multiskill_manager import MultiSkillManager


# Per-frame color palette for debug viz cylinder prototypes. Cycled if there
# are more reference frames than colors.
_DEBUG_VIZ_COLORS: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.2, 0.2),   # red
    (0.2, 0.4, 1.0),   # blue
    (0.2, 1.0, 0.2),   # green
    (1.0, 0.85, 0.2),  # yellow
    (0.2, 1.0, 1.0),   # cyan
    (1.0, 0.2, 1.0),   # magenta
)
_DEBUG_VIZ_NUM_SAMPLES: int = 32
# Cylinder radius used for the segment prototypes (metres).  The cylinder's
# native height is 1.0 m — per-segment length is applied via the z-scale
# component at ``visualize`` time.
_DEBUG_VIZ_CYLINDER_RADIUS: float = 0.005
_DEBUG_VIZ_EPS: float = 1e-6
_DEBUG_VIZ_PRIM_PATH: str = "/Visuals/traj_ref_segments"
# Frames whose name contains any of these substrings are skipped during
# debug-viz discovery.  Default hides the hand trajectories (named
# ``*_wrist_yaw_link`` on the G1); clear the tuple to show them again.
_DEBUG_VIZ_HIDDEN_FRAME_SUBSTRINGS: tuple[str, ...] = ("wrist", "hand")
# Spheres drawn at the end of each visualized frame's trajectory.  Per-env
# radius is ``max(_DEBUG_VIZ_END_SPHERE_DIST_FRAC * chord,
# _DEBUG_VIZ_END_SPHERE_MIN_RADIUS)`` where ``chord`` is
# ``||last_cp - first_cp||`` of the **current domain's** Bezier for that
# frame.  Matches the scale used by the
# ``frame_deviation_from_reference`` termination — set this fraction and
# min equal to the termination's ``max_frac`` / ``min_dist`` to visualize
# the actual cutoff.
_DEBUG_VIZ_END_SPHERE_DIST_FRAC: float = 0.25
_DEBUG_VIZ_END_SPHERE_MIN_RADIUS: float = 0.1


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
    # ``_debug_markers`` is lazily constructed once frame discovery in
    # ``_post_init`` has populated ``_debug_frame_names``; ``_debug_vis_pending``
    # remembers a ``set_debug_vis(True)`` request that arrived before then.
    _debug_markers: object | None = None
    _debug_vis_pending: bool = False

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
            if any(sub in frame for sub in _DEBUG_VIZ_HIDDEN_FRAME_SUBSTRINGS):
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
        # Per-frame RGB colour, cycled through the palette.  One marker
        # prototype per frame uses this as its ``PreviewSurfaceCfg``
        # diffuse colour, so the resulting cylinders inherit the per-frame
        # tint without needing per-instance colour at ``visualize`` time.
        self._debug_frame_colors: list[tuple[float, float, float]] = [
            _DEBUG_VIZ_COLORS[i % len(_DEBUG_VIZ_COLORS)] for i in range(len(self._debug_frame_names))
        ]

        # If ``set_debug_vis(True)`` arrived before this point (the parent
        # ``CommandTerm.__init__`` calls it before ``_post_init`` runs),
        # construct the marker instance now that we know how many frame
        # prototypes to allocate.
        if self._debug_vis_pending and self._debug_markers is None:
            self._debug_markers = self._create_debug_markers()
            if self._debug_markers is not None:
                self._debug_markers.set_visibility(True)
            self._debug_vis_pending = False

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
        """Toggle the trajectory-segment marker visualizer.

        Uses :class:`isaaclab.markers.VisualizationMarkers` to render one
        cylinder per Bezier-sample segment per discovered reference frame.
        The actual per-step update happens in :meth:`_debug_vis_callback`.
        """
        if debug_vis:
            # Frame discovery happens in ``_post_init``, which the parent
            # ``CommandTerm.__init__`` calls *after* ``_set_debug_vis_impl``
            # — defer construction until we know the prototype layout.
            if self._debug_markers is None and hasattr(self, "_debug_frame_pos_idx"):
                self._debug_markers = self._create_debug_markers()
            if self._debug_markers is not None:
                self._debug_markers.set_visibility(True)
            else:
                self._debug_vis_pending = True
        else:
            self._debug_vis_pending = False
            if self._debug_markers is not None:
                self._debug_markers.set_visibility(False)

    def _create_debug_markers(self):
        """Instantiate the :class:`VisualizationMarkers` with one cylinder
        prototype per discovered frame.

        Returns ``None`` if Isaac Lab's markers / sim subpackages aren't
        importable at this point (most commonly when the simulator has
        not been launched yet — :meth:`_set_debug_vis_impl` will retry
        the next time it's called with ``True``).
        """
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        except Exception as exc:
            print(f"[WARN BatchedMultiSkillCommand] "
                  f"Could not import VisualizationMarkers: {exc}")
            return None

        prototypes: dict[str, object] = {}
        for name, color in zip(self._debug_frame_names, self._debug_frame_colors):
            prototypes[name] = sim_utils.CylinderCfg(
                radius=_DEBUG_VIZ_CYLINDER_RADIUS,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
            )
        # One sphere prototype per frame at native radius 1.0 m — per-env
        # radius is applied via the ``scales`` arg in ``_debug_vis_callback``.
        # Sphere prototype indices are ``F..2F-1``, matching the cylinder
        # frame order so frame ``i`` re-uses its colour for the end marker.
        for name, color in zip(self._debug_frame_names, self._debug_frame_colors):
            prototypes[f"_{name}_end_sphere"] = sim_utils.SphereCfg(
                radius=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
            )
        cfg = VisualizationMarkersCfg(prim_path=_DEBUG_VIZ_PRIM_PATH, markers=prototypes)
        try:
            return VisualizationMarkers(cfg)
        except Exception as exc:
            print(f"[WARN BatchedMultiSkillCommand] "
                  f"Could not create VisualizationMarkers: {exc}")
            return None

    def _debug_vis_callback(self, event) -> None:  # noqa: D401 — IsaacLab convention
        """Push one batched ``visualize`` call per tick.

        Per-env Bezier samples for every discovered frame are stacked
        into a flat list of ``num_envs × num_frames × (S − 1)`` cylinders;
        the prototype index of each cylinder picks the frame's colour.
        """
        # Frame discovery may not have run yet (parent ``__init__``
        # invokes ``_set_debug_vis_impl`` before ``_post_init``).
        if not hasattr(self, "_debug_frame_pos_idx"):
            return
        if not self.robot.is_initialized:
            return
        if self._debug_markers is None:
            self._debug_markers = self._create_debug_markers()
            if self._debug_markers is None:
                return
            self._debug_markers.set_visibility(True)

        manager = self.manager
        phase = manager.phase                                         # (N,)
        traj_idx = manager.get_current_trajectory_indices()           # (N,)
        domain_idx = manager._get_domain_indices(phase, traj_idx)     # (N,)

        coeffs_pos = manager.data["coeffs_pos"][traj_idx, domain_idx]  # (N, P, K+1)
        N, _, K1 = coeffs_pos.shape
        S = _DEBUG_VIZ_NUM_SAMPLES
        F = len(self._debug_frame_pos_idx)
        if F == 0 or N == 0 or S < 2:
            return

        # Bernstein-polynomial evaluation, vectorised across S samples.
        degree = K1 - 1
        coefs = manager._binomial_coeffs[degree]                      # (K+1,)
        i_vec = torch.arange(K1, device=self.device)                  # (K+1,)
        tau = torch.linspace(0.0, 1.0, S, device=self.device)         # (S,)
        tau_pow = tau.unsqueeze(1) ** i_vec                           # (S, K+1)
        one_minus_pow = (1.0 - tau).unsqueeze(1) ** (degree - i_vec)  # (S, K+1)
        weights = coefs * tau_pow * one_minus_pow                     # (S, K+1)
        local_outs = torch.einsum("sd,npd->nsp", weights, coeffs_pos) # (N, S, P)

        # Gather the (x, y, z) slice for every discovered frame at once.
        frame_idx = torch.stack(self._debug_frame_pos_idx, dim=0)     # (F, 3)
        local_pos = local_outs[:, :, frame_idx]                       # (N, S, F, 3)

        # Local → world via the env's current ref-frame anchor.
        anchor = self.ref_poses[:, :3]                                # (N, 3)
        ref_quat = self.ref_poses[:, 3:]                              # (N, 4) xyzw
        yaw_q = yaw_quat(ref_quat)                                    # (N, 4)
        yaw_q_exp = yaw_q.view(N, 1, 1, 4).expand(N, S, F, 4)
        anchor_exp = anchor.view(N, 1, 1, 3)
        world_pos = anchor_exp + quat_apply(yaw_q_exp, local_pos)     # (N, S, F, 3)

        # Per-segment endpoints, midpoint, direction, length.
        starts = world_pos[:, :-1, :, :]                              # (N, S-1, F, 3)
        ends = world_pos[:, 1:, :, :]
        centers = 0.5 * (starts + ends)                               # (N, S-1, F, 3)
        vecs = ends - starts
        lens = torch.linalg.norm(vecs, dim=-1).clamp(min=_DEBUG_VIZ_EPS)
        dirs = vecs / lens.unsqueeze(-1)                              # (N, S-1, F, 3)

        # Quaternion that rotates the cylinder's local +z onto ``dirs``.
        # ``axis = +z × dir = (-dir.y, dir.x, 0)`` is zero whenever ``dir``
        # is parallel to ±z — pick the +x axis as a fallback (any axis in
        # the xy-plane gives the correct ±π rotation).
        z_dot_dir = dirs[..., 2].clamp(-1.0, 1.0)
        angle = torch.acos(z_dot_dir)
        axis = torch.stack(
            [-dirs[..., 1], dirs[..., 0], torch.zeros_like(dirs[..., 0])],
            dim=-1,
        )
        axis_norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
        fallback = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand_as(axis)
        axis = torch.where(axis_norm > _DEBUG_VIZ_EPS, axis, fallback)
        quats = quat_from_angle_axis(angle.reshape(-1), axis.reshape(-1, 3))  # (M, 4) xyzw

        # Scale: cylinder native radius/height are baked into the prototype;
        # we only scale here, including stretching height along z by segment
        # length.
        scales = torch.ones_like(dirs)                                # (N, S-1, F, 3)
        scales[..., 2] = lens                                         # z scale = segment length

        # Marker index per cylinder == frame index in
        # ``self._debug_frame_names`` (== prototype dict insertion order).
        frame_indices = torch.arange(F, device=self.device, dtype=torch.long)
        marker_indices = frame_indices.view(1, 1, F).expand(N, S - 1, F).reshape(-1)

        translations = centers.reshape(-1, 3)
        scales_flat = scales.reshape(-1, 3)

        # One sphere per env per frame at the end of that frame's Bezier
        # samples.  Sphere prototype indices are ``F..2F-1``, matching the
        # cylinder frame order.  Radius matches the
        # ``frame_deviation_from_reference`` termination scale: per-env
        # per-frame ``max(frac * current_domain_chord, min_radius)``.
        frame_start = world_pos[:, 0, :, :]                                    # (N, F, 3)
        frame_end = world_pos[:, -1, :, :]                                     # (N, F, 3)
        frame_dist = torch.linalg.norm(frame_end - frame_start, dim=-1)        # (N, F)
        sphere_radius = torch.clamp(
            frame_dist * _DEBUG_VIZ_END_SPHERE_DIST_FRAC,
            min=_DEBUG_VIZ_END_SPHERE_MIN_RADIUS,
        )                                                                       # (N, F)
        sphere_translations = frame_end.reshape(-1, 3)                         # (N*F, 3)
        sphere_scales = sphere_radius.unsqueeze(-1).expand(N, F, 3).reshape(-1, 3)
        sphere_quats = quat_from_angle_axis(
            torch.zeros(N * F, device=self.device),
            torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(N * F, 3),
        )
        sphere_indices = (
            (torch.arange(F, device=self.device, dtype=torch.long) + F)
            .view(1, F)
            .expand(N, F)
            .reshape(-1)
        )

        translations = torch.cat([translations, sphere_translations], dim=0)
        quats = torch.cat([quats, sphere_quats], dim=0)
        scales_flat = torch.cat([scales_flat, sphere_scales], dim=0)
        marker_indices = torch.cat([marker_indices, sphere_indices], dim=0)

        self._debug_markers.visualize(
            translations=translations,
            orientations=quats,
            scales=scales_flat,
            marker_indices=marker_indices,
        )
