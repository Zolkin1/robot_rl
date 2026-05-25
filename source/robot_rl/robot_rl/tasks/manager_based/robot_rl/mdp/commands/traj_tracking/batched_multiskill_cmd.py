"""Batched multi-skill trajectory command using :class:`MultiSkillManager`."""

from __future__ import annotations

import torch
import warp as wp

from isaaclab.utils.math import quat_apply, quat_from_angle_axis, yaw_quat

from .base_trajectory_cmd import BaseTrajectoryCommand
from .manager_base import ManagerBase
from .multiskill_manager import MultiSkillManager
from .skill_bucket import (
    bucket_for_velocity,
    commit_pending_at_fire,
    step_skill_pending,
)
from .skill_state import CrossfadeState, PendingSkillChange
from .skill_transition import blend_outputs


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
# Master toggle for the end-of-trajectory spheres.  Set to False to hide
# them entirely (no prototype, no per-step viz emission).
_DEBUG_VIZ_SHOW_END_SPHERES: bool = False
# Flat disk drawn at each env's reference-pose anchor position so the
# anchor frame the trajectory is rendered around is visible at a glance.
_DEBUG_VIZ_SHOW_REF_POSE: bool = True
_DEBUG_VIZ_REF_POSE_RADIUS: float = 0.15
_DEBUG_VIZ_REF_POSE_HEIGHT: float = 0.005
_DEBUG_VIZ_REF_POSE_COLOR: tuple[float, float, float] = (1.0, 1.0, 1.0)


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

        # Per-env seconds since the last trajectory change (or fresh
        # episode reset).  Initialised large so the very first compute is
        # not treated as "just changed".  Read by
        # ``mdp.frame_deviation_from_reference`` to suppress its
        # termination during the post-transition grace window.
        self.time_since_traj_change_s = torch.full(
            (self.num_envs,), 1.0e6, device=self.device
        )

        # --- Skill-transition cross-fade state ---------------------------
        # Started by ``_apply_contact_gate`` when a gate fires and the
        # local pending queue commits for that env; the next
        # ``_transform_desired_outputs`` calls blend the (synthetic-
        # traj-idx) old trajectory output into the cached new trajectory
        # output, weighted by ``alpha = phi_elapsed / blend_end_phi``.
        # The phi delta read inside ``step()`` comes from
        # ``manager.last_phase_delta`` so no per-step phase snapshot is
        # needed on this class.
        self.transition = CrossfadeState(self.num_envs, device=self.device)

        # --- Skill ownership + bucket tables -----------------------------
        # The trajectory cmd owns the active ``skill_id`` (what the
        # ``MultiSkillManager`` filters trajectories on).  It is derived
        # each step from which velocity bucket the ramped
        # ``vel_target_b`` currently sits in — *not* from whatever bucket
        # the velocity cmd most recently sampled.  Cross-bucket changes
        # are queued in ``self.pending`` and committed when the contact
        # gate fires.
        self._terrain = self.env.scene[self.cfg.terrain_name]
        for required in ("skill_probs_at", "skill_list", "world_xy_to_cell"):
            if not hasattr(self._terrain, required):
                raise TypeError(
                    f"BatchedMultiSkillCommand requires the scene's "
                    f"'{self.cfg.terrain_name}' entity to expose '{required}'. "
                    f"Got {type(self._terrain).__name__}; this command needs a "
                    f"MetaTerrainImporter (or subclass)."
                )
        self._skill_list: list[str] = list(self._terrain.skill_list)
        bucket_keys = set(self.cfg.velocity_buckets.keys())
        skill_set = set(self._skill_list)
        if bucket_keys != skill_set:
            missing = skill_set - bucket_keys
            extra = bucket_keys - skill_set
            raise ValueError(
                f"velocity_buckets keys must equal terrain.skill_list. "
                f"missing buckets for skills: {sorted(missing)}; "
                f"extra buckets not in skill_list: {sorted(extra)}."
            )
        self._skill_lin_vel_x = torch.tensor(
            [self.cfg.velocity_buckets[s].lin_vel_x for s in self._skill_list],
            device=self.device, dtype=torch.float,
        )
        self._skill_lin_vel_y = torch.tensor(
            [self.cfg.velocity_buckets[s].lin_vel_y for s in self._skill_list],
            device=self.device, dtype=torch.float,
        )
        self._skill_ang_vel_z = torch.tensor(
            [self.cfg.velocity_buckets[s].ang_vel_z for s in self._skill_list],
            device=self.device, dtype=torch.float,
        )
        self.skill_id = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device,
        )
        self.pending = PendingSkillChange(self.num_envs, device=self.device)
        # Velocity cmd is registered alongside this term; resolution is
        # lazy because ``env.command_manager`` is only finalised after
        # all terms construct.
        self._vel_cmd = None
        # Register us as the manager's source-of-truth for skill_id.
        self.manager.set_skill_owner(self)

        # --- Precomputed orientation-quat index groups for SLERP ---------
        # For each body frame in the position outputs that has a complete
        # (ori_x, ori_y, ori_z, ori_w) quartet, store the 4 column indices
        # so the cross-fade can SLERP that quaternion instead of linearly
        # blending it.  Frames without all four ori components are skipped.
        ori_groups: list[list[int]] = []
        for name in self.ordered_pos_output_names:
            if ":ori_x" not in name:
                continue
            frame, _ = name.split(":", 1)
            if frame == "joint":
                continue
            try:
                quad = [
                    self.ordered_pos_output_names.index(f"{frame}:ori_{a}")
                    for a in ("x", "y", "z", "w")
                ]
            except ValueError:
                continue
            ori_groups.append(quad)
        if ori_groups:
            self._transition_quat_index_groups = torch.tensor(
                ori_groups, dtype=torch.long, device=self.device
            )
        else:
            self._transition_quat_index_groups = None

        # --- Contact-gate wiring -----------------------------------------
        self._gating_enabled = self.cfg.contact_gate_window_frac is not None

        # The gate-on-contact deferral relies on the contact gate firing
        # to drain the local pending queue.  If the gate is disabled but
        # ``gate_skill_change_on_contact`` is True, every cross-bucket
        # transition would defer to a gate that never fires and the env
        # would lock on its initial skill.  Fail hard so the operator
        # surfaces the cfg mismatch immediately instead of silently
        # mutating one of the two fields.
        if not self._gating_enabled and self.cfg.gate_skill_change_on_contact:
            raise ValueError(
                "Inconsistent cfg: contact_gate_window_frac is None but "
                "gate_skill_change_on_contact is True.  Pending skill "
                "changes would never commit.  Set "
                "contact_gate_window_frac to a non-None value OR set "
                "gate_skill_change_on_contact=False."
            )

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
            fire_ids = torch.cat([early_ids, late_ids])
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
            # Only the "new domain" snaps actually advance into a new
            # cycle / half-cycle — those are the contact-switch events.
            # End-of-previous-domain snaps are hold-back, not a contact
            # landing, so they don't trigger a skill commit.
            fire_ids = new_dom_ids

        # If any of the envs that just fired a gate has a pending skill
        # change buffered on the velocity command, commit it now and start
        # the cross-fade.
        if fire_ids.numel() > 0:
            self._commit_pending_skill_change_and_start_transition(fire_ids, cur_traj)

    def current_chord_per_frame(self, frame_pos_indices: torch.Tensor) -> torch.Tensor:
        """Per-env per-frame Bezier chord length of the current target.

        For envs not in an active transition this is just the new
        trajectory's current-domain ``||last_cp - first_cp||`` per frame.
        For envs that ARE transitioning, the chord is alpha-blended with
        the old trajectory's current-domain chord using the same alpha
        applied to ``y_des`` in :meth:`_transform_desired_outputs` —
        ``(1 - alpha) * old_chord + alpha * new_chord``.  That keeps the
        scale tensor used by
        :func:`mdp.frame_deviation_from_reference` self-consistent with
        the (possibly blended) ``y_des`` the same termination compares
        against; the threshold morphs from pure-old to pure-new as the
        blend progresses instead of jumping at gate-fire time.

        Args:
            frame_pos_indices: ``[F, 3]`` long tensor of column indices
                into ``ordered_pos_output_names`` (xyz per frame).

        Returns:
            ``[N, F]`` chord length per env per frame.
        """
        mgr = self.manager

        # New (target) chord — always.
        traj_idx = mgr.get_current_trajectory_indices()
        dom_idx = mgr._get_domain_indices(mgr.phase, traj_idx)
        coeffs = mgr.data["coeffs_pos"][traj_idx, dom_idx]            # (N, P, K+1)
        frame_coeffs = coeffs[:, frame_pos_indices, :]                # (N, F, 3, K+1)
        new_chord = torch.linalg.norm(
            frame_coeffs[..., -1] - frame_coeffs[..., 0], dim=-1
        )                                                             # (N, F)

        if (
            self.cfg.transition_blend_end_phi <= 0.0
            or not self.transition.active.any()
        ):
            return new_chord

        # Old (fading-out) chord for transitioning envs only.  Must use
        # the SAME phase source for the old domain lookup that
        # ``compute_blended_outputs_at`` uses for the old eval — else
        # the chord (termination scale) and the error (numerator
        # against y_des) live in different old-trajectory domains and
        # ``frame_deviation_from_reference`` mis-scales.  Concretely:
        # for periodic→perpetual the blended y_des uses old at
        # ``transition.old_phase`` (cycling walking), but a naïve chord
        # lookup using ``manager.phase`` (== 0 for perpetual new) would
        # land in walking's domain 0 (support foot, chord ≈ 0) regardless
        # of where the env actually is.  The decision matches the helper:
        # sync to manager.phase when new has gates, decouple to
        # transition.old_phase when new is perpetual.
        tx_ids = torch.where(self.transition.active)[0]                 # (M,)
        old_traj = self.transition.old_traj_idx[tx_ids]
        new_traj_tx = traj_idx[tx_ids]
        new_has_gates_tx = mgr._num_gates_per_traj[new_traj_tx] > 0
        phase_m = torch.where(
            new_has_gates_tx,
            mgr.phase[tx_ids],
            self.transition.old_phase[tx_ids],
        )
        old_dom = mgr._get_domain_indices(phase_m, old_traj)
        old_coeffs = mgr.data["coeffs_pos"][old_traj, old_dom]          # (M, P, K+1)
        old_frame_coeffs = old_coeffs[:, frame_pos_indices, :]          # (M, F, 3, K+1)
        old_chord = torch.linalg.norm(
            old_frame_coeffs[..., -1] - old_frame_coeffs[..., 0], dim=-1
        )                                                               # (M, F)

        alpha = self.transition.alpha(
            tx_ids, self.cfg.transition_blend_end_phi
        ).unsqueeze(-1)                                                  # (M, 1)

        blended = (1.0 - alpha) * old_chord + alpha * new_chord[tx_ids]
        new_chord[tx_ids] = blended
        return new_chord

    def _commit_pending_skill_change_and_start_transition(
        self, fire_ids: torch.Tensor, cur_traj_pre_commit: torch.Tensor,
    ) -> None:
        """Drain the local pending queue for ``fire_ids`` and start the
        cross-fade for envs whose commit actually flipped the skill.

        Args:
            fire_ids: ``[K]`` global env indices that just fired a contact
                gate (early + late snaps).
            cur_traj_pre_commit: ``[N]`` trajectory indices that were
                current *before* the commit, used as the fade-out source.
        """
        if fire_ids.numel() == 0:
            return
        pending_active_subset = self.pending.active[fire_ids]
        pending_skill_subset = self.pending.skill_id[fire_ids]
        active_subset = self.skill_id[fire_ids]
        new_active, new_pending_active, commit_mask = commit_pending_at_fire(
            torch.ones_like(pending_active_subset),  # all fire_ids fired by definition
            pending_active_subset,
            pending_skill_subset,
            active_subset,
        )
        if not commit_mask.any():
            return
        commit_ids = fire_ids[commit_mask]

        # Write back into the global buffers.
        self.skill_id[commit_ids] = new_active[commit_mask]
        self.pending.active[fire_ids] = new_pending_active

        # Snapshot the env's phase BEFORE rebuilding the cache (the new
        # trajectory may have different phase-advance rules — e.g.
        # perpetual snaps to 0 — and we want the OLD trajectory's clock
        # in the transition to continue from where the env actually was
        # at commit time, not from 0).
        initial_old_phase = self.manager.phase[commit_ids].clone()
        old_traj_commit = cur_traj_pre_commit[commit_ids]

        # Force the manager to rebuild its trajectory assignment with the
        # newly committed skill so we can read each env's NEW trajectory
        # index for the both-perpetual guard below.
        self.manager.invalidate_cache()
        new_traj_commit = self.manager.get_current_trajectory_indices()[commit_ids]

        # Guard: cross-fade alpha advances from progress on EITHER the
        # new or old trajectory's phase (see ``_transform_desired_outputs``
        # below).  If BOTH are perpetual neither side advances, alpha
        # never moves, and the blend silently stalls.  In the current
        # trajectory pool this combination never arises (only ``standing``
        # is perpetual), but surface it loudly if a future cfg creates
        # a perpetual→perpetual transition.
        old_gates = self.manager._num_gates_per_traj[old_traj_commit]
        new_gates = self.manager._num_gates_per_traj[new_traj_commit]
        both_perpetual = (old_gates == 0) & (new_gates == 0)
        if both_perpetual.any():
            bad_ids = commit_ids[both_perpetual]
            raise ValueError(
                "Cross-fade is not defined for perpetual→perpetual skill "
                "transitions (both old and new trajectories have zero "
                "contact gates, so the blend clock cannot advance).  "
                f"Affected env ids: {bad_ids.tolist()}.  Either configure "
                "one of the involved skills as periodic/episodic, or extend "
                "_commit_pending_skill_change_and_start_transition to "
                "handle this case explicitly (e.g. instant-flip)."
            )

        # Start the cross-fade with the pre-commit trajectory as the
        # fade-out source, seeded with the env's current phase so the
        # old-trajectory clock continues from where it was.
        self.transition.start(
            commit_ids, old_traj_commit, initial_old_phase,
        )

        # Re-arm the contact gate for these envs so ``next_gate_idx``
        # points at the new trajectory's gate layout.
        self.manager._reseed_gate_for_envs(commit_ids)

    # ------------------------------------------------------------------
    # Bucket-driven active-skill state machine
    # ------------------------------------------------------------------

    def _skill_for_velocity(
        self,
        vel: torch.Tensor,
        env_ids: torch.Tensor,
        xy_w: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the per-env desired skill_id given ``vel`` and the cell.

        Each env's skill is picked by ``argmin`` of per-bucket box-distance
        to ``vel`` over the env's *eligible* skills (eligibility from the
        composite importer's per-block ``skill_probs_at(xy)``). When ``vel``
        sits inside an eligible bucket that bucket wins (dist 0); when ``vel``
        is outside every eligible bucket the eligible bucket with the
        nearest edge wins. The latter is what flips the skill the instant
        the stance foot enters a block where the previous skill is no
        longer eligible — independent of where the velocity ramp currently
        sits.

        Args:
            vel: ``[K, 3]`` ``(vx, vy, wz)`` per env.
            env_ids: ``[K]`` global env indices (used for the cell lookup
                anchor when ``xy_w`` is not supplied).
            xy_w: Optional ``[K, 2]`` world-frame ``(x, y)`` override
                for the cell lookup.  Default uses ``self.ref_poses`` —
                stable across pelvis jitter mid-episode.  The reset
                path must override with ``env.scene.env_origins`` since
                ``ref_poses`` is stale at reset-event time.

        Returns:
            ``[K]`` long skill indices into ``terrain.skill_list``.
        """
        if xy_w is None:
            xy_w = self.ref_poses[env_ids, :2]
        eligible = self._terrain.skill_probs_at(xy_w).T > 0.0  # [S, K]
        return bucket_for_velocity(
            vel,
            eligible,
            self._skill_lin_vel_x,
            self._skill_lin_vel_y,
            self._skill_ang_vel_z,
        )

    def _resolve_vel_cmd(self) -> None:
        """Lazy-resolve the velocity command term on first compute."""
        if self._vel_cmd is None:
            self._vel_cmd = self.env.command_manager.get_term(
                self.cfg.velocity_command_name
            )

    def _update_skill_pending_from_velocity(self, env_ids: torch.Tensor) -> None:
        """Per-step: derive desired skill from ``vel_target_b`` and update
        ``self.skill_id`` / ``self.pending`` accordingly.

        Called at the top of :meth:`_pre_update_phase` before the manager
        cache rebuild so the trajectory-selection step sees the freshest
        ``skill_id``.

        Args:
            env_ids: ``[K]`` long indices of envs to process (typically the
                advance mask — envs that actually stepped this tick).
        """
        if env_ids.numel() == 0:
            return
        self._resolve_vel_cmd()
        vel = self._vel_cmd.vel_target_b[env_ids]
        desired = self._skill_for_velocity(vel, env_ids)

        active_subset = self.skill_id[env_ids]
        pa_subset = self.pending.active[env_ids]
        ps_subset = self.pending.skill_id[env_ids]
        a_out, pa_out, ps_out, tx_clear = step_skill_pending(
            desired,
            active_subset,
            pa_subset,
            ps_subset,
            self.cfg.gate_skill_change_on_contact,
        )

        # Per-env override: gate-on-contact requires the active trajectory
        # to *have* contact gates to ever drain pending.  Skills like
        # ``standing`` typically have zero gates per period — if we leave
        # those envs in the deferred path the pending queue would never
        # commit and the env would be stuck on standing while
        # ``vel_target_b`` ramps into a walking/running bucket.  For envs
        # whose active trajectory has no gates, flip immediately.
        cur_traj_full = self.manager.get_current_trajectory_indices()
        active_has_gates = self.manager._num_gates_per_traj[cur_traj_full[env_ids]] > 0
        force_instant = pa_out & ~active_has_gates
        if force_instant.any():
            a_out = torch.where(force_instant, ps_out, a_out)
            pa_out = pa_out & ~force_instant
            tx_clear = tx_clear | force_instant

        # Scatter back into global buffers.
        self.skill_id[env_ids] = a_out
        self.pending.active[env_ids] = pa_out
        self.pending.skill_id[env_ids] = ps_out
        if tx_clear.any():
            # Instant flips (gate-off mode or forced because the active
            # traj has no gates): clear any leftover cross-fade state for
            # envs that just had their active skill swapped under them.
            clear_ids = env_ids[tx_clear]
            self.transition.clear(clear_ids)
            # Force the manager to re-resolve trajectories with the new
            # skill before the next selection.
            self.manager.invalidate_cache()

    def reset_for_episode(self, env_ids: torch.Tensor) -> None:
        """Snap the active skill_id to whichever bucket contains the
        velocity cmd's freshly-set ``vel_target_b``, and clear any stale
        pending / cross-fade state.

        Called from :func:`mdp.events.resets.reset_on_reference` AFTER
        the velocity cmd's own ``reset_for_episode`` has run, so
        ``vel_target_b`` reflects the new episode's freshly-sampled
        velocity (snapped past the max-acc clamp on the reset path).

        Uses ``env.scene.env_origins`` for the cell lookup because
        ``ref_poses`` on this term is still stale from the previous
        episode at reset-event time.
        """
        env_ids_t = (
            env_ids if isinstance(env_ids, torch.Tensor)
            else torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        )
        if env_ids_t.numel() == 0:
            return
        self._resolve_vel_cmd()
        vel = self._vel_cmd.vel_target_b[env_ids_t]
        xy_spawn = self.env.scene.env_origins[env_ids_t, :2]
        new_active = self._skill_for_velocity(vel, env_ids_t, xy_w=xy_spawn)
        self.skill_id[env_ids_t] = new_active
        self.pending.clear(env_ids_t)
        self.transition.clear(env_ids_t)

    def compute_blended_outputs_at(
        self,
        phase: torch.Tensor,
        env_ids: torch.Tensor,
        y_pos_new: torch.Tensor | None = None,
        y_vel_new: torch.Tensor | None = None,
        alpha_override: torch.Tensor | None = None,
        old_phase_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Side-effect-free cross-fade-aware trajectory output evaluator.

        Returns the blended ``(y_pos, y_vel)`` the policy would see at
        the given ``phase`` for each row, accounting for any active
        cross-fade transition.  Used by both:

        - :meth:`_transform_desired_outputs` on the per-step update path
          (passing in the already-evaluated new-trajectory outputs and
          advancing the transition clock after the call).
        - :meth:`_debug_vis_callback` and any other debug consumer that
          wants the blended ``y_des`` at multiple phases per env
          (passing ``None`` for ``y_pos_new`` / ``y_vel_new`` lets the
          helper evaluate the new trajectory itself).

        Per-env state read: ``self.transition.active``,
        ``self.transition.old_traj_idx``, ``self.transition.phi_elapsed``
        via :meth:`CrossfadeState.alpha`.  None of these are mutated —
        the caller is responsible for calling :meth:`CrossfadeState.step`
        if the blend clock should advance.

        Args:
            phase: ``(K,)`` query phase per row.
            env_ids: ``(K,)`` long env indices per row.  Repeats are
                allowed when a single env is queried at multiple phases
                (the viz uses ``K = num_envs * S``).
            y_pos_new: Optional ``(K, P)`` pre-evaluated new-trajectory
                positions; saves a Bezier eval when the caller already
                has them.  ``None`` triggers an internal evaluation.
            y_vel_new: Optional ``(K, V)``; same semantics for velocities.
            alpha_override: Optional ``(K,)`` per-row blend weight in
                ``[0, 1]``.  ``None`` (default) uses each env's current
                ``transition.alpha`` — what production applies to its
                single ``y_des`` point at the current phase.  Pass an
                explicit per-row alpha when the caller wants a
                temporally-aligned view of ``y_des`` along a phi sweep
                (e.g. the debug viz, where alpha must ramp along the
                sweep to render the actual path the policy will trace
                through the upcoming domain).  Values outside ``[0, 1]``
                are clamped.
            old_phase_override: Optional ``(K,)`` per-row phase to
                evaluate the OLD trajectory at.  ``None`` (default)
                picks per-env: if the env's NEW trajectory has any
                contact gates (periodic / half-periodic / episodic),
                use the per-row query ``phase`` so old and new share
                phi (their sagittal-reflection boundaries stay
                aligned, avoiding sign-flip spikes in the blended
                output); otherwise (perpetual new, manager.phase
                locked at 0) fall back to ``self.transition.old_phase``
                — the independent old-trajectory clock — so the blend
                can still progress while old keeps animating.  Pass an
                explicit value when the caller wants direct control
                regardless of trajectory types (e.g. the debug viz
                uses per-sample ``phi_grid`` to keep the rendered
                cylinder stable within a domain).

        Returns:
            ``(y_pos_blended, y_vel_blended)`` of shape ``(K, P)`` /
            ``(K, V)``.  Non-transitioning rows pass through unchanged
            (returning ``y_pos_new`` / ``y_vel_new`` for those rows).
        """
        mgr = self.manager

        # Resolve any missing new-trajectory evaluations.
        if y_pos_new is None or y_vel_new is None:
            traj_idx_full = mgr.get_current_trajectory_indices()
            traj_idx_row = traj_idx_full[env_ids]
            new_dom = mgr._get_domain_indices(phase, traj_idx_row)
            new_T_dom = mgr.data["domain_times"][traj_idx_row, new_dom]
            new_tau = mgr._compute_normalized_tau(phase, traj_idx_row, new_dom)
            if y_pos_new is None:
                new_coeffs_pos = mgr.data["coeffs_pos"][traj_idx_row, new_dom]
                y_pos_new = mgr._compute_bezier_batched(
                    new_tau, new_coeffs_pos, new_T_dom, derivative=False
                )
            if y_vel_new is None:
                new_coeffs_vel = mgr.data["coeffs_vel"][traj_idx_row, new_dom]
                y_vel_new = mgr._compute_bezier_batched(
                    new_tau, new_coeffs_vel, new_T_dom, derivative=False
                )

        if (
            self.cfg.transition_blend_end_phi <= 0.0
            or not self.transition.active.any()
        ):
            return y_pos_new, y_vel_new

        tx_mask = self.transition.active[env_ids]
        if not tx_mask.any():
            return y_pos_new, y_vel_new

        # Evaluate the OLD trajectory.  Three possible phase sources,
        # picked per row:
        #
        # 1. Explicit caller override (``old_phase_override``).  The viz
        #    uses this to pass per-sample ``phi_grid`` so the rendered
        #    cylinder stays stable within a domain.
        # 2. The independent old-trajectory clock
        #    (``self.transition.old_phase``) — required when the NEW
        #    trajectory is perpetual (its ``manager.phase`` locked at 0
        #    means using the per-row query phase would freeze the old
        #    eval too and the blend couldn't progress).
        # 3. The per-row query ``phase`` (= ``manager.phase`` for
        #    production) — the simpler default for periodic↔periodic /
        #    half↔half / half↔periodic / episodic-involved transitions,
        #    where both trajectories share the same ``[0, 1]`` phi
        #    semantic.  Critically, syncing old to new's phase keeps
        #    any sagittal-reflection boundary (e.g. half-periodic
        #    phi=0.5) aligned between the two evaluations, so output
        #    columns don't sign-flip out of sync and the blend stays a
        #    well-defined convex combination in a single local frame.
        #
        # Picking sync vs decouple is per-env, based on whether the
        # current new trajectory has any contact gates (≡ is
        # periodic/half/episodic).  Zero gates ⇒ perpetual ⇒ decouple.
        tx_global = env_ids[tx_mask]
        old_traj = self.transition.old_traj_idx[tx_global]
        if old_phase_override is not None:
            old_phase_m = old_phase_override[tx_mask]
        else:
            new_traj_tx = mgr.get_current_trajectory_indices()[tx_global]
            new_has_gates = mgr._num_gates_per_traj[new_traj_tx] > 0
            old_phase_m = torch.where(
                new_has_gates,
                phase[tx_mask],
                self.transition.old_phase[tx_global],
            )
        old_dom = mgr._get_domain_indices(old_phase_m, old_traj)
        old_coeffs_pos = mgr.data["coeffs_pos"][old_traj, old_dom]
        old_coeffs_vel = mgr.data["coeffs_vel"][old_traj, old_dom]
        old_T_dom = mgr.data["domain_times"][old_traj, old_dom]
        old_tau = mgr._compute_normalized_tau(old_phase_m, old_traj, old_dom)
        y_pos_old = mgr._compute_bezier_batched(
            old_tau, old_coeffs_pos, old_T_dom, derivative=False
        )
        y_vel_old = mgr._compute_bezier_batched(
            old_tau, old_coeffs_vel, old_T_dom, derivative=False
        )

        # Per-row alpha.  ``None`` (production) → broadcast the env's
        # current transition alpha to every row sharing that env, which
        # matches what production applies to its single ``y_des`` point.
        # Override (viz) → use the caller-supplied per-row alpha so the
        # rendered curve can show the actual temporal path of ``y_des``
        # through the upcoming domain.
        if alpha_override is None:
            alpha = self.transition.alpha(tx_global, self.cfg.transition_blend_end_phi)
        else:
            alpha = alpha_override[tx_mask].clamp(0.0, 1.0)

        y_pos_new_tx = y_pos_new[tx_mask]
        y_vel_new_tx = y_vel_new[tx_mask]
        y_pos_blended = blend_outputs(
            y_pos_old, y_pos_new_tx, alpha, self._transition_quat_index_groups
        )
        # Velocities are 3-vectors (no quaternion columns), linear blend everywhere.
        y_vel_blended = blend_outputs(y_vel_old, y_vel_new_tx, alpha, None)

        y_pos_out = y_pos_new.clone()
        y_vel_out = y_vel_new.clone()
        y_pos_out[tx_mask] = y_pos_blended
        y_vel_out[tx_mask] = y_vel_blended
        return y_pos_out, y_vel_out

    def _transform_desired_outputs(
        self,
        phase: torch.Tensor,
        y_pos: torch.Tensor,
        y_vel: torch.Tensor,
        env_ids: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cross-fade the cached new-trajectory outputs with the old skill's
        trajectory outputs for envs in an active transition.

        Thin wrapper around :meth:`compute_blended_outputs_at` that
        additionally advances ``transition.phi_elapsed`` so the blend
        clock progresses each step.  Debug / viz consumers should call
        ``compute_blended_outputs_at`` directly to avoid the
        side-effect.
        """
        if self.cfg.transition_blend_end_phi <= 0.0 or not self.transition.active.any():
            return y_pos, y_vel

        if env_ids is None:
            env_ids_t = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids_t = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(
                env_ids, dtype=torch.long, device=self.device
            )
        tx_mask = self.transition.active[env_ids_t]
        if not tx_mask.any():
            return y_pos, y_vel

        # Blend via the shared helper.
        y_pos, y_vel = self.compute_blended_outputs_at(
            phase, env_ids_t, y_pos_new=y_pos, y_vel_new=y_vel,
        )

        # Advance per-env transition phi.  Use whichever of (new, old)
        # is actually progressing this step:
        #   * For periodic→periodic and most natural transitions the new
        #     trajectory's wrap-aware ``last_phase_delta`` is positive
        #     and dominates, preserving the original "blend over
        #     blend_end_phi of new-trajectory phase" semantic.
        #   * For periodic→perpetual the new trajectory's phase is
        #     locked at 0 — ``last_phase_delta`` is zero (or negative
        #     on the single snap step).  In that case fall back to the
        #     old trajectory's per-step advancement so the blend can
        #     still complete (in roughly one cycle of the OLD skill).
        # ``torch.where(new_delta > 0, ...)`` picks new when it's
        # advancing normally and the old's delta only when it isn't.
        tx_global = env_ids_t[tx_mask]
        new_delta = self.manager.last_phase_delta[tx_global]
        old_delta = self.transition.last_old_phase_delta[tx_global]
        phi_delta = torch.where(new_delta > 0, new_delta, old_delta.abs())
        self.transition.step(tx_global, phi_delta, self.cfg.transition_blend_end_phi)

        return y_pos, y_vel

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

        # Clear any leftover cross-fade state on freshly reset envs.  The
        # reset path doesn't go through CommandTerm.reset for this term, so
        # without this clear an env could re-enter training with stale
        # transition state.
        just_reset = ep_len == 0
        if just_reset.any():
            self.transition.clear(torch.where(just_reset)[0])

        in_window = (ep_len > 0) & (ep_len < self.env.max_episode_length)
        if self._last_compute_step is None:
            advance_mask = in_window
        else:
            advance_mask = in_window & (ep_len != self._last_compute_step)

        if not advance_mask.any():
            # Nothing to advance — skip the rest of the step.
            self._last_compute_step = ep_len.clone()
            return

        adv_ids = torch.where(advance_mask)[0]
        self.manager.update_phase(self.env.step_dt, env_ids=adv_ids)

        # Advance the cross-fade's old-trajectory clock independently.
        # The new (current) trajectory's clock was just advanced above
        # by ``manager.update_phase``; for a transition whose new
        # trajectory is perpetual (manager.phase forced to 0), the
        # alpha can't progress off ``last_phase_delta``.  Stepping the
        # old trajectory's phase here gives ``_transform_desired_outputs``
        # a non-zero ``last_old_phase_delta`` to fall back on so the
        # blend completes within roughly one cycle of the OLD skill.
        tx_active_in_adv = self.transition.active[adv_ids]
        if tx_active_in_adv.any():
            tx_in_adv = adv_ids[tx_active_in_adv]
            old_traj = self.transition.old_traj_idx[tx_in_adv]
            new_old_phase, old_delta = self.manager.advance_phase_for_traj(
                self.transition.old_phase[tx_in_adv], old_traj, self.env.step_dt,
            )
            self.transition.old_phase[tx_in_adv] = new_old_phase
            self.transition.last_old_phase_delta[:] = 0.0
            self.transition.last_old_phase_delta[tx_in_adv] = old_delta
        else:
            self.transition.last_old_phase_delta[:] = 0.0

        # Derive the active skill from where the velocity cmd has ramped
        # ``vel_target_b`` to.  Bucket transitions queue / clear pending;
        # actual ``skill_id`` flips happen later in the gate-fire path
        # (deferred) or directly inline (instant-flip mode).  Done BEFORE
        # the cache rebuild so the manager's trajectory selection sees
        # the freshest ``skill_id``.
        self._update_skill_pending_from_velocity(adv_ids)

        # Resolve current trajectory assignment after any phase mutations.
        # Populates ``manager._traj_changed`` against the previous-tick
        # snapshot.  Crucially this no longer commits the snapshot — see
        # ``MultiSkillManager.commit_traj_state``.
        self.manager.invalidate_cache()
        self.manager.get_current_trajectory_indices()

        if self._gating_enabled:
            # Unified reseed: arm the gate for any env that is either
            # disarmed (``next_gate_idx == -1``, includes the init case
            # where ``_traj_changed`` is False because the first cache
            # build initialises ``_prev_global_indices = new_indices``)
            # OR whose trajectory just changed (so the gate slots need to
            # point at the new trajectory's layout).  Restricted to envs
            # whose current trajectory actually has gates.
            traj_idx_full = self.manager.get_current_trajectory_indices()
            has_gates = self.manager._num_gates_per_traj[traj_idx_full] > 0
            needs_arm = self.manager.next_gate_idx == -1
            if self.manager._traj_changed is not None:
                needs_arm = needs_arm | self.manager._traj_changed
            needs_arm = needs_arm & has_gates
            if needs_arm.any():
                self.manager._reseed_gate_for_envs(torch.where(needs_arm)[0])

            contact_now = self._read_contact_now()
            self._apply_contact_gate(contact_now)

        # Update the post-traj-change grace timer.  Only advance the timer
        # for envs that actually stepped this tick (``advance_mask``);
        # idempotent-skip envs, fresh-reset envs, and done-this-step envs
        # didn't accumulate sim time so their timer must not move.  Then
        # reset to 0 for envs the manager just swapped a trajectory on in
        # the middle of an episode — covers both within-skill bucket swaps
        # and full skill changes since the commit step inside
        # ``_apply_contact_gate`` invalidates + rebuilds the cache, which
        # re-populates ``_traj_changed`` against the still-unchanged
        # ``_prev_global_indices``.  Fresh episode resets are excluded:
        # the reset path spawns the robot at the reference, so there is no
        # transition transient and the deviation termination should arm
        # immediately.  Placed AFTER the gate so we see post-commit
        # changes; placed BEFORE ``commit_traj_state`` so ``_traj_changed``
        # is still meaningful.  Read by
        # ``mdp.frame_deviation_from_reference`` to suppress termination
        # during the grace window after an in-episode transition.
        self.time_since_traj_change_s[advance_mask] = (
            self.time_since_traj_change_s[advance_mask] + self.env.step_dt
        )
        fresh_reset = ep_len == 0
        if fresh_reset.any():
            self.time_since_traj_change_s[fresh_reset] = 1.0e6
        traj_changed = self.manager._traj_changed
        if traj_changed is not None and traj_changed.any():
            in_episode_change = traj_changed & ~fresh_reset
            if in_episode_change.any():
                self.time_since_traj_change_s[in_episode_change] = 0.0

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
        # Gated by ``_DEBUG_VIZ_SHOW_END_SPHERES`` so prototypes and emission
        # stay consistent.
        if _DEBUG_VIZ_SHOW_END_SPHERES:
            for name, color in zip(self._debug_frame_names, self._debug_frame_colors):
                prototypes[f"_{name}_end_sphere"] = sim_utils.SphereCfg(
                    radius=1.0,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
                )
        # Ref-pose disk prototype.  Native radius/height are baked in here
        # (no per-instance scaling needed at ``visualize`` time).  Index is
        # captured for the per-step emission below.
        self._debug_ref_pose_marker_idx: int | None = None
        if _DEBUG_VIZ_SHOW_REF_POSE:
            self._debug_ref_pose_marker_idx = len(prototypes)
            prototypes["_ref_pose_disk"] = sim_utils.CylinderCfg(
                radius=_DEBUG_VIZ_REF_POSE_RADIUS,
                height=_DEBUG_VIZ_REF_POSE_HEIGHT,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=_DEBUG_VIZ_REF_POSE_COLOR
                ),
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

        N = phase.shape[0]
        S = _DEBUG_VIZ_NUM_SAMPLES
        F = len(self._debug_frame_pos_idx)
        if F == 0 or N == 0 or S < 2:
            return

        # Build an (N, S) grid of absolute phi values spanning each env's
        # current domain, then drive everything through
        # ``compute_blended_outputs_at`` — the SAME shared helper that
        # the production update path calls in ``_transform_desired_outputs``.
        # That way the viz cannot drift from the actual blend math: any
        # change to the production blend is automatically reflected here.
        boundaries = manager.data["domain_boundaries"][traj_idx]      # (N, D+1)
        totals = manager.data["total_time"][traj_idx]                 # (N,)
        t_start = boundaries.gather(1, domain_idx.unsqueeze(1)).squeeze(1)
        t_end = boundaries.gather(1, (domain_idx + 1).unsqueeze(1)).squeeze(1)
        phi_start = t_start / totals                                  # (N,)
        phi_end = t_end / totals                                      # (N,)
        # ``_get_domain_indices`` (``searchsorted(right=False)``) and
        # ``_compute_normalized_tau`` (``phi >= 0.5`` folding for
        # half-periodic) handle boundary phi values asymmetrically:
        # ``_get_domain_indices`` puts ``phi == boundary`` into the lower
        # domain, while the half-periodic fold puts ``phi == 0.5`` into
        # the second half.  At the exact reflection boundary (``phi=0.5``
        # is the end of one domain and the start of the next half), these
        # disagree and the resulting tau goes negative, producing a wild
        # Bezier extrapolation that visually coincides with another
        # sample (closing the rendered cylinder loop).  Same risk at
        # ``phi=phi_start`` (start of a domain landing in the previous).
        # Nudge both endpoints fractionally inward so all samples land in
        # the current domain unambiguously and on the expected side of
        # the half-periodic fold.
        tau = torch.linspace(0.0, 1.0, S, device=self.device).clone()  # (S,)
        tau[0] = 1.0e-4
        tau[-1] = 1.0 - 1.0e-4
        phi_grid = phi_start.unsqueeze(1) + tau.unsqueeze(0) * (
            phi_end - phi_start
        ).unsqueeze(1)                                                # (N, S)

        # Flatten to (N*S,) for the shared helper.  Each row carries one
        # (env, phase) query; the helper handles new/old traj eval and the
        # cross-fade blend identically to the production path.
        phi_flat = phi_grid.reshape(-1)                                # (N*S,)
        env_ids_flat = (
            torch.arange(N, device=self.device).unsqueeze(1).expand(N, S).reshape(-1)
        )

        # Per-sample alpha so the rendered curve traces the *actual*
        # path of y_des through the upcoming domain — at each sweep
        # sample, alpha = (phi_elapsed_now + (phi(sample) - phi_now)) /
        # blend_end_phi.  Sample at phi_now gets the env's current
        # alpha (matching the policy's live y_des); samples ahead in
        # phi get the alpha they'll have when the policy reaches them;
        # samples behind get the alpha they had earlier in this
        # transition.  Clamped to [0, 1] inside the helper.
        if (
            self.cfg.transition_blend_end_phi > 0.0
            and self.transition.active.any()
        ):
            phi_elapsed_now = self.transition.phi_elapsed             # (N,)
            blend_end = self.cfg.transition_blend_end_phi
            alpha_grid = (
                phi_elapsed_now.unsqueeze(1)
                + (phi_grid - phase.unsqueeze(1))
            ) / blend_end                                              # (N, S)
            alpha_flat = alpha_grid.reshape(-1)                        # (N*S,)
        else:
            alpha_flat = None

        # Override the helper's old-trajectory phase source with the
        # per-sample phi grid (same as new's per-sample phase).  Without
        # this, the helper falls back to ``self.transition.old_phase``
        # which is one value per env that advances each step — even
        # when the new-trajectory domain is fixed within a step.  That
        # made the rendered cylinder drift during a transition; using
        # the same per-sample phi for both old and new keeps the
        # cylinder stable across steps within a domain (and identical
        # to the pre-perpetual-fix viz behaviour the user expects).
        y_pos_flat, _ = self.compute_blended_outputs_at(
            phi_flat, env_ids_flat,
            alpha_override=alpha_flat,
            old_phase_override=phi_flat,
        )
        local_outs = y_pos_flat.view(N, S, -1)                         # (N, S, P)

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
        # Gated by ``_DEBUG_VIZ_SHOW_END_SPHERES``.
        if _DEBUG_VIZ_SHOW_END_SPHERES:
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

        # One flat disk per env at the reference-pose anchor.  Native radius
        # and height live on the prototype, so scale is identity.  Orient by
        # the ref-pose yaw so the disk's local +z stays world-up (the
        # prototype is a thin upright cylinder; yaw-only rotation keeps it
        # flat).
        if (
            _DEBUG_VIZ_SHOW_REF_POSE
            and getattr(self, "_debug_ref_pose_marker_idx", None) is not None
        ):
            ref_pose_translations = anchor                                # (N, 3)
            ref_pose_quats = yaw_q                                        # (N, 4) xyzw
            ref_pose_scales = torch.ones(N, 3, device=self.device)
            ref_pose_indices = torch.full(
                (N,),
                self._debug_ref_pose_marker_idx,
                device=self.device,
                dtype=torch.long,
            )
            translations = torch.cat([translations, ref_pose_translations], dim=0)
            quats = torch.cat([quats, ref_pose_quats], dim=0)
            scales_flat = torch.cat([scales_flat, ref_pose_scales], dim=0)
            marker_indices = torch.cat([marker_indices, ref_pose_indices], dim=0)

        self._debug_markers.visualize(
            translations=translations,
            orientations=quats,
            scales=scales_flat,
            marker_indices=marker_indices,
        )
