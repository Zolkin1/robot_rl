"""Configuration for the batched multi-skill trajectory command."""

from dataclasses import field

from isaaclab.utils import configclass

from ..multiskill_velocity_commands_cfg import VelocityBucketCfg
from .base_trajectory_cmd_cfg import BaseTrajectoryCommandCfg


@configclass
class BatchedMultiSkillCommandCfg(BaseTrajectoryCommandCfg):
    """Configuration for :class:`BatchedMultiSkillCommand`.

    ``path`` should point to a top-level folder whose subdirectories are
    skills::

        path/
        ├── walking/
        │   ├── walk_20.yaml
        │   └── walk_40.yaml
        └── running/
            ├── run_160.yaml
            └── run_180.yaml

    Each subfolder name becomes the skill name.  If ``path`` itself
    contains YAML files (no subfolders), it is treated as a single
    ``"default"`` skill.
    """

    class_type: type | str = (
        "robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking"
        ".batched_multiskill_cmd:BatchedMultiSkillCommand"
    )

    contact_gate_window_frac: float | None = 0.2
    """Fraction of the local domain (phi distance from the previous gate to
    this gate) used as the early-fire window.  With ``W=0.1`` a half-period
    gate gets a 0.05 phi early window and a full-period / episodic gate
    gets 0.10.  Inside the early window an expected contact snaps the
    phase forward into the new domain.  The late side is unbounded — any
    expected contact past the gate snaps the phase backward to the start
    of the new domain, no matter how late.  The gate stays armed until
    that contact lands; the manager's monotonic ``gate_rel_phi`` handles
    cycle wraps so phi=1.0 gates work without special-casing.  Set
    ``hold_on_late_contact=True`` for the hold-at-boundary variant.  Set
    to ``None`` to disable contact gating entirely."""

    contact_sensor_names: tuple[str, ...] = ("left_foot_contact", "right_foot_contact")
    """Names of the ContactSensor scene entities used by the contact gate.
    Their combined body list must cover every reference frame in the loaded
    trajectories' ``contact_bodies``.  Each contact body must appear in
    exactly one sensor.  Only used when ``contact_gate_window_frac`` is not
    ``None``."""

    contact_force_threshold: float = 1.0
    """Net contact-force magnitude (N) above which a contact body is
    considered in contact for gating purposes."""

    transition_blend_end_phi: float = 1.0
    """Phi window over which the post-skill-change cross-fade ramps from the
    old skill's trajectory to the new skill's trajectory.  At gate-fire
    time the per-env ``alpha_blend = 0`` (pure old); after
    ``transition_blend_end_phi`` worth of (new-trajectory) phase has
    elapsed, ``alpha_blend = 1`` and the env tracks pure new output.
    Set to ``0`` to disable cross-fading entirely (gate-only behaviour:
    skill commit at the gate, but no blend afterwards)."""

    hold_on_late_contact: bool = False
    """If ``True``, when phase has crossed the gate boundary without an
    expected contact each step pulls the phase back to
    ``gate_phi - eps_phi`` — end of the previous (old) domain — until
    contact arrives, at which point the phase snaps forward into the new
    domain.  If ``False`` (default), phase advances naturally past the
    gate; an expected contact at any point past the gate produces a
    backward snap to the start of the new domain (no domain change).
    The gate stays armed indefinitely until contact lands.  Only used
    when ``contact_gate_window_frac`` is not ``None``."""

    velocity_buckets: dict[str, VelocityBucketCfg] = field(default_factory=dict)
    """Per-skill velocity ranges.  Keys must equal ``terrain.skill_list``
    exactly.  The trajectory cmd uses these tables to derive its active
    skill each step from the live ``vel_target_b`` (bucket lookup), and
    the velocity cmd reads them via public properties to sample within
    a bucket on resample.  Source of truth lives here because the
    skill in use by the policy is determined by which bucket the ramped
    velocity is in — not by what the velocity cmd most recently sampled."""

    gate_skill_change_on_contact: bool = True
    """If True (default), bucket-crossings detected on the live
    ``vel_target_b`` are buffered in a pending slot and only committed
    to ``skill_id`` when the contact gate fires.  Set to False to flip
    the active skill instantly when the bucket changes (legacy
    behaviour; no cross-fade window).  Requires
    ``contact_gate_window_frac`` to be non-None when True (otherwise
    pending changes would never drain)."""

    velocity_command_name: str = "base_velocity"
    """Name of the velocity command term whose ``vel_target_b`` this
    trajectory cmd reads each step to derive its active skill via the
    bucket lookup."""

    terrain_name: str = "terrain"
    """Scene entity name of the (meta) terrain importer.  Used to read
    the per-cell ``skill_probs`` mask that constrains which buckets are
    eligible candidates when ``vel_target_b`` would otherwise match
    multiple buckets (e.g. ``stair_up (0.4, 0.4)`` overlapping
    ``walk_forward (0.1, 1.5)``)."""

    debug_skill_marker_height: float = 1.5
    """Z offset (m) above each env's pelvis at which the active-skill
    debug marker is drawn when ``debug_vis=True``."""

    debug_skill_marker_radius: float = 0.08
    """Radius (m) of the active-skill debug sphere marker."""

    skill_query_buffer: float = 0.0
    """Extra distance (m) added to the predicted swing-foot landing xy
    before it's used as the terrain-skill cell-lookup anchor in
    :meth:`BatchedMultiSkillCommand._ref_xy_at_next_gate`.  Applied purely
    in the ref-frame *forward* direction (local +x, yaw-aligned to the
    stance foot's heading ≈ direction of travel), so the value maps
    directly to the toe-ahead-of-ankle distance: a 0.15 m toe overhang is
    covered by ``skill_query_buffer = 0.15`` (plus any margin).  This is
    deliberately NOT applied along the full swing displacement vector —
    that vector carries a large lateral component (≈ half the stance
    width), which for slower gaits would shrink the buffer's forward
    reach well below its nominal value.  Useful when the ankle reference
    lands just short of a terrain feature (e.g. a stair riser) that the
    toe would actually touch — the buffered query then triggers the
    terrain-aware skill switch in time.  Set to ``0.0`` to disable (query
    at the bare ankle prediction).

    The look-ahead is *asymmetric*: it only applies while the ankle
    prediction is still on a flat (non-terrain) block, so it promotes onto
    a stair when approaching but never pulls the skill off a stair early
    when exiting (where the toe clears onto the flat top a step before the
    ankle leaves the last tread)."""
