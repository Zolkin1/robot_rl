"""Configuration for the batched multi-skill trajectory command."""

from isaaclab.utils import configclass

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
