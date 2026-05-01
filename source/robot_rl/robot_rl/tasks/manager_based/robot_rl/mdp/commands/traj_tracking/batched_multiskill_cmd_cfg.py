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

    contact_sensor_names: list[str] = ("left_foot_contact", "right_foot_contact")
    """Names of the ContactSensor scene entities used by the contact gate.
    Their combined body list must cover every reference frame in the loaded
    trajectories' ``contact_bodies``.  Each contact body must appear in
    exactly one sensor.  Only used when ``contact_gate_window_frac`` is not
    ``None``."""

    contact_force_threshold: float = 1.0
    """Net contact-force magnitude (N) above which a contact body is
    considered in contact for gating purposes."""

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

    track_traj_stats: bool = True
    """If ``True``, the underlying :class:`MultiSkillManager` allocates a
    per-trajectory CLF stats tracker (:class:`TrajectoryCLFStats`) updated
    once per step from :meth:`log_v_on_phasing_var`. Required for adaptive
    trajectory sampling."""

    traj_stats_alpha: float = 0.005
    """EMA factor for the per-trajectory tracker (matches ``skill_v_logs``)."""

    traj_stats_reset_warmup: int = 2
    """Frames after each env reset that are excluded from the tracker."""

    traj_stats_transition_warmup: int = 3
    """Frames after each skill transition that are excluded from the tracker."""
