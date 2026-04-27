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

    smooth_transitions: bool = True
    """Align the trajectory clock on skill changes so the new skill starts at
    a phi that matches the current reference frame in contact and preserves
    the fractional period position.  When ``False``, the per-env clock keeps
    advancing linearly and the new trajectory is evaluated at that raw time
    (the old pre-smoothing behaviour)."""

    contact_gate_window_frac: float | None = 0.2
    """Fraction of the gated period (half-period for half-periodic, full
    period for full-periodic / episodic) over which the next expected foot
    contact will trigger advancement to the next period.  Inside the window
    a contact event snaps the clock to the period boundary; past the
    boundary without contact, the clock holds at the boundary until contact
    lands.  Set to ``None`` to disable contact gating (clock advances purely
    on time, original behaviour)."""

    contact_sensor_names: list[str] = ("left_foot_contact", "right_foot_contact")
    """Names of the ContactSensor scene entities used by the contact gate.
    Their combined body list must cover every reference frame in the loaded
    trajectories' ``contact_bodies``.  Each contact body must appear in
    exactly one sensor.  Only used when ``contact_gate_window_frac`` is not
    ``None``."""

    contact_force_threshold: float = 1.0
    """Net contact-force magnitude (N) above which a contact body is
    considered in contact for gating purposes."""
