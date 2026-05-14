from isaaclab.utils import configclass

from .velocity_commands_cfg import VelocityTrackingCommandCfg


@configclass
class VelocityBucketCfg:
    """Velocity ranges for one named skill.

    Per-env skill is sampled from the terrain importer's per-cell
    ``skill_probs``; velocity is then drawn uniformly from this bucket's
    ranges.
    """

    lin_vel_x: tuple[float, float] = (0.0, 0.0)
    """Range for the linear-x velocity command (m/s)."""

    lin_vel_y: tuple[float, float] = (0.0, 0.0)
    """Range for the linear-y velocity command (m/s)."""

    ang_vel_z: tuple[float, float] = (0.0, 0.0)
    """Range for the angular-z velocity command (rad/s)."""

    heading: tuple[float, float] | None = None
    """Range for the heading command (rad). If None, uses the parent config's heading range."""


@configclass
class MultiskillVelocityTrackingCommandCfg(VelocityTrackingCommandCfg):
    """Velocity tracking command config with per-skill velocity ranges.

    The per-env skill assignment is drawn at resample time from the terrain
    importer's ``skill_probs[:, r, c]`` for each env's current cell ``(r, c)``.
    The chosen skill indexes into ``velocity_buckets`` to give a uniform
    velocity range to sample from.

    The terrain importer must expose ``skill_probs``, ``skill_list``, and
    ``world_xy_to_cell`` (i.e. be a
    :class:`MetaTerrainImporter` subclass). The keys of
    ``velocity_buckets`` must equal the importer's ``skill_list`` exactly.
    """

    class_type: type | str = "{DIR}.multiskill_velocity_commands:MultiskillVelocityTrackingCommand"

    terrain_name: str = "terrain"
    """Scene field name under which the meta terrain importer is registered."""

    trajectory_command_name: str = "traj_ref"
    """Name of the trajectory command term whose ``ref_poses[:, :2]`` is used
    as the world-frame ``(x, y)`` for the terrain cell lookup on mid-episode
    resamples.  Reference frames are stable across the per-step jitter of
    the robot's pelvis and reflect where the policy is conceptually
    anchored, so using them avoids cell-misclassification when the pelvis
    drifts off-cell mid-stride.  The reset path still uses
    ``env.scene.env_origins`` since ``ref_poses`` is stale at reset-event
    time.  The velocity command also reads the trajectory cmd's
    ``velocity_buckets`` cfg from this term at construction time —
    the bucket ranges live on the trajectory cmd because they're the
    source-of-truth for which skill the policy is actually executing."""

    skill_transition_prob: float | None = None
    """If set, the fraction of resamples that force a skill transition (new
    skill ≠ previous skill). The non-transitioning fraction (1 −
    skill_transition_prob) keeps the env's previous skill and only resamples
    the velocity within that skill's bucket. If None, skills are sampled
    independently from the per-env probability vector each resample."""

    max_acc_frac: float | None = None
    """Fraction of envs that have the ``max_acc`` clamp applied. If None, no
    envs are clamped (commanded velocity snaps directly to the sampled
    target). If set, must be in [0, 1] and that fraction of envs use
    ``max_acc`` while the rest are unclamped. The per-env assignment persists
    for the duration of an episode and is re-rolled on reset."""

    rel_standing_envs: float = 0.0
    """Not used — standing is achieved via a zero-velocity bucket whose skill
    is selected by the terrain. Kept for parent compatibility."""
