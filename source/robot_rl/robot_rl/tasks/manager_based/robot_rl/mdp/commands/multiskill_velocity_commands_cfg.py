from isaaclab.utils import configclass

from .velocity_commands_cfg import VelocityTrackingCommandCfg


@configclass
class VelocityBucketCfg:
    """A velocity bucket assigning a fraction of envs to a specific velocity range."""

    percentage: float = 0.0
    """Fraction of environments assigned to this bucket (0.0 to 1.0)."""

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
    """Velocity tracking command config with per-bucket velocity ranges for multiskill training.

    Each velocity bucket specifies a fraction of environments and the velocity range
    they should sample from. Any remaining fraction (1.0 - sum of bucket percentages)
    uses the default uniform range from ``ranges``.

    Standing behavior is achieved via a zero-velocity bucket rather than ``rel_standing_envs``.
    """

    class_type: type | str = "{DIR}.multiskill_velocity_commands:MultiskillVelocityTrackingCommand"

    velocity_buckets: list[VelocityBucketCfg] = []
    """List of velocity buckets. Percentages should sum to <= 1.0.
    Any remainder is assigned to the default uniform range from ``ranges``."""

    skill_transition_prob: float | None = None
    """If set, fraction of resamples that force a skill transition (new bucket != current bucket).
    The non-transitioning fraction (1 - skill_transition_prob) keeps the env's previous bucket and
    only resamples the velocity within that bucket. If None, buckets are sampled independently from
    the configured percentages each resample."""

    max_acc_frac: float | None = None
    """Fraction of envs that have the ``max_acc`` clamp applied. If None, no envs are clamped
    (commanded velocity snaps directly to the sampled target). If set, must be in [0, 1] and that
    fraction of envs use ``max_acc`` while the rest are unclamped. The per-env assignment persists
    for the duration of an episode and is re-rolled on reset."""

    rel_standing_envs: float = 0.0
    """Not used -- standing is a zero-velocity bucket. Kept for parent compatibility."""

    adaptive_sample_fraction: float = 0.0
    """Fraction of envs (per resample) whose commanded velocity is overridden
    to oversample trajectories the policy currently struggles with.

    For the chosen subset, a global trajectory is sampled with probability
    proportional to ``(mean_v + eps) ** adaptive_sample_beta`` (read from
    ``MultiSkillManager.traj_stats``), and the env's
    ``(lin_vel_x, lin_vel_y, ang_vel_z)`` are set to that trajectory's
    conditioning values. The deterministic argmin in
    ``MultiSkillManager._select_trajectories`` then assigns that trajectory.

    Set to ``0.0`` (default) to disable adaptive sampling — the resample
    behaves exactly as before. Must be in ``[0, 1]``.
    """

    adaptive_sample_beta: float = 1.0
    """Temperature for the adaptive sampler: weights are
    ``(mean_v + eps) ** beta``. ``beta=0`` collapses to uniform sampling
    over flat-terrain trajectories (useful for ablation); ``beta=1`` is
    proportional to mean V; larger values sharpen further."""

    adaptive_sample_eps: float = 1e-3
    """Small constant added to ``mean_v`` before exponentiation, so cold
    start (all zero) gives a uniform distribution rather than NaN."""

    multiskill_term_name: str = "traj_ref"
    """Name of the multiskill trajectory command term in the command manager.
    Used to look up the :class:`MultiSkillManager` for adaptive sampling."""
