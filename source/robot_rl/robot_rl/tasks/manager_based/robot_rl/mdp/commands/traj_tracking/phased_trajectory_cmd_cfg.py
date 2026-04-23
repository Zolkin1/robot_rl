"""Configuration for :class:`PhasedTrajectoryCommand`."""

from isaaclab.utils import configclass

from .base_trajectory_cmd_cfg import BaseTrajectoryCommandCfg


@configclass
class PhasedTrajectoryCommandCfg(BaseTrajectoryCommandCfg):
    """Configuration shared by phased (single-skill) trajectory commands.

    Adds the hold-phi fields on top of
    :class:`BaseTrajectoryCommandCfg`.  ``percent_hold_phi`` controls the
    per-episode random hold of the trajectory at t=0.
    ``hold_phi_threshold`` is the commanded base-velocity magnitude below
    which the phase boundary-hold engages.  ``phasing_boundaries`` sets
    which boundary crossing (1st, 2nd, ...) locks the phase.
    """

    percent_hold_phi: float = -1
    hold_phi_threshold: float = -1
    phasing_boundaries: float = 1
