"""Configuration for the single-skill library trajectory command."""

from isaaclab.utils import configclass

from .base_trajectory_cmd_cfg import BaseTrajectoryCommandCfg


@configclass
class LibraryCommandCfg(BaseTrajectoryCommandCfg):
    """Configuration for :class:`LibraryCommand`.

    ``path`` should point to a folder of trajectory YAML files (a single skill).
    """

    class_type: type | str = (
        "robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking"
        ".library_cmd:LibraryCommand"
    )
