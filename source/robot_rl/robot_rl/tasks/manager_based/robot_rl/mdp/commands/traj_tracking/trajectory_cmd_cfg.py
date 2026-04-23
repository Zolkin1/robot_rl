"""Configuration for :class:`TrajectoryCommand`."""

from isaaclab.utils import configclass

from .phased_trajectory_cmd_cfg import PhasedTrajectoryCommandCfg


@configclass
class TrajectoryCommandCfg(PhasedTrajectoryCommandCfg):
    """Configuration for the single-skill trajectory command.

    ``manager_type`` selects the manager to instantiate: ``"trajectory"``
    loads a single trajectory YAML via :class:`TrajectoryManager`;
    ``"library"`` loads a folder of YAML files via :class:`LibraryManager`.
    """

    class_type: type | str = (
        "robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking"
        ".trajectory_cmd:TrajectoryCommand"
    )

    manager_type: str = ""
    heuristic_func = None
