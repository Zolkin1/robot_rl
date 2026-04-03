from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass


@configclass
class TrajectoryCommandCfg(CommandTermCfg):
    """Configuration for trajectory commands."""

    class_type: type | str = (
        "robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_cmd:TrajectoryCommand"
    )
    asset_name: str = "robot"
    contact_bodies: list[str] = None
    manager_type: str = ""
    conditioner_generator_name: str = ""
    num_outputs: int = -1
    path: str = ""
    hf_repo: str = None
    Q_weights: list[float] = None
    R_weights: list[float] = None
    resampling_time_range: tuple[float, float] = (5.0, 15.0)    # TODO: How can I remove this?
    heuristic_func = None
    random_start_time_max: float = -1
    percent_hold_phi: float = -1
    hold_phi_threshold: float = -1
    phasing_boundaries: float = 1