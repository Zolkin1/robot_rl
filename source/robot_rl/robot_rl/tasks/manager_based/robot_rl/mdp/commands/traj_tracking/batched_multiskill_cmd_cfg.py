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
