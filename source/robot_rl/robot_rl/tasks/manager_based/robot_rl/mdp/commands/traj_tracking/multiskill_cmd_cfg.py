"""Configuration for the MultiSkill trajectory command."""

from isaaclab.utils import configclass

from .trajectory_cmd_cfg import TrajectoryCommandCfg


@configclass
class MultiSkillCommandCfg(TrajectoryCommandCfg):
    """Configuration for multi-skill trajectory commands.

    Extends :class:`TrajectoryCommandCfg`.  The inherited ``path`` field
    points to a top-level folder whose subdirectories are skills::

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

    Set ``hf_repo`` to download from HuggingFace automatically.
    """

    class_type: type | str = (
        "robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking"
        ".multiskill_cmd:MultiSkillCommand"
    )
