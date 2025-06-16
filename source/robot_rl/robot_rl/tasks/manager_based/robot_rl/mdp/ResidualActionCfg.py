from isaaclab.managers.action_manager import ActionTermCfg, ActionTerm
from isaaclab.utils import configclass
from robot_rl.tasks.manager_based.robot_rl.mdp import residual_actions
from dataclasses import MISSING
@configclass
class ResidualActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = residual_actions.ResidualAction
    scale: float = 0.1
    preserve_order: bool = False
    joint_names: list[str] = MISSING
    