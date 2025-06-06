import math
import torch

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm

from .g1_rough_env_lip_cfg import G1RoughLipEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip

##
# Environment configuration
##
@configclass
class G1FlatLipEnvCfg(G1RoughLipEnvCfg):
    """Configuration for the G1 Flat environment."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        ##
        # Commands
        ##
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (0, 0)   # For now, making the heading match the velocity

        ##
        # Rewards
        ##
        # -- Task

        # self.rewards.lin_vel_z_l2.weight = 0.0
        # self.rewards.track_lin_vel_xy_exp.weight = 2.5
        # self.rewards.track_ang_vel_z_exp.weight = 0.75


class G1FlatRefTrackingEnvCfg(G1FlatLipEnvCfg):
    """Configuration for the G1 Flat environment."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.rewards.clf_reward = None
        self.rewards.clf_decreasing_condition = None


class G1FlatLipEnvCfg_PLAY(G1FlatLipEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        # self.events.push_robot = None
        self.events.push_robot.interval_range_s = (5.0,5.0)
        self.events.reset_base.params["pose_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0,0)} #(-3.14, 3.14)},
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.3, 0.3)