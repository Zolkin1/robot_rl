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

from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import (HumanoidEnvCfg, HumanoidEventsCfg,
                                                                    HumanoidRewardCfg, PERIOD)
from robot_rl.tasks.manager_based.robot_rl import mdp

##
# Pre-defined configs
##
from robot_rl.assets.robots.g1_21j import G1_MINIMAL_CFG  # isort: skip


##
# LIP Specific Constants
##
WDES = 0.6 #0.2 #0.25

##
# Lip specific rewards
##
class G1RoughLipRewards(HumanoidRewardCfg):
    """Rewards specific to LIP Model"""
    lip_gait_tracking = RewTerm(
        func=mdp.lip_gait_tracking,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "period": PERIOD,
            "std": 0.2,
            "nom_height": 0.78,
            "Tswing": PERIOD/2.,
            "command_name": "base_velocity",
            "wdes": WDES,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        }
    )

    lip_feet_tracking = RewTerm(
        func=mdp.lip_feet_tracking,
        weight=10.0,
        params={
            "period": PERIOD,
            "std": 0.2,
            "Tswing": PERIOD/2.,
            "feet_bodies": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        }
    )

class G1RoughLipEventsCfg(HumanoidEventsCfg):
    # Calculate new step location on a fixed interval
    update_step_location = EventTerm(func=mdp.compute_step_location_local,
                                     mode="interval",
                                     interval_range_s=(PERIOD / 2., PERIOD / 2.),
                                     is_global_time=False,
                                     params={
                                         "nom_height": 0.78,
                                         "Tswing": PERIOD / 2.,
                                         "command_name": "base_velocity",
                                         "wdes": WDES,
                                         "feet_bodies": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
                                     })
    # Do on reset
    reset_update_set_location = EventTerm(func=mdp.compute_step_location_local,
                                          mode="reset",
                                          params={
                                              "nom_height": 0.78,
                                              "Tswing": PERIOD / 2.,
                                              "command_name": "base_velocity",
                                              "wdes": WDES,
                                              "feet_bodies": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
                                          })
##
# Environment configuration
##
@configclass
class G1RoughLipEnvCfg(HumanoidEnvCfg):
    """Configuration for the G1 Flat environment."""
    rewards: G1RoughLipRewards = G1RoughLipRewards()
    events: G1RoughLipEventsCfg = G1RoughLipEventsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis_link"

        # No height scanner for now
        self.scene.height_scanner = None


        ##
        # Randomization
        ##
        # self.events.push_robot = None
        self.events.push_robot.params["velocity_range"] = {"x": (-1, 1), "y": (-1, 1), "roll": (-0.4, 0.4),
                                                           "pitch": (-0.4, 0.4), "yaw": (-0.4, 0.4)}
        # self.events.push_robot.params["velocity_range"] = {"x": (-0, 0), "y": (-0, 0), "roll": (-0.0, 0.0),
        #                                                    "pitch": (-0., 0.), "yaw": (-0.0, 0.0)}
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.add_base_mass.params["operation"] = "scale"
        # self.events.randomize_ground_contact_friction.params["static_friction_range"] = (0.1, 1.25)
        # self.events.randomize_ground_contact_friction.params["dynamic_friction_range"] = (0.1, 1.25)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14,3.14)}, #(-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        ##
        # Commands
        ##
        self.commands.base_velocity.ranges.lin_vel_x = (-1, 1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        ##
        # Terminations
        ##
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["pelvis_link"]

        ##
        # Rewards
        ##
        self.rewards.track_lin_vel_xy_exp.weight = 5.0 #1
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight =  -2.0 # TODO reduce this maybe?
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_vel_l2.weight = -1.0e-3
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_pos_limits.weight = -5.0
        self.rewards.alive.weight = 0.15
        self.rewards.contact_no_vel.weight = -0.2
        self.rewards.joint_deviation_hip.weight = -1.0
        self.rewards.height_torso.weight = -20 #-10.0
        self.rewards.feet_clearance.weight = -20.0
        self.rewards.phase_contact.weight = 0 #0.25

        # TODO: Add the footstep location rewards
        self.rewards.lip_gait_tracking.weight = 2
        self.rewards.lip_feet_tracking.weight = 3 #10.0

        self.rewards.joint_deviation_arms.weight = -0.5             # Arms regularization
        self.rewards.joint_deviation_torso.weight = -1.0

        self.rewards.height_torso.params["target_height"] = 0.75
        self.rewards.feet_clearance.params["target_height"] = 0.12