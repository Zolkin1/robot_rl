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

from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidEnvCfg

##
# Pre-defined configs
##
from robot_rl.assets.robots.g1_21j import G1_MINIMAL_CFG  # isort: skip

##
# Environment configuration
##
@configclass
class G1RoughEnvCfg(HumanoidEnvCfg):
    """Configuration for the G1 Flat environment."""
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
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.add_base_mass.params["operation"] = "scale"
        # self.events.randomize_ground_contact_friction.params["static_friction_range"] = (0.1, 1.25)
        # self.events.randomize_ground_contact_friction.params["dynamic_friction_range"] = (0.1, 1.25)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5) # 0 - 1
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4,0.4) #(-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        ##
        # Terminations
        ##
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["pelvis_link"]

        ##
        # Rewards
        ##
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
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
        self.rewards.height_torso.weight = -10.0
        self.rewards.feet_clearance.weight = -20.0
        self.rewards.phase_contact.weight = 0.25

        self.rewards.joint_deviation_arms.weight = -0.5             # Arms regularization
        self.rewards.joint_deviation_torso.weight = -1.0

        self.rewards.height_torso.params["target_height"] = 0.75
        self.rewards.feet_clearance.params["target_height"] = 0.12



        # # -- Regularization
        # self.rewards.dof_torques_l2.weight = -1e-4                  # Joint torques
        # # self.rewards.torque_lim.weight = -1e-2                      # Torque limits
        # self.rewards.joint_vel.weight = -1e-3                       # Joint velocity
        # self.rewards.dof_pos_limits.weight = -1 #-10                    # Joint limits
        # self.rewards.joint_reg.weight = 0. #0.25                    # Regularize positions of leg joints (relative to a nominal)
        # self.rewards.ang_vel_xy_l2.weight = -0.05                   # Base x-y angular velocity
        # self.rewards.lin_vel_z_l2.weight = -2.0                     # Base z linear velocity
        # self.rewards.flat_orientation_l2.weight = -1.5              # Tilting
        # self.rewards.action_rate_l2.weight = -0.005                 # Action smoothing
        # self.rewards.joint_deviation_hip.weight = -1.0              # Hip yaw and roll regularization
        # self.rewards.joint_deviation_arms.weight = -0.5             # Arms regularization
        # self.rewards.joint_deviation_torso.weight = -1.0
        # self.rewards.phase_feet_contacts.weight = 0.25 #1.               # Contact location
        # self.rewards.height_torso.weight = -2.                     # Base height
        # self.rewards.height_torso.params["target_height"] = 0.76
        # self.rewards.feet_clearance.weight = -20.
        # self.rewards.feet_clearance.params["target_height"] = 0.1
        # self.rewards.feet_slide.weight = -0.3
        # self.rewards.dof_acc_l2.weight = -1.25e-7
        # self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        # )
        # self.rewards.dof_torques_l2.weight = -1.5e-7
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        # )
        #
        # # -- Task
        # self.rewards.track_lin_vel_xy_exp.weight = 2 #2.5
        # self.rewards.track_ang_vel_z_exp.weight = 0.5 #0.75  # 0
        #
        # # -- Unused
        # self.rewards.track_heading.weight = 0.                     # Base heading
        # self.rewards.feet_air_time.weight = 0.
