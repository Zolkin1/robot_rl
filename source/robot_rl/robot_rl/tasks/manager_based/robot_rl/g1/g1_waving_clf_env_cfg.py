# Copyright 2026 Zachary Olkin. All rights reserved.

from isaaclab.utils import configclass
# from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_cfg import GaitLibraryHZDCommandCfg
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidCommandsCfg
# from robot_rl.tasks.manager_based.robot_rl.g1.g1_flat_env_hzd_cfg import G1FlatHZDEnvCfg
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import (HumanoidEnvCfg, HumanoidCommandsCfg,
                                                                    HumanoidRewardCfg)
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from robot_rl.tasks.manager_based.robot_rl import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from robot_rl.tasks.manager_based.robot_rl.terrains.rough import ROUGH_SLOPED_FOR_FLAT_HZD_CFG
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from robot_rl.assets.robots.g1_21j import G1_MINIMAL_CFG  # isort: skip
import math

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_cmd_cfg import TrajectoryCommandCfg
from .g1_trajopt_obs import G1TrajOptObservationsCfg
from .g1_trajopt_reward import G1TrajOptCLFRewards

##
# Lyapunov Weights
##
WAVING_Q_weights = {}
WAVING_Q_weights["com:pos_x"] = [25.0, 250.0]
WAVING_Q_weights["com:pos_y"] = [500.0, 20.0]
WAVING_Q_weights["com:pos_z"] = [250.0, 10.0]

WAVING_Q_weights["left_ankle_roll_link:pos_x"] = [1500.0, 50.0]
WAVING_Q_weights["left_ankle_roll_link:pos_y"] = [1500.0, 50.0]
WAVING_Q_weights["left_ankle_roll_link:pos_z"] = [2500.0, 50.0]
WAVING_Q_weights["left_ankle_roll_link:ori_x"] = [30.0, 1.0]
WAVING_Q_weights["left_ankle_roll_link:ori_y"] = [150.0, 1.0]
WAVING_Q_weights["left_ankle_roll_link:ori_z"] = [400.0, 10.0]

WAVING_Q_weights["right_ankle_roll_link:pos_x"] = [1500.0, 50.0]
WAVING_Q_weights["right_ankle_roll_link:pos_y"] = [1500.0, 50.0]
WAVING_Q_weights["right_ankle_roll_link:pos_z"] = [2500.0, 50.0]
WAVING_Q_weights["right_ankle_roll_link:ori_x"] = [30.0, 1.0]
WAVING_Q_weights["right_ankle_roll_link:ori_y"] = [150.0, 1.0]
WAVING_Q_weights["right_ankle_roll_link:ori_z"] = [400.0, 10.0]

WAVING_Q_weights["pelvis_link:pos_x"] = [25.0, 250.0]
WAVING_Q_weights["pelvis_link:pos_y"] = [500.0, 20.0]
WAVING_Q_weights["pelvis_link:pos_z"] = [250.0, 10.0]
WAVING_Q_weights["pelvis_link:ori_x"] = [300.0, 20.0]
WAVING_Q_weights["pelvis_link:ori_y"] = [250.0, 10.0]
WAVING_Q_weights["pelvis_link:ori_z"] = [300.0, 30.0]

WAVING_Q_weights["joint:waist_yaw_joint"] = [100.0, 1.0]
WAVING_Q_weights["joint:left_elbow_joint"] = [50.0, 1.0]
WAVING_Q_weights["joint:left_shoulder_pitch_joint"] = [50.0, 1.0]
WAVING_Q_weights["joint:left_shoulder_roll_joint"] = [50.0, 1.0]
WAVING_Q_weights["joint:left_shoulder_yaw_joint"] = [50.0, 1.0]
WAVING_Q_weights["joint:right_elbow_joint"] = [50.0, 1.0]
WAVING_Q_weights["joint:right_shoulder_pitch_joint"] = [50.0, 1.0]
WAVING_Q_weights["joint:right_shoulder_roll_joint"] = [50.0, 1.0]
WAVING_Q_weights["joint:right_shoulder_yaw_joint"] = [50.0, 1.0]

WAVING_Q_weights["right_wrist_yaw_link:pos_x"] = [500.0, 50.0]
WAVING_Q_weights["right_wrist_yaw_link:pos_y"] = [500.0, 50.0]
WAVING_Q_weights["right_wrist_yaw_link:pos_z"] = [500.0, 50.0]
WAVING_Q_weights["right_wrist_yaw_link:ori_x"] = [150.0, 10.0]
WAVING_Q_weights["right_wrist_yaw_link:ori_y"] = [150.0, 10.0]
WAVING_Q_weights["right_wrist_yaw_link:ori_z"] = [150.0, 10.0]

WAVING_Q_weights["left_wrist_yaw_link:pos_x"] = [500.0, 50.0]
WAVING_Q_weights["left_wrist_yaw_link:pos_y"] = [500.0, 50.0]
WAVING_Q_weights["left_wrist_yaw_link:pos_z"] = [500.0, 50.0]
WAVING_Q_weights["left_wrist_yaw_link:ori_x"] = [150.0, 10.0]
WAVING_Q_weights["left_wrist_yaw_link:ori_y"] = [150.0, 10.0]
WAVING_Q_weights["left_wrist_yaw_link:ori_z"] = [150.0, 10.0]
# WAVING_Q_weights = [
#     25.0,   250.0,      # com_x pos, vel
#     500.0,   20.0,      # com_y pos, vel
#     650.0,   10.0,      # com_z pos, vel
#     300.0,    20.0,     # pelvis_roll pos, vel
#     250.0,    10.0,     # pelvis_pitch pos, vel
#     300.0,    30.0,     # pelvis_yaw pos, vel
#     1500.0, 50.0,       # swing_x pos, vel
#     1500.0,  50.0,      # swing_y pos, vel
#     2500.0, 50.0,       # swing_z pos, vel
#     30.0,    1.0,       # swing_ori_roll pos, vel
#     150.0,    1.0,       # swing_ori_pitch pos, vel
#     400.0,    10.0,     # swing_ori_yaw pos, vel
#     1500.0, 50.0,       # stance_x pos, vel
#     1500.0,  50.0,      # stance_y pos, vel
#     2500.0, 50.0,       # stance_z pos, vel
#     30.0,    1.0,       # stance_ori_roll pos, vel
#     150.0,    1.0,       # stance_ori_pitch pos, vel
#     400.0,    10.0,     # swing_ori_yaw pos, vel
#     100.0,    1.0,      # waist_yaw pos, vel
#     50.0,1.0, #left shoulder pitch
#     50.0,1.0, #left shoulder roll
#     50.0,1.0, #left shoulder yaw
#     50.0,1.0, #left elbow
#     50.0,1.0, #right shoulder pitch
#     50.0,1.0, #right shoulder roll
#     50.0,1.0, #right shoulder yaw
#     50.0,1.0, #right elbow
# ]

WAVING_R_weights = {}
WAVING_R_weights["com:pos_x"] = [0.1]
WAVING_R_weights["com:pos_y"] = [0.1]
WAVING_R_weights["com:pos_z"] = [0.1]

WAVING_R_weights["left_ankle_roll_link:pos_x"] = [0.05]
WAVING_R_weights["left_ankle_roll_link:pos_y"] = [0.05]
WAVING_R_weights["left_ankle_roll_link:pos_z"] = [0.05]
WAVING_R_weights["left_ankle_roll_link:ori_x"] = [0.02]
WAVING_R_weights["left_ankle_roll_link:ori_y"] = [0.02]
WAVING_R_weights["left_ankle_roll_link:ori_z"] = [0.02]

WAVING_R_weights["right_ankle_roll_link:pos_x"] = [0.05]
WAVING_R_weights["right_ankle_roll_link:pos_y"] = [0.05]
WAVING_R_weights["right_ankle_roll_link:pos_z"] = [0.05]
WAVING_R_weights["right_ankle_roll_link:ori_x"] = [0.02]
WAVING_R_weights["right_ankle_roll_link:ori_y"] = [0.02]
WAVING_R_weights["right_ankle_roll_link:ori_z"] = [0.02]

WAVING_R_weights["pelvis_link:pos_x"] = [0.05]
WAVING_R_weights["pelvis_link:pos_y"] = [0.05]
WAVING_R_weights["pelvis_link:pos_z"] = [0.05]
WAVING_R_weights["pelvis_link:ori_x"] = [0.05]
WAVING_R_weights["pelvis_link:ori_y"] = [0.05]
WAVING_R_weights["pelvis_link:ori_z"] = [0.05]

WAVING_R_weights["joint:waist_yaw_joint"] = [0.01]
WAVING_R_weights["joint:left_elbow_joint"] = [0.01]
WAVING_R_weights["joint:left_shoulder_pitch_joint"] = [0.01]
WAVING_R_weights["joint:left_shoulder_roll_joint"] = [0.01]
WAVING_R_weights["joint:left_shoulder_yaw_joint"] = [0.01]
WAVING_R_weights["joint:right_elbow_joint"] = [0.01]
WAVING_R_weights["joint:right_shoulder_pitch_joint"] = [0.01]
WAVING_R_weights["joint:right_shoulder_roll_joint"] = [0.01]
WAVING_R_weights["joint:right_shoulder_yaw_joint"] = [0.01]

WAVING_R_weights["right_wrist_yaw_link:pos_x"] = [0.05]
WAVING_R_weights["right_wrist_yaw_link:pos_y"] = [0.05]
WAVING_R_weights["right_wrist_yaw_link:pos_z"] = [0.05]
WAVING_R_weights["right_wrist_yaw_link:ori_x"] = [0.05]
WAVING_R_weights["right_wrist_yaw_link:ori_y"] = [0.05]
WAVING_R_weights["right_wrist_yaw_link:ori_z"] = [0.05]

WAVING_R_weights["left_wrist_yaw_link:pos_x"] = [0.05]
WAVING_R_weights["left_wrist_yaw_link:pos_y"] = [0.05]
WAVING_R_weights["left_wrist_yaw_link:pos_z"] = [0.05]
WAVING_R_weights["left_wrist_yaw_link:ori_x"] = [0.05]
WAVING_R_weights["left_wrist_yaw_link:ori_y"] = [0.05]
WAVING_R_weights["left_wrist_yaw_link:ori_z"] = [0.05]
# WAVING_R_weights = [
#         0.1, 0.1, 0.1,      # CoM inputs: allow moderate effort
#         0.05,0.05,0.05,     # pelvis inputs: lower torque priority
#         0.05,0.05,0.05,     # swing foot linear inputs
#         0.02,0.02,0.02,     # swing foot orientation inputs: small adjustments
#         0.05, 0.05, 0.05,   # stance foot linear inputs
#         0.02, 0.02, 0.02,   # stance foot orientation inputs: small adjustments
#         0.1,0.01,0.01,
#         0.01,0.01,0.01,
#         0.01,0.01,0.01,
#     ]

##
# Commands
##
@configclass
class G1WavingCommandsCfg(HumanoidCommandsCfg):
    """Configuration for gait library commands."""
    traj_ref = TrajectoryCommandCfg(
        contact_bodies = [".*_ankle_roll_link"],

        manager_type="trajectory",
        hf_repo = "zolkin/robot_rl",
        path = "trajectories/waving/waving_config_solution.yaml",

        conditioner_generator_name = "base_velocity",
        num_outputs = 27,
        Q_weights = WAVING_Q_weights,
        R_weights = WAVING_R_weights,
    )

##
# Curriculums
##
@configclass
class G1WavingCLFCurriculumCfg:
    """Curriculum terms for the MDP."""

    clf_curriculum = CurrTerm(func=mdp.clf_curriculum, params={"update_interval": 1000, "min_val": 20.0})

@configclass
class G1WavingCLFEnvCfg(HumanoidEnvCfg):
    """Configuration for the G1 environment with gait library."""
    commands: G1WavingCommandsCfg = G1WavingCommandsCfg()
    observations: G1TrajOptObservationsCfg = G1TrajOptObservationsCfg()
    rewards: G1TrajOptCLFRewards = G1TrajOptCLFRewards()
    curriculum: G1WavingCLFCurriculumCfg = G1WavingCLFCurriculumCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        ##
        # Commands
        ##
        # Configure velocity ranges for different gaits
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)  # Allow full range
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0) # TODO: put back: (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (0,0)

        self.commands.gait_period = None

        ##
        # Randomization
        ##
        # self.events.push_robot.params["velocity_range"] = {
        #     "x": (-1, 1),
        #     "y": (-1, 1),
        #     "roll": (-0.4, 0.4),
        #     "pitch": (-0.4, 0.4),
        #     "yaw": (-0.4, 0.4),
        # }
        self.events.push_robot = None

        # self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis_link"]
        # self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        # self.events.add_base_mass.params["operation"] = "scale"
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
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

        self.events.base_external_force_torque = None

        ##
        # Rewards
        ##
        self.rewards.holonomic_constraint.params["command_name"] = "traj_ref"
        self.rewards.holonomic_constraint_vel.params["command_name"] = "traj_ref"

        self.rewards.clf_reward.params = {
            "command_name": "traj_ref",
            "max_eta_err": 0.25,
        }
        self.rewards.clf_decreasing_condition.params = {
            "command_name": "traj_ref",
            "alpha": 0.5,
            "eta_max": 0.25,
            "eta_dot_max": 0.3,
        }
        self.rewards.clf_decreasing_condition.weight = -1
        # self.rewards.clf_decreasing_condition = None

        ##
        # Terminations
        ##
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"

        ##
        # Curriculum
        ##
        self.curriculum.clf_curriculum = None
        self.curriculum.terrain_levels = None

        self.events.reset_base.params["pose_range"]["yaw"] = (0,0)

        # self.curriculum.clf_curriculum.params = {
        #     "min_max_err": (0.1,0.1),
        #     "scale": (0.001,0.001),
        #     "update_interval": 20000
        # }

        # self.rewards.clf_decreasing_condition = None

        ##
        # Terrain
        ##
        # self.scene.terrain.terrain_generator = ROUGH_SLOPED_FOR_FLAT_HZD_CFG
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no curriculum
        self.curriculum.terrain_levels = None
        self.curriculum.clf_curriculum = None

@configclass
class G1WavingCLFECEnvCfg(G1WavingCLFEnvCfg):
    """Configuration for the G1 environment with gait library."""
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        #both front and back 1.14
        #just front: 0.616
        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (0.616,0.616),
                "operation": "add",
            }
        )

@configclass
class G1WavingCLFEnvCfg_PLAY(G1WavingCLFEnvCfg):
    """Configuration for the G1 environment with gait library."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.terrain.size = (3,3)
        self.scene.terrain.border_width = 0.0
        self.scene.terrain.num_rows = 3
        self.scene.terrain.num_cols = 2
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None

        self.events.randomize_ground_contact_friction = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)  # Allow full range
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0,0)

        # self.events.reset_base.params["pose_range"]["yaw"] = (-3.14,3.14)
        # self.events.reset_base.params["pose_range"]["x"] = (-3,3)
        # self.events.reset_base.params["pose_range"]["y"] = (-3,3)