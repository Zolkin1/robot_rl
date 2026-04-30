from isaaclab.utils import configclass
# from robot_rl.tasks.manager_based.robot_rl.mdp.commands.clf_cmd.hzd_cfg import GaitLibraryHZDCommandCfg
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidCommandsCfg
# from robot_rl.tasks.manager_based.robot_rl.g1.g1_flat_env_hzd_cfg import G1FlatHZDEnvCfg
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import (HumanoidEnvCfg, HumanoidCommandsCfg,
                                                                    HumanoidRewardCfg, HumanoidTerminationCfg)
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from robot_rl.tasks.manager_based.robot_rl import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as TerminationTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from source.robot_rl.robot_rl.tasks.manager_based.robot_rl.terrains.config.rough import ROUGH_SLOPED_FOR_FLAT_HZD_CFG
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from robot_rl.assets.robots.g1_21j import G1_MINIMAL_CFG  # isort: skip
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
import math

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_cmd_cfg import TrajectoryCommandCfg
from .g1_trajopt_obs import G1TrajOptObservationsCfg
from .g1_trajopt_reward import G1TrajOptCLFRewards

##
# Lyapunov Weights
##
BOWING_Q_weights = {}
BOWING_Q_weights["com:pos_x"] = [5.0, 1.0]
BOWING_Q_weights["com:pos_y"] = [5.0, 1.0]
BOWING_Q_weights["com:pos_z"] = [5.0, 1.0]

BOWING_Q_weights["left_ankle_roll_link:pos_x"] = [10.0, 2.0]
BOWING_Q_weights["left_ankle_roll_link:pos_y"] = [10.0, 2.0]
BOWING_Q_weights["left_ankle_roll_link:pos_z"] = [10.0, 2.0]
BOWING_Q_weights["left_ankle_roll_link:ori_x"] = [10.0, 2.0]
BOWING_Q_weights["left_ankle_roll_link:ori_y"] = [10.0, 2.0]
BOWING_Q_weights["left_ankle_roll_link:ori_z"] = [10.0, 2.0]

BOWING_Q_weights["right_ankle_roll_link:pos_x"] = [10.0, 2.0]
BOWING_Q_weights["right_ankle_roll_link:pos_y"] = [10.0, 2.0]
BOWING_Q_weights["right_ankle_roll_link:pos_z"] = [10.0, 2.0]
BOWING_Q_weights["right_ankle_roll_link:ori_x"] = [10.0, 2.0]
BOWING_Q_weights["right_ankle_roll_link:ori_y"] = [10.0, 2.0]
BOWING_Q_weights["right_ankle_roll_link:ori_z"] = [10.0, 2.0]

BOWING_Q_weights["joint:left_hip_roll_joint"] = [4.0, 1.0]
BOWING_Q_weights["joint:left_hip_pitch_joint"] = [4.0, 1.0]
BOWING_Q_weights["joint:left_hip_yaw_joint"] = [4.0, 1.0]
BOWING_Q_weights["joint:left_knee_joint"] = [4.0, 1.0]
BOWING_Q_weights["joint:left_ankle_roll_joint"] = [4.0, 1.0]
BOWING_Q_weights["joint:left_ankle_pitch_joint"] = [4.0, 1.0]
BOWING_Q_weights["joint:right_hip_roll_joint"] = [4.0, 1.0]
BOWING_Q_weights["joint:right_hip_pitch_joint"] = [4.0, 1.0]
BOWING_Q_weights["joint:right_hip_yaw_joint"] = [4.0, 1.0]
BOWING_Q_weights["joint:right_knee_joint"] = [4.0, 1.0]
BOWING_Q_weights["joint:right_ankle_roll_joint"] = [4.0, 1.0]
BOWING_Q_weights["joint:right_ankle_pitch_joint"] = [4.0, 1.0]

BOWING_Q_weights["pelvis_link:pos_x"] = [5.0, 1.0]
BOWING_Q_weights["pelvis_link:pos_y"] = [5.0, 1.0]
BOWING_Q_weights["pelvis_link:pos_z"] = [5.0, 1.0]
BOWING_Q_weights["pelvis_link:ori_x"] = [15.0, 3.0]
BOWING_Q_weights["pelvis_link:ori_y"] = [15.0, 3.0]
BOWING_Q_weights["pelvis_link:ori_z"] = [15.0, 3.0]

BOWING_Q_weights["joint:waist_yaw_joint"] = [8.0, 1.0]
BOWING_Q_weights["joint:left_elbow_joint"] = [8.0, 1.0]
BOWING_Q_weights["joint:left_shoulder_pitch_joint"] = [8.0, 1.0]
BOWING_Q_weights["joint:left_shoulder_roll_joint"] = [8.0, 1.0]
BOWING_Q_weights["joint:left_shoulder_yaw_joint"] = [8.0, 1.0]
BOWING_Q_weights["joint:right_elbow_joint"] = [8.0, 1.0]
BOWING_Q_weights["joint:right_shoulder_pitch_joint"] = [8.0, 1.0]
BOWING_Q_weights["joint:right_shoulder_roll_joint"] = [8.0, 1.0]
BOWING_Q_weights["joint:right_shoulder_yaw_joint"] = [8.0, 1.0]

BOWING_Q_weights["right_wrist_yaw_link:pos_x"] = [10.0, 2.0]
BOWING_Q_weights["right_wrist_yaw_link:pos_y"] = [10.0, 2.0]
BOWING_Q_weights["right_wrist_yaw_link:pos_z"] = [10.0, 2.0]
BOWING_Q_weights["right_wrist_yaw_link:ori_x"] = [0.0, 0.0]
BOWING_Q_weights["right_wrist_yaw_link:ori_y"] = [0.0, 0.0]
BOWING_Q_weights["right_wrist_yaw_link:ori_z"] = [0.0, 0.0]

BOWING_Q_weights["left_wrist_yaw_link:pos_x"] = [10.0, 2.0]
BOWING_Q_weights["left_wrist_yaw_link:pos_y"] = [10.0, 2.0]
BOWING_Q_weights["left_wrist_yaw_link:pos_z"] = [10.0, 2.0]
BOWING_Q_weights["left_wrist_yaw_link:ori_x"] = [0.0, 0.0]
BOWING_Q_weights["left_wrist_yaw_link:ori_y"] = [0.0, 0.0]
BOWING_Q_weights["left_wrist_yaw_link:ori_z"] = [0.0, 0.0]

BOWING_R_weights = {}
BOWING_R_weights["com:pos_x"] = [0.01]
BOWING_R_weights["com:pos_y"] = [0.01]
BOWING_R_weights["com:pos_z"] = [0.01]

BOWING_R_weights["left_ankle_roll_link:pos_x"] = [0.005]
BOWING_R_weights["left_ankle_roll_link:pos_y"] = [0.005]
BOWING_R_weights["left_ankle_roll_link:pos_z"] = [0.005]
BOWING_R_weights["left_ankle_roll_link:ori_x"] = [0.002]
BOWING_R_weights["left_ankle_roll_link:ori_y"] = [0.002]
BOWING_R_weights["left_ankle_roll_link:ori_z"] = [0.002]

BOWING_R_weights["right_ankle_roll_link:pos_x"] = [0.005]
BOWING_R_weights["right_ankle_roll_link:pos_y"] = [0.005]
BOWING_R_weights["right_ankle_roll_link:pos_z"] = [0.005]
BOWING_R_weights["right_ankle_roll_link:ori_x"] = [0.002]
BOWING_R_weights["right_ankle_roll_link:ori_y"] = [0.002]
BOWING_R_weights["right_ankle_roll_link:ori_z"] = [0.002]

BOWING_R_weights["joint:left_hip_roll_joint"] = [0.001]
BOWING_R_weights["joint:left_hip_pitch_joint"] = [0.001]
BOWING_R_weights["joint:left_hip_yaw_joint"] = [0.001]
BOWING_R_weights["joint:left_knee_joint"] = [0.001]
BOWING_R_weights["joint:left_ankle_roll_joint"] = [0.001]
BOWING_R_weights["joint:left_ankle_pitch_joint"] = [0.001]
BOWING_R_weights["joint:right_hip_roll_joint"] = [0.001]
BOWING_R_weights["joint:right_hip_pitch_joint"] = [0.001]
BOWING_R_weights["joint:right_hip_yaw_joint"] = [0.001]
BOWING_R_weights["joint:right_knee_joint"] = [0.001]
BOWING_R_weights["joint:right_ankle_roll_joint"] = [0.001]
BOWING_R_weights["joint:right_ankle_pitch_joint"] = [0.001]

BOWING_R_weights["pelvis_link:pos_x"] = [0.005]
BOWING_R_weights["pelvis_link:pos_y"] = [0.005]
BOWING_R_weights["pelvis_link:pos_z"] = [0.005]
BOWING_R_weights["pelvis_link:ori_x"] = [0.005]
BOWING_R_weights["pelvis_link:ori_y"] = [0.005]
BOWING_R_weights["pelvis_link:ori_z"] = [0.005]

BOWING_R_weights["joint:waist_yaw_joint"] = [0.001]
BOWING_R_weights["joint:left_elbow_joint"] = [0.001]
BOWING_R_weights["joint:left_shoulder_pitch_joint"] = [0.001]
BOWING_R_weights["joint:left_shoulder_roll_joint"] = [0.001]
BOWING_R_weights["joint:left_shoulder_yaw_joint"] = [0.001]
BOWING_R_weights["joint:right_elbow_joint"] = [0.001]
BOWING_R_weights["joint:right_shoulder_pitch_joint"] = [0.001]
BOWING_R_weights["joint:right_shoulder_roll_joint"] = [0.001]
BOWING_R_weights["joint:right_shoulder_yaw_joint"] = [0.001]

BOWING_R_weights["right_wrist_yaw_link:pos_x"] = [0.005]
BOWING_R_weights["right_wrist_yaw_link:pos_y"] = [0.005]
BOWING_R_weights["right_wrist_yaw_link:pos_z"] = [0.005]
BOWING_R_weights["right_wrist_yaw_link:ori_x"] = [0.005]
BOWING_R_weights["right_wrist_yaw_link:ori_y"] = [0.005]
BOWING_R_weights["right_wrist_yaw_link:ori_z"] = [0.005]

BOWING_R_weights["left_wrist_yaw_link:pos_x"] = [0.005]
BOWING_R_weights["left_wrist_yaw_link:pos_y"] = [0.005]
BOWING_R_weights["left_wrist_yaw_link:pos_z"] = [0.005]
BOWING_R_weights["left_wrist_yaw_link:ori_x"] = [0.005]
BOWING_R_weights["left_wrist_yaw_link:ori_y"] = [0.005]
BOWING_R_weights["left_wrist_yaw_link:ori_z"] = [0.005]

##
# Commands
##
@configclass
class G1BowingCommandsCfg(HumanoidCommandsCfg):
    """Configuration for gait library commands."""
    traj_ref = TrajectoryCommandCfg(
        contact_bodies = [".*_ankle_roll_link"],

        manager_type="trajectory",
        hf_repo = "zolkin/robot_rl",
        path = "trajectories/bowing/bow_forward_config_solution.yaml",

        conditioner_generator_name = "base_velocity",
        Q_weights = BOWING_Q_weights,
        R_weights = BOWING_R_weights,
        random_start_time_max = -1.0, #1.0,
        percent_hold_phi = 0.0, #0.05,
    )

##
# Terminations
##
@configclass
class G1BowingTerminationsCfg(HumanoidTerminationCfg):
    """Termination terms for the MDP."""

    pass
    # base_orientation = TerminationTerm(func=mdp.base_orientation, params={
    #     "cmd_name": "traj_ref",
    #     "roll_limit_deg": 20,
    #     "pitch_limit_deg": 20,
    # })


##
# Curriculums
##
@configclass
class G1BowingCLFCurriculumCfg:
    """Curriculum terms for the MDP."""

    clf_curriculum = CurrTerm(func=mdp.clf_curriculum, params={"update_interval": 15000, "min_max_err": (0.15, 0.15, 0.15), "scale": (0.05, 0.05, 0.05)})

@configclass
class G1BowingCLFEnvCfg(HumanoidEnvCfg):
    """Configuration for the G1 environment with gait library."""
    commands: G1BowingCommandsCfg = G1BowingCommandsCfg()
    observations: G1TrajOptObservationsCfg = G1TrajOptObservationsCfg()
    rewards: G1TrajOptCLFRewards = G1TrajOptCLFRewards()
    curriculum: G1BowingCLFCurriculumCfg = G1BowingCLFCurriculumCfg()
    terminations = G1BowingTerminationsCfg = G1BowingTerminationsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        self.episode_length_s = 2.5 #2.8 #10

        ##
        # Scene
        ##
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        ##
        # Commands
        ##
        # Configure velocity ranges for different gaits
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0)
        self.commands.base_velocity.ranges.heading = (0,0)

        self.commands.gait_period = None

        ##
        # Randomization
        ##
        self.events.push_robot.params["velocity_range"] = {
            "x": (-1, 1),
            "y": (-1, 1),
            "roll": (-0.4, 0.4),
            "pitch": (-0.4, 0.4),
            "yaw": (-0.4, 0.4),
        }
        # self.events.push_robot = None

        self.events.add_base_mass.params["asset_cfg"].body_names = ["waist_yaw_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.add_base_mass.params["operation"] = "scale"
        # self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (0.85, 1.15)
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
        
        self.events.base_com.params["asset_cfg"].body_names = "waist_yaw_link"
        self.events.base_com.params["com_range"] = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)}

        ##
        # Observations
        ##
        self.observations.policy.ref_traj = None
        self.observations.policy.act_traj = None
        self.observations.policy.ref_traj_vel = None
        self.observations.policy.act_traj_vel = None
        self.observations.policy.root_quat = None
        self.observations.policy.base_lin_vel = None
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
        self.rewards.clf_decreasing_condition.weight = 0 #-1
        # self.rewards.clf_decreasing_condition = None

        ##
        # Terminations
        ##
        # self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"

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

@configclass
class G1BowingCLFECEnvCfg(G1BowingCLFEnvCfg):
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
class G1BowingCLFEnvCfg_PLAY(G1BowingCLFEnvCfg):
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

        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        self.events.randomize_ground_contact_friction = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        
        
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)  # Allow full range
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0)

        # self.events.reset_base.params["pose_range"]["yaw"] = (-3.14,3.14)
        # self.events.reset_base.params["pose_range"]["x"] = (-3,3)
        # self.events.reset_base.params["pose_range"]["y"] = (-3,3)