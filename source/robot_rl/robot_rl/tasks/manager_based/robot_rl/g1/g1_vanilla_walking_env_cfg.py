# Copyright 2026 Zachary Olkin. All rights reserved.

import math
from isaaclab.utils import configclass

from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import (HumanoidEnvCfg, HumanoidRewardCfg)

##
# Pre-defined configs
##
from robot_rl.assets.robots.g1_21j import G1_MINIMAL_CFG  # isort: skip
from isaaclab.sensors import  RayCasterCfg, patterns
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from robot_rl.tasks.manager_based.robot_rl import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import ObservationGroupCfg as ObsGroup

##
# Vanilla Rewards
##
@configclass
class G1VanillaWalkingRewardsCfg(HumanoidRewardCfg):
    ##
    # Termination
    ###
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    ##
    # Tracking
    ##
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)

    ##
    # Feet
    ##
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
        },
    )

    phase_contact = RewTerm(
        func=mdp.phase_contact,
        weight=0.18,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )

    ##
    # Penalize deviation from default of the joints that are not essential for locomotion
    ##
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0,  # -0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_yaw_joint")},
    )

    ##
    # Torso Height
    ##
    height_torso = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": 0.78},
    )

    undesired_contacts = None

    ##
    # Regularization
    ##
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)

    ##
    # Feet
    ##
    feet_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=0.0,
        params={
            "target_height": 0.08,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    contact_no_vel = RewTerm(
        func=mdp.contact_no_vel,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

##
# Observations
##
@configclass
class G1VanillaWalkingObservationsCfg():
    """Observation specifications for the G1 Flat environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},
                                    scale=(2.0, 2.0, 2.0))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), scale=0.05)

        actions = ObsTerm(func=mdp.last_action)
        # Phase clock
        sin_phase = ObsTerm(func=mdp.sin_phase, params={"command_name": "gait_period"})
        cos_phase = ObsTerm(func=mdp.cos_phase, params={"command_name": "gait_period"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},
                                    scale=(2.0, 2.0, 2.0))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)

        sin_phase = ObsTerm(func=mdp.sin_phase, params={"command_name": "gait_period"})
        cos_phase = ObsTerm(func=mdp.cos_phase, params={"command_name": "gait_period"})

        contact_state = ObsTerm(
            func=mdp.contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

##
# Environment configuration
##
@configclass
class G1VanillaWalkingEnvCfg(HumanoidEnvCfg):
    """Configuration for the G1 Rough environment."""
    observations: G1VanillaWalkingObservationsCfg = G1VanillaWalkingObservationsCfg()
    rewards: G1VanillaWalkingRewardsCfg = G1VanillaWalkingRewardsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None

        # no terrain curriculum
        self.curriculum.terrain_levels = None


        ##
        # Randomization
        ##
        # -- Push Params -- #
        self.events.push_robot.params["velocity_range"] = {
            "x": (-1, 1),
            "y": (-1, 1),
            "roll": (-0.4, 0.4),
            "pitch": (-0.4, 0.4),
            "yaw": (-0.4, 0.4),
        }
        # -- Base Mass Params -- #
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.add_base_mass.params["operation"] = "scale"

        # -- Reset Params -- #
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
        # Commands
        ##
        self.commands.base_velocity.ranges.lin_vel_x = (-0.75,0.75) #(-1.0, 1.0) # 0 - 1
        self.commands.base_velocity.ranges.lin_vel_y = (0.0,0.0) #(-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5,0.5) #(-1.0, 1.0) #(-1.0, 1.0)

        ##
        # Terminations
        ##
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"


        ##
        # Rewards
        ##
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0 # TODO reduce this maybe?
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_vel_l2.weight = -1.0e-3
        self.rewards.action_rate_l2.weight = -0.01
        # self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_pos_limits.weight = -5.0
        self.rewards.alive.weight = 0.15
        self.rewards.contact_no_vel.weight = -0.2
        self.rewards.joint_deviation_hip.weight = -1.0
        self.rewards.height_torso.weight = -10.0
        self.rewards.feet_clearance.weight = -20.0
        self.rewards.phase_contact.weight = 0.25

        self.rewards.joint_deviation_arms.weight = -0.5  # Arms regularization
        self.rewards.joint_deviation_torso.weight = -1.0

        self.rewards.height_torso.params["target_height"] = 0.75
        self.rewards.feet_clearance.params["target_height"] = 0.12

# Extra Compute (EC)
@configclass
class G1VanillaWalkingECEnvCfg(G1VanillaWalkingEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (0.616,0.616),
                "operation": "add",
            }
        )