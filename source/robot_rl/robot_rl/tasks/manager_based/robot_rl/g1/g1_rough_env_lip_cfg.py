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
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg

from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import (HumanoidEnvCfg, HumanoidEventsCfg,
                                                                    HumanoidRewardCfg, PERIOD)

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg  #Inherit from the base envs

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.mdp.cmd_cfg import HLIPCommandCfg
##
# Pre-defined configs
##
from robot_rl.assets.robots.g1_21j import G1_MINIMAL_CFG  # isort: skip

#

@configclass
class G1RoughLipCommandsCfg(CommandsCfg):
    """Commands for the G1 Flat environment."""   
    hlip_ref = HLIPCommandCfg()


@configclass
class G1RoughLipObservationsCfg(ObservationsCfg):
    """Observation specifications for the G1 Flat environment."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Observations for policy group."""
        base_lin_vel = None     # Removed - no sensor
        height_scan = None      # Removed - not supported yet

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),history_length=1,scale=0.25)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},history_length=1,scale=(2.0,2.0,0.25))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),history_length=1,scale=0.05)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),history_length=1)

        # Phase clock
        sin_phase = ObsTerm(func=mdp.sin_phase, params={"period": PERIOD})
        cos_phase = ObsTerm(func=mdp.cos_phase, params={"period": PERIOD})

        # des_foot_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "hlip_ref"},history_length=1,scale=(1.0,1.0))

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1),history_length=1,scale=1.0)
        foot_vel = ObsTerm(func=mdp.foot_vel, params={"command_name": "hlip_ref"},scale=1.0)
        foot_ang_vel = ObsTerm(func=mdp.foot_ang_vel, params={"command_name": "hlip_ref"},scale=1.0)
        ref_traj = ObsTerm(func=mdp.ref_traj, params={"command_name": "hlip_ref"},scale=1.0)
        act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "hlip_ref"},scale=1.0)
        ref_traj_vel = ObsTerm(func=mdp.ref_traj_vel, params={"command_name": "hlip_ref"},scale=0.1)
        act_traj_vel = ObsTerm(func=mdp.act_traj_vel, params={"command_name": "hlip_ref"},scale=0.1)
        # v_dot = ObsTerm(func=mdp.v_dot, params={"command_name": "hlip_ref"},clip=(-1000.0,1000.0),scale=0.001)
        # v = ObsTerm(func=mdp.v, params={"command_name": "hlip_ref"},clip=(0.0,500.0),scale=0.01)
        height_scan = None      # Removed - not supported yet


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# Lip specific rewards
##
class G1RoughLipRewards(HumanoidRewardCfg):
    """Rewards specific to LIP Model"""

    holonomic_constraint = RewTerm(
        func=mdp.holonomic_constraint,
        weight=4.0,
        params={
            "command_name": "hlip_ref",
            "z_offset": 0.036,
        }
    )

    holonomic_constraint_vel = RewTerm(
        func=mdp.holonomic_constraint_vel,
        weight=2.0,
        params={
            "command_name": "hlip_ref",
        }
    )

    # reference_tracking = RewTerm(
    #     func=mdp.reference_tracking,
    #     weight=5.0,
    #     params={
    #         "command_name": "hlip_ref",
    #         "term_std": [
    #             0.2, 0.2, 0.1,    # com x, y, z
    #             0.5, 0.5, 0.5,    # pelvis roll, pitch, yaw
    #             0.1, 0.1, 0.05,   # swing foot x, y, z
    #             0.5, 0.5, 0.5     # swing foot roll, pitch, yaw
    #             ],
    #         "term_weight": [1.0,3.0,4.0, #com x,y,z
    #                         5.0,5.0,1.0, #pelvis roll, pitch, yaw
    #                         15.0,15.0,20.0, #swing foot x,y,z
    #                         1.0,1.0,3.0, #swing foot roll, pitch, yaw
    #                       ]
    #     }
    # )

    # reference_vel_tracking = RewTerm(
    #     func=mdp.reference_vel_tracking,
    #     weight=5.0,
    #     params={
    #         "command_name": "hlip_ref",
    #         "term_std": [
    #             0.3, 0.4, 0.1,    # COM velocity x, y, z  — less sensitive to x/y, tighter for z (often near 0)
    #             0.5, 0.5, 0.5,    # Pelvis angular velocity roll, pitch, yaw — wide range, low precision needed
    #             0.2, 0.2, 0.2,    # Swing foot linear velocity x, y, z — more critical
    #             0.5, 0.5, 0.5     # Swing foot angular velocity roll, pitch, yaw — usually sloppy
    #         ],
    #         "term_weight": [
    #             0.2, 3.0, 2.0,    # COM velocity x, y, z — prioritize planar motion
    #             1.0, 1.0, 5.0,    # Pelvis angular vel roll, pitch, yaw — soft regularization
    #             5.0, 5.0, 1.0,    # Swing foot linear vel x, y, z — z (vertical swing timing) is crucial
    #             0.0, 0.0, 5.0     # Swing foot angular vel — low priority unless you're doing precision landings
    #         ]

    #     }
    # )


    clf_reward = RewTerm(
        func=mdp.clf_reward,
        weight=10.0,
        params={
            "command_name": "hlip_ref",
            "max_clf": 50.0,
        }
    )

    clf_decreasing_condition = RewTerm(
        func=mdp.clf_decreasing_condition,
        weight=-2.0,
        params={
            "command_name": "hlip_ref",
            "max_clf_decreasing": 100.0,
        }
    )

    # track_lin_vel_x_exp = RewTerm(
    #     func=mdp.track_lin_vel_x_exp,
    #     weight=2.0,
    #     params={
    #         "command_name": "base_velocity",
    #         "std": 0.5,
    #     }
    # )


@configclass
class G1RoughLipEnvCfg(HumanoidEnvCfg):
    """Configuration for the G1 Flat environment."""
    rewards: G1RoughLipRewards = G1RoughLipRewards()
    # events: G1RoughLipEventsCfg = G1RoughLipEventsCfg()
    observations: G1RoughLipObservationsCfg = G1RoughLipObservationsCfg()
    commands: G1RoughLipCommandsCfg = G1RoughLipCommandsCfg()
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
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.reset_base.params = {
            
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0,0)}, #(-3.14, 3.14)},
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
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0,1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0,1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5,0.5)

        ##
        # Terminations
        ##
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["pelvis_link"]

        ##
        # Rewards
        ##
        self.rewards.feet_air_time = None
        self.rewards.phase_contact = None
        self.rewards.lin_vel_z_l2 = None
        # self.rewards.height_torso = None
        self.rewards.feet_clearance = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.termination_penalty = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.joint_deviation_hip = None
        self.rewards.contact_no_vel = None
        self.rewards.alive = None
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        # self.rewards.track_ang_vel_z_exp.weight = 1.0
 
        # torque, acc, vel, action rate regularization
        self.rewards.dof_torques_l2.weight = -1.0e-5
        # self.rewards.dof_pos_limits.weight = -5.0
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_vel_l2.weight = -1.0e-5
        self.rewards.action_rate_l2.weight = -0.001
        # self.rewards.joint_deviation_arms.weight = -1.0             # Arms regularization
        # self.rewards.joint_deviation_torso.weight = -1.0
        
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_torso = None
        self.rewards.dof_pos_limits = None
        # self.rewards.dof_vel_l2 = None
        # self.rewards.dof_acc_l2 = None
        # self.rewards.dof_torques_l2 = None
        # self.rewards.action_rate_l2 = None  
        self.rewards.height_torso = None
        
        
        # self.rewards.alive.weight = 0.15
        # self.rewards.contact_no_vel.weight = -0.2
        # self.rewards.lip_gait_tracking.weight = 2
        # self.rewards.joint_deviation_hip.weight = -0.0
        # self.rewards.ang_vel_xy_l2.weight = -0.05
        # self.rewards.height_torso.weight = -1.0 #-10.0
        # self.rewards.feet_clearance.weight = -20.0
        # self.rewards.lin_vel_z_l2.weight =  -2.0 
        # self.rewards.track_lin_vel_xy_exp.weight = 3.5 #1
        # self.rewards.phase_contact.weight = 0 #0.25
        
        
        # self.rewards.lip_feet_tracking.weight = 10.0 #10.0
        # self.rewards.flat_orientation_l2.weight = -1.0
        # self.rewards.height_torso.params["target_height"] = 0.75
        # self.rewards.feet_clearance.params["target_height"] = 0.12