# Copyright 2026 Zachary Olkin. All rights reserved.

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg

from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import (HumanoidEnvCfg, HumanoidCommandsCfg,
                                                                    HumanoidRewardCfg)
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg  #Inherit from the base envs

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.cmd_cfg import HLIPCommandCfg

##
# Pre-defined configs
from robot_rl.assets.robots.g1_21j import G1_MINIMAL_CFG  # isort: skip


##
# Commands
##
@configclass
class G1LipCLFCommandsCfg(HumanoidCommandsCfg):
    """Commands for the G1 Flat environment."""   
    hlip_ref = HLIPCommandCfg()

##
# Curriculums
##
@configclass
class G1LipCLFCurriculumCfg:
    """Curriculum terms for the MDP."""

    clf_curriculum = CurrTerm(func=mdp.clf_curriculum, params={"update_interval": 1000, "min_val": 20.0})

##
# Rewards
##
@configclass
class G1LipCLFRewards(HumanoidRewardCfg):
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


    clf_reward = RewTerm(
        func=mdp.clf_reward,
        weight=10.0,
        params={
            "command_name": "hlip_ref",
            "max_eta_err": 0.25,
        }
    )

    clf_decreasing_condition = RewTerm(
        func=mdp.clf_decreasing_condition,
        weight=-2.0,
        params={
            "command_name": "hlip_ref",
            "alpha": 0.5,
            "eta_max": 0.2,
            "eta_dot_max":0.3,
        }
    )

##
# Observations
##
@configclass
class G1LipCLFObservationsCfg():
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

        foot_vel = ObsTerm(func=mdp.foot_vel, params={"command_name": "hlip_ref"}, scale=1.0)
        foot_ang_vel = ObsTerm(func=mdp.foot_ang_vel, params={"command_name": "hlip_ref"}, scale=1.0)
        ref_traj = ObsTerm(
            func=mdp.ref_traj,
            params={"command_name": "hlip_ref"}
        )
        act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "hlip_ref"}, scale=1.0)
        ref_traj_vel = ObsTerm(func=mdp.ref_traj_vel, params={"command_name": "hlip_ref"}, clip=(-20.0, 20.0,),
                               scale=1.0)
        act_traj_vel = ObsTerm(func=mdp.act_traj_vel, params={"command_name": "hlip_ref"}, clip=(-20.0, 20.0,),
                               scale=1.0)
        height_scan = None  # Removed - not supported yet

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class G1LipCLFEnvCfg(HumanoidEnvCfg):
    """Configuration for the G1 Flat environment."""
    rewards: G1LipCLFRewards = G1LipCLFRewards()
    observations: G1LipCLFObservationsCfg = G1LipCLFObservationsCfg()
    commands: G1LipCLFCommandsCfg = G1LipCLFCommandsCfg()
    curriculum: G1LipCLFCurriculumCfg = G1LipCLFCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        ##
        # Scene
        ##
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

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
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.add_base_mass.params["operation"] = "scale"
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
        self.commands.base_velocity.ranges.lin_vel_x = (-0.75,0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (0,0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5,0.5)

        ##
        # Terminations
        ##
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"

        
        ##
        # Rewards
        ##
        # CLF Related
        self.rewards.clf_reward.params = {
            "command_name": "hlip_ref",
            "max_eta_err": 0.3,
        }

        self.rewards.clf_decreasing_condition.params = {
            "command_name": "hlip_ref",
            "alpha": 1.0,
            "eta_max": 0.2,
            "eta_dot_max": 0.3,
        }

        # torque, acc, vel, action rate regularization
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.01

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

# Extra compute (EC)
@configclass
class G1LipCLFECEnvCfg(G1LipCLFEnvCfg):
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

@configclass
class G1LipCLFEnvCfg_PLAY(G1LipCLFEnvCfg):
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
