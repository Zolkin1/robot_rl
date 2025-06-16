import math
import torch

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import  RayCasterCfg, patterns
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import TerminationsCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from .g1_rough_env_lip_cfg import G1RoughLipEnvCfg, G1RoughLipRewards, G1RoughLipObservationsCfg
from robot_rl.tasks.manager_based.robot_rl.terrains.rough import STAIR_CFG

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg  #Inherit from the base envs

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidEventsCfg
from robot_rl.tasks.manager_based.robot_rl.mdp.stair_cfg import StairHLIPCommandCfg
##
# Pre-defined configs
##
from robot_rl.assets.robots.g1_21j import G1_MINIMAL_CFG  # isort: skip

#

@configclass
class G1StairObservationsCfg(G1RoughLipObservationsCfg):
    """Observation specifications for the G1 Flat environment."""
    @configclass
    class PolicyCfg(G1RoughLipObservationsCfg.PolicyCfg):
      height_scan = None
      sin_phase = None
      height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            scale=1,
            clip=(-1.0, 1.0)
        )
      sin_phase = ObsTerm(
            func=mdp.stair_sin_phase,
            params={"command_name": "hlip_ref"},
        )
      cos_phase = ObsTerm(
            func=mdp.stair_cos_phase,
            params={"command_name": "hlip_ref"},
        )
      step_duration = ObsTerm(
            func=mdp.step_duration,
            params={"command_name": "hlip_ref"},
        )
      
    @configclass
    class CriticCfg(G1RoughLipObservationsCfg.CriticCfg):
        height_scan = None
        sin_phase = None
        cos_phase = None
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            scale=1,
            clip=(-1.0, 1.0)
        )
        sin_phase = ObsTerm(
            func=mdp.stair_sin_phase,
            params={"command_name": "hlip_ref"},
        )
        cos_phase = ObsTerm(
            func=mdp.stair_cos_phase,
            params={"command_name": "hlip_ref"},
        )

        step_duration = ObsTerm(
            func=mdp.step_duration,
            params={"command_name": "hlip_ref"},
        )
        
        contact_state = ObsTerm(
            func=mdp.contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class G1StairCommandsCfg(CommandsCfg):
    """Commands for the G1 Flat environment."""   
    hlip_ref = StairHLIPCommandCfg()


@configclass
class G1StairRewardsCfg(G1RoughLipRewards):
    """Rewards for the G1 Flat environment."""
    holonomic_constraint = None
    holonomic_constraint_stair: RewTerm = RewTerm(
        func=mdp.holonomic_constraint_stair,
        params={"command_name": "hlip_ref"},
        weight=4.0,
    )

    swing_foot_contact = RewTerm(
        func=mdp.swing_foot_contact_penalty,
        params={"command_name": "hlip_ref",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        weight=-2.0,
    )

@configclass
class G1StairsTerminationCfg(TerminationsCfg):
    """Events for the G1 Flat environment."""
    no_progress = DoneTerm(
        func=mdp.no_progress,
        params={},
       )

@configclass
class G1StairEnvCfg(G1RoughLipEnvCfg):
    """Configuration for the G1 Flat environment."""
    commands: G1StairCommandsCfg = G1StairCommandsCfg()
    rewards: G1StairRewardsCfg = G1StairRewardsCfg()
    terminations: G1StairsTerminationCfg = G1StairsTerminationCfg()
    observations: G1StairObservationsCfg = G1StairObservationsCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()


        self.curriculum.clf_curriculum = None
        ##
        # Scene
        ##
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # No height scanner for now
     
        self.scene.terrain.terrain_type = "generator"

        self.scene.terrain.terrain_generator = STAIR_CFG
        # self.scene.terrain.terrain_generator.max_init_terrain_level = 2.0
        # self.curriculum.terrain_levels = None
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels)
        self.scene.height_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            attach_yaw_only=True,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.5, 1.5]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis_link"


      
        ##
        # Randomization
        ##
        # self.events.push_robot = None
        self.events.push_robot.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "roll": (-0.4, 0.4),
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
            
            "pose_range": {"x": (0.0,0.0), "y": (0.0,0.0), "yaw": (0,0)},
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
        self.commands.base_velocity.ranges.lin_vel_x = (0.4,0.75)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0,0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0,0.0)

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
        # self.rewards.alive = None
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None

        self.rewards.clf_reward.params["max_clf"] = 100.0
        self.rewards.clf_decreasing_condition.params["max_clf_decreasing"] = 100.0
        # self.rewards.track_ang_vel_z_exp.weight = 1.0
 
        # torque, acc, vel, action rate regularization
        # self.rewards.dof_torques_l2.weight = -1.0e-5
        # self.rewards.dof_pos_limits.weight = -1.0
        # self.rewards.dof_acc_l2.weight = -2.5e-7
        # self.rewards.dof_vel_l2.weight = -1.0e-5
        # self.rewards.action_rate_l2.weight = -0.001
        # self.rewards.joint_deviation_arms.weight = -1.0             # Arms regularization
        # self.rewards.joint_deviation_torso.weight = -1.0
        
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_torso = None
        self.rewards.dof_pos_limits = None
        self.rewards.dof_vel_l2 = None
        self.rewards.dof_acc_l2 = None
        self.rewards.dof_torques_l2 = None
        self.rewards.action_rate_l2 = None  
        self.rewards.height_torso = None
        


class G1StairPlay_EnvCfg(G1StairEnvCfg):

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.events.reset_base.params["pose_range"] = {"x": (0,0), "y": (0,0), "yaw": (0,0)} #(-3.14, 3.14)},
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.commands.hlip_ref.debug_vis = False

        STAIR_CFG.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.1,0.1)
        self.scene.terrain.terrain_generator = STAIR_CFG
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 2
        # self.scene.terrain.terrain_generator.max_init_terrain_level = (.,3.)

        self.commands.base_velocity.ranges.lin_vel_x = (0.5,0.6)
        