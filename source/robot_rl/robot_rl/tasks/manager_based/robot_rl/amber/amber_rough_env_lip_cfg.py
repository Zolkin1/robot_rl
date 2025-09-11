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

from robot_rl.tasks.manager_based.robot_rl.amber import mdp
from .amber_rough_env_cfg import AmberRoughEnvCfg
from robot_rl.tasks.manager_based.robot_rl.amber.amber_env_cfg import (
    PERIOD,
    AmberEventsCfg,
    WDES,
    AmberObservationsCfg,
    AmberRewardCfg
    )
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
    EventCfg,
)
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
Nom_ht= 0.77
half_cycle= True
visualise_flag= False
##
# Pre-defined configs
##
@configclass
class AmberRoughLipObsCfg(AmberObservationsCfg):
    @configclass
    class PolicyCfg(AmberObservationsCfg.PolicyCfg):
        future_feet = ObsTerm(
            func    = mdp.desired_foot_targets_obs,
            history_length = 1,
            scale   = 1.0,
            params  = {
                "Ts":           PERIOD/2.0,
                "nom_height":   Nom_ht,
                "wdes":         0,
                "command_name": "base_velocity",
                "asset_cfg":    SceneEntityCfg("robot",body_names=["left_shin","right_shin"],),
                "debug":        False,
                "visualize":    False,
                "half_cycle":   half_cycle,
            },
        )
    
        # current_feet = ObsTerm(
        #     func           = mdp.current_foot_positions,
        #     history_length = 1,
        #     noise          = Unoise(n_min=-0.01, n_max=0.01),
        #     scale          = 1.0,   # no extra scaling
        # )
        # velocity_commands=None
    @configclass
    class CriticCfg(AmberObservationsCfg.CriticCfg):
        # inherit everything the base critic had...
        future_feet = ObsTerm(
            func    = mdp.desired_foot_targets_obs,
            history_length = 1,
            scale   = 1.0,
            params  = {
                "Ts":           PERIOD/2.0,
                "nom_height":   Nom_ht,
                "wdes":         0,
                "command_name": "base_velocity",
                "asset_cfg":    SceneEntityCfg("robot",body_names=["left_shin","right_shin"],),
                "debug":        False,
                "visualize":False,
                "half_cycle":   half_cycle,
            },
        )
        # current_feet = ObsTerm(
        #     func=mdp.current_foot_positions,
        #     history_length=1,
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        #     scale=1.0,
        # )
        # velocity_commands=None
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

##
# LIP Specific Constants
##

##
# Lip specific rewards
##
class AmberRoughLipRewards(AmberRewardCfg):
    """Rewards specific to LIP Model"""
    # lip_gait_tracking = RewTerm(
    #     func=mdp.lip_gait_tracking,
    #     weight=0.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "period": PERIOD,
    #         "std": 0.2,
    #         "nom_height": 0.78,
    #         "Tswing": PERIOD/2.,
    #         "command_name": "base_velocity",
    #         "wdes": WDES,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #     }
    # )
    rcs_phase_with_place = RewTerm(
        func=mdp.rcs_phase_reward_with_placement,
        weight=40.0,
        params={
            "Ts":                PERIOD/2,
            "left_sensor_name":  "contact_forces_left",
            "right_sensor_name": "contact_forces_right",
            "force_thresh":      1.0,
            "sigma":             0.05,
            "asset_cfg":         SceneEntityCfg("robot",body_names=["left_shin","right_shin"],),
            "debug":             False,
            "visualise":         visualise_flag,
            "half_cycle":   half_cycle,
        },
    )

    #PLOTTING REWARDS
    left_target = RewTerm(
        func    = mdp.left_foot_target_reward,
        weight  = 1e-10,           # tune as you like
        params  = {
            "Ts":            PERIOD/2.0,
            "nom_height":    1.36,
            "wdes":          0.0,
            "command_name":  "base_velocity",
            "asset_cfg":     SceneEntityCfg("robot", body_names=["left_shin","right_shin"]),
            "debug":         False,
            "visualize":     False,  # or True if you want spheres
            "half_cycle":    half_cycle,
        },
    )
    right_target = RewTerm(
        func    = mdp.right_foot_target_reward,
        weight  = 1e-10,           # tune as you like
        params  = {
            "Ts":            PERIOD/2.0,
            "nom_height":    1.36,
            "wdes":          0.0,
            "command_name":  "base_velocity",
            "asset_cfg":     SceneEntityCfg("robot", body_names=["left_shin","right_shin"]),
            "debug":         False,
            "visualize":     False,  # or True if you want spheres
            "half_cycle":    half_cycle,
        },
    )
    curr_x_L = RewTerm(
        func   = mdp.current_left_foot_x,
        weight  = 1e-10,  # or however you want to normalize
        params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_shin","right_shin"])
        }
    )
    curr_z_L = RewTerm(
        func   = mdp.current_left_foot_z,
        weight  = 1e-10,  # or however you want to normalize
        params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_shin","right_shin"])
        }
    )
    curr_z_R = RewTerm(
        func   = mdp.current_right_foot_z,
        weight  = 1e-10,  # or however you want to normalize
        params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_shin","right_shin"])
        }
    )
    curr_x_R = RewTerm(
        func   = mdp.current_right_foot_x,
        weight  = 1e-10,
        params = {
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_shin","right_shin"])
        }
    )
    # future_feet = RewTerm(
    #         func    = mdp.desired_foot_targets_obs,
    #         weight=20.0,
    #         params  = {
    #             "Ts":           PERIOD/2.0,
    #             "nom_height":   1.36,
    #             "wdes":         0,
    #             "command_name": "base_velocity",
    #             "asset_cfg":    SceneEntityCfg("robot",body_names=["left_shin","right_shin"],),
    #             "debug":        False,
    #             "visualize":True
    #         },
    #     )
    # lip_feet_tracking = RewTerm(
    #     func=mdp.lip_feet_tracking,
    #     weight=10.0,
    #     params={
    #         "period": PERIOD,
    #         "std": 0.2,
    #         "Tswing": PERIOD/2.,
    #         "feet_bodies": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #     }
    # )




@configclass
class AmberRoughLipEventsCfg(AmberEventsCfg):
    # Calculate new step location on a fixed interval
    # update_step_location = EventTerm(func=mdp.compute_step_location_local_amber,
    #                                  mode="interval",
    #                                  interval_range_s=(PERIOD / 2., PERIOD / 2.),
    #                                  is_global_time=False,
    #                                  params={
    #                                      "nom_height": 1.3,
    #                                      "Tswing": PERIOD / 2.,
    #                                      "command_name": "base_velocity",
    #                                      "wdes": WDES,
    #                                      "feet_bodies"  : SceneEntityCfg(
    #                                         "robot",
    #                                         body_names=["left_shin","right_shin"]
    #                                     ),
    #                                  })
    
    LIP_plotting = EventTerm(
        func=mdp.debug_plot_and_print,
        mode="step",
        params={
            "Ts":            PERIOD/2.0,
            "nom_height":    1.36,
            "wdes":          0.0,
            "command_name":  "base_velocity",
            "asset_cfg":     SceneEntityCfg("robot", body_names=["left_shin","right_shin"]),
            "debug":         True,
            "visualize":     True,
        },
    )
    # Do on reset
    # reset_update_set_location = EventTerm(func=mdp.compute_step_location_local,
    #                                       mode="reset",
    #                                       params={
    #                                           "nom_height": 0.78,
    #                                           "Tswing": PERIOD / 2.,
    #                                           "command_name": "base_velocity",
    #                                           "wdes": WDES,
    #                                           "feet_bodies": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #                                       })
##
# Environment configuration
##
@configclass
class AmberRoughLipEnvCfg(AmberRoughEnvCfg):
    """Configuration for the Amber Flat environment."""
    # events: AmberRoughLipEventsCfg = AmberRoughLipEventsCfg()
    observations: AmberRoughLipObsCfg = AmberRoughLipObsCfg()
    rewards : AmberRoughLipRewards = AmberRoughLipRewards()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()


        # ##
        # # Scene
        # ##
        # self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis_link"

        # # No height scanner for now
        # self.scene.height_scanner = None


        # ##
        # # Randomization
        # ##
        # # self.events.push_robot = None
        # self.events.push_robot.params["velocity_range"] = {"x": (-1, 1), "y": (-1, 1), "roll": (-0.4, 0.4),
        #                                                    "pitch": (-0.4, 0.4), "yaw": (-0.4, 0.4)}
        # # self.events.push_robot.params["velocity_range"] = {"x": (-0, 0), "y": (-0, 0), "roll": (-0.0, 0.0),
        # #                                                    "pitch": (-0., 0.), "yaw": (-0.0, 0.0)}
        # self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis_link"]
        # self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        # self.events.add_base_mass.params["operation"] = "scale"
        # # self.events.randomize_ground_contact_friction.params["static_friction_range"] = (0.1, 1.25)
        # # self.events.randomize_ground_contact_friction.params["dynamic_friction_range"] = (0.1, 1.25)
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["pelvis_link"]
        # self.events.reset_base.params = {
        #     "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14,3.14)}, #(-3.14, 3.14)},
        #     "velocity_range": {
        #         "x": (0.0, 0.0),
        #         "y": (0.0, 0.0),
        #         "z": (0.0, 0.0),
        #         "roll": (0.0, 0.0),
        #         "pitch": (0.0, 0.0),
        #         "yaw": (0.0, 0.0),
        #     },
        # }

        # ##
        # # Commands
        # ##
        # self.commands.base_velocity.ranges.lin_vel_x = (-1, 1)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ##
        # # Terminations
        # ##
        # self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"
        # # self.terminations.base_contact.params["sensor_cfg"].body_names = ["pelvis_link"]

        # ##
        # # Rewards
        # ##
        # self.rewards.track_lin_vel_xy_exp.weight = 5.0 #1
        # self.rewards.track_ang_vel_z_exp.weight = 0.5
        # self.rewards.lin_vel_z_l2.weight =  -2.0 # TODO reduce this maybe?
        # self.rewards.ang_vel_xy_l2.weight = -0.05
        # self.rewards.dof_torques_l2.weight = -1.0e-5
        # self.rewards.dof_acc_l2.weight = -2.5e-7
        # self.rewards.dof_vel_l2.weight = -1.0e-3
        # self.rewards.action_rate_l2.weight = -0.01
        # self.rewards.feet_air_time.weight = 0.0
        # self.rewards.flat_orientation_l2.weight = -1.0
        # self.rewards.dof_pos_limits.weight = -5.0
        # self.rewards.alive.weight = 0.15
        # self.rewards.contact_no_vel.weight = -0.2
        # self.rewards.joint_deviation_hip.weight = -1.0
        # self.rewards.height_torso.weight = -20 #-10.0
        # self.rewards.feet_clearance.weight = -20.0
        # self.rewards.phase_contact.weight = 0 #0.25

        # # TODO: Add the footstep location rewards
        # self.rewards.lip_gait_tracking.weight = 2
        # self.rewards.lip_feet_tracking.weight = 3 #10.0

        # self.rewards.joint_deviation_arms.weight = -0.5             # Arms regularization
        # self.rewards.joint_deviation_torso.weight = -1.0

        # self.rewards.height_torso.params["target_height"] = 0.75
        # self.rewards.feet_clearance.params["target_height"] = 0.12