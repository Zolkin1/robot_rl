# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg, ObservationsCfg, EventCfg   #Inherit from the base envs
import isaaclab.sim as sim_utils
from robot_rl.tasks.manager_based.robot_rl.mdp.cmd_cfg import HLIPCommandCfg

from . import mdp

##
# Pre-defined configs
##

##
# Scene definition
##


##
# MDP settings
##

# Constants (do this better)
cf = HLIPCommandCfg()
PERIOD = cf.gait_period

@configclass
class HumanoidActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)


@configclass
class HumanoidObservationsCfg(ObservationsCfg):
    """Observation specifications for the G1 Flat environment."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Observations for policy group."""
        base_lin_vel = None     # Removed - no sensor
        height_scan = None      # Removed - not supported yet

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),history_length=1,scale=0.25)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},history_length=1,scale=(2.0,2.0,2.0))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),history_length=1,scale=0.05)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),history_length=1)

        # Phase clock
        sin_phase = ObsTerm(func=mdp.sin_phase, params={"period": PERIOD})
        cos_phase = ObsTerm(func=mdp.cos_phase, params={"period": PERIOD})

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1),history_length=1,scale=2.0)
        height_scan = None      # Removed - not supported yet


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class HumanoidEventsCfg(EventCfg):
    """Event configuration."""
    randomize_ground_contact_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "static_friction_range": (0.1, 1.25),
            "dynamic_friction_range": (0.1, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,  # ensures dynamic friction <= static friction
        },
    )

@configclass
class HumanoidRewardCfg(RewardsCfg):
    """Reward terms for the MDP."""

    ##
    # Termination
    ###
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    ##
    # Tracking
    ##
    # track_lin_vel_xy_exp = RewTerm(
    #     func=mdp.track_lin_vel_xy_yaw_frame_exp,
    #     weight=1.0,
    #     params={"command_name": "base_velocity", "std": 0.5},
    # )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    # )

    # # Track the heading
    # track_heading = RewTerm(
    #     func=mdp.track_heading,
    #     weight=1.0,
    #     params={"command_name": "base_velocity", "std": 0.2},
    # )

    ##
    # Feet
    ##
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": PERIOD / 2.,
        },
    )

    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.3,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #     },
    # )

    # phase_feet_contacts = RewTerm(
    #     func=mdp.phase_feet_contacts,
    #     weight=0, #10
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "period": period,
    #         "std": 0.2,
    #         "nom_height": 0.78,
    #         "Tswing": period/2.,
    #         "command_name": "base_velocity",
    #         "wdes": 0.3,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #     }
    # )

    phase_contact = RewTerm(
        func=mdp.phase_contact,
        weight=0.18,
        params={
            # "command_name": "base_velocity",
            "Tswing": PERIOD/2.,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )

    ##
    # Joint limits
    ##
    # Penalize ankle and knee joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        # params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )

    ##
    # Penalize deviation from default of the joints that are not essential for locomotion
    ##
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1, # TODO: Move to l2
        weight=0, #-0.2,
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
# Environment configuration
##
@configclass
class HumanoidEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: HumanoidRewardCfg = HumanoidRewardCfg()
    observations: HumanoidObservationsCfg = HumanoidObservationsCfg()
    events: HumanoidEventsCfg = HumanoidEventsCfg()
    actions: HumanoidActionsCfg = HumanoidActionsCfg()

    # TODO: Is this the right way to do this? How do I reset these?
    # current_des_step: torch.Tensor = torch.zeros(1)
    # control_count: int = 0

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # self.control_count = 0


    # def __prepare_tensors__(self):
    #     """Move tensors to GPU"""
    #     # self.rewards.joint_reg.params["joint_des"] = torch.tensor(
    #     #     self.rewards.joint_reg.params["joint_des"],
    #     #     device=self.sim.device
    #     # )
    #     #
    #     # self.rewards.joint_reg.params["joint_weight"] = torch.tensor(
    #     #     self.rewards.joint_reg.params["joint_weight"],
    #     #     device=self.sim.device
    #     # )

    #     self.current_des_step = torch.zeros(self.scene.num_envs, 3, device=self.sim.device)
    #     self.com_lin_vel_avg = torch.zeros(self.scene.num_envs, 3, device=self.sim.device)


    # def define_markers(self) -> VisualizationMarkers:
    #     """Define markers with various different shapes."""
    #     self.footprint_cfg = VisualizationMarkersCfg(
    #         prim_path="/Visuals/footprint",
    #         markers={
    #             "des_foot": sim_utils.CuboidCfg(
    #                 size=(0.2, 0.065, 0.018),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #             ),
    #             # "stance_foot": sim_utils.CuboidCfg(
    #             #     size=(0.2, 0.065, 0.018),
    #             #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    #             # ),
    #         }
    #     )
    #     self.footprint_visualizer = VisualizationMarkers(self.footprint_cfg)

    # # def post_physics_step(self):
    #     super().post_physics_step()
    #
    #     # Re-compute the desired foot step location
    #     if ()