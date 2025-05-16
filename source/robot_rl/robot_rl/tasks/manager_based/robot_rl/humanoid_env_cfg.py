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
period = 0.8  # (0.4 s swing phase)


@configclass
class HumanoidActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)


@configclass
class HumanoidObservationsCfg(ObservationsCfg):
    """Observation specifications for the G1 Flat environment."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Observations for policy group."""
        base_lin_vel = None     # Removed - no sensor
        height_scan = None      # Removed - not supported yet

        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),history_length=1,scale=0.05)

        # Phase clock
        sin_phase = ObsTerm(func=mdp.sin_phase, params={"period": period})
        cos_phase = ObsTerm(func=mdp.cos_phase, params={"period": period})

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class HumanoidEventsCfg(EventCfg):
    """Event configuration."""
    # Calculate new step location on a fixed interval
    update_step_location = EventTerm(func=mdp.compute_step_location,
                                    mode="interval",
                                    interval_range_s=(period/2., period/2.),
                                    is_global_time=False,
                                    params={
                                        "nom_height": 0.78,
                                        "Tswing": period/2.,
                                        "command_name": "base_velocity",
                                        "wdes": 0.4,
                                        "feet_bodies": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
                                        })

# @configclass
# class HumanoidCommandCfg(CommandCfg):
#     """Command configuration"""
#     base_velocity = mdp.UniformVelocityCommandCfg(
#         asset_name="robot",
#         resampling_time_range=(10.0, 10.0),
#         rel_standing_envs=0.02,
#         rel_heading_envs=1.0,
#         heading_command=True,
#         heading_control_stiffness=0.5,
#         debug_vis=True,
#         ranges=mdp.UniformVelocityCommandCfg.Ranges(
#             lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
#         ),
#     )

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

    # Track the heading
    track_heading = RewTerm(
        func=mdp.track_heading,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )

    ##
    # Feet
    ##
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": period/2.,
        },
    )
    # TODO: Try removing
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    phase_feet_contacts = RewTerm(
        func=mdp.phase_feet_contacts,
        weight=10,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "period": period,
            "std": 0.2,
            "nom_height": 0.78,
            "Tswing": period/2.,
            "command_name": "base_velocity",
            "wdes": 0.3,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        }
    )

    ##
    # Joint limits
    ##
    # Penalize ankle and knee joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])},
    )

    ##
    # Penalize deviation from default of the joints that are not essential for locomotion
    ##
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
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
    # TODO: Determine if this will work (see docs note)
    # torque_lim = RewTerm(
    #     func=mdp.applied_torque_limits,
    #     weight=0,
    #     params=
    # )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=0,
    )
    joint_reg = RewTerm(
        func=mdp.joint_pos_target,
        weight=0.75,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            # Joint Order:
            # L Hip Pitch
            # L Hip Roll
            # L Hip Yaw
            # L Knee
            # L Ankle Pitch
            # L Ankle Roll
            # R Hip Pitch
            # R Hip Roll
            # R Hip Yaw
            # R Knee
            # R Ankle Pitch
            # R Ankle Roll
            # Waist Yaw
            # L Shoulder Pitch
            # L Shoulder Roll
            # L Shoulder Yaw
            # L Elbow
            # R Shoulder Pitch
            # R Shoulder Roll
            # R Shoulder Yaw
            # R Elbow
            "joint_des": [-0.42, 0, 0, 0.81, -0.4, 0,
                          -0.42, 0, 0, 0.81, -0.4, 0,
                          0,
                          0, 0.27, 0, 0.5,
                          0, -0.27, 0, 0.5,],
            "std": 0.1,
            "joint_weight": [0.0, 10., 10., 0.0, 1., 1.,
                             0.0, 10., 10., 0.0, 1., 1.,
                             1.,
                             1., 1., 1., 1.,
                             1., 1., 1., 1.,],
        }
    )

    feet_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=0.0,
        params={
            "target_height": 0.08,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

# @configclass
# class HumanoidVizCfg(VisualizationMarkersCfg):


##
# Environment configuration
##
@configclass
class HumanoidEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: HumanoidRewardCfg = HumanoidRewardCfg()
    observations: HumanoidObservationsCfg = HumanoidObservationsCfg()
    events: HumanoidEventsCfg = HumanoidEventsCfg()

    # TODO: Is this the right way to do this? How do I reset these?
    # current_des_step: torch.Tensor = torch.zeros(1)
    # control_count: int = 0

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.control_count = 0


    def __prepare_tensors__(self):
        """Move tensors to GPU"""
        self.rewards.joint_reg.params["joint_des"] = torch.tensor(
            self.rewards.joint_reg.params["joint_des"],
            device=self.sim.device
        )

        self.rewards.joint_reg.params["joint_weight"] = torch.tensor(
            self.rewards.joint_reg.params["joint_weight"],
            device=self.sim.device
        )

        self.current_des_step = torch.zeros(self.scene.num_envs, 3, device=self.sim.device)


    def define_markers(self) -> VisualizationMarkers:
        """Define markers with various different shapes."""
        self.footprint_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/footprint",
            markers={
                "foot": sim_utils.CuboidCfg(
                    size=(0.2, 0.065, 0.018),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                # "right": sim_utils.CuboidCfg(
                #     size=(0.2, 0.065, 0.018),
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                # ),
            }
        )
        self.footprint_visualizer = VisualizationMarkers(self.footprint_cfg)

    # def post_physics_step(self):
    #     super().post_physics_step()
    #
    #     # Re-compute the desired foot step location
    #     if ()