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

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg, ObservationsCfg    #Inherit from the base envs

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
    leg_joint_reg = RewTerm(
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
            "joint_weight": [1., 1., 1., 1., 1., 1.,
                             1., 1., 1., 1., 1., 1.,
                             1.,
                             1., 1., 1., 1.,
                             1., 1., 1., 1.,],
        }
    )


##
# Environment configuration
##
@configclass
class HumanoidEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: HumanoidRewardCfg = HumanoidRewardCfg()
    observations: HumanoidObservationsCfg = HumanoidObservationsCfg()

    # TODO: How to load in a custom G1 model?

    def __post_init__(self):
        # post init of parent
        super().__post_init__()


    def __prepare_tensors__(self):
        """Move tesnors to GPU"""
        self.rewards.leg_joint_reg.params["joint_des"] = torch.tensor(
            self.rewards.leg_joint_reg.params["joint_des"],
            device=self.sim.device
        )

        self.rewards.leg_joint_reg.params["joint_weight"] = torch.tensor(
            self.rewards.leg_joint_reg.params["joint_weight"],
            device=self.sim.device
        )