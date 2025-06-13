# amber_env_cfg.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg, SceneEntityCfg

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
    EventCfg,
)
from .amber5 import AMBER_CFG
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


@configclass
class AmberActionsCfg:
    """Action space: commanded joint‐position targets for the 4 Amber joints."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["q1_left", "q2_left", "q1_right", "q2_right"],
        scale=1.0,
        use_default_offset=True,
    )

# TODO: Try playing with the period for the lip model
PERIOD = 0.8 #0.6 #0.8  # (0.4 s swing phase)


@configclass
class AmberObservationsCfg(ObservationsCfg):
    """Observations for Amber: for both policy and critic."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # no base linear vel sensor (we only command it, not observe it directly)
        base_lin_vel = None # not avvail
        height_scan = None

        # angular velocity around Y (planar pitch rate; mdp.base_ang_vel returns [wx, wy, wz])
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            history_length=1,
            scale=0.5,
        )

        # the commanded forward velocity (so the policy knows the target)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            history_length=1,
            scale=2.0,
        )

        # joint velocities and joint positions (relative to default stance)
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            history_length=1,
            scale=0.1,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=1,
        )

        # Phase clock
        sin_phase = ObsTerm(func=mdp.sin_phase, params={"period": PERIOD})
        cos_phase = ObsTerm(func=mdp.cos_phase, params={"period": PERIOD})

    @configclass
    class CriticCfg(PolicyCfg):
        # allow critic to also see the actual forward speed
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=1,
            scale=2.0,
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class AmberRewardCfg(RewardsCfg):
    """Keep the default velocity‐tracking rewards, but turn off unwanted terms."""

    # big penalty on fall (pelvis contact)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # track forward velocity (x) – using the base class's XY function is fine since y is always zero
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity",
                        "std": 0.5,      },      # required parameter},
    )

    # no need to track angular yaw (z) or sideways velocity
    track_ang_vel_z = None

    # small alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=0.1)


@configclass
class AmberEventsCfg(EventCfg):
    """You can insert random pushes or mass‐perturbation here if desired."""
    pass


@configclass
class AmberEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Environment config for planar Amber to track forward speed."""

    # plug in our robot asset and all the MDP parts
    # scene: sim_utils.SceneCfg = None  # placeholder to satisfy dataclass
    actions: AmberActionsCfg = AmberActionsCfg()
    observations: AmberObservationsCfg = AmberObservationsCfg()
    rewards: AmberRewardCfg = AmberRewardCfg()
    events: AmberEventsCfg = AmberEventsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = AMBER_CFG.replace(prim_path="{ENV_REGEX_NS}/Amber")


        # self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
        #         "contact_forces",       # the name of the contact‐force sensor manager
        #         body_names=["torso"]     # your link name
        #     )
        # CONTACT FORCES/RESET
        #Scene entity not accepted
        # self.scene.contact_forces = SceneEntityCfg(
        #     "contact_forces", body_names=["torso"]
        # )
        # self.scene.contact_sensors = SceneEntityCfg(
        #     "contact_sensors", body_names=["torso"]
        # )

        # self.terminations.base_contact = TerminationTermCfg(
        # sensor_cfg=SceneEntityCfg("contact_forces", body_names=["torso"])
        # )
        self.scene.contact_forces = None

        self.scene.contact_sensors = None

        # 2) Disable any reward/termination terms that refer to those sensors
        self.terminations.base_contact        = None
        self.rewards.feet_air_time            = None
        self.rewards.undesired_contacts       = None
        # swap in the Amber robot (must match the prim_path used in your scene)
        self.events.add_base_mass = None
        # self.terminations.base_contact.params["sensor_cfg"] = SceneEntityCfg(
        #     "contact_forces",
        #     body_names=["torso"],            # Amber actually has “torso” not “pelvis”
        # )

        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso"]

        # likewise, if you use the default “base_contact” termination:
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["torso"]




    # def define_markers(self) -> VisualizationMarkers:
    # """Define markers with various different shapes."""
    # self.footprint_cfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/footprint",
    #     markers={
    #         "des_foot": sim_utils.CuboidCfg(
    #             size=(0.2, 0.065, 0.018),
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #         ),
    #         # "stance_foot": sim_utils.CuboidCfg(
    #         #     size=(0.2, 0.065, 0.018),
    #         #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    #         # ),
    #     }
    # )
    #     self.footprint_visualizer = VisualizationMarkers(self.footprint_cfg)