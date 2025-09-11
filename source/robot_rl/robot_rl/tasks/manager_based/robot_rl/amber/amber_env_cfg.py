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
# from isaaclab.managers.reset_manager import ResetCallback
from isaaclab.assets import AssetBaseCfg
import robot_rl.tasks.manager_based.robot_rl.amber.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm


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
# from isaaclab.sim import schemas, schemas_cfg
# from pxr import UsdPhysics             # need this to apply the API
# import omni.usd

# def set_torso_mass(mass_kg: float, env_id: int = 0):
#     stage      = omni.usd.get_context().get_stage()
#     torso_path = f"/World/envs/env_{env_id}/Amber/amber3_PF/torso"

#     # Ensure MassAPI exists; modify_mass_properties() returns False otherwise
#     prim = stage.GetPrimAtPath(torso_path)
#     if not UsdPhysics.MassAPI(prim):
#         UsdPhysics.MassAPI.Apply(prim)                          # :contentReference[oaicite:2]{index=2}

#     cfg = schemas_cfg.MassPropertiesCfg(mass=mass_kg)           # only this attr is set :contentReference[oaicite:3]{index=3}
#     schemas.modify_mass_properties(torso_path, cfg, stage)      # returns True on success :contentReference[oaicite:4]{index=4}

@configclass
class AmberActionsCfg:
    """Action space: commanded joint‐position targets for the 4 Amber joints."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["q1_left", "q2_left", "q1_right", "q2_right"],
        scale=1.0,
    )


# TODO: Try playing with the period for the lip model
PERIOD = 0.8 #0.6 #0.8  # (0.4 s swing phase)
WDES=0.0


@configclass
class CommandsCfg:
    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=0.0,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1, -1), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0., 0.)
    #     ),
    # )
    base_velocity = mdp.LoopVelocityCommandCfg(
        asset_name="robot",
        sequence=[-0.1, -0.5, -1.0, -1.2, -0.7],    # your custom cycle
        resampling_time_range=(5.0, 5.0),          # 3 seconds per value
        debug_vis=True,
    )

@configclass
class AmberObservationsCfg(ObservationsCfg):
    """Observations for Amber: for both policy and critic."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # no base linear vel sensor (we only command it, not observe it directly)
        base_lin_vel = None # not avvail
        height_scan = None

        # future_feet = ObsTerm(
        #     func            = mdp.future_foot_targets_lip,
        #     history_length  = 1,
        #     noise           = None,        # or Unoise() if you like
        #     scale           = 1.0,         # optional scaling
        # )
        # base_ang_vel= None
        # angular velocity around Y (planar pitch rate; mdp.base_ang_vel returns [wx, wy, wz])
        base_ang_vel = ObsTerm(
            func  = mdp.base_ang_vel_amber,
            noise = Unoise(n_min=-0.2, n_max=0.2),
            history_length = 1,
            scale = 1.0,          # adjust if you need re-scaling
        )
        projected_gravity = ObsTerm(
            func  = mdp.projected_gravity_amber,
            noise = Unoise(n_min=-0.05, n_max=0.05),
            history_length = 1,
            scale = 1.0,         # adjust if you need scaling
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
            func=mdp.base_lin_vel_amber,
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
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)


    ## -------------Functions for plotting rewards:
    curr_vel = RewTerm(
        func=mdp.curr_vel_fn,
        weight=1e-10,
        params={"command_name": "base_velocity",
                        "std": 0.4,      },      # required parameter},
    )
    cmd_vel_given = RewTerm(
        func=mdp.cmd_vel_fn,
        weight=1e-10,
        params={"command_name": "base_velocity",
                        "std": 0.4,      },      # required parameter},
    )



    ## Reward functions:
    # track forward velocity (x) – using the base class's XY function is fine since y is always zero
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_x_amber,
        weight=120,
        params={"command_name": "base_velocity",
                        "std": 0.4,      },      # required parameter},
    )
    # punishes joint angles
    joint_angles = RewTerm(
        func=mdp.track_joint_angles_exp,
        weight=-6.0,           # <— tune this scalar to control strength
        params={"std": 0.5,
                "threshold_deg":2,
                },  # <— how “wide” your kernel is
    )
    joint_symmetry_reward = RewTerm(
        func   = mdp.joint_symmetry_cycle,
        weight = 5.0,                 # tune to blend with other rewards
        params = {
            "Ts":              PERIOD/2,   # seconds per step
            "std":             0.15,  # width for exp decay
            "diff_threshold":  0.2,  # radians tolerance before penalty
            "reward_good":     2.0,
            "penalty_slope":   2.5,   # strength of penalty beyond threshold
            "debug":           False,
        },
    )
    stride_consistency = RewTerm(
        func   = mdp.stride_consistency_penalty,
        weight = 10.0,                 # scale relative to your other terms
        params = {
            "left_sensor_name":  "contact_forces_left",
            "right_sensor_name": "contact_forces_right",
            "force_thresh":      1.0,
            "max_penalty":       0.8,   # metres
            "debug":             True,
        },
    )
    # torso angle
    torso_orientation = RewTerm(
        func   = mdp.torso_rotation_term,
        weight = 2.0,
        params = {
            "asset_cfg":             SceneEntityCfg("robot", body_names=["torso"]),
            "reward_window_deg":     7.0,
            "penalty_threshold_deg": 15.0,
            "penalty_cap":           20.0,
        },
    )
    #rewards stance swing phase
    phase_contact = RewTerm(
        func   = mdp.foot_phase_contact_amber,
        weight = 10.0,   # positive weight rewards in‐phase contacts
        params = {
            "period":             0.8,
            "command_name":       "base_velocity",
            "left_sensor_name":   "contact_forces_left",
            "right_sensor_name":  "contact_forces_right",
            "force_thresh":       1.0,
            "cmd_thresh":         0.005,
        },
    )
    phase_contact_per_cycle = RewTerm(
        func   = mdp.foot_phase_cycle_reward,
        weight = 10.0,      # positive; function already returns ± values
        params = {
            "period":             0.8,
            "command_name":       "base_velocity",
            "left_sensor_name":   "contact_forces_left",
            "right_sensor_name":  "contact_forces_right",
            "force_thresh":       1.0,
            "cmd_thresh":         0.05,
        },
    )
    # rewards consecutive foot placements
    alternation_contact = RewTerm(
        func   = mdp.alternation_contact_reward,
        weight = 20.0,    # positive to reward alternation, negative for repeats
        params = {
            "left_sensor_name":  "contact_forces_left",
            "right_sensor_name": "contact_forces_right",
        },
    )
    # rewards progressive foots
    progressive_step = RewTerm(
        func   = mdp.alternative_linear_cycle,
        weight = 6,
        params = {
            "command_name":      "base_velocity",
            "left_sensor_name":  "contact_forces_left",
            "right_sensor_name": "contact_forces_right",
            "min_cmd_speed":     0.05,
            "penalty_bad":       -5.0,
            "max_step":       0.2,
        },
    )
    # punishes more than 1 reward per cycle:
    foot_cycle_sym = RewTerm(
        func   = mdp.foot_contact_cycle_reward,
        weight = 10.0,    # positive: reward per-cycle correctness, penalty built into func
        params = {
            "period":           PERIOD,
            "left_sensor_name": "contact_forces_left",
            "right_sensor_name":"contact_forces_right",
            "contact_thresh":    1,
            "expo_rate":         0.5,   # tune how steeply penalties grow
        },
    )
    # foot_clearance = RewTerm(
    #     func=mdp.foot_clearance_amber,
    #     weight=1.0,   # tune so its scale matches your other rewards
    #     params={
    #         "target_height": 0.1,            # desired clearance (m)
    #         "left_sensor_name":  "contact_forces_left",
    #         "right_sensor_name": "contact_forces_right",
    #     },
    # )
    symmetric_foot_airtime = RewTerm(
        func   = mdp.foot_air_time_symmetry,
        weight = -1.0,   # negative → penalize asymmetry
        params = {
            "period":            0.8,
            "left_sensor_name":  "contact_forces_left",
            "right_sensor_name": "contact_forces_right",
            "contact_thresh":     1,
            "diff_threshold":     3,
            "reward_good":     2,
        },
    )
    feet_no_slip_condition = RewTerm(
        func   = mdp.contact_no_vel_amber,
        weight = -15,                 # negative → subtracts squared-speed cost
        params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_shin", "right_shin"]
            ),
            "left_sensor_name":  "contact_forces_left",
            "right_sensor_name": "contact_forces_right",
        },
    )
    com_step_reward = RewTerm(
        func   = mdp.contact_step_com_reward,
        weight = 10.0,                 # tune to match the scale of your other terms
        params = {
            "command_name":     "base_velocity",
            "left_sensor_name":  "contact_forces_left",
            "right_sensor_name": "contact_forces_right",
            "force_thresh":      1.0,      # contact detection threshold (N)
            "max_reward_dist":   0.30,     # clamp distance in metres
            "min_cmd_speed":     0.05,     # below this |cmd_x| treat as no-motion
            "debug":             False,    # True to print each landing event
        },
    )
    # penalises contact acc to time exponentially
    paper_cycle_reward = RewTerm(
        func   = mdp.rcs_phase_reward_no_pos,
        weight = 1.0,          # keep at 1.0 (formula already has 9×)
        params = {
            "Ts":                PERIOD/2,
            "left_sensor_name":  "contact_forces_left",
            "right_sensor_name": "contact_forces_right",
            "force_thresh":      1.0,
            "debug":             False,
        },
    )
    
    # no need to track angular yaw (z) or sideways velocity
    track_ang_vel_z = None
    # small alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=0.1)


@configclass
class AmberEventsCfg(EventCfg):
    """You can insert random pushes or mass‐perturbation here if desired."""
    # print_foot_positions = EventTerm(
    #     func=mdp.print_foot_positions,
    #     mode="interval",
    #     interval_range_s=(PERIOD/2., PERIOD/2.),
    #     is_global_time=False,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )
@configclass
class AmberEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Environment config for planar Amber to track forward speed."""

    # plug in our robot asset and all the MDP parts
    # scene: sim_utils.SceneCfg = None  # placeholder to satisfy dataclass
    actions: AmberActionsCfg = AmberActionsCfg()
    observations: AmberObservationsCfg = AmberObservationsCfg()
    rewards: AmberRewardCfg = AmberRewardCfg()
    events: AmberEventsCfg = AmberEventsCfg()
    commands: CommandsCfg =CommandsCfg()
    def __post_init__(self):
        super().__post_init__()
        # set_torso_mass(10)
        ## uncomment the below code when commenting CommandsCfg
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1) # 0 - 1
        # self.commands.base_velocity.ranges.lin_vel_y = (0,0) #(-1.0, 1.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0, 0)
