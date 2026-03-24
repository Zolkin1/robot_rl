# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from robot_rl.assets.robots.double_integrator import DOUBLE_INTEGRATOR_CFG
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_cmd_cfg import TrajectoryCommandCfg

from . import mdp


##
# Scene
##
@configclass
class DISceneCfg(InteractiveSceneCfg):
    """Scene configuration for the double integrator."""

    robot: ArticulationCfg = DOUBLE_INTEGRATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ground_plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##
@configclass
class DIActionsCfg:
    """Action specifications for the MDP."""

    force = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["slide_joint"],
        scale=1.0,
    )


@configclass
class DIObservationsCfg:
    """Observation specifications for the double integrator."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        position = ObsTerm(func=mdp.di_position)
        velocity = ObsTerm(func=mdp.di_velocity)
        target_position = ObsTerm(func=mdp.di_target_position)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        position = ObsTerm(func=mdp.di_position)
        velocity = ObsTerm(func=mdp.di_velocity)
        target_position = ObsTerm(func=mdp.di_target_position)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class DITerminationCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_of_bounds = DoneTerm(
        func=mdp.di_out_of_bounds,
        params={"max_distance": 10000.0},
    )


@configclass
class DIEventsCfg:
    """Event configuration."""

    reset_scene_to_default = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_joint_state = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-5.0, 5.0),
            "velocity_range": (-2.0, 2.0),
        },
    )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="ball"),
    #         "mass_distribution_params": (5.0, 5.0),
    #         "operation": "add",
    #     },
    # )

DI_Q_weights = {}
DI_Q_weights["joint:slide_joint"] = [1.0, 1.0]

DI_R_weights = {}
DI_R_weights["joint:slide_joint"] = [1.0]

@configclass
class DICommandCfg:
    """Command configuration."""
    # Needed for the CLF
    traj_ref = TrajectoryCommandCfg(
        contact_bodies=["ball"],
        manager_type="trajectory",
        hf_repo="zolkin/robot_rl",
        path="trajectories/double_integrator/double_integrator_origin.yaml",
        num_outputs=1,
        Q_weights=DI_Q_weights,
        R_weights=DI_R_weights,
        hold_phi_threshold=0.0,
        heuristic_func=None,
    )

@configclass
class DIRewardCfg:
    """Reward terms for the MDP. (placeholder — rewards to be added later)"""
    # clf_reward = None
    clf_reward = RewTerm(
        func=mdp.clf_reward,
        weight=10.0,
        params={
            "command_name": "traj_ref",
            "max_eta_err": 0.5,
        }
    )

    clf_decreasing_condition = RewTerm(
        func=mdp.clf_decreasing_condition,
        weight=-5.0,
        params={
            "command_name": "traj_ref",
            "alpha": 0.02, #0.5,
            "eta_max": 0.5,
            "eta_dot_max": 0.75,
        }
    )


##
# Environment configuration
##
@configclass
class DIEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the double integrator environment."""

    scene: DISceneCfg = DISceneCfg(num_envs=4096, env_spacing=2.5)
    observations: DIObservationsCfg = DIObservationsCfg()
    actions: DIActionsCfg = DIActionsCfg()
    commands: DICommandCfg = DICommandCfg()
    rewards: DIRewardCfg = DIRewardCfg()
    terminations: DITerminationCfg = DITerminationCfg()
    events: DIEventsCfg = DIEventsCfg()
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 4.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
