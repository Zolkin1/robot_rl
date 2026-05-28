import math
from dataclasses import MISSING

##
# Isaac Lab
##
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.scene import InteractiveSceneCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab_tasks.utils import PresetCfg
from isaaclab_newton.sensors import ContactSensorCfg as NewtonContactSensorCfg
from isaaclab_physx.sensors import ContactSensorCfg as PhysXContactSensorCfg

from robot_rl.sensors import RecursiveContactSensorCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Robot RL
##
from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_cmd_cfg import TrajectoryCommandCfg
from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.assets.robots import G1_ACTION_SCALE
from robot_rl.assets.robots.g1_21j import G1_CFG  # isort: skip

from .clf_weights import WALKING_Q_weights, WALKING_R_weights
from ..mdp.commands.velocity_commands_cfg import VelocityTrackingCommandCfg

##
# Actions
##
@configclass
class G1ClfTrackingActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=G1_ACTION_SCALE, use_default_offset=True)


@configclass
class ClfTrackingBaseEnvContactSensorCfg(PresetCfg):

    def __init__(self, prim_path):
        self.default = PhysXContactSensorCfg(prim_path=prim_path, history_length=3,
                                        track_air_time=True)
        self.newton = NewtonContactSensorCfg(prim_path=prim_path, history_length=3,
                                        track_air_time=True)
        self.physx = self.default

@configclass
class G1ClfTrackingSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = MISSING

    # # sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )

    left_thigh_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/left_hip_pitch_link/.*")
    right_thigh_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/right_hip_pitch_link/.*")
    # Match the ``waist_yaw_link`` rigid body itself (the torso box collider),
    # NOT ``.../waist_yaw_link/.*`` — the trailing ``/.*`` matches only the
    # *descendants* of waist_yaw_link (the shoulder→elbow→wrist arm chain),
    # so it never saw a torso contact (a faceplant loads waist_yaw_link).
    torso_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/waist_yaw_link")
    left_elbow_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/waist_yaw_link/left_shoulder_pitch_link/left_shoulder_roll_link/left_shoulder_yaw_link/.*")
    right_elbow_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/waist_yaw_link/right_shoulder_pitch_link/right_shoulder_roll_link/right_shoulder_yaw_link/.*")
    left_foot_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/left_hip_pitch_link/left_hip_roll_link/left_hip_yaw_link/left_knee_link/left_ankle_pitch_link/.*")
    right_foot_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/right_hip_pitch_link/right_hip_roll_link/right_hip_yaw_link/right_knee_link/right_ankle_pitch_link/.*")
    # Shin == the ``knee_link`` rigid body (the shank between the knee joint
    # and the ankle).  Match the link itself (no trailing ``/.*``) so the
    # sensor sees only the shin and not its foot descendants — used for a
    # dedicated shin-contact termination on the stairs.
    left_shin_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/left_hip_pitch_link/left_hip_roll_link/left_hip_yaw_link/left_knee_link")
    right_shin_contact = ClfTrackingBaseEnvContactSensorCfg("{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/right_hip_pitch_link/right_hip_roll_link/right_hip_yaw_link/right_knee_link")

    #RecursiveContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/Geometry/.*", history_length=3, track_air_time=True)

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

##
# Commands
##
@configclass
class G1ClfTrackingCommandCfg:
    # TODO: Migrate to BatchedMultiSkillCommandCfg.  Single-skill
    # TrajectoryCommandCfg is slated for deletion; see
    # ``g1_clf_multiskill_base.G1MultiSkillCommandsCfg`` for the
    # equivalent multi-skill setup (folder of skill subfolders + skill
    # conditioner instead of ``manager_type="library"``).  Sin/cos phase
    # observations need a multi-skill replacement as well — see
    # ``observations._phased_cmd``.
    traj_ref = TrajectoryCommandCfg(
        contact_bodies = [".*_ankle_roll_link"],

        manager_type="library",
        hf_repo = "zolkin/robot_rl",
        path = "trajectories/walking",

        conditioner_generator_name = "base_velocity",
        Q_weights = WALKING_Q_weights,
        R_weights = WALKING_R_weights,
        hold_phi_threshold = 0.1,
        heuristic_func=None,
        phasing_boundaries=4,
    )

    base_velocity = VelocityTrackingCommandCfg(
        asset_name="robot",
        resampling_time_range=(7.0, 10.0), #(10.0, 10.0),
        rel_standing_envs=0.05, #0.05, #0.02,
        rel_closed_loop=0.5, #0.55,
        rel_closed_loop_yaw=0.25,
        rel_open_loop=0.2,
        debug_vis=False,
        ranges=VelocityTrackingCommandCfg.VelRanges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
            y_pos_offset=(-0.5, 0.5),
            y_kp=(1.2, 1.8),
            y_kd=(0.2, 0.4),
        ))


##
# Observations
##
@configclass
class G1ClfTrackingObservationsCfg:
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
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), scale=0.05)
        actions = ObsTerm(func=mdp.last_action, clip=(-20.0,20.0))

        # TODO: Move everything to the multiskill phase as its general and can even work for single skills
        # Phase clock
        sin_phase = ObsTerm(func=mdp.ref_sin_phase, params={"command_name": "traj_ref"})
        cos_phase = ObsTerm(func=mdp.ref_cos_phase, params={"command_name": "traj_ref"})


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0)
        root_quat = ObsTerm(func=mdp.root_quat_w, scale=1.0)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},
                                    scale=(2.0, 2.0, 2.0))

        joint_pos = ObsTerm(func=mdp.joint_pos_rel,)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action, clip=(-20.0,20.0))

        # Phase clock
        sin_phase = ObsTerm(func=mdp.ref_sin_phase, params={"command_name": "traj_ref"})
        cos_phase = ObsTerm(func=mdp.ref_cos_phase, params={"command_name": "traj_ref"})

        ref_traj = ObsTerm(func=mdp.ref_traj, params={"command_name": "traj_ref"})
        act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "traj_ref"})
        ref_traj_vel = ObsTerm(func=mdp.ref_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))
        act_traj_vel = ObsTerm(func=mdp.act_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))

        height_scan = None  # Removed - not supported yet

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class G1ClfTrackingEventsCfg:
    # pass
    reset_on_ref = EventTerm(
        func=mdp.reset_on_reference,
        mode="reset",
        params={"command_name": "traj_ref",
                "base_frame_name": "pelvis_link",
                "conditioner_command_name": "base_velocity",
                "special_val": 0.0,     # Sometimes start on the running traj
                "rel_envs_on_special": 0.0,
                "rel_envs_on_ref": 1.0, #0.5,
                "joint_add_range": [0.0, 0.0]} #[-0.1, 0.1]}
    )

    # As of Isaac v3.0.0 beta there is no way to randomize dynamic and viscous friction that I can see.
    # Randomize joint friction
    joint_friction_params = EventTerm(
        func=mdp.randomize_joint_parameters_multi_friction,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "static_friction_distribution_params": (0.3, 1.6),
                "dynamic_friction_distribution_params": (0.3, 1.2),
                "viscous_friction_distribution_params": (0.01, 0.1),
                "operation": "add"},
    )

    # Randomize armature
    joint_armature_params = EventTerm(
        func=mdp.randomize_joint_parameters_multi_friction,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "armature_distribution_params": (0.95, 1.05),
                "operation": "scale"},
    )

    # PD Gain randomization
    gain_randomization = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform"
        },
    )

    # Adjust torso mass
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
            "mass_distribution_params": (0.85, 1.15),  # (-5.0, 5.0),
            "operation": "scale",
        },
    )

    # Randomize calibration errors
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    randomize_ground_contact_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.2),  # Consider upping this
            "num_buckets": 64,
            "make_consistent": True,  # ensures dynamic friction <= static friction
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.75, 0.75), "y": (-0.75, 0.75)}},
    )

@configclass
class G1ClfTrackingRewardCfg:
    ##
    # Joint limits
    ##
    # Penalize ankle and knee joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
    )

    ##
    # Torque
    ##
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)

    torque_lims = RewTerm(
        func=mdp.torque_limits,
        weight=-1.0,
    )

    ##
    # Undesired Contacts
    ##
    undesired_contacts = RewTerm(
        func=mdp.multiple_undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfgs": [SceneEntityCfg("left_thigh_contact",),
                            SceneEntityCfg("right_thigh_contact",),
                            SceneEntityCfg("torso_contact",),
                            SceneEntityCfg("left_elbow_contact",),
                            SceneEntityCfg("right_elbow_contact",),],
            "threshold": 1.0,
        },
    )

    # Shin (knee_link) contacts — split out from ``undesired_contacts`` so the
    # weight can be tuned independently for stair scenarios where the shin
    # banging the riser is a particularly bad failure mode.  Uses the
    # dedicated shin sensors that match only ``knee_link`` (feet excluded).
    shin_contacts = RewTerm(
        func=mdp.multiple_undesired_contacts,
        weight=-0.75,
        params={
            "sensor_cfgs": [SceneEntityCfg("left_shin_contact",),
                            SceneEntityCfg("right_shin_contact",),],
            "threshold": 1.0,
        },
    )

    ##
    # Action rate
    ##
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    ##
    # CLF Tracking
    ##
    clf_reward = RewTerm(
        func=mdp.clf_reward,
        weight=10.0,
        params={
            "command_name": "traj_ref",
            "max_eta_err": 6.0,
        }
    )

    ##
    # CLF Stability
    ##
    clf_decreasing_condition = RewTerm(
        func=mdp.clf_decreasing_condition,
        weight=-5.0,
        params={
            "command_name": "traj_ref",
            "alpha": 0.02,
            "eta_max": 6,
            "eta_dot_max": 12,
        }
    )

    ##
    # Goal conditioned rewards
    ##
    xy_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity",
                "std": 0.5, }
    )

    yaw_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity",
                "std": 0.5, }
    )

@configclass
class G1ClfTrackingTerminationCfg:
    """Handle the terminations."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = None
    base_height = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.25,
        }
    )

##
# Normal Config
##
@configclass
class G1ClfTrackingEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: G1ClfTrackingSceneCfg = G1ClfTrackingSceneCfg(num_envs=4096, env_spacing=2.5)

    observations: G1ClfTrackingObservationsCfg = G1ClfTrackingObservationsCfg()

    actions: G1ClfTrackingActionsCfg = G1ClfTrackingActionsCfg()

    commands: G1ClfTrackingCommandCfg = G1ClfTrackingCommandCfg()

    rewards: G1ClfTrackingRewardCfg = G1ClfTrackingRewardCfg()

    events: G1ClfTrackingEventsCfg = G1ClfTrackingEventsCfg()

    terminations: G1ClfTrackingTerminationCfg = G1ClfTrackingTerminationCfg()

    curriculum: None

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics = PhysxCfg(
            gpu_max_rigid_patch_count=10 * 2**15,
            gpu_total_aggregate_pairs_capacity=2**22,
        )


##
#
##
@configclass
class G1ClfTrackingEcEnvCfg(G1ClfTrackingEnvCfg):
    """Configuration for the G1 environment with gait library."""
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        #both front and back 1.14
        #just front: 0.616
        self.events.add_plate_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="waist_yaw_link"),
                "mass_distribution_params": (0.616,0.616),
                "operation": "add",
            }
        )


# @configclass
# class G1ClfTrackingEnvCfg_PLAY(G1ClfTrackingEnvCfg):
#     """Configuration for the G1 environment with gait library."""
#
#     def __post_init__(self):
#         # Post init of parent
#         super().__post_init__()
#
#         self.scene.num_envs = 2
#         self.scene.env_spacing = 2.5
#         self.observations.policy.enable_corruption = False
#         self.scene.terrain.size = (3, 3)
#         self.scene.terrain.border_width = 0.0
#         self.scene.terrain.num_rows = 3
#         self.scene.terrain.num_cols = 2
#
#         self.episode_length_s = 8.0
#
#         self.events.randomize_ground_contact_friction = None
#         self.events.add_base_mass = None
#         self.events.base_com = None
#         self.events.base_external_force_torque = None
#         self.events.push_robot = None
#         self.events.gain_randomization = None
#
#         self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
#         self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
#         self.commands.base_velocity.ranges.ang_vel_z = (-0.75, 0.75)
