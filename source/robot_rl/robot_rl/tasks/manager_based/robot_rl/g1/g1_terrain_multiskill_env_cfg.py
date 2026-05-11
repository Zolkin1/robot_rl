import copy

from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ..mdp.commands.multiskill_velocity_commands_cfg import VelocityBucketCfg

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.g1.g1_clf_tracking_base import G1ClfTrackingSceneCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_clf_multiskill_base import G1MultiSkillCLFEnvCfg, G1MultiSkillObservationCfg
from robot_rl.tasks.manager_based.robot_rl.terrains.config.terrain_cfgs import STAIR_WALK_CFG
from robot_rl.tasks.manager_based.robot_rl.terrains.meta_stair_importer_cfg import MetaStairTerrainImporterCfg


# Body-contact sensors (declared on the base scene cfg) that should terminate
# the episode when they touch the terrain.  Feet are intentionally excluded.
_TERRAIN_CONTACT_TERMINATION_SENSORS = (
    "torso_contact",
    "left_thigh_contact",
    "right_thigh_contact",
    "left_elbow_contact",
    "right_elbow_contact",
)


@configclass
class G1TerrainMultiskillSceneCfg(G1ClfTrackingSceneCfg):
    """Configuration for the terrain scene with a legged robot."""
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/pelvis_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.0, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

@configclass
class G1TerrainMultiskillObservationCfg(G1MultiSkillObservationCfg):

    @configclass
    class PolicyCfg(G1MultiSkillObservationCfg.PolicyCfg):
        sin_phase = None
        cos_phase = None

        # Trying the actor with the traj ref and traj actual
        ref_traj = ObsTerm(func=mdp.ref_traj, params={"command_name": "traj_ref"})
        act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "traj_ref"})
        ref_traj_vel = ObsTerm(func=mdp.ref_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))
        act_traj_vel = ObsTerm(func=mdp.act_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

    @configclass
    class CriticCfg(G1MultiSkillObservationCfg.CriticCfg):
        sin_phase = None
        cos_phase = None

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

    @configclass
    class StudentCfg(PolicyCfg):
        ref_traj = None
        act_traj = None
        ref_traj_vel = None
        act_traj_vel = None

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    student: StudentCfg = StudentCfg()

@configclass
class G1TerrainMultiskillCLFEnvCfg(G1MultiSkillCLFEnvCfg):
    """Config for the G1 for multiple skills with terrain."""

    scene: G1TerrainMultiskillSceneCfg = G1TerrainMultiskillSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: G1TerrainMultiskillObservationCfg = G1TerrainMultiskillObservationCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        ##
        # Commands
        ##
        self.commands.traj_ref.path = "trajectories/retargeted/2026-05-11_12-34-50_merged"

        # Configure velocity ranges for different gaits
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 3.7)
        # self.commands.base_velocity.ranges.lin_vel_y = (0, 0) #(-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (0, 0) #(-0.75, 0.75)
        self.commands.base_velocity.ranges.heading = (-3.14,3.14)
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)

        # Per-skill velocity ranges.  Keys MUST equal the terrain importer's
        # ``skill_list`` exactly (validated at construction); the per-env skill
        # is sampled from ``terrain.skill_probs`` and indexes into this dict.
        # lin_vel_y / ang_vel_z default to (0, 0) — bump them later when the
        # tasks need lateral / turning motion.
        self.commands.base_velocity.velocity_buckets = {
            "standing":     VelocityBucketCfg(lin_vel_x=(0.0, 0.1)),
            "walk_forward": VelocityBucketCfg(lin_vel_x=(0.1, 1.5)),
            "running":      VelocityBucketCfg(lin_vel_x=(1.5, 3.7)),
            "stair_up":     VelocityBucketCfg(lin_vel_x=(0.4, 0.4)),
        }


        ##
        # Rewards
        ##

        ## CLF Based
        self.rewards.clf_reward.params["max_eta_err"] = 4.0
        self.rewards.clf_decreasing_condition.params["eta_max"] = 4.0
        self.rewards.clf_decreasing_condition.params["eta_dot_max"] = 9.0

        # Velocity Tracking
        self.rewards.xy_vel.params["std"] = 0.4
        self.rewards.yaw_vel.params["std"] = 0.4

        ##
        # Events
        ##
        # Update push forces
        # TODO: Should probably make it so i only get pushes on flat ground
        self.events.push_robot = None


        ##
        # Terrain
        ##
        base_terrain = self.scene.terrain
        self.scene.terrain = MetaStairTerrainImporterCfg(
            prim_path=base_terrain.prim_path,
            terrain_type="generator",
            terrain_generator=STAIR_WALK_CFG,
            max_init_terrain_level=0,
            collision_group=base_terrain.collision_group,
            physics_material=base_terrain.physics_material,
            visual_material=base_terrain.visual_material,
            debug_vis=base_terrain.debug_vis,
            skill_list=["stair_up", "walk_forward", "running", "standing"],
        )

        self.commands.base_velocity.debug_vis = False

        ##
        # Terminations: end the episode if any non-foot body takes a hard hit.
        # Threshold is set high enough that incidental self-collision doesn't
        # trip it — only a real impact against the staircase will.
        ##
        for sensor_name in _TERRAIN_CONTACT_TERMINATION_SENSORS:
            setattr(
                self.terminations,
                f"{sensor_name}_terrain",
                DoneTerm(
                    func=mdp.illegal_terrain_contact,
                    params={"sensor_cfg": SceneEntityCfg(sensor_name), "threshold": 50.0},
                ),
            )

        self.terminations.base_orientation = DoneTerm(
            func=mdp.base_orientation,
            params={"cmd_name": "traj_ref", "roll_limit_deg": 65.0, "pitch_limit_deg": 65.0},
        )


@configclass
class G1TerrainMultiskillCLFDistillationEnvCfg(G1TerrainMultiskillCLFEnvCfg):
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        self.commands.base_velocity.resampling_time_range = (2.0, 6.0)
        # Per-skill velocity ranges are inherited from the parent
        # ``G1TerrainMultiskillCLFEnvCfg``.  The old percentage-based bucket
        # list no longer applies — under the terrain-driven sampler, the
        # per-env skill distribution comes from ``terrain.skill_probs``
        # rather than from cfg percentages.  Override individual buckets
        # here (e.g. ``self.commands.base_velocity.velocity_buckets["running"]
        # .lin_vel_x = (2.0, 3.5)``) if distillation needs tighter ranges.
        self.commands.base_velocity.max_acc = 1.0
        self.commands.base_velocity.max_acc_frac = 1.0

        self.commands.base_velocity.skill_transition_prob = 0.6

        self.commands.traj_ref.contact_gate_window_frac = 0.1


@configclass
class G1TerrainMultiskillCLFTransformerDistillationEnvCfg(G1TerrainMultiskillCLFDistillationEnvCfg):
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Set uniform history for all policy obs terms (each timestep = 1 token)
        history_length = 50
        self.observations.student.base_ang_vel.history_length = history_length
        self.observations.student.projected_gravity.history_length = history_length
        self.observations.student.velocity_commands.history_length = history_length
        self.observations.student.joint_pos.history_length = history_length
        self.observations.student.joint_vel.history_length = history_length
        self.observations.student.actions.history_length = history_length

@configclass
class G1TerrainMultiskillCLFEnvCfgPlay(G1TerrainMultiskillCLFEnvCfg):
    """Configuration for the G1 running gait library play environment."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 3.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_x = (1.1, 3.7)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.75, 0.75)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.75, 0.75)
        self.commands.base_velocity.ranges.resampling_time_range=(2.0, 2.0) #(4.0, 4.0) #(3.0, 4.0)
        self.commands.base_velocity.skill_transition_prob = 1.0
        self.commands.base_velocity.debug_vis = False

        self.episode_length_s = 15.0 #10.0 #4.0 #6.0


        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        # Shrink the play terrain.  The actual grid is driven by the
        # ``terrain_generator`` (STAIR_WALK_CFG) — overriding fields on the
        # importer cfg has no effect.  Deepcopy first so we don't mutate the
        # shared module-level config used by the training env.
        self.scene.terrain.terrain_generator = copy.deepcopy(
            self.scene.terrain.terrain_generator
        )
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 2
        self.scene.terrain.terrain_generator.border_width = 0.0
        self.scene.terrain.terrain_generator.inter_column_borders = []

        self.events.randomize_ground_contact_friction = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.gain_randomization = None
        # self.events.joint_friction_params = None  # Can't use this - friction goes out of distribution

        self.commands.traj_ref.debug_vis = True
        self.commands.base_velocity.debug_vis = True #False
        self.scene.height_scanner.debug_vis = True