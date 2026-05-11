from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.g1.g1_clf_tracking_base import G1ClfTrackingSceneCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_clf_multiskill_base import G1MultiSkillCLFEnvCfg, G1MultiSkillObservationCfg
from robot_rl.tasks.manager_based.robot_rl.terrains.config.terrain_cfgs import LONG_STAIRS_CFG
from robot_rl.tasks.manager_based.robot_rl.terrains.meta_stair_importer_cfg import MetaStairTerrainImporterCfg

from ..mdp.commands.multiskill_velocity_commands_cfg import VelocityBucketCfg

# Body-contact sensors (declared on the base scene cfg) that should terminate the
# episode when they touch the terrain. Feet are intentionally excluded.
_TERRAIN_CONTACT_TERMINATION_SENSORS = (
    "torso_contact",
    "left_thigh_contact",
    "right_thigh_contact",
    "left_elbow_contact",
    "right_elbow_contact",
)
# TODO: Should probably add the head and/or hands to this list

@configclass
class G1StairsTrackingSceneCfg(G1ClfTrackingSceneCfg):
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
class G1StairsObservationCfg(G1MultiSkillObservationCfg):

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
class G1StairsCLFEnvCfg(G1MultiSkillCLFEnvCfg):
    """Config for the G1 for walking."""
    scene: G1StairsTrackingSceneCfg = G1StairsTrackingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: G1StairsObservationCfg = G1StairsObservationCfg()


    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        ##
        # Commands
        ##
        # Full speed from fast (0.0-1.4)
        self.commands.traj_ref.path = "trajectories/retargeted/2026-04-30_14-43-20_stairs_up"

        # Configure velocity ranges for different gaits
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)  # (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0)  # (-0.75, 0.75)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Skill keys must equal the terrain importer's ``skill_list`` (set
        # below) — only ``stair_up`` is declared by ``LONG_STAIRS_CFG``'s
        # sub-terrains, so that's the only bucket needed here.
        self.commands.base_velocity.velocity_buckets = {
            "stair_up": VelocityBucketCfg(lin_vel_x=(0.4, 0.4)),
        }

        self.commands.base_velocity.debug_vis = False

        ##
        # Events
        ##
        self.events.push_robot = None

        ##
        # Rewards
        ##
        # Velocity Tracking
        self.rewards.xy_vel.params["std"] = 0.3
        self.rewards.yaw_vel.params["std"] = 0.3

        self.rewards.clf_reward.params["max_eta_err"] = 4.0 # 10
        self.rewards.clf_decreasing_condition.params["eta_max"] = 4.0 # 10
        self.rewards.clf_decreasing_condition.params["eta_dot_max"] = 9.0 # 20

        ##
        # Observations
        ##

        ##
        # Terrain — swap the inherited TerrainImporterCfg for the meta variant
        # so ``terrain_meta_data["project"]`` is available downstream (rewards,
        # CLF, etc.). Inherit the materials from the base scene's terrain.
        ##
        base_terrain = self.scene.terrain
        self.scene.terrain = MetaStairTerrainImporterCfg(
            prim_path=base_terrain.prim_path,
            terrain_type="generator",
            terrain_generator=LONG_STAIRS_CFG,
            max_init_terrain_level=0,
            collision_group=base_terrain.collision_group,
            physics_material=base_terrain.physics_material,
            visual_material=base_terrain.visual_material,
            debug_vis=base_terrain.debug_vis,
            # Match the skills declared by ``LONG_STAIRS_CFG``'s ``stairs_up``
            # sub-terrain so the post-process skill_probs check passes.
            skill_list=["stair_up"],
        )

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

        # self.terminations.base_orientation = DoneTerm(
        #     func=mdp.base_orientation,
        #     params={"cmd_name": "traj_ref", "roll_limit_deg": 65.0, "pitch_limit_deg": 65.0},
        # )

        self.terminations.frame_drift = DoneTerm(
          func=mdp.frame_deviation_from_reference,
          params={
              "cmd_name": "traj_ref",
              "frame_names": ["pelvis_link", "left_ankle_roll_link", "right_ankle_roll_link"],
              "max_frac": 0.3, #0.25,
              "min_dist": 0.1,
          },
        )

@configclass
class G1StairsCLFEnvCfg_PLAY(G1StairsCLFEnvCfg):
    """Configuration for the G1 environment with gait library."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # One staircase per env, laid out in a single row so robots render
        # side-by-side. These fields live on TerrainGeneratorCfg, not
        # TerrainImporterCfg.
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = self.scene.num_envs

        self.episode_length_s = 20.0 #2.0

        self.events.randomize_ground_contact_friction = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.gain_randomization = None

        self.commands.traj_ref.debug_vis = True
        self.commands.base_velocity.debug_vis = True #False
        self.scene.height_scanner.debug_vis = True

