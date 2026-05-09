import math

from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from ..mdp.commands.multiskill_velocity_commands_cfg import VelocityBucketCfg

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.g1.g1_clf_tracking_base import G1ClfTrackingSceneCfg
from robot_rl.tasks.manager_based.robot_rl.g1.g1_clf_multiskill_base import G1MultiSkillCLFEnvCfg, G1MultiSkillObservationCfg


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
        # TODO: Pull the correct trajectories!
        self.commands.traj_ref.path = "trajectories/retargeted/2026-04-10_11-41-19_merged"

        # TODO: Make it so that i have velocity dependent on terrain
        # Configure velocity ranges for different gaits
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 3.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0) #(-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0) #(-0.75, 0.75)
        self.commands.base_velocity.ranges.heading = (-3.14,3.14)
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)


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
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None

        self.commands.base_velocity.debug_vis = False


@configclass
class G1TerrainMultiskillCLFDistillationEnvCfg(G1TerrainMultiskillCLFEnvCfg):
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        self.commands.base_velocity.resampling_time_range = (2.0, 6.0)
        self.commands.base_velocity.velocity_buckets=[
            VelocityBucketCfg(percentage=0.10, lin_vel_x=(0.0, 0.1)),   # Standing
            VelocityBucketCfg(percentage=0.45, lin_vel_x=(0.1, 1.5)),   # Walking
            VelocityBucketCfg(percentage=0.45, lin_vel_x=(1.5, 3.7)),   # Running
        ]
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
        self.scene.terrain.size = (3, 3)
        self.scene.terrain.border_width = 0.0
        self.scene.terrain.num_rows = 3
        self.scene.terrain.num_cols = 2

        self.events.randomize_ground_contact_friction = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.gain_randomization = None
        # self.events.joint_friction_params = None  # Can't use this - friction goes out of distribution
