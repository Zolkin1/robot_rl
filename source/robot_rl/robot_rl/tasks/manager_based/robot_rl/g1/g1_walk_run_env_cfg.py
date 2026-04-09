import math

from isaaclab.utils import configclass

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from robot_rl.tasks.manager_based.robot_rl import mdp

from .g1_clf_tracking_base import G1ClfTrackingEnvCfg, G1ClfTrackingObservationsCfg, G1ClfTrackingCommandCfg
from ..mdp.commands.multiskill_velocity_commands_cfg import MultiskillVelocityTrackingCommandCfg, VelocityBucketCfg
from ..mdp.commands.velocity_commands_cfg import VelocityTrackingCommandCfg

@configclass
class G1WalkRunObservationCfg(G1ClfTrackingObservationsCfg):

    @configclass
    class PolicyCfg(G1ClfTrackingObservationsCfg.PolicyCfg):
        sin_phase = None
        cos_phase = None

        multiskill_phases = ObsTerm(func=mdp.multiskill_phase, params={"frequency_list": [1.0/(2*0.299), 1.0/(2*0.46)], "command_name": "traj_ref"} )

        # actions = ObsTerm(func=mdp.last_action, clip=(-20.0,20.0), history_length=5)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=3)


        # Trying the actor with the traj ref and traj actual
        # ref_traj = ObsTerm(func=mdp.ref_traj, params={"command_name": "traj_ref"})
        # act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "traj_ref"})
        # ref_traj_vel = ObsTerm(func=mdp.ref_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))
        # act_traj_vel = ObsTerm(func=mdp.act_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))

    @configclass
    class CriticCfg(G1ClfTrackingObservationsCfg.CriticCfg):
        sin_phase = None
        cos_phase = None

        multiskill_phases = ObsTerm(func=mdp.multiskill_phase, params={"frequency_list": [1.0/(2*0.299), 1.0/(2*0.46)], "command_name": "traj_ref"} )

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
class G1WalkRunCommandsCfg(G1ClfTrackingCommandCfg):
    base_velocity = MultiskillVelocityTrackingCommandCfg(
        asset_name="robot",
        resampling_time_range=(7.0, 10.0), #(10.0, 10.0),
        rel_closed_loop=0.55, #0.55,
        rel_closed_loop_yaw=0.25,
        rel_open_loop=0.2,
        debug_vis=True,
        ranges=VelocityTrackingCommandCfg.VelRanges(
            lin_vel_x=(0.0, 3.7),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
            y_pos_offset=(-0.5, 0.5),
            y_kp=(1.2, 1.8),
            y_kd=(0.2, 0.4),
        ),
        velocity_buckets = [
            VelocityBucketCfg(percentage=0.48, lin_vel_x=(0.11, 1.49)),     # Walking
            VelocityBucketCfg(percentage=0.48, lin_vel_x=(1.51, 3.7)),      # Running
            VelocityBucketCfg(percentage=0.04, lin_vel_x=(0, 0.99)),           # Standing
        ]
    )


@configclass
class G1WalkRunCLFEnvCfg(G1ClfTrackingEnvCfg):
    observations: G1WalkRunObservationCfg = G1WalkRunObservationCfg()
    commands: G1WalkRunCommandsCfg = G1WalkRunCommandsCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        ##
        # Commands
        ##
        self.commands.traj_ref.path = "trajectories/retargeted/2026-04-01_11-24-07_merged_walk_run"

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
        self.rewards.clf_reward.params["max_eta_err"] = 10.0
        self.rewards.clf_decreasing_condition.params["eta_max"] = 10.0
        self.rewards.clf_decreasing_condition.params["eta_dot_max"] = 20.0

        # Velocity Tracking
        self.rewards.xy_vel.params["std"] = 0.5
        self.rewards.yaw_vel.params["std"] = 0.5

        ##
        # Events
        ##
        # Update push forces
        self.events.push_robot.params['velocity_range'] = {"x": (-0.75, 0.75), "y": (-0.75, 0.75)}


        ##
        # Terrain
        ##
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None

@configclass
class G1WalkRunCLFEnvCfgPlay(G1WalkRunCLFEnvCfg):
    """Configuration for the G1 running gait library play environment."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (3.6, 3.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_x = (1.1, 3.7)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.75, 0.75)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.75, 0.75)
        self.commands.base_velocity.ranges.resampling_time_range=(5.0, 5.0) #(4.0, 4.0) #(3.0, 4.0)
        self.commands.base_velocity.debug_vis = False

        self.episode_length_s = 10.0 #4.0 #6.0


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
