import math

from isaaclab.utils import configclass

from .g1_clf_multiskill_base import G1MultiSkillCLFEnvCfg
from ..mdp.commands.multiskill_velocity_commands_cfg import VelocityBucketCfg

@configclass
class G1WalkRunCLFEnvCfg(G1MultiSkillCLFEnvCfg):

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        ##
        # Commands
        ##
        self.commands.traj_ref.path = "trajectories/retargeted/2026-04-10_11-41-19_merged"

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
        self.rewards.clf_reward.params["max_eta_err"] = 6.0 # 10
        self.rewards.clf_decreasing_condition.params["eta_max"] = 6.0 # 10
        self.rewards.clf_decreasing_condition.params["eta_dot_max"] = 15.0 # 20

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
class G1WalkRunCLFDistillationEnvCfg(G1WalkRunCLFEnvCfg):
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False
        # TODO: Just for debugging
        # self.observations.student.enable_corruption = False

        self.commands.base_velocity.resampling_time_range = (2.0, 4.0)
        self.commands.base_velocity.velocity_buckets = [
            VelocityBucketCfg(percentage=0.45, lin_vel_x=(0.11, 1.49)),     # Walking
            VelocityBucketCfg(percentage=0.45, lin_vel_x=(1.51, 3.7)),      # Running
            VelocityBucketCfg(percentage=0.10, lin_vel_x=(0, 0.09)),        # Standing
        ]

        self.commands.base_velocity.skill_transition_prob = 0.9

        # For now just distill walking — override velocity buckets
        # (ranges.lin_vel_x alone has no effect because buckets consume 100% of envs)
        # self.commands.base_velocity.velocity_buckets = [
        #     VelocityBucketCfg(percentage=0.95, lin_vel_x=(0.11, 1.4)),   # Walking
        #     VelocityBucketCfg(percentage=0.05, lin_vel_x=(0.0, 0.10)),   # Standing
        # ]
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.4)

@configclass
class G1WalkRunCLFTransformerDistillationEnvCfg(G1WalkRunCLFDistillationEnvCfg):
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

        # One hot testing
        # self.observations.student.skill_one_hot.history_length = history_length

        # For the privledged student test
        # self.observations.student.ref_traj.history_length = history_length
        self.observations.student.act_traj.history_length = history_length
        # self.observations.student.ref_traj_vel.history_length = history_length
        self.observations.student.act_traj_vel.history_length = history_length

@configclass
class G1WalkRunCLFEnvCfgPlay(G1WalkRunCLFEnvCfg):
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
