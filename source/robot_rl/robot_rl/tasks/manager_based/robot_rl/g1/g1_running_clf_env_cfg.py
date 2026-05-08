from isaaclab.utils import configclass

from .g1_clf_tracking_base import G1ClfTrackingEnvCfg


# TODO: Migrate to BatchedMultiSkillCommandCfg (see
# ``g1_clf_multiskill_base.G1MultiSkillCommandsCfg``).  Inherits the
# single-skill ``traj_ref = TrajectoryCommandCfg(...)`` from
# ``G1ClfTrackingEnvCfg``, which is slated for deletion.
@configclass
class G1RunningCLFEnvCfg(G1ClfTrackingEnvCfg):  #G1RunningGaitLibraryEnvCfg
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        ##
        # Commands
        ##
        self.commands.traj_ref.path = "trajectories/retargeted/2026-03-31_15-13-34_running_retargeted_config_full_speeds"

        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.rel_closed_loop = 0.55

        # Configure velocity ranges for different gaits
        self.commands.base_velocity.ranges.lin_vel_x = (1.6, 3.6)  # Allow full range
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0) #(-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0) #(-0.75, 0.75)
        self.commands.base_velocity.ranges.heading = (-3.14,3.14)
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)

        # self.commands.base_velocity.ranges.lin_vel_y = (-0.75, 0.75)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.75, 0.75)

        ##
        # Rewards
        ##

        ## CLF Based
        self.rewards.clf_reward.params["max_eta_err"] = 12.0
        self.rewards.clf_decreasing_condition.params["eta_max"] = 12.0
        self.rewards.clf_decreasing_condition.params["eta_dot_max"] = 24.0

        # Velocity Tracking
        self.rewards.xy_vel.params["std"] = 0.75
        self.rewards.yaw_vel.params["std"] = 0.75

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
class G1RunningCLFEnvCfgPlay(G1RunningCLFEnvCfg):
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

@configclass
class G1RunningCLFEnvCfgExperiment(G1RunningCLFEnvCfg):
    """Configuration for the G1 running gait library play environment."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (3.0, 3.0) #(3.6, 3.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0) #(-0.75, 0.75)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0) #(-0.75, 0.75)
        self.commands.base_velocity.ranges.resampling_time_range=(20.0, 20.0)
        self.commands.base_velocity.debug_vis = False

        # TODO: Can play with this
        self.commands.base_velocity.rel_open_loop = 1.0
        self.commands.base_velocity.rel_closed_loop = 0.0
        self.commands.base_velocity.rel_closed_loop_yaw = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0

        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        self.events.push_robot = None