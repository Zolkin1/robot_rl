from isaaclab.utils import configclass

from robot_rl.tasks.manager_based.robot_rl.g1.g1_clf_tracking_base import G1ClfTrackingEnvCfg

# TODO: I should really put all the single skill cfgs in a single file just to clean up all the env cfg files.

# TODO: Migrate to BatchedMultiSkillCommandCfg (see
# ``g1_clf_multiskill_base.G1MultiSkillCommandsCfg``).  Inherits the
# single-skill ``traj_ref = TrajectoryCommandCfg(...)`` from
# ``G1ClfTrackingEnvCfg``, which is slated for deletion.
@configclass
class G1WalkingCLFEnvCfg(G1ClfTrackingEnvCfg):
    """Config for the G1 for walking."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        ##
        # Commands
        ##
        # Partial speed from fast (0.8-1.4)
        # self.commands.traj_ref.path = "trajectories/retargeted/2026-03-27_09-43-45_walk_forward_retargeted_config"

        # Full speed from fast (0.0-1.4)
        self.commands.traj_ref.path = "trajectories/retargeted/2026-03-31_14-47-00_walk_forward_retargeted_config_full_walking_speed"

        # TODO: Try next
        # self.commands.traj_ref.path = "trajectories/retargeted/2026-04-01_10-48-18_walk_forward_retargeted_config_full_speed_wider_stance"

        # Configure velocity ranges for different gaits
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.4)  # Allow full range
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0) #(-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0) #(-0.75, 0.75)
        self.commands.base_velocity.ranges.heading = (-3.14,3.14)
        self.commands.base_velocity.resampling_time_range = (3.0, 4.0)

        ##
        # Observations
        ##
        self.observations.policy.joint_pos.history_length = 6
        self.observations.policy.sin_phase = None
        self.observations.policy.cos_phase = None

        self.observations.critic.sin_phase = None
        self.observations.critic.cos_phase = None

        ##
        # Terrain
        ##
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None

@configclass
class G1WalkingCLFTransformerEnvCfg(G1WalkingCLFEnvCfg):
    """Config for G1 walking with causal transformer policy.

    Sets uniform history_length on all policy observation terms so the
    transformer can reshape the flat observation into a token sequence.
    """

    def __post_init__(self):
        # Must call parent to get walking-specific setup
        # (removes sin/cos phase, sets commands, terrain, etc.)
        super().__post_init__()

        # Set uniform history for all policy obs terms (each timestep = 1 token)
        history_length = 10
        self.observations.policy.base_ang_vel.history_length = history_length
        self.observations.policy.projected_gravity.history_length = history_length
        self.observations.policy.velocity_commands.history_length = history_length
        self.observations.policy.joint_pos.history_length = history_length
        self.observations.policy.joint_vel.history_length = history_length
        self.observations.policy.actions.history_length = history_length


@configclass
class G1WalkingCLFEnvCfg_PLAY(G1WalkingCLFEnvCfg):
    """Configuration for the G1 environment with gait library."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()
        
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.scene.terrain.size = (3,3)
        self.scene.terrain.border_width = 0.0
        self.scene.terrain.num_rows = 3
        self.scene.terrain.num_cols = 2

        self.episode_length_s = 8.0

        self.events.randomize_ground_contact_friction = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.gain_randomization = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.4)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-0.75, 0.75)
        self.commands.base_velocity.debug_vis = False

