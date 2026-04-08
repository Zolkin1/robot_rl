from isaaclab.utils import configclass

from isaaclab.managers import ObservationTermCfg as ObsTerm

from robot_rl.tasks.manager_based.robot_rl import mdp

from .g1_clf_tracking_base import G1ClfTrackingEnvCfg, G1ClfTrackingObservationsCfg

@configclass
class G1WalkRunObservationCfg(G1ClfTrackingObservationsCfg):

    @configclass
    class PolicyCfg(G1ClfTrackingObservationsCfg.PolicyCfg):
        sin_phase = None
        cos_phase = None

        multiskill_phases = ObsTerm(func=mdp.multiskill_phase, params={"frequency_list": [1.0/(2*0.299), 1.0/(2*0.46)]} )

    @configclass
    class CriticCfg(G1ClfTrackingObservationsCfg.CriticCfg):
        sin_phase = None
        cos_phase = None

        multiskill_phases = ObsTerm(func=mdp.multiskill_phase, params={"frequency_list": [1.0/(2*0.299), 1.0/(2*0.46)]} )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class G1WalkRunCLFEnvCfg(G1ClfTrackingEnvCfg):
    observations: G1WalkRunObservationCfg = G1WalkRunObservationCfg()

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        ##
        # Commands
        ##
        self.commands.traj_ref.path = "trajectories/retargeted/2026-04-01_11-24-07_merged_walk_run"

        # Configure velocity ranges for different gaits
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 3.6)  # Allow full range
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
