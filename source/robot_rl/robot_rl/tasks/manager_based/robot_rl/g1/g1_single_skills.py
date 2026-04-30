from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.g1.g1_clf_multiskill_base import G1MultiSkillCLFEnvCfg
from robot_rl.tasks.manager_based.robot_rl.terrains import LONG_STAIRS_CFG

# Body-contact sensors (declared on the base scene cfg) that should terminate the
# episode when they touch the terrain. Feet are intentionally excluded.
_TERRAIN_CONTACT_TERMINATION_SENSORS = (
    "torso_contact",
    "left_thigh_contact",
    "right_thigh_contact",
    "left_elbow_contact",
    "right_elbow_contact",
)


@configclass
class G1StairsCLFEnvCfg(G1MultiSkillCLFEnvCfg):
    """Config for the G1 for walking."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        ##
        # Commands
        ##
        # Full speed from fast (0.0-1.4)
        self.commands.traj_ref.path = "trajectories/retargeted/2026-04-30_14-43-20_stairs_up"

        # Configure velocity ranges for different gaits
        self.commands.base_velocity.ranges.lin_vel_x = (0.4, 0.4)  # Allow full range
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0)  # (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0)  # (-0.75, 0.75)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        ##
        # Observations
        ##

        ##
        # Terrain
        ##
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = LONG_STAIRS_CFG
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.height_scanner = None

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

@configclass
class G1StairsCLFEnvCfg_PLAY(G1StairsCLFEnvCfg):
    """Configuration for the G1 environment with gait library."""

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # One staircase per env (overrides the 1x1 default in LONG_STAIRS_CFG).
        # These fields live on TerrainGeneratorCfg, not TerrainImporterCfg.
        self.scene.terrain.terrain_generator.num_cols = self.scene.num_envs

        self.episode_length_s = 8.0

        self.events.randomize_ground_contact_friction = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.gain_randomization = None

        self.commands.base_velocity.debug_vis = False

