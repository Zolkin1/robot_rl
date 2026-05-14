import math

from isaaclab.utils import configclass

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm

from robot_rl.tasks.manager_based.robot_rl import mdp
from .g1_clf_multiskill_base import G1MultiSkillCLFEnvCfg, G1MultiSkillObservationCfg
from ..mdp.commands.multiskill_velocity_commands_cfg import VelocityBucketCfg
from ..terrains.meta_terrain_generator_cfg import MetaTerrainGeneratorCfg
from ..terrains.meta_terrain_importer_cfg import MetaTerrainImporterCfg
from ..terrains.trimesh.flat_cfg import MeshFlatTerrainCfg


# Flat-only multiskill terrain: a single sub-terrain whose ``skill_probs``
# advertise walking / running / standing so the multiskill velocity command
# can sample any of those three on a flat-ground env.  Used by
# :class:`G1WalkRunCLFEnvCfg` below.
FLAT_MULTISKILL_CFG = MetaTerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=0.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshFlatTerrainCfg(proportion=1.0),
    },
)

@configclass
class G1WalkRunCLFEnvCfg(G1MultiSkillCLFEnvCfg):

    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        ##
        # Commands
        ##
        self.commands.traj_ref.path = "trajectories/retargeted/2026-05-11_12-34-50_merged"

        # Configure velocity ranges for different gaits
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 3.7)
        self.commands.base_velocity.ranges.lin_vel_y = (0, 0) #(-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (0, 0) #(-0.75, 0.75)
        self.commands.base_velocity.ranges.heading = (-3.14,3.14)
        self.commands.base_velocity.resampling_time_range = (4.0, 8.0)

        # Per-skill velocity ranges.  Keys must equal the terrain importer's
        # ``skill_list`` (validated at construction).  The trajectory cmd
        # owns this dict: its active skill_id is derived each step from
        # which bucket ``vel_target_b`` is in.  The velocity cmd reads
        # the same dict to sample within a bucket on resample.
        self.commands.traj_ref.velocity_buckets = {
            "standing":     VelocityBucketCfg(lin_vel_x=(0.0, 0.1)),
            "walk_forward": VelocityBucketCfg(lin_vel_x=(0.1, 1.5)),
            "running":      VelocityBucketCfg(lin_vel_x=(1.5, 3.7)),
        }

        # Required pairing with ``gate_skill_change_on_contact=True``
        # (default): contact gate must be enabled for the pending-skill
        # queue to drain.  ``BatchedMultiSkillCommand`` raises at init
        # otherwise.
        self.commands.traj_ref.contact_gate_window_frac = 0.1

        # Note: Having the accerletaion in the teacher training seems to hurt the behavior
        # TODO: Consider using a max acc on some % (like 10) of the teacher so its not OOD
        self.commands.base_velocity.max_acc = 1.0
        self.commands.base_velocity.max_acc_frac = 1.0 #0.2

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
        # The multiskill velocity command requires a ``MetaTerrainImporter``
        # so it can read ``terrain.skill_probs`` for cell-based skill
        # sampling.  Use a flat-only multiskill terrain — semantically the
        # same "flat ground everywhere" as the old ``terrain_type="plane"``
        # but exposes the skill machinery the cmd expects.
        base_terrain = self.scene.terrain
        self.scene.terrain = MetaTerrainImporterCfg(
            prim_path=base_terrain.prim_path,
            terrain_type="generator",
            terrain_generator=FLAT_MULTISKILL_CFG,
            max_init_terrain_level=0,
            collision_group=base_terrain.collision_group,
            physics_material=base_terrain.physics_material,
            visual_material=base_terrain.visual_material,
            debug_vis=base_terrain.debug_vis,
            skill_list=["walk_forward", "running", "standing"],
        )
        self.scene.height_scanner = None

        self.commands.base_velocity.debug_vis = False

        ##
        # Terminations
        ##
        self.terminations.frame_drift = DoneTerm(
          func=mdp.frame_deviation_from_reference,
          params={
              "debug": False,
              "cmd_name": "traj_ref",
              "frame_names": ["pelvis_link", "left_ankle_roll_link", "right_ankle_roll_link"],
              "max_frac": 0.3, #0.25,
              "min_dist": 0.1,
              "grace_period_s": 0.5, #2.0,    # TODO: This probably makes critic learning hard as sometimes these states terminate and sometimes they don't.
                                        #   Depends on when traj was switched, which isn't an observation currently.
          },
        )


@configclass
class G1WalkRunCLFTransformerRLEnvCfg(G1WalkRunCLFEnvCfg):
    """Env cfg to run RL with the transformer."""
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        # Set uniform history for all policy obs terms (each timestep = 1 token)
        history_length = 25
        self.observations.policy.base_ang_vel.history_length = history_length
        self.observations.policy.projected_gravity.history_length = history_length
        self.observations.policy.velocity_commands.history_length = history_length
        self.observations.policy.joint_pos.history_length = history_length
        self.observations.policy.joint_vel.history_length = history_length
        self.observations.policy.actions.history_length = history_length
        self.observations.policy.ref_traj.history_length = history_length
        self.observations.policy.act_traj.history_length = history_length
        self.observations.policy.ref_traj_vel.history_length = history_length
        self.observations.policy.act_traj_vel.history_length = history_length

        self.observations.critic.base_ang_vel.history_length = history_length
        self.observations.critic.projected_gravity.history_length = history_length
        self.observations.critic.velocity_commands.history_length = history_length
        self.observations.critic.joint_pos.history_length = history_length
        self.observations.critic.joint_vel.history_length = history_length
        self.observations.critic.actions.history_length = history_length
        self.observations.critic.ref_traj.history_length = history_length
        self.observations.critic.act_traj.history_length = history_length
        self.observations.critic.ref_traj_vel.history_length = history_length
        self.observations.critic.act_traj_vel.history_length = history_length
        self.observations.critic.base_lin_vel.history_length = history_length
        self.observations.critic.root_quat.history_length = history_length


@configclass
class _SharedTrunkPolicyCfg(G1MultiSkillObservationCfg.PolicyCfg):
    """Policy obs augmented with privileged base_lin_vel + root_quat.

    Used by :class:`G1WalkRunCLFSharedTransformerRLEnvCfg`. The env is a
    teacher policy that will be distilled, so leaking privileged info to
    the actor is intentional — it lets the shared transformer trunk feed a
    single latent into both heads.
    """
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)
    root_quat = ObsTerm(func=mdp.root_quat_w, scale=1.0)


@configclass
class G1WalkRunCLFSharedTransformerRLEnvCfg(G1WalkRunCLFEnvCfg):
    """Walk-run env for the shared-trunk causal-transformer teacher.

    Differences from :class:`G1WalkRunCLFTransformerRLEnvCfg`:
    - Policy obs group includes the privileged terms ``base_lin_vel`` and
      ``root_quat`` so actor and critic see identical obs (293 dims/step
      with history_length=25). This is intentional — the env trains a
      teacher policy that gets distilled to a non-privileged student.
    - history_length=25 on every term across both policy and critic groups
      (so the transformer can reshape every term to (history, single_dim)).
    """
    def __post_init__(self):
        super().__post_init__()
        # Replace policy with the privileged-augmented variant before
        # setting histories, so the new terms get the same history_length.
        self.observations.policy = _SharedTrunkPolicyCfg()

        history_length = 25
        policy_terms = (
            "base_ang_vel", "projected_gravity", "velocity_commands",
            "joint_pos", "joint_vel", "actions",
            "ref_traj", "act_traj", "ref_traj_vel", "act_traj_vel",
            "base_lin_vel", "root_quat",
        )
        for name in policy_terms:
            getattr(self.observations.policy, name).history_length = history_length

        critic_terms = (
            "base_lin_vel", "base_ang_vel", "root_quat",
            "projected_gravity", "velocity_commands",
            "joint_pos", "joint_vel", "actions",
            "ref_traj", "act_traj", "ref_traj_vel", "act_traj_vel",
        )
        for name in critic_terms:
            getattr(self.observations.critic, name).history_length = history_length


@configclass
class G1WalkRunCLFDistillationEnvCfg(G1WalkRunCLFEnvCfg):
    def __post_init__(self):
        # Post init of parent
        super().__post_init__()

        self.observations.policy.enable_corruption = False
        self.observations.critic.enable_corruption = False

        self.commands.base_velocity.resampling_time_range = (2.0, 6.0)
        # ``velocity_buckets`` inherited from the base cfg (dict keyed by
        # skill name).  Override individual buckets here if distillation
        # needs tighter ranges.
        self.commands.base_velocity.max_acc = 1.0
        self.commands.base_velocity.max_acc_frac = 1.0

        self.commands.base_velocity.skill_transition_prob = 0.6
        # ``contact_gate_window_frac`` is already set on the base cfg.


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

@configclass
class G1WalkRunCLFEnvCfgPlay(G1WalkRunCLFEnvCfg):
    """Configuration for the G1 running gait library play environment."""

    def __post_init__(self):
        super().__post_init__()

        import copy

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 3.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # NOTE: ``resampling_time_range`` lives on the cfg root, not on
        # ``ranges`` — setting ``ranges.resampling_time_range`` silently
        # creates an unused attribute.
        self.commands.base_velocity.resampling_time_range = (4.0, 4.0)
        self.commands.base_velocity.skill_transition_prob = 1.0
        self.commands.base_velocity.debug_vis = True
        self.commands.traj_ref.debug_vis = True

        self.episode_length_s = 15.0 #10.0 #4.0 #6.0


        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        # Shrink the play terrain.  The actual grid is driven by the
        # ``terrain_generator`` (FLAT_MULTISKILL_CFG) — overriding fields
        # on the importer cfg has no effect.  Deepcopy first so we don't
        # mutate the shared module-level generator used by training.
        self.scene.terrain.terrain_generator = copy.deepcopy(
            self.scene.terrain.terrain_generator
        )
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 2
        self.scene.terrain.terrain_generator.border_width = 20.0

        self.events.randomize_ground_contact_friction = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.gain_randomization = None
        # self.events.joint_friction_params = None  # Can't use this - friction goes out of distribution
