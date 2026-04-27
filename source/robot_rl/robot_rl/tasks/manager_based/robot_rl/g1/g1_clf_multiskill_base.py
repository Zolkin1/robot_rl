import math

from isaaclab.utils import configclass

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from robot_rl.tasks.manager_based.robot_rl import mdp

from .g1_clf_tracking_base import G1ClfTrackingEnvCfg, G1ClfTrackingObservationsCfg, G1ClfTrackingCommandCfg
from ..mdp.commands.traj_tracking.batched_multiskill_cmd_cfg import BatchedMultiSkillCommandCfg
from ..mdp.commands.multiskill_velocity_commands_cfg import MultiskillVelocityTrackingCommandCfg, VelocityBucketCfg
from ..mdp.commands.velocity_commands_cfg import VelocityTrackingCommandCfg

from .clf_weights import WALKING_Q_weights, WALKING_R_weights


@configclass
class G1MultiSkillObservationCfg(G1ClfTrackingObservationsCfg):

    @configclass
    class PolicyCfg(G1ClfTrackingObservationsCfg.PolicyCfg):
        sin_phase = None
        cos_phase = None

        # Trying the actor with the traj ref and traj actual
        ref_traj = ObsTerm(func=mdp.ref_traj, params={"command_name": "traj_ref"})
        act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "traj_ref"})
        ref_traj_vel = ObsTerm(func=mdp.ref_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))
        act_traj_vel = ObsTerm(func=mdp.act_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))

    @configclass
    class UnPrivilegedPolicyCfg(G1ClfTrackingObservationsCfg.PolicyCfg):
        sin_phase = None
        cos_phase = None

    @configclass
    class CriticCfg(G1ClfTrackingObservationsCfg.CriticCfg):
        sin_phase = None
        cos_phase = None

    @configclass
    class StudentCfg(PolicyCfg):
        # skill_one_hot = ObsTerm(func=mdp.skill_one_hot, params={"command_name": "base_velocity"})
        ref_traj = None
        act_traj = None
        ref_traj_vel = None
        act_traj_vel = None

    policy: PolicyCfg = PolicyCfg()
    unpriv_policy: UnPrivilegedPolicyCfg = UnPrivilegedPolicyCfg()
    critic: CriticCfg = CriticCfg()
    student: StudentCfg = StudentCfg()

@configclass
class G1MultiSkillCommandsCfg(G1ClfTrackingCommandCfg):
    traj_ref = BatchedMultiSkillCommandCfg(
        contact_bodies = [".*_ankle_roll_link"],

        hf_repo = "zolkin/robot_rl",
        path = "trajectories/retargeted/2026-04-10_11-41-19_merged",

        conditioner_generator_name = "base_velocity",
        Q_weights = WALKING_Q_weights,
        R_weights = WALKING_R_weights,

        contact_gate_window_frac=0.1,
    )

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
            VelocityBucketCfg(percentage=0.04, lin_vel_x=(0, 0.09)),        # Standing
        ]
    )

@configclass
class G1MultiSkillCLFEnvCfg(G1ClfTrackingEnvCfg):
    observations: G1MultiSkillObservationCfg = G1MultiSkillObservationCfg()
    commands: G1MultiSkillCommandsCfg = G1MultiSkillCommandsCfg()
