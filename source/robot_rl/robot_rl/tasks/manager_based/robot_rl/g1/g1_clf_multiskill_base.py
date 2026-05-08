import math

from isaaclab.utils import configclass

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from robot_rl.tasks.manager_based.robot_rl import mdp

from .g1_clf_tracking_base import G1ClfTrackingEnvCfg, G1ClfTrackingObservationsCfg, G1ClfTrackingCommandCfg, \
    G1ClfTrackingRewardCfg
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
    critic: CriticCfg = CriticCfg()
    student: StudentCfg = StudentCfg()

@configclass
class G1MultiSkillRewardCfg(G1ClfTrackingRewardCfg):
    undesired_contacts = RewTerm(
        func=mdp.multiple_undesired_contacts,
        weight=-0.5,
        params={
            "sensor_cfgs": [SceneEntityCfg("left_thigh_contact",),
                            SceneEntityCfg("right_thigh_contact",),
                            SceneEntityCfg("torso_contact",),
                            SceneEntityCfg("left_elbow_contact",),
                            SceneEntityCfg("right_elbow_contact",),],
            "threshold": 1.0,
        },
    )
@configclass
class G1MultiSkillCommandsCfg(G1ClfTrackingCommandCfg):
    traj_ref = BatchedMultiSkillCommandCfg(
        contact_bodies = [".*_ankle_roll_link"],

        hf_repo = "zolkin/robot_rl",
        path = "trajectories/retargeted/2026-04-10_11-41-19_merged",

        conditioner_generator_name = "base_velocity",
        Q_weights = WALKING_Q_weights,
        R_weights = WALKING_R_weights,

        contact_gate_window_frac=None, #0.2,   # NOTE: Making this too big (like 0.2) can lead to spurious detections.
        hold_on_late_contact=True, #False, #True,  # TODO: need to consider that for False, the reference frame is
        # updated to be the current point's position, but this is in the air, and on the snap
        # back things are not adjusted. Maybe this is why I don't get late triggers with this or
        # maybe this is hurting? Needs investigation. Distillation may be able to handle this.
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
        velocity_buckets=[
            VelocityBucketCfg(percentage=0.10, lin_vel_x=(0.0, 0.1)),   # Standing
            VelocityBucketCfg(percentage=0.45, lin_vel_x=(0.1, 1.5)),   # Walking
            VelocityBucketCfg(percentage=0.45, lin_vel_x=(1.5, 3.7)),   # Running
        ]
    )

@configclass
class G1MultiSkillCLFEnvCfg(G1ClfTrackingEnvCfg):
    observations: G1MultiSkillObservationCfg = G1MultiSkillObservationCfg()
    commands: G1MultiSkillCommandsCfg = G1MultiSkillCommandsCfg()
