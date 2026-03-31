import math
from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from robot_rl.tasks.manager_based.robot_rl import mdp

@configclass
class G1TrajMimicRewards:
    torque_lims = RewTerm(
        func=mdp.torque_limits,
        weight=-1.0,
    )

    # Base
    base_pos = RewTerm(
        func=mdp.base_pos_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 0.2}
    )
    base_ori = RewTerm(
        func=mdp.base_ori_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 0.3}
    )
    base_lin_vel = RewTerm(
        func=mdp.base_lin_vel_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 0.3}
    )
    base_ang_vel = RewTerm(
        func=mdp.base_ang_vel_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 1.0}
    )

    # Joints
    joint_pos = RewTerm(
        func=mdp.joint_pos_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 0.2 * math.sqrt(21)}
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 3.0 * math.sqrt(21)}
    )

    # Bodies
    body_pos = RewTerm(
        func=mdp.body_pos_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 0.1 * math.sqrt(4)}
    )
    body_ori = RewTerm(
        func=mdp.body_ori_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 0.2 * math.sqrt(4)}
    )
    body_lin_vel = RewTerm(
        func=mdp.body_lin_vel_reward,
        weight=1.0,
        params={"command_name": "traj_ref",
                "sigma": 1.0 * math.sqrt(4)}
    )
    body_ang_vel = RewTerm(
        func=mdp.body_ang_vel_reward,
        weight=1.0,  # 0.0,
        params={"command_name": "traj_ref",
                "sigma": 0.5 * math.sqrt(4)}
    )

    # Goal conditioned rewards
    xy_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity",
                "std": 0.5, }
    )

    yaw_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity",
                "std": 0.5, }
    )