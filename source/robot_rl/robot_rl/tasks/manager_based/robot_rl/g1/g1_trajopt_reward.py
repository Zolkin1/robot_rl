# Copyright 2026 Zachary Olkin. All rights reserved.

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import (HumanoidEnvCfg, HumanoidCommandsCfg,
                                                                    HumanoidRewardCfg)
from robot_rl.tasks.manager_based.robot_rl import mdp
from isaaclab.managers import SceneEntityCfg

##
# Rewards
##
@configclass
class G1TrajOptCLFRewards(HumanoidRewardCfg):
    """Rewards specific to LIP Model"""

    holonomic_constraint = RewTerm(
        func=mdp.holonomic_constraint,
        weight=4.0,
        params={
            "command_name": "traj_ref",
            "z_offset": 0.036,
        }
    )

    holonomic_constraint_vel = RewTerm(
        func=mdp.holonomic_constraint_vel,
        weight=2.0,
        params={
            "command_name": "traj_ref",
        }
    )


    clf_reward = RewTerm(
        func=mdp.clf_reward,
        weight=10.0,
        params={
            "command_name": "traj_ref",
            "max_eta_err": 0.8, #0.25,
        }
    )

    clf_decreasing_condition = RewTerm(
        func=mdp.clf_decreasing_condition,
        weight=-2.0,
        params={
            "command_name": "traj_ref",
            "alpha": 0.5,
            "eta_max": 0.2,
            "eta_dot_max":0.3,
        }
    )

    # Penalize unwanted self collisions
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )

    # Penalize missing the contact schedule
    # TODO: Test
    # contact_schedule_penalty = RewTerm(
    #     func=mdp.contact_schedule_penalty,
    #     weight=-3.0,
    #     params={"command_name": "traj_ref",
    #             "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #             "weight_scalar": 1.0,},
    # )