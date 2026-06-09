# Copyright 2026 Zachary Olkin. All rights reserved.

from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from robot_rl.tasks.manager_based.robot_rl import mdp
from isaaclab.managers import SceneEntityCfg

##
# Observations
##
@configclass
class G1TrajOptObservationsCfg():
    """Observation specifications for the G1 Flat environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},
                                    scale=(2.0, 2.0, 2.0))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), scale=0.05)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.0, n_max=1.0), scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        # Phase clock
        sin_phase = ObsTerm(func=mdp.ref_sin_phase, params={"command_name": "traj_ref"})
        cos_phase = ObsTerm(func=mdp.ref_cos_phase, params={"command_name": "traj_ref"})

        ## Teacher only terms
        # ref_traj = ObsTerm(func=mdp.ref_traj, params={"command_name": "traj_ref"})
        # act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "traj_ref"})
        # ref_traj_vel = ObsTerm(func=mdp.ref_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))
        # act_traj_vel = ObsTerm(func=mdp.act_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))

        # root_quat = ObsTerm(func=mdp.root_quat_w, scale=1.0)
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)

        # Try a trajectory error observation
        # traj_error = ObsTerm(func=mdp.traj_error, params={"command_name": "traj_ref"})


        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0)
        root_quat = ObsTerm(func=mdp.root_quat_w, scale=1.0)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},
                                    scale=(2.0, 2.0, 2.0))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        actions = ObsTerm(func=mdp.last_action)
        # Phase clock
        sin_phase = ObsTerm(func=mdp.ref_sin_phase, params={"command_name": "traj_ref"})
        cos_phase = ObsTerm(func=mdp.ref_cos_phase, params={"command_name": "traj_ref"})

        contact_state = ObsTerm(
            func=mdp.contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )

        ref_traj = ObsTerm(func=mdp.ref_traj, params={"command_name": "traj_ref"})
        act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "traj_ref"})
        ref_traj_vel = ObsTerm(func=mdp.ref_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))
        act_traj_vel = ObsTerm(func=mdp.act_traj_vel, params={"command_name": "traj_ref"}, clip=(-20.0, 20.0,))

        height_scan = None  # Removed - not supported yet


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()