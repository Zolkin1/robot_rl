from typing import Tuple
import torch
import tensordict


def _tile_multiplier(multiplier: torch.Tensor, obs_size: int) -> torch.Tensor:
    """Tile a per-timestep multiplier to cover the full (possibly history-stacked) obs slice.

    When history_length > 1, the obs term is flattened as [t0_feat, t1_feat, ..., tH_feat].
    This tiles the single-timestep multiplier H times so element-wise multiply works.

    Args:
        multiplier: 1D tensor for a single timestep (e.g. [-1, 1, -1] for 3-dim obs).
        obs_size: Total flattened size of this obs term (single_dim * history_length).

    Returns:
        Tiled multiplier of shape [obs_size].
    """
    single_dim = multiplier.shape[0]
    n_repeats = obs_size // single_dim
    return multiplier.repeat(n_repeats)


def _switch_g1_joints_with_history(flat_joints: torch.Tensor, single_dim: int = 21) -> torch.Tensor:
    """Apply joint switching that handles history-stacked joint observations.

    Args:
        flat_joints: [batch, history_length * single_dim] flattened joint obs.
        single_dim: Number of joints per timestep (default 21).

    Returns:
        Switched joints with same shape as input.
    """
    obs_size = flat_joints.shape[-1]
    if obs_size == single_dim:
        return _switch_g1_joints(flat_joints)

    # Reshape to [batch * history_length, single_dim], switch, reshape back
    batch = flat_joints.shape[0]
    reshaped = flat_joints.reshape(-1, single_dim)
    switched = _switch_g1_joints(reshaped)
    return switched.reshape(batch, obs_size)


def symmetric_data_augmentation_episodic(env, obs: tensordict.TensorDict, actions: torch.Tensor) -> Tuple[tensordict.TensorDict, torch.Tensor]:
    """Augment the data for the RSL RL data augmentation.

    obs: Tensor of shape [batch, num_obs]
    actions: Tensor of shape [batch, num_actions]
    env: RL vec env

    Flip the observation and actions. Handles history-stacked observations
    by tiling multipliers or reshaping before applying per-timestep transforms.
    """

    # Can pull the remapping matrix R from the trajectory manager from the command from the env

    if obs is not None:
        device = obs.device

        batch_size = obs.batch_size[0]

        obs_aug = obs.repeat(2)

        cmd = env.unwrapped.command_manager.get_term("traj_ref")

        # Original observations
        obs_aug["policy"][:batch_size] = obs["policy"][:batch_size]

        for group in ["policy", "critic"]:
            obs_idx = 0
            for i, name in enumerate(env.unwrapped.observation_manager.active_terms[group]):
                obs_size = 0
                if name == "base_ang_vel":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    multiplier = _tile_multiplier(torch.tensor([-1, 1, -1], device=device), obs_size)

                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        obs[group][:, obs_idx:obs_idx + obs_size] * multiplier)
                elif name == "base_lin_vel":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    multiplier = _tile_multiplier(torch.tensor([1, -1, 1], device=device), obs_size)

                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        obs[group][:, obs_idx:obs_idx + obs_size] * multiplier
                    )
                elif name == "projected_gravity":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    multiplier = _tile_multiplier(torch.tensor([1, -1, 1], device=device), obs_size)

                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        obs[group][:, obs_idx:obs_idx + obs_size] * multiplier
                    )
                elif name == "velocity_commands":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    multiplier = _tile_multiplier(torch.tensor([1, -1, -1], device=device), obs_size)

                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        obs[group][:, obs_idx:obs_idx + obs_size] * multiplier
                    )
                elif name == "joint_pos" or name == "joint_vel" or name == "actions":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        _switch_g1_joints_with_history(obs[group][:, obs_idx:obs_idx + obs_size])
                    )
                elif name == "sin_phase" or name == "cos_phase":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = obs[group][:, obs_idx:obs_idx + obs_size]
                elif name == "ref_traj" or name == "act_traj":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        cmd.get_symmetric_traj(obs[group][:, obs_idx:obs_idx + obs_size], "pos")
                    )
                elif name == "ref_traj_vel" or name == "act_traj_vel":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        cmd.get_symmetric_traj(obs[group][:, obs_idx:obs_idx + obs_size], "vel")
                    )
                elif name == "root_quat":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    multiplier = _tile_multiplier(torch.tensor([-1, 1, -1, 1], device=device), obs_size)
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        obs[group][:, obs_idx:obs_idx + obs_size] * multiplier
                    )
                elif name == "contact_state":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        cmd.get_symmetric_contacts(obs[group][:, obs_idx:obs_idx + obs_size])
                    )

                # TODO: Add height map support

                obs_idx += obs_size
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]

        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)

        # Original actions
        actions_aug[:batch_size] = actions

        actions_aug[batch_size:] = _switch_g1_joints(actions)
    else:
        actions_aug = None

    return (obs_aug, actions_aug)

def symmetric_data_augmentation_half_periodic(env, obs: tensordict.TensorDict, actions: torch.Tensor) -> Tuple[tensordict.TensorDict, torch.Tensor]:
    """Augment the data for the RSL RL data augmentation.

    obs: Tensor of shape [batch, num_obs]
    actions: Tensor of shape [batch, num_actions]
    env: RL vec env

    Flip the observation and actions. Handles history-stacked observations
    by tiling multipliers or reshaping before applying per-timestep transforms.
    """

    # Can pull the remapping matrix R from the trajectory manager from the command from the env

    if obs is not None:
        device = obs.device

        batch_size = obs.batch_size[0]

        obs_aug = obs.repeat(2)

        cmd = env.unwrapped.command_manager.get_term("traj_ref")

        # Original observations
        obs_aug["policy"][:batch_size] = obs["policy"][:batch_size]

        for group in ["policy", "critic"]:
            obs_idx = 0
            for i, name in enumerate(env.unwrapped.observation_manager.active_terms[group]):
                obs_size = 0
                if name == "base_ang_vel":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    multiplier = _tile_multiplier(torch.tensor([-1, 1, -1], device=device), obs_size)

                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        obs[group][:, obs_idx:obs_idx + obs_size] * multiplier)
                elif name == "base_lin_vel":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    multiplier = _tile_multiplier(torch.tensor([1, -1, 1], device=device), obs_size)

                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        obs[group][:, obs_idx:obs_idx + obs_size] * multiplier
                    )
                elif name == "projected_gravity":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    multiplier = _tile_multiplier(torch.tensor([1, -1, 1], device=device), obs_size)

                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        obs[group][:, obs_idx:obs_idx + obs_size] * multiplier
                    )
                elif name == "velocity_commands":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    multiplier = _tile_multiplier(torch.tensor([1, -1, -1], device=device), obs_size)

                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        obs[group][:, obs_idx:obs_idx + obs_size] * multiplier
                    )
                elif name == "joint_pos" or name == "joint_vel" or name == "actions":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        _switch_g1_joints_with_history(obs[group][:, obs_idx:obs_idx + obs_size])
                    )
                elif name == "sin_phase" or name == "cos_phase":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = -1*obs[group][:, obs_idx:obs_idx + obs_size]
                elif name == "ref_traj" or name == "act_traj":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        cmd.get_symmetric_traj(obs[group][:, obs_idx:obs_idx + obs_size], "pos")
                    )
                elif name == "ref_traj_vel" or name == "act_traj_vel":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        cmd.get_symmetric_traj(obs[group][:, obs_idx:obs_idx + obs_size], "vel")
                    )
                elif name == "root_quat":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    multiplier = _tile_multiplier(torch.tensor([-1, 1, -1, 1], device=device), obs_size)
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        obs[group][:, obs_idx:obs_idx + obs_size] * multiplier
                    )
                elif name == "contact_state":
                    obs_size = env.unwrapped.observation_manager.group_obs_term_dim[group][i][0]
                    obs_aug[group][batch_size:, obs_idx:obs_idx + obs_size] = (
                        cmd.get_symmetric_contacts(obs[group][:, obs_idx:obs_idx + obs_size])
                    )

                # TODO: Add height map support

                obs_idx += obs_size
    else:
        obs_aug = None

    if actions is not None:
        batch_size = actions.shape[0]

        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)

        # Original actions
        actions_aug[:batch_size] = actions

        actions_aug[batch_size:] = _switch_g1_joints(actions)
    else:
        actions_aug = None

    return (obs_aug, actions_aug)

def _switch_g1_joints(joints: torch.Tensor) -> torch.Tensor:
    """
    Reflection the joint values about the sagittal plane.

    IsaacSim ordering:
    [
        left_hip_pitch_joint, right_hip_pitch_joint, waist_yaw_joint,
        left_hip_roll_joint, right_hip_roll_joint, left_shoulder_pitch_joint,
        right_shoulder_pitch_joint, left_hip_yaw_joint, right_hip_yaw_joint,
        left_shoulder_roll_joint, right_shoulder_roll_joint, left_knee_joint,
        right_knee_joint, left_shoulder_yaw_joint, right_shoulder_yaw_joint,
        left_ankle_pitch_joint, right_ankle_pitch_joint, left_elbow_joint,
        right_elbow_joint, left_ankle_roll_joint, right_ankle_roll_joint
    ]

    Map all left -> right and right -> left
    Negate all roll and yaw joints.

    Left leg:
    [0, 3, 7, 11, 15, 19]

    Right Leg:
    [1, 4, 8, 12, 16, 20]

    Left Arm:
    [5, 9, 13, 17]

    Right Arm:
    [6, 10, 14, 18]

    Waist Yaw:
    [2]

    """
    joints_switched = torch.zeros_like(joints)

    left_leg = [0, 3, 7, 11, 15, 19]
    right_leg = [1, 4, 8, 12, 16, 20]
    left_arm = [5, 9, 13, 17]
    right_arm = [6, 10, 14, 18]
    waist_yaw = [2]

    # left leg <-- right leg
    joints_switched[:, left_leg] = joints[:, right_leg]

    # left arm <-- right arm
    joints_switched[:, left_arm] = joints[:, right_arm]

    # right leg <-- left leg
    joints_switched[:, right_leg] = joints[:, left_leg]

    # right arm <-- left arm
    joints_switched[:, right_arm] = joints[:, left_arm]

    # Waist yaw
    joints_switched[:, waist_yaw] = joints[:, waist_yaw]

    # Negate roll and yaw joints
    joints_switched[:, [2, 3, 4, 7, 8, 9, 10, 13, 14, 19, 20]] *= -1.0

    return joints_switched