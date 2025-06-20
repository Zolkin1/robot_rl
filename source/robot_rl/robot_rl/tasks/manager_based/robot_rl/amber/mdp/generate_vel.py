# robot_rl/tasks/manager_based/robot_rl/amber/mdp.py

import torch
# from isaaclab.utils.noise import AdditiveUniformNoise  # if you like adding noise

def sampled_forward_only(env, params):
    """
    A custom velocity‐command generator that only samples an x‐velocity.
    Returns a tensor of shape (n_envs, 3): [vx, 0, 0] for each env.
    """
    # number of parallel envs
    n = env.num_envs

    # you can pull ranges from params or from the env cfg:
    low, high = params.get("x_range", env.cfg.commands.base_velocity.ranges.lin_vel_x)

    # uniformly sample in [low, high)
    vx = torch.rand(n, device=env.device) * (high - low) + low

    # pack into (vx, vy=0, yaw_rate=0)
    zeros = torch.zeros_like(vx)
    return torch.stack([vx, zeros, zeros], dim=1)
