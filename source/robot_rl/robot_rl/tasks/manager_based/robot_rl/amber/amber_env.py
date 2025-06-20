# amber_env.py
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils
        
import numpy as np
import gymnasium as gym
import torch
# from isaaclab.managers import ResetCallback
# from .amber_flat_env_cfg import AmberFlatEnvCfg






class NaNResetWrapper(gym.Wrapper):
    """
    Wrapper that catches any NaN or super‐large (> vel_threshold) rewards,
    zeroes them out, and immediately terminates/reset those sub-envs.
    """

    def __init__(self, env, vel_threshold=5000.0):
        super().__init__(env)
        self.vel_threshold = vel_threshold

    def step(self, action):
        # perform the normal step
        obs, reward, terminated, truncated, info = super().step(action)

        # if reward is a torch.Tensor (possibly on GPU), do everything in PyTorch
        if torch.is_tensor(reward):
            mask = torch.isnan(reward) | (reward.abs() > self.vel_threshold)
            if mask.any():
                idxs = mask.nonzero(as_tuple=True)[0].cpu().tolist()
                print(f"[NaNResetWrapper] bad reward in envs {idxs}, zeroing and resetting them")

                # zero out the bad rewards
                reward = reward.clone()
                reward[mask] = 0.0

                # force termination on those sub-envs
                terminated = terminated.clone()
                truncated  = truncated.clone()
                terminated[mask] = True
                truncated [mask] = False

        else:
            # fallback if reward is a NumPy array
            bad = np.isnan(reward) | (np.abs(reward) > self.vel_threshold)
            if np.any(bad):
                idxs = np.nonzero(bad)[0].tolist()
                print(f"[NaNResetWrapper] bad reward in envs {idxs}, zeroing and resetting them")

                reward = reward.copy()
                reward[bad] = 0.0

                terminated = np.array(terminated, copy=True)
                truncated  = np.array(truncated,  copy=True)
                terminated[bad] = True
                truncated [bad] = False

        return obs, reward, terminated, truncated, info