# amber_obs.py
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

def body_link_vel_xy(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(name="robot")
) -> torch.Tensor:
    """Return the current link-3 (index 3) world-frame XY velocity, vs command."""
    asset = env.scene[asset_cfg.name]
    cmd   = env.command_manager.get_command(command_name)[:, :2]
    # pick out link index 3 (your “base_link”) and its first two coords:
    act   = asset.data.body_link_lin_vel_w[:, 3, :2]
    return torch.cat([cmd, act], dim=1)   # or however you want to pack it
