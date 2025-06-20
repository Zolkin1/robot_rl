import gymnasium as gym
from . import agents  # if you end up giving Amber its own agent configs
# from .amber_env import AmberFlatEnv

gym.register(
    id="Amber-flat-vel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amber_flat_env_cfg:AmberFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

# If you make a LIP version:
gym.register(
    id="Amber-flat-lip-vel",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.amber_flat_env_lip_cfg:AmberFlatLipEnvCfg",
        "rsl_rl_cfg_entry_point": "<your_agents_module>.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_cfg_entry_point": "<your_agents_module>:skrl_ppo_cfg.yaml",
    },
)
