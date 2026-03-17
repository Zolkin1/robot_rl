import gymnasium as gym

from . import agents

# Guard to prevent multiple registrations
_registered = False

if not _registered:
    ## =========================================
    # Double Integrator with CLF-RL
    ## =========================================
    gym.register(
        id="double-integrator",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.double_integrator_cfg:DIEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )
