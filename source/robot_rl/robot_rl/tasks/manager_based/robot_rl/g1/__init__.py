import gymnasium as gym

from . import agents

# Guard to prevent multiple registrations
_registered = False

##
# Register Gym environments.
##

if not _registered:
    ## =========================================
    # Walking Trajectory Optimization
    ## =========================================
    gym.register(
        id="G1-walking-clf",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walking_clf_env_cfg:G1WalkingCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="G1-walking-clf-symmetric",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walking_clf_env_cfg:G1WalkingCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricHalfPeriodicPPORunnerCfg",
        }
    )

    # Custom network architecture
    gym.register(
        id="G1-walking-clf-custom-arch",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walking_clf_env_cfg:G1WalkingCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricMoEPPORunnerCfg",
        }
    )

    # Extra compute (EC)
    gym.register(
        id="G1-walking-clf-ec",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walking_clf_env_cfg:G1WalkingCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    # Play
    gym.register(
        id="G1-walking-clf-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walking_clf_env_cfg:G1WalkingCLFEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    ## =========================================
    # Running Trajectory Optimization
    ## =========================================
    gym.register(
        id="G1-running-clf",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_running_clf_env_cfg:G1RunningCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        }
    )

    gym.register(
        id="G1-running-clf-symmetric",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_running_clf_env_cfg:G1RunningCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricHalfPeriodicPPORunnerCfg",
        }
    )

    # Play
    gym.register(
        id="G1-running-clf-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_running_clf_env_cfg:G1RunningCLFEnvCfgPlay",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        }
    )

    # Play
    gym.register(
        id="G1-running-clf-experiment",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_running_clf_env_cfg:G1RunningCLFEnvCfgExperiment",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        }
    )

    ## =========================================
    # Walk + Run Trajectory Optimization
    ## =========================================
    gym.register(
        id="G1-walk-run-clf-symmetric",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walk_run_env_cfg:G1WalkRunCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricHalfPeriodicPPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_distillation_cfg:G1WalkRunDistillationRunnerCfg"
            ),
        }
    )

    # Custom network architecture
    gym.register(
        id="G1-walk-run-clf-custom-arch",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walk_run_env_cfg:G1WalkRunCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricMoEPPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_distillation_cfg:G1WalkRunDistillationRunnerCfg"
            ),
        }
    )

    # Play
    gym.register(
        id="G1-walk-run-clf-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walk_run_env_cfg:G1WalkRunCLFEnvCfgPlay",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        }
    )

    gym.register(
        id="G1-walk-run-clf-custom-arch-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walk_run_env_cfg:G1WalkRunCLFEnvCfgPlay",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricMoEPPORunnerCfg",
        }
    )

    ## =========================================
    # Bowing Trajectory Optimization
    ## =========================================
    gym.register(
        id="G1-bow_forward-clf",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_bow_forward_clf_env_cfg:G1BowingCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        }
    )

    gym.register(
        id="G1-bow_forward-clf-symmetric",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_bow_forward_clf_env_cfg:G1BowingCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricEpisodicPPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_distillation_cfg:G1BowingDistillationRunnerCfg"
            ),
        }
    )

    # Play
    gym.register(
        id="G1-bow_forward-clf-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_bow_forward_clf_env_cfg:G1BowingCLFEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        }
    )

    ## =========================================
    # Bend Up Trajectory Optimization
    ## =========================================
    gym.register(
        id="G1-bend_up-clf",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_bend_up_clf_env_cfg:G1BendUpCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        }
    )

    gym.register(
        id="G1-bend_up-clf-symmetric",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_bend_up_clf_env_cfg:G1BendUpCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricEpisodicPPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_distillation_cfg:G1BowingDistillationRunnerCfg"
            ),
        }
    )

    # Play
    gym.register(
        id="G1-bend_up-clf-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_bend_up_clf_env_cfg:G1BendUpCLFEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        }
    )

    _registered = True


