import gymnasium as gym

from . import agents

# Guard to prevent multiple registrations
_registered = False

##
# Register Gym environments.
##

if not _registered:
    ## =========================================
    # Vanilla, heuristic RL
    ## =========================================
    gym.register(
        id="G1-vanilla-walking",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_vanilla_walking_env_cfg:G1VanillaWalkingEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        },
    )

    # Extra compute (EC)
    gym.register(
        id="G1-vanilla-walking-ec",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_vanilla_walking_env_cfg:G1VanillaWalkingECEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        },
    )

    # TODO: Add a play


    ## =========================================
    # LIP-CLF RL
    ## =========================================
    gym.register(
        id="G1-lip-clf",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_lip_clf_env_cfg:G1LipCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        },
    )

    # Extra compute (EC)
    gym.register(
        id="G1-lip-clf-ec",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_lip_clf_env_cfg:G1LipCLFECEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        },
    )

    # Play
    gym.register(
        id="G1-lip-ref-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_flat_env_lip_cfg:G1FlatLipEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        },
    )

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
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
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

    # Extra compute (EC)
    gym.register(
        id="G1-walking-clf-ec",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walking_clf_env_cfg:G1WalkingCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
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
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
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
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
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
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
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
    # Waving Trajectory Optimization
    ## =========================================
    gym.register(
        id="G1-waving-clf",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_waving_clf_env_cfg:G1WavingCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        }
    )

    # Play
    gym.register(
        id="G1-waving-clf-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_waving_clf_env_cfg:G1WavingCLFEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
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
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
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
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
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
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
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
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
        }
    )

    _registered = True


