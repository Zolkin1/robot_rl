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

    # Transformer Architecture
    gym.register(
        id="G1-walking-clf-transformer",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walking_clf_env_cfg:G1WalkingCLFTransformerEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CausalTransformerPPORunnerCfg",
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
    # Walking + Depth Camera (flat scan / image)
    ## =========================================
    gym.register(
        id="G1-walking-clf-depth-scan",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_depth_env_cfg:G1WalkingCLFDepthScanEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="G1-walking-clf-depth-image",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_depth_env_cfg:G1WalkingCLFDepthImageEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="G1-walking-clf-depth-scan-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_depth_env_cfg:G1WalkingCLFDepthScanEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        },
    )

    gym.register(
        id="G1-walking-clf-depth-image-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_depth_env_cfg:G1WalkingCLFDepthImageEnvCfg_PLAY",
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
    #########
    ## MLP ##
    #########
    gym.register(
        id="G1-walk-run-clf-symmetric",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walk_run_env_cfg:G1WalkRunCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultiSkillSymmetricHalfPeriodicPPORunnerCfg",
        }
    )

    # Play
    gym.register(
        id="G1-walk-run-clf-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walk_run_env_cfg:G1WalkRunCLFEnvCfgPlay",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultiSkillSymmetricHalfPeriodicPPORunnerCfg",
        }
    )

    #########
    ## MoE ##
    #########
    gym.register(
        id="G1-walk-run-clf-sym-moe",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walk_run_env_cfg:G1WalkRunCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricMoEPPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_distillation_cfg:G1MultiskillMoE2MLPDistillationRunnerCfg"
            ),
        }
    )

    ##################
    ## Distillation ##
    ##################
    # MLP -> MLP distillation
    gym.register(
        id="G1-walk-run-clf-distill-mlp2mlp",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walk_run_env_cfg:G1WalkRunCLFDistillationEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricHalfPeriodicPPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_distillation_cfg:G1MulitskillMLP2MLPDistillationRunner"
            ),
        }
    )

    # MLP teacher -> LSTM student
    gym.register(
        id="G1-walk-run-clf-distill-mlp2lstm",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walk_run_env_cfg:G1WalkRunCLFDistillationEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricHalfPeriodicPPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_distillation_cfg:G1MultiskillMLP2LSTMDistillationRunnerCfg"
            ),
        }
    )

    # MLP teacher -> Transformer student
    gym.register(
        id="G1-walk-run-clf-distill-mlp2transformer",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walk_run_env_cfg:G1WalkRunCLFTransformerDistillationEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricHalfPeriodicPPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_distillation_cfg:G1MultiskillMLP2TransformerDistillationRunnerCfg"
            ),
        }
    )

    # MLP teacher -> Transformer student with sagittal symmetry augmentation
    gym.register(
        id="G1-walk-run-clf-sym-distill-mlp2transformer",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walk_run_env_cfg:G1WalkRunCLFTransformerDistillationEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricHalfPeriodicPPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_distillation_cfg:G1MultiskillMLP2TransformerSymmetricDistillationRunnerCfg"
            ),
        }
    )

    # MLP teacher -> Transformer student with mixed on/off-policy rollout
    gym.register(
        id="G1-walk-run-clf-distill-mixed-mlp2transformer",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walk_run_env_cfg:G1WalkRunCLFTransformerDistillationEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricHalfPeriodicPPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_distillation_cfg:G1MultiskillMixedPolicyMLP2TransformerDistillationRunnerCfg"
            ),
        }
    )

    # MoE teacher -> MoE student
    gym.register(
        id="G1-walk-run-clf-sym-distill-moe2moe",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walk_run_env_cfg:G1WalkRunCLFDistillationEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricMoEPPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_distillation_cfg:G1MultiskillMoE2MoEDistillationRunnerCfg"
            ),
        }
    )

    # MLP teacher -> MoE student
    gym.register(
        id="G1-walk-run-clf-sym-distill-mlp2moe",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walk_run_env_cfg:G1WalkRunCLFDistillationEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricHalfPeriodicPPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_distillation_cfg:G1MultiskillMLP2MoEDistillationRunnerCfg"
            ),
        }
    )

    # TODO: Test
    # Hybrid DAgger + PPO distillation, MLP -> Transformer
    gym.register(
        id="G1-walk-run-clf-distill-hybrid",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_walk_run_env_cfg:G1WalkRunCLFTransformerDistillationEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricHalfPeriodicPPORunnerCfg",
            "rsl_rl_distillation_cfg_entry_point": (
                f"{agents.__name__}.rsl_rl_distillation_cfg:G1HybridDistillationRunnerCfg"
            ),
        }
    )

    ########
    # Play #
    ########
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

    ## =========================================
    # Stairs Terrain
    ## =========================================
    gym.register(
        id="G1-stairs-clf",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_stairs_clf_env_cfg:G1StairsCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricHalfPeriodicPPORunnerCfg",
        },
    )

    gym.register(
        id="G1-stairs-clf-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_stairs_clf_env_cfg:G1StairsCLFEnvCfg_PLAY",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricHalfPeriodicPPORunnerCfg",
        },
    )

    # Multiskill terrain
    gym.register(
        id="G1-multiskill-terrain-clf",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_terrain_multiskill_env_cfg:G1TerrainMultiskillCLFEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricHalfPeriodicPPORunnerCfg",
        },
    )

    gym.register(
        id="G1-multiskill-terrain-clf-play",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_terrain_multiskill_env_cfg:G1TerrainMultiskillCLFEnvCfgPlay",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SymmetricHalfPeriodicPPORunnerCfg",
        },
    )
    _registered = True


