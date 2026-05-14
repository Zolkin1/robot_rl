# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import MISSING
from typing import Literal
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlMLPModelCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)
from .symmetry_functions import (
    symmetric_data_augmentation_episodic,
    symmetric_data_augmentation_half_periodic
)

###################################################
# MLP
###################################################
##
# Default MLP Runner
##
@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 200
    experiment_name = "g1"
    empirical_normalization = False
    logger="wandb"
    wandb_project = "robot_rl"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        obs_normalization=False,
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    resume = False
    resume_path = None
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }

##
# Symmetric MLP Runner
##
@configclass
class SymmetricEpisodicPPORunnerCfg(PPORunnerCfg):
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=symmetric_data_augmentation_episodic
        )
    )

##
# Symmetric MLP Runner for Half Periodic
##
@configclass
class SymmetricHalfPeriodicPPORunnerCfg(PPORunnerCfg):
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=symmetric_data_augmentation_half_periodic
        )
    )

# Multiskill
@configclass
class MultiSkillSymmetricHalfPeriodicPPORunnerCfg(PPORunnerCfg):
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=symmetric_data_augmentation_half_periodic
        )
    )

###################################################
# MoE
###################################################
##
# Symmetric MoE Runner
##
@configclass
class SymmetricMoEPPORunnerCfg(PPORunnerCfg):
    """PPO runner config using the custom MLP model (placeholder for future MoE)."""
    experiment_name = "g1_multiskill_arch"
    actor = RslRlMLPModelCfg(
        class_name="robot_rl.network.moe_network:MoEModel",
        hidden_dims=[256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        obs_normalization=False,
    )
    # TODO: Try just making the critic an MLP only
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    # critic = RslRlMLPModelCfg(
    #     class_name="robot_rl.network.moe_network:MoEModel",
    #     hidden_dims=[256, 128],
    #     activation="elu",
    #     obs_normalization=False,
    # )

    algorithm = RslRlPpoAlgorithmCfg(
        class_name="robot_rl.network.moe_ppo.MoEPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=symmetric_data_augmentation_episodic
        )
    )

###################################################
# Transformer
###################################################
##
# Symmetric Causal Transformer Model
##
@configclass
class RslRlCausalTransformerModelCfg(RslRlMLPModelCfg):
    """Configuration for the Causal Transformer model."""
    class_name: str = "robot_rl.network.transformer_network:CausalTransformerModel"
    single_obs_dim: int = 72
    history_length: int = 50
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.0
    # Indices below are tied to PolicyCfg declaration order in g1_clf_tracking_base.py
    use_velocity_embedding: bool = False
    velocity_command_start: int = 6
    velocity_command_dim: int = 3

##
# Symmetric Causal Transformer Runner
##
@configclass
class CausalTransformerPPORunnerCfg(PPORunnerCfg):
    """PPO runner config using a causal transformer actor."""
    experiment_name = "g1_causal_transformer"
    obs_groups = {
        "actor": ["unpriv_policy"],
        "critic": ["critic"],
    }
    actor = RslRlCausalTransformerModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        obs_normalization=False,
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, data_augmentation_func=symmetric_data_augmentation_episodic
        )
    )



##
# Walk-Run Causal Transformer Runner (actor + critic both transformer)
##
@configclass
class WalkRunCausalTransformerPPORunnerCfg(MultiSkillSymmetricHalfPeriodicPPORunnerCfg):
    """PPO runner for the walk-run env with a 25-step causal transformer for
    both actor and critic.

    Pairs with :class:`G1WalkRunCLFTransformerRLEnvCfg`, which sets
    ``history_length=25`` on every term in the ``policy`` and ``critic``
    observation groups.  ``single_obs_dim`` for each model must equal the
    per-step flattened dim of its group (group total dim / 25).  Start with
    the placeholders below, run once, read the per-group obs dim from the
    obs-manager log, divide by 25, and fill in the real values.
    """
    experiment_name = "g1_walk_run_causal_transformer"
    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }
    actor = RslRlCausalTransformerModelCfg(
        history_length=25,
        single_obs_dim=286,  # policy group: 7150 / 25
        hidden_dims=[256, 128],
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.0,
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
        obs_normalization=False,
    )
    critic = RslRlCausalTransformerModelCfg(
        history_length=25,
        single_obs_dim=293,  # critic group: 7325 / 25 (extra 7 dims from base_lin_vel + root_quat)
        hidden_dims=[256, 128],
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.0,
        activation="elu",
        obs_normalization=False,
    )
