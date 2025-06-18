# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from dataclasses import MISSING
from typing import Literal
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class CustomPPOActorCriticCfg:
    """Configuration for the PPO actor-critic networks."""

    class_name: str = "ActorCriticCNN"
    """The policy class name. Default is ActorCritic."""

    init_noise_std: float = MISSING
    """The initial noise standard deviation for the policy."""

    noise_std_type: Literal["scalar", "log"] = "scalar"
    """The type of noise standard deviation for the policy. Default is scalar."""

    actor_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the actor network."""

    critic_hidden_dims: list[int] = MISSING
    """The hidden dimensions of the critic network."""

    activation: str = MISSING
    """The activation function for the actor and critic networks."""

    height_map_shape: tuple[int, int, int] = (1, 16,16)
    """The shape of the height map (C, H, W)."""

    proprio_dim: int = 75
    """The dimension of the proprioceptive features."""

@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 200
    experiment_name = "g1"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
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
    resume_path = None


@configclass
class StairPPOCfg(PPORunnerCfg):
    # resume = False
    # resume_path = None
    policy = None
    policy = CustomPPOActorCriticCfg(
        class_name="ActorCriticCNN",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        height_map_shape=(1, 25,25),
    )
    resume = True
    resume_path = "/home/amy/gitrepo/robot_rl/logs/g1_policies/height-scan-flat/g1/2025-06-18_13-27-29/model_3600.pt"



@configclass
class StairCNNPPOCfg(PPORunnerCfg):
    resume = False
    resume_path = None
    policy = None
    policy = CustomPPOActorCriticCfg(
        class_name="ActorCriticCNN",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        height_map_shape=(1, 25,25),
    )
