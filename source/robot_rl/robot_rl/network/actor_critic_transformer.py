# Copyright 2026 Zachary Olkin. All rights reserved.

import torch
import torch.nn as nn
from torch.distributions import Normal
from math import prod
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation


class ActorCriticTransformer(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        height_map_shape,   # (C, H, W)
        actor_dim: int,     # full actor obs length (including hmap)
        critic_dim: int,    # full critic obs length (including hmap)
        num_actions: int,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        **kwargs,
    ):
        
        # ----------------- Dimensions and constants -----------------
        self.height_map_shape = height_map_shape
        self.hmap_flat = prod(height_map_shape)
        C, H, W = height_map_shape
        self.num_tokens = H * W
        self.token_dim = 16  # Transformer embedding size

        self.actor_proprio_dim = actor_dim - self.hmap_flat
        self.critic_proprio_dim = critic_dim - self.hmap_flat

        # ----------------- Update input dims to policy --------------
        actor_input_dim = self.token_dim + self.actor_proprio_dim
        critic_input_dim = self.token_dim + self.critic_proprio_dim

        super().__init__(
            num_actor_obs=actor_input_dim,
            num_critic_obs=critic_input_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )

        

        # ----------------- CNN-style Token Embedding ----------------
        self.hmap_embed = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),   # (25,25) → (12,12)
            nn.Conv2d(C, self.token_dim, kernel_size=3, padding=1),
        )
        self.critic_hmap_embed = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),   # (25,25) → (12,12)
            nn.Conv2d(C, self.token_dim, kernel_size=3, padding=1),
        )

        # ----------------- Transformers -----------------------------
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.token_dim, nhead=1, dim_feedforward=32,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        critic_encoder_layer = nn.TransformerEncoderLayer(d_model=self.token_dim, nhead=1, dim_feedforward=32,batch_first=True)
        self.critic_transformer = nn.TransformerEncoder(critic_encoder_layer, num_layers=1)

    

    def _embed_hmap(self, x: torch.Tensor, embed_layer: nn.Module) -> torch.Tensor:
        # Input: (B, C, H, W)
        x = embed_layer(x)  # -> (B, D, H, W)
        x = x.flatten(2).transpose(1, 2)  # -> (B, N, D), tokens
        return x

    def _features(self, flat_obs: torch.Tensor) -> torch.Tensor:
        B = flat_obs.shape[0]
        hmap = flat_obs[:, :self.hmap_flat].reshape(B, *self.height_map_shape)
        proprio = flat_obs[:, self.hmap_flat:]

        tokens = self._embed_hmap(hmap, self.hmap_embed)
        encoded = self.transformer(tokens)  # (B, N, D)
        pooled = encoded.mean(dim=1)       # (B, D)

        return torch.cat([pooled, proprio], dim=-1)

    def critic_features(self, flat_obs: torch.Tensor) -> torch.Tensor:
        B = flat_obs.shape[0]
        hmap = flat_obs[:, :self.hmap_flat].reshape(B, *self.height_map_shape)
        proprio = flat_obs[:, self.hmap_flat:]

        tokens = self._embed_hmap(hmap, self.critic_hmap_embed)
        encoded = self.critic_transformer(tokens)
        pooled = encoded.mean(dim=1)

        return torch.cat([pooled, proprio], dim=-1)

    def update_distribution(self, obs):
        feat = self._features(obs)
        mean = self.actor(feat)
        std = self.std.expand_as(mean) if self.noise_std_type == "scalar" \
              else torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act_inference(self, obs):
        return self.actor(self._features(obs))

    def evaluate(self, critic_obs, **kwargs):
        return self.critic(self.critic_features(critic_obs))
