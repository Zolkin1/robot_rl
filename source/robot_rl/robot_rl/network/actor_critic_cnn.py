# my_modules/cnn_actor_critic.py   (excerpt)

from math import prod
import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.utils import resolve_nn_activation

class ActorCriticCNN(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        height_map_shape,   # (C, H, W)
        actor_dim: int,               # actor-side proprio
        critic_dim: int,                # full critic obs length (after flatten)
        num_actions: int,
        actor_hidden_dims=[256, 256],
        critic_hidden_dims=[256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type="scalar",
        **kwargs,
    ):
        
        # ---------- Common sizes -----------------------------------------
        self.height_map_shape = height_map_shape
        self.hmap_flat = prod(height_map_shape)           # N_hmap

        C, H, W = height_map_shape
        act_fn = resolve_nn_activation(activation)
        cnn_out = 32 


        # ---------- Dimensions for actor / critic -----------------------
        actor_proprio_dim = actor_dim - self.hmap_flat
        actor_in_dim = cnn_out + actor_proprio_dim


        critic_proprio_dim = critic_dim - self.hmap_flat
        if critic_proprio_dim < 0:
            raise ValueError(
                f"critic_dim ({critic_dim}) is smaller than the flattened "
                f"height-map size ({self.hmap_flat}).")

        critic_in_dim  = cnn_out + critic_proprio_dim
    
        # ---------- Call superclass to build heads ----------------------
        super().__init__(
            num_actor_obs=actor_in_dim,
            num_critic_obs=critic_in_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        

        # ---------- CNN trunk -------------------------------------------
        # --- CNN feature extractor ---
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 16, 3, stride=2, padding=1), act_fn,   # (C, 41, 41) → (16, 21, 21)
            nn.Conv2d(16, 32, 3, stride=2, padding=1), act_fn,  # → (32, 11, 11)
            nn.Conv2d(32, 32, 3, stride=2, padding=1), act_fn,  # → (32, 6, 6)
            nn.AdaptiveAvgPool2d(1),                            # → (32, 1, 1)
            nn.Flatten(),                                       # → (32,)
        )


        

    # ------------- existing helper & overrides stay unchanged ----------
    def _features(self, flat_obs: torch.Tensor) -> torch.Tensor:
        B = flat_obs.shape[0]
        hmap = flat_obs[:, : self.hmap_flat].reshape(B, *self.height_map_shape)
        prop = flat_obs[:, self.hmap_flat :]
        return torch.cat([self.cnn(hmap), prop], dim=-1)

    def update_distribution(self, obs):
        feat = self._features(obs)
        mean = self.actor(feat)
        std  = self.std.expand_as(mean) if self.noise_std_type == "scalar" \
               else torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act_inference(self, obs):          # for deployment
        return self.actor(self._features(obs))

    def evaluate(self, critic_obs, **kw):  # value function
        return self.critic(self._features(critic_obs))
