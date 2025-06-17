from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.tasks.base import Task, TaskConfig
from judo.tasks.cost_functions import (
    quadratic_norm,
    smooth_l1_norm,
)

# TODO: Adjust to match Isaac
XML_PATH = str(MODEL_PATH / "xml/cartpole.xml")


@dataclass
class SpcMpcCartpoleConfig(TaskConfig):
    """Reward configuration for the cartpole task."""

    Q: np.ndarray       # Diagonal for now
    q: np.ndarray


class SpcMpcCartpole(Task[SpcMpcCartpoleConfig]):
    """Defines the cartpole balancing task."""

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the cartpole task."""
        super().__init__(model_path, sim_model_path=sim_model_path)
        self.reset()

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        config: SpcMpcCartpoleConfig,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Simple quadratic reward.

        Returns:
            A list of rewards shaped (batch_size,) where reward at index i represents the reward for that batched traj
        """

        # Compute cost: cost_i = sum_t x_{i,t}^T Q x_{i,t}
        # Shape: [N, timesteps].sum(1)
        quadratic_terms = np.einsum('ntk,kl,ntl->nt', states, np.diag(config.Q), states).sum(1)

        linear_terms = (states * config.q).sum(-1).sum(-1)

        return -0.5*quadratic_terms - linear_terms

    def reset(self) -> None:
        """Resets the model to a default (random) state."""
        self.data.qpos = np.array([1.0, np.pi]) + np.random.randn(2)
        self.data.qvel = 1e-1 * np.random.randn(2)
        mujoco.mj_forward(self.model, self.data)