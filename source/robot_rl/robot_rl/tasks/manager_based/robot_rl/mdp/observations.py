from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv


def sin_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    phase = torch.tensor(2*torch.pi * (env.sim.current_time / period))
    sphase = torch.sin(phase)

    sphase = torch.ones((env.num_envs, 1), device=env.device) * sphase;

    return sphase

def cos_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    phase = torch.tensor(2*torch.pi * (env.sim.current_time / period))
    cphase = torch.cos(phase)

    cphase = torch.ones((env.num_envs, 1), device=env.device) * cphase

    return cphase

def is_ground_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    sp = sin_phase(env, period)
    cp = cos_phase(env, period)

    return torch.tensor([(sp < 0.0), (cp < 0.0)])