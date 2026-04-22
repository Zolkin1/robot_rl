from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.observations import generated_commands

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def ref_traj(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    ref_traj = cmd.y_des.clone()
    return ref_traj

def act_traj(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    act_traj = cmd.y_act.clone()
    return act_traj

def ref_traj_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    ref_traj_vel = cmd.dy_des
    return ref_traj_vel

def act_traj_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    act_traj_vel = cmd.dy_act
    return act_traj_vel


def ref_sin_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)

    phase = 2*torch.pi * cmd.get_phasing_var()

    sphase = torch.sin(phase)
    if sphase.ndim == 1:
        # [B] → [B, 1]
        sphase = sphase.unsqueeze(-1)

    return sphase

def ref_cos_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)

    phase = 2*torch.pi * cmd.get_phasing_var()

    cphase = torch.cos(phase)
    if cphase.ndim == 1:
        # [B] → [B, 1]
        cphase = cphase.unsqueeze(-1)
    return cphase

def skill_one_hot(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """One-hot encoding of the active velocity bucket (skill) for each environment.

    Reads ``bucket_id`` from a :class:`MultiskillVelocityTrackingCommand` term and converts it
    into a one-hot vector of length ``len(cfg.velocity_buckets)``. Envs assigned to the implicit
    default bucket (when bucket percentages do not sum to 1) are returned as all-zeros.

    Returns a tensor of shape ``(num_envs, num_buckets)``.
    """
    cmd = env.command_manager.get_term(command_name)
    num_buckets = len(cmd.cfg.velocity_buckets)
    one_hot = torch.zeros(env.num_envs, num_buckets, device=env.device)
    in_bucket = cmd.bucket_id < num_buckets
    one_hot[in_bucket] = torch.nn.functional.one_hot(
        cmd.bucket_id[in_bucket], num_classes=num_buckets
    ).float()
    return one_hot


def multiskill_phase(env: ManagerBasedRLEnv, frequency_list: list[float], command_name: str) -> torch.Tensor:
    """Create phasing variables at different frequencies to cover a range.

    Returns a tensor of shape (num_envs, 2 * num_frequencies) with sin and cos
    values interleaved: [sin_f0, cos_f0, sin_f1, cos_f1, ...].
    """
    # TODO: Make the frequency list read from the traj_ref command
    frequencies = torch.tensor(frequency_list, device=env.device)   # Hz
    num_freq = len(frequencies)

    ## DEBUG
    # cmd = env.command_manager.get_term(command_name)
    # print(f"Command phasing var: {cmd.get_phasing_var()}")
    ## DEBUG END

    # TODO: Should align this with the sampled reset time
    t = env.episode_length_buf * env.step_dt

    ## DEBUG
    # full_period = [2*0.299, 2*0.46]
    # phasing_vars = [(t % full_period[i])/(full_period[i]) for i in range(num_freq)]
    # commanded_velocity = env.command_manager.get_command("base_velocity")
    # print(f"Obs phasing var: {phasing_vars}\nCommanded velocity: {commanded_velocity}\n")
    ## DEBUG END

    sp = torch.zeros(env.num_envs, num_freq, device=env.device)
    cp = torch.zeros(env.num_envs, num_freq, device=env.device)

    for i in range(num_freq):
        sp[:, i] = torch.sin(2 * torch.pi * frequencies[i] * t)
        cp[:, i] = torch.cos(2 * torch.pi * frequencies[i] * t)

    return torch.cat([sp, cp], dim=-1)
