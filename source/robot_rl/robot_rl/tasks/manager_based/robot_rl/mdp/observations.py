from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv


def foot_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    left_foot_vel = cmd.robot.data.body_lin_vel_w[:,cmd.feet_bodies_idx[0],:]
    right_foot_vel = cmd.robot.data.body_lin_vel_w[:,cmd.feet_bodies_idx[1],:]

    foot_vel = torch.cat([left_foot_vel, right_foot_vel], dim=-1)

    return foot_vel

def foot_ang_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    left_foot_ang_vel = cmd.robot.data.body_ang_vel_w[:,cmd.feet_bodies_idx[0],:]
    right_foot_ang_vel = cmd.robot.data.body_ang_vel_w[:,cmd.feet_bodies_idx[1],:]

    foot_ang_vel = torch.cat([left_foot_ang_vel, right_foot_ang_vel], dim=-1)

    return foot_ang_vel

def ref_traj(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    ref_traj = cmd.y_out
    return ref_traj

def act_traj(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    act_traj = cmd.y_act
    return act_traj


def ref_traj_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    ref_traj_vel = cmd.dy_out
    return ref_traj_vel

def act_traj_vel(env: ManagerBasedRLEnv, command_name:str = "hlip_ref") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    act_traj_vel = cmd.dy_act
    return act_traj_vel

def joint_pos_des(env: ManagerBasedRLEnv, cmd_name:str) -> torch.Tensor:
    cmd = env.command_manager.get_term(cmd_name)
    joint_pos_des = cmd.joint_pos_des
    return joint_pos_des

def v_dot(env: ManagerBasedRLEnv, command_name:str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    v_dot = cmd.vdot.unsqueeze(-1)
    return v_dot

def v(env: ManagerBasedRLEnv, command_name:str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    v = cmd.v.unsqueeze(-1)
    return v
        

def stair_sin_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    phase = 2*torch.pi * cmd.tp
    sphase = torch.sin(phase)
    if sphase.ndim == 1:
        # [B] → [B, 1]
        sphase = sphase.unsqueeze(-1)

    return sphase

def stair_cos_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    phase = 2*torch.pi * cmd.tp
    cphase = torch.cos(phase)
    if cphase.ndim == 1:
        # [B] → [B, 1]
        cphase = cphase.unsqueeze(-1)
    return cphase

def sin_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    phase = torch.tensor(2*torch.pi * (env.sim.current_time / period))
    sphase = torch.sin(phase)

    sphase = torch.ones((env.num_envs, 1), device=env.device) * sphase

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

def step_location(env: ManagerBasedRLEnv) -> torch.Tensor:
    foot_pos = env.cfg.current_des_step
    return foot_pos