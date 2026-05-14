from __future__ import annotations

import torch
import warp as wp
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp.observations import generated_commands

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.phased_trajectory_cmd import (
    PhasedTrajectoryCommand,
)

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


# TODO: Delete ``_phased_cmd``, ``ref_sin_phase``, ``ref_cos_phase``
# along with :class:`PhasedTrajectoryCommand`.  The multi-skill bases
# use :func:`multiskill_phase` (below) which reads the multi-skill
# command's phase directly — replace any g1 single-skill obs that still
# wires up sin/cos phase via ``_phased_cmd`` before removing.
def _phased_cmd(env: ManagerBasedRLEnv, command_name: str) -> PhasedTrajectoryCommand:
    cmd = env.command_manager.get_term(command_name)
    if not isinstance(cmd, PhasedTrajectoryCommand):
        raise TypeError(
            f"Phase observation requires a PhasedTrajectoryCommand for "
            f"'{command_name}', got {type(cmd).__name__}."
        )
    return cmd


def ref_sin_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = _phased_cmd(env, command_name)

    phase = 2*torch.pi * cmd.get_phasing_var()

    sphase = torch.sin(phase)
    if sphase.ndim == 1:
        # [B] → [B, 1]
        sphase = sphase.unsqueeze(-1)

    return sphase

def ref_cos_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    cmd = _phased_cmd(env, command_name)

    phase = 2*torch.pi * cmd.get_phasing_var()

    cphase = torch.cos(phase)
    if cphase.ndim == 1:
        # [B] → [B, 1]
        cphase = cphase.unsqueeze(-1)
    return cphase

def grace_period_active(
    env: ManagerBasedRLEnv,
    command_name: str,
    grace_period_s: float,
    one_hot: bool = True,
) -> torch.Tensor:
    """Indicator for whether the post-trajectory-change grace period is active.

    Reads ``time_since_traj_change_s`` from the named command term (populated
    by :class:`BatchedMultiSkillCommand._pre_update_phase`) and compares it
    against the configured grace window. The flag is True while
    ``time_since_traj_change_s < grace_period_s``.

    Args:
        env: The Isaac Lab RL environment.
        command_name: Trajectory command term exposing
            ``time_since_traj_change_s``.
        grace_period_s: Grace-window length in seconds. Pair with the value
            on the matching ``frame_deviation_from_reference`` termination
            to keep them in sync.
        one_hot: When True, return a ``[num_envs, 2]`` one-hot
            ``[not_in_grace, in_grace]``. When False, return a
            ``[num_envs, 1]`` 0/1 flag.

    Returns:
        Float tensor of shape ``[num_envs, 2]`` (one-hot) or
        ``[num_envs, 1]`` (single flag). If the command term doesn't expose
        ``time_since_traj_change_s`` (e.g. single-skill commands), reports
        "not in grace" for all envs.
    """
    cmd = env.command_manager.get_term(command_name)
    time_since = getattr(cmd, "time_since_traj_change_s", None)
    if time_since is None:
        return torch.zeros(
            (env.num_envs, 2 if one_hot else 1),
            device=env.device,
        )
    in_grace = time_since < grace_period_s
    if one_hot:
        return torch.nn.functional.one_hot(in_grace.long(), num_classes=2).float()
    return in_grace.float().unsqueeze(-1)


def skill_one_hot(env: ManagerBasedRLEnv, command_name: str = "traj_ref") -> torch.Tensor:
    """One-hot encoding of the trajectory cmd's active skill for each env.

    Reads ``skill_id`` from a :class:`BatchedMultiSkillCommand` term and
    converts it into a one-hot vector of length
    ``len(cfg.velocity_buckets)``.  The active skill is bucket-derived
    from the live ``vel_target_b`` and lags the velocity cmd's sampled
    bucket across the gate-on-contact deferral window — this is the
    skill the policy is actually executing.

    Returns a tensor of shape ``(num_envs, num_buckets)``.
    """
    cmd = env.command_manager.get_term(command_name)
    num_buckets = len(cmd.cfg.velocity_buckets)
    one_hot = torch.zeros(env.num_envs, num_buckets, device=env.device)
    in_bucket = cmd.skill_id < num_buckets
    one_hot[in_bucket] = torch.nn.functional.one_hot(
        cmd.skill_id[in_bucket], num_classes=num_buckets
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
