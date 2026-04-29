# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a checkpoint on a multiskill task and report per-trajectory Lyapunov error.

Mirrors the env/policy construction in ``play.py`` but strips video,
JIT/ONNX export, MoE hooks, real-time pacing, and plotting. Runs the
policy deterministically for a fixed number of steps (in parallel across
all envs), then reads the per-trajectory mean V from
:class:`DataLogger` and writes a sorted CSV next to the checkpoint.
"""

import argparse
import contextlib
import csv
import importlib.metadata as metadata
import os
import sys

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import robot_rl  # noqa: F401 — register gym environments
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip
from data_logger import DataLogger  # isort: skip

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(
    description="Per-trajectory Lyapunov eval for an RSL-RL checkpoint."
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument(
    "--steps", type=int, default=2000,
    help="Number of rollout steps to evaluate.",
)
parser.add_argument(
    "--csv_path", type=str, default=None,
    help="Where to write the per-trajectory CSV. Defaults to "
         "<checkpoint_dir>/eval_lyapunov.csv.",
)
parser.add_argument(
    "--top_k", type=int, default=20,
    help="Number of hardest trajectories to print to stdout.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run the rollout and write a per-trajectory Lyapunov report."""
    with launch_simulation(env_cfg, args_cli):
        import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401

        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] No pre-trained checkpoint available for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            if agent_cfg.class_name == "DistillationRunner" and getattr(args_cli, "teacher_experiment", None):
                play_log_path = os.path.abspath(
                    os.path.join("logs", "rsl_rl", args_cli.teacher_experiment)
                )
            else:
                play_log_path = log_root_path
            resume_path = get_checkpoint_path(play_log_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent
            env = multi_agent_to_single_agent(env)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO] Loading checkpoint: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        logger = DataLogger(env)
        obs = env.get_observations()

        print(f"[INFO] Rolling out for {args_cli.steps} steps across {env_cfg.scene.num_envs} envs...")
        for t in range(args_cli.steps):
            with torch.inference_mode():
                actions = policy(obs, stochastic_output=False)
                obs, _, dones, _ = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
            logger.collect_step(env)

        data, meta = logger.finalize()
        env.close()

        if "per_traj_v_mean" not in data:
            print("[ERROR] No per-trajectory data collected. Is this a multiskill task?")
            return

        means = np.asarray(data["per_traj_v_mean"])
        counts = np.asarray(data["per_traj_v_count"])
        names = meta.get("per_traj_names") or [f"traj_{i}" for i in range(len(means))]

        order = np.argsort(-means)
        csv_path = args_cli.csv_path or os.path.join(log_dir, "eval_lyapunov.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["rank", "trajectory", "mean_v", "count"])
            for rank, i in enumerate(order):
                writer.writerow([rank, names[i], f"{means[i]:.6f}", int(counts[i])])
        print(f"[INFO] Wrote per-trajectory report to {csv_path}")

        k = min(int(args_cli.top_k), len(order))
        print(f"\nTop {k} hardest trajectories by mean V:")
        print(f"  {'rank':>4}  {'trajectory':<40}  {'mean_v':>10}  {'count':>8}")
        for rank, i in enumerate(order[:k]):
            print(f"  {rank:>4}  {names[i]:<40}  {means[i]:>10.4f}  {int(counts[i]):>8}")


if __name__ == "__main__":
    main()
