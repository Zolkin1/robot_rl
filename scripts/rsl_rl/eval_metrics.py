# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless, massively-parallel metrics eval for an RSL-RL checkpoint.

Rolls out a trained policy across many envs (like training, but in eval mode)
for a fixed number of steps and aggregates the same metrics that are logged
during training, so policies can be understood, compared, and reconciled with
training trends *after* training is over.

Two complementary views are produced:

* ``training_style`` — the exact reset-batch averages the RSL-RL logger writes
  to wandb/tensorboard during training (``Episode_Reward/<term>``,
  ``Episode_Termination/<term>`` including the per-skill ``frame_drift_<skill>``
  breakdowns, and command metrics such as ``error_vel_xy``). Reproduced by
  averaging ``extras["log"]`` over every (post-warmup) step, mirroring
  :class:`rsl_rl.utils.logger.Logger`.
* ``per_episode`` — true per-completed-episode distributions (mean/std/min/max
  and 10/50/90 percentiles) of episode length, total return, and per reward
  term, plus termination-cause counts. Accumulated by us because
  ``env.step()`` resets terminated envs internally before returning.

Results are written as JSON + CSV next to the checkpoint and printed as a
console table. Per-skill breakdowns are included automatically when the env's
``traj_ref`` command exposes skills (multiskill tasks).

Modeled on ``eval_lyapunov.py`` (headless rollout) and ``play.py`` (env/policy
construction).
"""

import argparse
import contextlib
import csv
import importlib.metadata as metadata
import json
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

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


parser = argparse.ArgumentParser(
    description="Headless parallel metrics eval for an RSL-RL checkpoint."
)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments to roll out in parallel.")
parser.add_argument("--task", type=str, default=None)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument(
    "--max_steps", type=int, default=6000,
    help="Number of env steps to roll out before aggregating.",
)
parser.add_argument(
    "--warmup_steps", type=int, default=200,
    help="Episodes that complete before this step are discarded (startup transient).",
)
parser.add_argument(
    "--stochastic", action="store_true", default=False,
    help="Sample actions like training instead of using the deterministic mean action.",
)
parser.add_argument(
    "--no_per_skill", action="store_true", default=False,
    help="Disable the per-skill metric breakdown even if the env exposes skills.",
)
parser.add_argument(
    "--output_dir", type=str, default=None,
    help="Where to write the report. Defaults to <checkpoint_dir>/eval_metrics/.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# force headless: this is a batch metrics job, never render.
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

installed_version = metadata.version("rsl-rl-lib")


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    """Return mean/std/min/max and 10/50/90 percentiles for a 1-D sample.

    Args:
        values: 1-D array of per-episode samples.

    Returns:
        Dictionary of summary statistics. ``count`` is always present; the rest
        are present only when ``values`` is non-empty.
    """
    count = int(values.size)
    if count == 0:
        return {"count": 0}
    return {
        "count": count,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
    }


class EvalMetrics:
    """Accumulates training-style aggregates and per-episode distributions.

    Mirrors the RSL-RL logger's reset-batch averaging for the
    ``training_style`` view, and maintains its own per-env accumulators (since
    ``env.step()`` resets terminated envs before returning) for the
    ``per_episode`` and ``per_skill`` views.
    """

    def __init__(self, env: RslRlVecEnvWrapper, warmup_steps: int, per_skill: bool):
        """Set up accumulators from the wrapped env's managers.

        Args:
            env: The RSL-RL-wrapped manager-based env.
            warmup_steps: Episodes completing before this global step are dropped.
            per_skill: Whether to attempt per-skill bucketing.
        """
        self._unwrapped = env.unwrapped
        self._rm = self._unwrapped.reward_manager
        self._tm = self._unwrapped.termination_manager
        self._dt: float = self._unwrapped.step_dt
        self._num_envs: int = self._unwrapped.num_envs
        self._device = self._unwrapped.device
        self._warmup_steps = warmup_steps

        self.reward_terms: list[str] = list(self._rm.active_terms)
        self.termination_terms: list[str] = list(self._tm.active_terms)
        self.max_episode_length_s: float = self._unwrapped.max_episode_length_s

        # -- training-style running sums over extras["log"] (reset-batch means)
        self._log_sum: dict[str, float] = {}
        self._log_count: dict[str, int] = {}

        # -- per-env accumulators for the current (in-progress) episode
        n, t = self._num_envs, len(self.reward_terms)
        self._cur_len = torch.zeros(n, dtype=torch.long, device=self._device)
        self._cur_return = torch.zeros(n, dtype=torch.float, device=self._device)
        self._cur_term_return = torch.zeros((n, t), dtype=torch.float, device=self._device)

        # -- collected per-episode samples (chunks of CPU tensors, concatenated at the end)
        self._len_chunks: list[torch.Tensor] = []
        self._ret_chunks: list[torch.Tensor] = []
        self._term_ret_chunks: list[torch.Tensor] = []
        self._cause_chunks: list[torch.Tensor] = []  # bool (n_done, n_term_terms)
        self._terminated_chunks: list[torch.Tensor] = []
        self._timeout_chunks: list[torch.Tensor] = []
        self._skill_chunks: list[torch.Tensor] = []

        # -- per-skill support (best-effort)
        self._skill_cmd = None
        self.skill_names: list[str] | None = None
        if per_skill:
            self._setup_skill_tracking()

    def _setup_skill_tracking(self) -> None:
        """Detect a skill-aware command term and cache the ordered skill names."""
        try:
            cmd = self._unwrapped.command_manager.get_term("traj_ref")
        except Exception:
            # Per-skill breakdown is best-effort: any failure to find/inspect the
            # command term simply disables it rather than aborting the eval.
            return
        if not hasattr(cmd, "skill_id"):
            return
        names = getattr(cmd, "_skill_list", None) or getattr(cmd, "skill_labels", None)
        if not names:
            return
        self._skill_cmd = cmd
        self.skill_names = list(names)

    def _current_skill_ids(self) -> torch.Tensor | None:
        """Return the per-env active skill index, or ``None`` if unavailable."""
        if self._skill_cmd is None:
            return None
        return self._skill_cmd.skill_id

    def record_step(self, rew: torch.Tensor, dones: torch.Tensor, extras: dict, global_step: int) -> None:
        """Accumulate one env step and flush any episodes that just completed.

        Must be called immediately after ``env.step()`` so the reward and
        termination managers still hold this step's per-term values.

        Args:
            rew: Per-env reward for this step (shape ``(num_envs,)``).
            dones: Per-env done flags from the wrapper (terminated | truncated).
            extras: The ``extras`` dict returned by the wrapper.
            global_step: The current rollout step index (for warmup gating).
        """
        past_warmup = global_step >= self._warmup_steps

        # (a) training-style aggregate: average extras["log"] over every step.
        if past_warmup:
            log = extras.get("episode") or extras.get("log")
            if log:
                for key, value in log.items():
                    val = value.item() if isinstance(value, torch.Tensor) else float(value)
                    self._log_sum[key] = self._log_sum.get(key, 0.0) + val
                    self._log_count[key] = self._log_count.get(key, 0) + 1

        # (b) per-episode accumulation. _step_reward is value/dt, so *dt recovers
        # the per-term contribution that Episode_Reward sums over the episode.
        self._cur_len += 1
        self._cur_return += rew
        self._cur_term_return += self._rm._step_reward * self._dt

        done_mask = dones.bool()
        if not bool(done_mask.any()):
            return
        idx = done_mask.nonzero(as_tuple=False).squeeze(-1)

        if past_warmup:
            self._len_chunks.append(self._cur_len[idx].cpu())
            self._ret_chunks.append(self._cur_return[idx].cpu())
            self._term_ret_chunks.append(self._cur_term_return[idx].cpu())
            # termination cause: which term(s) fired on the terminal step.
            self._cause_chunks.append(self._tm._term_dones[idx].cpu())
            self._terminated_chunks.append(self._tm.terminated[idx].cpu())
            self._timeout_chunks.append(self._tm.time_outs[idx].cpu())
            skill_ids = self._current_skill_ids()
            if skill_ids is not None:
                self._skill_chunks.append(skill_ids[idx].cpu())

        # reset accumulators for the envs that just reset.
        self._cur_len[idx] = 0
        self._cur_return[idx] = 0.0
        self._cur_term_return[idx] = 0.0

    # -- aggregation -------------------------------------------------------

    def _training_style(self) -> dict[str, float]:
        """Reproduce the per-key reset-batch means logged during training."""
        return {key: self._log_sum[key] / self._log_count[key] for key in sorted(self._log_sum)}

    def _termination_breakdown(self, causes: np.ndarray, n_episodes: int) -> dict[str, dict[str, float]]:
        """Per-term fraction/count of completed episodes that fired that term."""
        out: dict[str, dict[str, float]] = {}
        for i, name in enumerate(self.termination_terms):
            fired = int(causes[:, i].sum()) if n_episodes else 0
            out[name] = {
                "count": fired,
                "fraction": (fired / n_episodes) if n_episodes else 0.0,
            }
        return out

    def aggregate(self) -> dict:
        """Build the full nested result dictionary from collected samples."""
        n_episodes = sum(int(c.numel()) for c in self._len_chunks)

        result: dict = {
            "training_style": self._training_style(),
            "per_episode": {},
            "per_skill": {},
        }
        result["total_completed_episodes"] = n_episodes
        if n_episodes == 0:
            return result

        lengths = torch.cat(self._len_chunks).numpy().astype(np.float64)
        returns = torch.cat(self._ret_chunks).numpy().astype(np.float64)
        term_returns = torch.cat(self._term_ret_chunks).numpy().astype(np.float64)  # (N, n_terms)
        causes = torch.cat(self._cause_chunks).numpy()  # bool (N, n_term_terms)
        terminated = torch.cat(self._terminated_chunks).numpy()
        timeouts = torch.cat(self._timeout_chunks).numpy()

        per_ep: dict = {
            "episode_length": _summary_stats(lengths),
            "total_return": _summary_stats(returns),
            "reward_terms": {
                name: _summary_stats(term_returns[:, i]) for i, name in enumerate(self.reward_terms)
            },
            "termination_causes": self._termination_breakdown(causes, n_episodes),
            "termination_kind": {
                "terminated": float(terminated.sum() / n_episodes),
                "timeout": float(timeouts.sum() / n_episodes),
            },
        }
        result["per_episode"] = per_ep

        # convenient top-level summary scalars (correspond to Train/mean_* in training).
        result["summary"] = {
            "mean_episode_reward": per_ep["total_return"]["mean"],
            "mean_episode_length": per_ep["episode_length"]["mean"],
            "total_completed_episodes": n_episodes,
        }

        # per-skill breakdown
        if self._skill_chunks and self.skill_names is not None:
            skills = torch.cat(self._skill_chunks).numpy()
            for sid, sname in enumerate(self.skill_names):
                mask = skills == sid
                count = int(mask.sum())
                if count == 0:
                    continue
                result["per_skill"][sname] = {
                    "count": count,
                    "episode_length": _summary_stats(lengths[mask]),
                    "total_return": _summary_stats(returns[mask]),
                    "reward_terms": {
                        name: _summary_stats(term_returns[mask, i])
                        for i, name in enumerate(self.reward_terms)
                    },
                    "termination_causes": self._termination_breakdown(causes[mask], count),
                }
        return result


# -- output helpers --------------------------------------------------------


def _flatten(prefix: str, obj, rows: list[tuple[str, float]]) -> None:
    """Flatten a nested dict of scalars into ``(path, value)`` rows."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(f"{prefix}/{key}" if prefix else key, value, rows)
    elif isinstance(obj, (int, float)):
        rows.append((prefix, float(obj)))


def write_outputs(result: dict, output_dir: str, meta: dict) -> None:
    """Write the JSON report and flat CSV summaries.

    Args:
        result: The aggregated metrics dictionary.
        output_dir: Directory to write into (created if missing).
        meta: Run metadata to embed under ``meta`` in the JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    payload = {"meta": meta, **result}

    json_path = os.path.join(output_dir, "eval_metrics.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[INFO] Wrote JSON report to {json_path}")

    # flat summary CSV (everything except per_skill, which gets its own file)
    rows: list[tuple[str, float]] = []
    for section in ("summary", "training_style", "per_episode"):
        _flatten(section, result.get(section, {}), rows)
    csv_path = os.path.join(output_dir, "eval_metrics_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for name, value in rows:
            writer.writerow([name, f"{value:.6f}"])
    print(f"[INFO] Wrote summary CSV to {csv_path}")

    # per-skill CSV
    if result.get("per_skill"):
        skill_rows: list[tuple[str, str, float]] = []
        for skill, data in result["per_skill"].items():
            flat: list[tuple[str, float]] = []
            _flatten("", data, flat)
            for name, value in flat:
                skill_rows.append((skill, name, value))
        skill_csv = os.path.join(output_dir, "eval_metrics_per_skill.csv")
        with open(skill_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["skill", "metric", "value"])
            for skill, name, value in skill_rows:
                writer.writerow([skill, name, f"{value:.6f}"])
        print(f"[INFO] Wrote per-skill CSV to {skill_csv}")


def print_report(result: dict, meta: dict) -> None:
    """Pretty-print a human-readable summary table to stdout."""
    bar = "=" * 76
    print(f"\n{bar}\nEVAL METRICS  |  {meta['task']}\n{bar}")
    print(f"  checkpoint : {meta['checkpoint']}")
    print(f"  envs       : {meta['num_envs']}   steps: {meta['max_steps']}   "
          f"warmup: {meta['warmup_steps']}   action: {meta['action_mode']}")
    print(f"  episodes   : {result.get('total_completed_episodes', 0)} completed (post-warmup)")

    if result.get("total_completed_episodes", 0) == 0:
        print("  [WARN] No episodes completed after warmup — nothing to summarize.")
        return

    s = result["summary"]
    print(f"  mean return: {s['mean_episode_reward']:.3f}   "
          f"mean ep length: {s['mean_episode_length']:.1f} steps")

    pe = result["per_episode"]
    print(f"\n  {'Reward term':<34}{'mean':>10}{'std':>10}{'p10':>9}{'p90':>9}")
    print("  " + "-" * 70)
    for name, st in pe["reward_terms"].items():
        print(f"  {name:<34}{st['mean']:>10.3f}{st['std']:>10.3f}{st['p10']:>9.3f}{st['p90']:>9.3f}")

    print(f"\n  {'Termination cause':<40}{'count':>10}{'fraction':>12}")
    print("  " + "-" * 62)
    for name, st in sorted(pe["termination_causes"].items(), key=lambda kv: -kv[1]["fraction"]):
        if st["count"] == 0:
            continue
        print(f"  {name:<40}{st['count']:>10}{st['fraction']:>12.3f}")
    tk = pe["termination_kind"]
    print(f"  {'(terminated / timeout)':<40}{'':>10}{tk['terminated']:>6.3f} / {tk['timeout']:.3f}")

    # command tracking errors (from training_style log keys)
    ts = result["training_style"]
    cmd_keys = [k for k in ts if k.startswith("error_") or "Metrics" in k]
    if cmd_keys:
        print("\n  Command metrics:")
        for k in cmd_keys:
            print(f"    {k:<40}{ts[k]:>12.4f}")

    if result.get("per_skill"):
        print(f"\n  {'Per-skill':<20}{'episodes':>10}{'mean ret':>12}{'mean len':>12}")
        print("  " + "-" * 54)
        for skill, data in result["per_skill"].items():
            print(f"  {skill:<20}{data['count']:>10}"
                  f"{data['total_return']['mean']:>12.3f}"
                  f"{data['episode_length']['mean']:>12.1f}")
    print(bar + "\n")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Roll out the checkpoint and write the metrics report."""
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
                play_log_path = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.teacher_experiment))
            else:
                play_log_path = log_root_path
            resume_path = get_checkpoint_path(play_log_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        checkpoint_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = checkpoint_dir
        output_dir = args_cli.output_dir or os.path.join(checkpoint_dir, "eval_metrics")

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

        collector = EvalMetrics(env, warmup_steps=args_cli.warmup_steps, per_skill=not args_cli.no_per_skill)
        action_mode = "stochastic" if args_cli.stochastic else "deterministic"
        print(f"[INFO] Rolling out for {args_cli.max_steps} steps across "
              f"{env.unwrapped.num_envs} envs ({action_mode} actions)...")

        obs = env.get_observations()
        for step in range(args_cli.max_steps):
            with torch.inference_mode():
                actions = policy(obs, stochastic_output=args_cli.stochastic)
                obs, rew, dones, extras = env.step(actions)
                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
            collector.record_step(rew, dones, extras, step)
            if (step + 1) % 500 == 0:
                print(f"  ...{step + 1}/{args_cli.max_steps} steps")

        result = collector.aggregate()
        env.close()

        meta = {
            "task": args_cli.task,
            "checkpoint": resume_path,
            "num_envs": env_cfg.scene.num_envs,
            "max_steps": args_cli.max_steps,
            "warmup_steps": args_cli.warmup_steps,
            "action_mode": action_mode,
            "seed": agent_cfg.seed,
            "step_dt": collector._dt,
            "max_episode_length_s": collector.max_episode_length_s,
        }
        print_report(result, meta)
        write_outputs(result, output_dir, meta)
        print(f"[INFO] Done. Report directory: {output_dir}")


if __name__ == "__main__":
    main()
