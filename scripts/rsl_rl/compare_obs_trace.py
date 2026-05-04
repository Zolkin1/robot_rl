"""Standalone observation/state trace + diff tool for cross-commit comparison.

Usage::

    # Step 1 — generate a trace on the current commit:
    #   (this needs IsaacLab; same launch flags as play.py)
    python compare_obs_trace.py run \\
        --task=<task-name> --checkpoint=<path/to/model.pt> \\
        --output=trace_NEW.pkl --steps=200 --seed=42

    # Step 2 — checkout the OLD commit, run the same command with a different output:
    python compare_obs_trace.py run \\
        --task=<task-name> --checkpoint=<path/to/model.pt> \\
        --output=trace_OLD.pkl --steps=200 --seed=42

    # Step 3 — diff the two traces (no IsaacLab needed):
    python compare_obs_trace.py diff --old=trace_OLD.pkl --new=trace_NEW.pkl

This script is paste-compatible with both pre-24531ad (time-based command) and
post-24531ad (phase-based command) checkouts. Fields that don't exist in one
or the other are silently skipped.

Determinism: pass the same ``--seed`` and ``--checkpoint`` to both runs. Use
``--num_envs=1`` for clean apples-to-apples step traces.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys


# ---------------------------------------------------------------------------
# Mode 1: diff (no IsaacLab dependency)
# ---------------------------------------------------------------------------

def cmd_diff(argv) -> None:
    """Compare two saved traces element-wise and print a summary."""
    import numpy as np

    parser = argparse.ArgumentParser(prog="compare_obs_trace.py diff")
    parser.add_argument("--old", required=True, help="Path to OLD trace pickle")
    parser.add_argument("--new", required=True, help="Path to NEW trace pickle")
    args = parser.parse_args(argv)

    with open(args.old, "rb") as f:
        old = pickle.load(f)
    with open(args.new, "rb") as f:
        new = pickle.load(f)

    print(f"\nOLD: {args.old} ({len(old['steps'])} steps)")
    print(f"NEW: {args.new} ({len(new['steps'])} steps)")
    print(f"OLD commit: {old.get('commit', '<unknown>')}")
    print(f"NEW commit: {new.get('commit', '<unknown>')}")

    n_steps = min(len(old["steps"]), len(new["steps"]))
    print(f"\nComparing first {n_steps} step(s).")

    # Collect all keys ever present
    all_keys: set[str] = set()
    for s in old["steps"][:n_steps]:
        all_keys.update(s.keys())
    for s in new["steps"][:n_steps]:
        all_keys.update(s.keys())

    # Tabular summary, sorted
    rows: list[tuple[str, int, str, float | None, int | None]] = []
    for key in sorted(all_keys):
        n_diff = 0
        n_only_old = 0
        n_only_new = 0
        max_abs_diff: float | None = None
        first_diff_t: int | None = None
        shape_mismatch = False

        for t in range(n_steps):
            ov = old["steps"][t].get(key)
            nv = new["steps"][t].get(key)
            if ov is None and nv is None:
                continue
            if ov is None and nv is not None:
                n_only_new += 1
                continue
            if nv is None and ov is not None:
                n_only_old += 1
                continue
            # Both present — compare
            ov_a = np.asarray(ov)
            nv_a = np.asarray(nv)
            if ov_a.shape != nv_a.shape:
                shape_mismatch = True
                n_diff += 1
                if first_diff_t is None:
                    first_diff_t = t
                continue
            if ov_a.dtype.kind in ("i", "u", "b"):
                if not np.array_equal(ov_a, nv_a):
                    n_diff += 1
                    if first_diff_t is None:
                        first_diff_t = t
            else:
                d = float(np.abs(ov_a.astype(np.float64) - nv_a.astype(np.float64)).max())
                if d > 1e-9:
                    n_diff += 1
                    if first_diff_t is None:
                        first_diff_t = t
                    if max_abs_diff is None or d > max_abs_diff:
                        max_abs_diff = d

        # Tag the row with summary
        marker = "OK" if n_diff == 0 and not n_only_old and not n_only_new else "DIFF"
        if shape_mismatch:
            marker = "SHAPE"
        rows.append((key, n_diff, marker, max_abs_diff, first_diff_t))
        if n_only_old or n_only_new:
            rows.append((key + " (presence)", n_only_old + n_only_new,
                          f"OLD-only={n_only_old} NEW-only={n_only_new}",
                          None, None))

    # Print
    print()
    print(f"{'KEY':<40} {'STATUS':<10} {'N_DIFF':>8} {'FIRST_T':>8} {'MAX_ABS_DIFF':>14}")
    print("-" * 84)
    for key, n_diff, marker, max_abs, first_t in rows:
        max_s = f"{max_abs:.6e}" if max_abs is not None else ""
        first_s = str(first_t) if first_t is not None else ""
        print(f"{key:<40} {marker:<10} {n_diff:>8} {first_s:>8} {max_s:>14}")

    # Highlight the key with the earliest divergence on a non-int field
    earliest = [r for r in rows if r[3] is not None and r[4] is not None]
    if earliest:
        earliest.sort(key=lambda r: r[4])
        print()
        print("Earliest divergence:")
        for r in earliest[:5]:
            print(f"  {r[0]} at t={r[4]} (max_abs_diff so far: {r[3]:.6e})")


# ---------------------------------------------------------------------------
# Mode 2: run (uses IsaacLab) — modeled after scripts/rsl_rl/play.py
# ---------------------------------------------------------------------------

def cmd_run(argv) -> None:
    """Run the policy for N steps and save a per-step state trace.

    Captures (per step):
      - obs[group] for every active observation group (raw, pre-normalisation).
      - cmd.y_des, cmd.y_act, cmd.dy_des, cmd.dy_act.
      - cmd.v, cmd.vdot.
      - cmd.current_domain, cmd.cur_ref_frame_idx, cmd.ref_poses.
      - Action sampled by the policy (deterministic mode).
      - Reward total (scalar) and per-term contributions if exposed.
      - Manager phase / time state — uses safe getattr so it works in OLD
        (no ``manager.phase``) and NEW (no ``cmd.init_time_offset``).
    """
    # --- Heavyweight imports (only in run mode) --------------------------
    import contextlib
    import importlib.metadata as metadata
    import time

    import gymnasium as gym
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

    import isaaclab_tasks  # noqa: F401
    import robot_rl  # noqa: F401 — register gym envs
    from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
    from isaaclab_tasks.utils.hydra import hydra_task_config

    # local
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cli_args  # noqa: E402

    with contextlib.suppress(ImportError):
        import isaaclab_tasks_experimental  # noqa: F401

    # --- Build a compatible argparse object expected by hydra_task_config ---
    # ``hydra_task_config`` reads from a global ``args_cli``; we need the same
    # args structure play.py uses.  We construct it manually from our own.
    parent_parser = argparse.ArgumentParser(add_help=False)
    # Trace-specific args (the rest are added by cli_args + launcher).
    parent_parser.add_argument("--task", type=str, required=True)
    parent_parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
    parent_parser.add_argument("--num_envs", type=int, default=1)
    parent_parser.add_argument("--seed", type=int, default=42)
    parent_parser.add_argument("--output", type=str, required=True)
    parent_parser.add_argument("--steps", type=int, default=200)
    parent_parser.add_argument("--commit_label", type=str, default="",
                               help="Free-form label written into the trace metadata.")
    cli_args.add_rsl_rl_args(parent_parser)  # adds --checkpoint, --resume, etc.
    add_launcher_args(parent_parser)         # adds --device, --disable_fabric, etc.

    args_cli, hydra_args = parent_parser.parse_known_args(argv)
    if not args_cli.checkpoint:
        print("[ERROR] --checkpoint is required (path to model.pt)")
        sys.exit(2)
    sys.argv = [sys.argv[0]] + hydra_args  # hydra needs only its overrides

    installed_version = metadata.version("rsl-rl-lib")

    @hydra_task_config(args_cli.task, args_cli.agent)
    def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
             agent_cfg: RslRlBaseRunnerCfg):
        with launch_simulation(env_cfg, args_cli):
            with contextlib.suppress(ImportError):
                import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401

            agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
            env_cfg.scene.num_envs = args_cli.num_envs
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
            env_cfg.seed = args_cli.seed
            env_cfg.sim.device = (args_cli.device if args_cli.device is not None
                                  else env_cfg.sim.device)

            resume_path = retrieve_file_path(args_cli.checkpoint)
            log_dir = os.path.dirname(resume_path)
            env_cfg.log_dir = log_dir

            env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
            if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
                from isaaclab.envs import multi_agent_to_single_agent
                env = multi_agent_to_single_agent(env)
            env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

            print(f"[INFO] Loading checkpoint: {resume_path}")
            if agent_cfg.class_name == "OnPolicyRunner":
                runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None,
                                         device=agent_cfg.device)
            elif agent_cfg.class_name == "DistillationRunner":
                runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None,
                                             device=agent_cfg.device)
            else:
                raise ValueError(f"Unsupported runner: {agent_cfg.class_name}")
            runner.load(resume_path)

            policy = runner.get_inference_policy(device=env.unwrapped.device)

            # --- Trace setup ---
            unwrapped = env.unwrapped
            cmd_term_name = None
            for name in ("traj_ref", "hlip_ref"):
                if name in unwrapped.command_manager.active_terms:
                    cmd_term_name = name
                    break
            if cmd_term_name is None:
                raise RuntimeError("No trajectory command term found.")
            cmd = unwrapped.command_manager.get_term(cmd_term_name)

            steps: list[dict] = []
            obs = env.get_observations()
            obs_dict = unwrapped.observation_manager.compute()  # raw per-group obs

            for step_idx in range(args_cli.steps):
                with torch.inference_mode():
                    actions = policy(obs)

                # Capture per-step state BEFORE stepping (i.e. the state the
                # policy saw + acted on this step).
                step_state: dict = {}

                # Raw observations per group
                for group, val in obs_dict.items():
                    if isinstance(val, dict):
                        # Group with multiple terms
                        for term, t_val in val.items():
                            step_state[f"obs/{group}/{term}"] = t_val.detach().cpu().numpy()
                    else:
                        step_state[f"obs/{group}"] = val.detach().cpu().numpy()

                # Action
                step_state["action"] = actions.detach().cpu().numpy()

                # Cmd term per-step state (safe getattr — handles OLD/NEW)
                for attr in ("y_des", "y_act", "dy_des", "dy_act", "v", "vdot",
                              "current_domain", "cur_ref_frame_idx", "ref_poses",
                              "traj_time"):
                    val = getattr(cmd, attr, None)
                    if val is not None and hasattr(val, "detach"):
                        step_state[f"cmd/{attr}"] = val.detach().cpu().numpy()

                # Manager state (NEW only — phase, gate state)
                manager = getattr(cmd, "manager", None)
                if manager is not None:
                    for attr in ("phase", "next_gate_idx", "gate_rel_phi"):
                        val = getattr(manager, attr, None)
                        if val is not None and hasattr(val, "detach"):
                            step_state[f"mgr/{attr}"] = val.detach().cpu().numpy()

                # Episode length / done state
                if hasattr(unwrapped, "episode_length_buf"):
                    step_state["ep_len"] = unwrapped.episode_length_buf.detach().cpu().numpy()

                steps.append(step_state)

                # Step the env
                with torch.inference_mode():
                    obs, rewards, dones, _ = env.step(actions)
                    if hasattr(rewards, "detach"):
                        steps[-1]["reward"] = rewards.detach().cpu().numpy()
                    if hasattr(dones, "detach"):
                        steps[-1]["done"] = dones.detach().cpu().numpy()
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    obs_dict = unwrapped.observation_manager.compute()

            # --- Save trace ---
            trace = {
                "commit": args_cli.commit_label,
                "task": args_cli.task,
                "checkpoint": args_cli.checkpoint,
                "seed": args_cli.seed,
                "num_envs": args_cli.num_envs,
                "n_steps": len(steps),
                "steps": steps,
                "metadata": {
                    "step_dt": float(unwrapped.step_dt),
                    "cmd_term_name": cmd_term_name,
                },
            }

            os.makedirs(os.path.dirname(os.path.abspath(args_cli.output)) or ".",
                         exist_ok=True)
            with open(args_cli.output, "wb") as f:
                pickle.dump(trace, f)
            print(f"[INFO] Trace ({len(steps)} steps) → {args_cli.output}")

            env.close()

    main()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Manual dispatch on sys.argv[1] — argparse subparsers + REMAINDER doesn't
    # play well with --foo=bar style options, so we just route to the right
    # mode and let it own argv parsing.
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        print("\nSubcommands: run, diff")
        sys.exit(0)

    cmd = sys.argv[1]
    rest = sys.argv[2:]
    if cmd == "diff":
        cmd_diff(rest)
    elif cmd == "run":
        cmd_run(rest)
    else:
        print(f"Unknown subcommand: {cmd!r}. Use 'run' or 'diff'.")
        sys.exit(2)


if __name__ == "__main__":
    main()
