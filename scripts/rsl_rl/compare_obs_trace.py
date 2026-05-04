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

def _grid_overlay_plot(
    out_path: str,
    title: str,
    time_s,
    old_arr,
    new_arr,
    names,
    n_cols: int = 4,
    y_label: str = "value",
) -> None:
    """Save a grid plot overlaying OLD vs NEW per dimension.

    ``old_arr`` / ``new_arr`` are ``[T, D]`` numpy arrays.  ``names`` provides
    per-dimension titles (length ``D``); short ones are auto-padded.
    """
    import os as _os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_dims = old_arr.shape[1]
    n_rows = max(1, (n_dims + n_cols - 1) // n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 2.6 * n_rows))
    axs_flat = np.array(axs).flatten()
    fig.suptitle(title, fontsize=14)

    T_old = old_arr.shape[0]
    T_new = new_arr.shape[0]
    t_old = time_s[:T_old] if time_s is not None else np.arange(T_old)
    t_new = time_s[:T_new] if time_s is not None else np.arange(T_new)

    for i in range(n_dims):
        ax = axs_flat[i]
        ax.plot(t_old, old_arr[:, i], label="OLD", linewidth=1.4)
        ax.plot(t_new, new_arr[:, i], label="NEW", linestyle="--", linewidth=1.4)
        name = names[i] if i < len(names) else f"dim {i}"
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("Time (s)" if time_s is not None else "step", fontsize=8)
        ax.set_ylabel(y_label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    for i in range(n_dims, len(axs_flat)):
        axs_flat[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    _os.makedirs(_os.path.dirname(_os.path.abspath(out_path)) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Wrote plot: {out_path}")


def cmd_diff(argv) -> None:
    """Compare two saved traces element-wise and print a summary."""
    import numpy as np

    parser = argparse.ArgumentParser(prog="compare_obs_trace.py diff")
    parser.add_argument("--old", required=True, help="Path to OLD trace pickle")
    parser.add_argument("--new", required=True, help="Path to NEW trace pickle")
    parser.add_argument("--plot", type=str, default=None,
                        help="Output directory for OLD-vs-NEW overlay plots "
                             "(commands.png, joint_pos.png). Skipped if not set.")
    parser.add_argument("--env_id", type=int, default=0,
                        help="Which env index to plot when num_envs>1 (default 0).")
    parser.add_argument("--snapshot_steps", type=str, default="0,1,5,50",
                        help="Comma-separated timesteps to print per-key abs-diff "
                             "snapshots for (default: '0,1,5,50'). Useful to "
                             "distinguish 'diverged from t=0' (RNG/init mismatch) "
                             "from 'drifted apart over time' (semantic dynamics).")
    args = parser.parse_args(argv)
    snapshot_steps = [int(s) for s in args.snapshot_steps.split(",") if s.strip()]

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

    # ---- Per-timestep snapshot table -----------------------------------
    # All keys present in either trace are included; rows where every
    # snapshot column is zero or missing are suppressed to keep the table
    # readable.  Keys are grouped (cmd/, obs/policy/, obs/critic/, etc.)
    # and sorted within each group.
    snapshot_steps_valid = [t for t in snapshot_steps if 0 <= t < n_steps]
    if snapshot_steps_valid:
        # Discover every key present in any of the snapshot timesteps in
        # either trace.
        snap_keys: set[str] = set()
        for t in snapshot_steps_valid:
            snap_keys.update(old["steps"][t].keys())
            snap_keys.update(new["steps"][t].keys())

        # Group by top-level prefix for readability: cmd/, obs/policy/,
        # obs/critic/, obs/student/, mgr/, robot/, then everything else.
        def _group_of(k: str) -> str:
            if k.startswith("obs/policy/"): return "1_obs_policy"
            if k.startswith("obs/critic/"): return "2_obs_critic"
            if k.startswith("obs/student/"): return "3_obs_student"
            if k.startswith("obs/"): return "4_obs_other"
            if k.startswith("cmd/"): return "0_cmd"
            if k.startswith("mgr/"): return "5_mgr"
            if k.startswith("robot/"): return "6_robot"
            return "9_misc"

        sorted_keys = sorted(snap_keys, key=lambda k: (_group_of(k), k))

        print()
        print(f"--- Per-timestep abs-diff snapshots (max over dims, env={0}) ---")
        header = f"{'KEY':<36}" + "".join(f"t={t:>5}".rjust(14) for t in snapshot_steps_valid)
        print(header)
        print("-" * len(header))

        last_group = None
        for key in sorted_keys:
            cells: list[str] = []
            any_nonzero = False
            any_present = False
            for t in snapshot_steps_valid:
                ov = old["steps"][t].get(key)
                nv = new["steps"][t].get(key)
                if ov is None and nv is None:
                    cells.append(f"{'-':>14}")
                    continue
                if ov is None or nv is None:
                    cells.append(f"{'<missing>':>14}")
                    any_present = True
                    continue
                any_present = True
                ov_a = np.asarray(ov)
                nv_a = np.asarray(nv)
                if ov_a.shape != nv_a.shape:
                    cells.append(f"{'<shape>':>14}")
                    continue
                if ov_a.dtype.kind in ("i", "u", "b"):
                    d_int = int(np.sum(ov_a != nv_a))
                    cells.append(f"i:n_diff={d_int:>3}".rjust(14))
                    if d_int > 0:
                        any_nonzero = True
                else:
                    d_f = float(np.abs(ov_a.astype(np.float64) - nv_a.astype(np.float64)).max())
                    cells.append(f"{d_f:>14.4e}")
                    if d_f > 1e-12:
                        any_nonzero = True
            if not any_present:
                continue
            # Suppress rows that are all zero across the snapshot timesteps —
            # keeps the table readable.  ``done`` and ``ep_len`` get suppressed.
            if not any_nonzero:
                continue
            grp = _group_of(key)
            if grp != last_group:
                print(f"{'-' * 36} {grp.split('_', 1)[1] if '_' in grp else grp}")
                last_group = grp
            print(f"{key:<36}" + "".join(cells))

    # ---- Optional overlay plots (commands + joint positions) -----------
    if args.plot is None:
        return

    def _stack_field(trace, key, env_id):
        """Return [T, D] numpy array for ``trace[steps][:][key][env_id, :]``.

        Returns None if the key is missing from any step.
        """
        rows_local: list = []
        for s in trace["steps"]:
            v = s.get(key)
            if v is None:
                return None
            v = np.asarray(v)
            if v.ndim == 1:
                rows_local.append(v[env_id : env_id + 1])
            elif v.ndim == 2:
                rows_local.append(v[env_id])
            else:
                rows_local.append(v[env_id].reshape(-1))
        return np.stack(rows_local, axis=0)

    env_id = args.env_id
    dt = old.get("metadata", {}).get("step_dt") or new.get("metadata", {}).get("step_dt")
    n_plot = min(len(old["steps"]), len(new["steps"]))
    time_s = (np.arange(n_plot) * dt) if dt is not None else None

    pos_names = (old.get("metadata", {}).get("ordered_pos_output_names")
                 or new.get("metadata", {}).get("ordered_pos_output_names")
                 or [])
    vel_names = (old.get("metadata", {}).get("ordered_vel_output_names")
                 or new.get("metadata", {}).get("ordered_vel_output_names")
                 or [])
    joint_names = (old.get("metadata", {}).get("joint_names")
                   or new.get("metadata", {}).get("joint_names")
                   or [])

    # ---- Commands plot (y_des / y_act / dy_des / dy_act per dim) -------
    for key, label, names in (
        ("cmd/y_des", "Reference Position (y_des)", pos_names),
        ("cmd/y_act", "Measured Position (y_act)", pos_names),
        ("cmd/dy_des", "Reference Velocity (dy_des)", vel_names),
        ("cmd/dy_act", "Measured Velocity (dy_act)", vel_names),
    ):
        old_arr = _stack_field(old, key, env_id)
        new_arr = _stack_field(new, key, env_id)
        if old_arr is None or new_arr is None:
            print(f"[WARN] Skipping plot for {key}: missing in one trace.")
            continue
        n = min(old_arr.shape[0], new_arr.shape[0])
        out = os.path.join(args.plot, f"{key.replace('/', '_')}.png")
        _grid_overlay_plot(
            out_path=out,
            title=f"{label} — OLD vs NEW (env {env_id})",
            time_s=time_s[:n] if time_s is not None else None,
            old_arr=old_arr[:n],
            new_arr=new_arr[:n],
            names=names,
            y_label="position" if "y_des" in key or "y_act" in key else "velocity",
        )

    # ---- Joint positions plot ------------------------------------------
    old_jp = _stack_field(old, "robot/joint_pos", env_id)
    new_jp = _stack_field(new, "robot/joint_pos", env_id)
    if old_jp is not None and new_jp is not None:
        n = min(old_jp.shape[0], new_jp.shape[0])
        out = os.path.join(args.plot, "robot_joint_pos.png")
        _grid_overlay_plot(
            out_path=out,
            title=f"Robot Joint Positions — OLD vs NEW (env {env_id})",
            time_s=time_s[:n] if time_s is not None else None,
            old_arr=old_jp[:n],
            new_arr=new_jp[:n],
            names=joint_names,
            y_label="rad",
        )
    else:
        print("[WARN] Skipping joint_pos plot: 'robot/joint_pos' missing in one or both traces.")

    # ---- Base velocity command plot ------------------------------------
    old_bv = _stack_field(old, "cmd/base_velocity", env_id)
    new_bv = _stack_field(new, "cmd/base_velocity", env_id)
    if old_bv is not None and new_bv is not None:
        n = min(old_bv.shape[0], new_bv.shape[0])
        # Standard channel names: lin_x, lin_y, ang_z (+ optional heading).
        bv_names = ["lin_vel_x", "lin_vel_y", "ang_vel_z", "heading"][:old_bv.shape[1]]
        out = os.path.join(args.plot, "cmd_base_velocity.png")
        _grid_overlay_plot(
            out_path=out,
            title=f"Commanded Base Velocity — OLD vs NEW (env {env_id})",
            time_s=time_s[:n] if time_s is not None else None,
            old_arr=old_bv[:n],
            new_arr=new_bv[:n],
            names=bv_names,
            n_cols=min(len(bv_names), 4),
            y_label="m/s or rad/s",
        )
    else:
        print("[WARN] Skipping base_velocity plot: 'cmd/base_velocity' missing in one or both traces.")

    # ---- Domain info plot (phasing var + current domain) --------------
    # Two stacked subplots: phasing variable (top) and current domain
    # (bottom), OLD vs NEW overlaid.  Mirrors play_plots.plot_domain_info
    # but as a comparison plot.
    old_phi = _stack_field(old, "cmd/phasing_var", env_id)
    new_phi = _stack_field(new, "cmd/phasing_var", env_id)
    old_dom = _stack_field(old, "cmd/current_domain", env_id)
    new_dom = _stack_field(new, "cmd/current_domain", env_id)
    if (old_phi is not None and new_phi is not None
            and old_dom is not None and new_dom is not None):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = min(old_phi.shape[0], new_phi.shape[0],
                old_dom.shape[0], new_dom.shape[0])
        t_axis = (time_s[:n] if time_s is not None else np.arange(n))
        fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        fig.suptitle(f"Domain Info — OLD vs NEW (env {env_id})", fontsize=14)
        # Top: phasing var
        axs[0].plot(t_axis, old_phi[:n].squeeze(), label="OLD", linewidth=1.4)
        axs[0].plot(t_axis, new_phi[:n].squeeze(), label="NEW",
                    linestyle="--", linewidth=1.4)
        axs[0].set_title("Phasing Var")
        axs[0].set_ylabel("phi")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(fontsize=8)
        # Bottom: current domain
        axs[1].plot(t_axis, old_dom[:n].squeeze(), label="OLD",
                    linewidth=1.4, drawstyle="steps-post")
        axs[1].plot(t_axis, new_dom[:n].squeeze(), label="NEW",
                    linestyle="--", linewidth=1.4, drawstyle="steps-post")
        axs[1].set_title("Current Domain")
        axs[1].set_ylabel("domain idx")
        axs[1].set_xlabel("Time (s)" if time_s is not None else "step")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(fontsize=8)
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        out = os.path.join(args.plot, "cmd_domain_info.png")
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Wrote plot: {out}")
    else:
        print("[WARN] Skipping domain_info plot: 'cmd/phasing_var' or "
              "'cmd/current_domain' missing in one or both traces.")

    # ---- Reference frame pose plot -------------------------------------
    # ref_poses is [N, 7] = [x, y, z, qx, qy, qz, qw] world-frame anchor.
    # Also overlays which ref-frame body each env is currently using
    # (cur_ref_frame_idx) — labeled with frame names if metadata provides
    # them.  Useful for spotting unexpected swaps mid-cycle.
    old_rp = _stack_field(old, "cmd/ref_poses", env_id)
    new_rp = _stack_field(new, "cmd/ref_poses", env_id)
    old_ridx = _stack_field(old, "cmd/cur_ref_frame_idx", env_id)
    new_ridx = _stack_field(new, "cmd/cur_ref_frame_idx", env_id)
    if old_rp is not None and new_rp is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = min(old_rp.shape[0], new_rp.shape[0])
        rp_names = ["pos_x", "pos_y", "pos_z",
                    "ori_x", "ori_y", "ori_z", "ori_w"][:old_rp.shape[1]]
        n_dims = old_rp.shape[1]
        n_cols = 4
        n_pose_rows = max(1, (n_dims + n_cols - 1) // n_cols)
        # Add one extra row for the ref-frame identity panel if we have it.
        has_ridx = old_ridx is not None and new_ridx is not None
        n_total_rows = n_pose_rows + (1 if has_ridx else 0)

        ref_frame_names = (old.get("metadata", {}).get("ref_frames")
                            or new.get("metadata", {}).get("ref_frames")
                            or [])

        fig, axs = plt.subplots(n_total_rows, n_cols,
                                 figsize=(4.5 * n_cols, 2.6 * n_total_rows))
        axs = np.atleast_2d(axs)
        fig.suptitle(f"Reference Frame Pose — OLD vs NEW (env {env_id})",
                      fontsize=14)
        t_axis = (time_s[:n] if time_s is not None else np.arange(n))

        # Top rows: per-component overlay
        for i in range(n_dims):
            r, c = i // n_cols, i % n_cols
            ax = axs[r, c]
            ax.plot(t_axis, old_rp[:n, i], label="OLD", linewidth=1.4)
            ax.plot(t_axis, new_rp[:n, i], label="NEW",
                    linestyle="--", linewidth=1.4)
            ax.set_title(rp_names[i] if i < len(rp_names) else f"dim {i}",
                          fontsize=9)
            ax.set_xlabel("Time (s)" if time_s is not None else "step",
                           fontsize=8)
            ax.set_ylabel("m / quat", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(fontsize=7)
        # Hide unused cells in the pose-component rows.
        for i in range(n_dims, n_pose_rows * n_cols):
            r, c = i // n_cols, i % n_cols
            axs[r, c].set_visible(False)

        # Bottom row (if available): ref-frame identity step plot, spanning
        # all columns by hiding the others and making one wide.
        if has_ridx:
            # Use the first cell of the bottom row, hide the rest.
            for c in range(1, n_cols):
                axs[n_pose_rows, c].set_visible(False)
            ax_id = axs[n_pose_rows, 0]
            # Repurpose this single cell as a wide span by adjusting width.
            pos = ax_id.get_position()
            # Compute combined width across all 4 columns
            right_pos = axs[n_pose_rows, n_cols - 1].get_position()
            ax_id.set_position([pos.x0, pos.y0,
                                  right_pos.x1 - pos.x0, pos.height])

            old_ridx_int = old_ridx[:n].astype(int).reshape(-1)
            new_ridx_int = new_ridx[:n].astype(int).reshape(-1)
            ax_id.step(t_axis, old_ridx_int, where="post",
                        label="OLD", linewidth=1.4)
            ax_id.step(t_axis, new_ridx_int, where="post",
                        label="NEW", linestyle="--", linewidth=1.4)
            n_frames = max(int(max(old_ridx_int.max(), new_ridx_int.max())) + 1,
                            len(ref_frame_names), 1)
            ax_id.set_yticks(range(n_frames))
            if ref_frame_names:
                # Truncate each label to keep legible
                ax_id.set_yticklabels(
                    [(s[:18] + "…") if len(s) > 19 else s
                     for s in ref_frame_names[:n_frames]],
                    fontsize=7,
                )
            ax_id.set_ylim(-0.5, n_frames - 0.5)
            ax_id.set_xlabel("Time (s)" if time_s is not None else "step",
                              fontsize=9)
            ax_id.set_title("Reference Frame In Use (cur_ref_frame_idx)",
                              fontsize=10)
            ax_id.grid(True, alpha=0.3)
            ax_id.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        out = os.path.join(args.plot, "cmd_ref_poses.png")
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Wrote plot: {out}")
    else:
        print("[WARN] Skipping ref_poses plot: 'cmd/ref_poses' missing in one or both traces.")


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
                        # Group with multiple terms (rare; usually it's a concat tensor)
                        for term, t_val in val.items():
                            step_state[f"obs/{group}/{term}"] = t_val.detach().cpu().numpy()
                    else:
                        step_state[f"obs/{group}"] = val.detach().cpu().numpy()

                # Also capture each obs term separately by re-querying through
                # the ObservationManager's per-term API (the concatenated
                # ``obs_dict`` above hides which sub-vector belongs to which
                # term, making cross-commit per-term diffs impossible).
                try:
                    obs_mgr = unwrapped.observation_manager
                    for group_name in obs_mgr.active_terms:
                        terms = obs_mgr.active_terms[group_name]
                        # ObservationManager has a per-group ordered list of
                        # term names; read each term's tensor directly.
                        for term_name in terms:
                            try:
                                t_val = obs_mgr._compute_obs_term(group_name, term_name) \
                                    if hasattr(obs_mgr, "_compute_obs_term") else None
                                if t_val is None:
                                    # Fall back: re-run the term's func against env.
                                    cfg = obs_mgr._group_obs_term_cfgs[group_name][
                                        list(terms).index(term_name)
                                    ]
                                    t_val = cfg.func(unwrapped, **(cfg.params or {}))
                                if hasattr(t_val, "detach"):
                                    step_state[f"obs/{group_name}/{term_name}"] = (
                                        t_val.detach().cpu().numpy()
                                    )
                            except Exception:
                                pass
                except Exception:
                    pass

                # Action
                step_state["action"] = actions.detach().cpu().numpy()

                # Cmd term per-step state (safe getattr — handles OLD/NEW)
                for attr in ("y_des", "y_act", "dy_des", "dy_act", "v", "vdot",
                              "current_domain", "cur_ref_frame_idx", "ref_poses",
                              "traj_time", "phasing_var"):
                    val = getattr(cmd, attr, None)
                    if val is not None and hasattr(val, "detach"):
                        step_state[f"cmd/{attr}"] = val.detach().cpu().numpy()

                # Phasing var fallback: compute from manager + traj_time when
                # the cmd term doesn't expose ``phasing_var`` directly (e.g.
                # multiskill phase-based command). Mirrors ``data_logger.py``.
                if "cmd/phasing_var" not in step_state:
                    try:
                        manager = getattr(cmd, "manager", None)
                        traj_time = getattr(cmd, "traj_time", None)
                        if manager is not None and traj_time is not None:
                            phi = manager.get_phasing_var(traj_time)
                            step_state["cmd/phasing_var"] = phi.detach().cpu().numpy()
                    except Exception:
                        pass

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

                # Robot joint positions (raw, in robot's own joint order — for
                # the joint-position plot in diff mode).
                try:
                    robot = unwrapped.scene.articulations["robot"]
                    jp = robot.data.joint_pos
                    if hasattr(jp, "to_torch"):
                        jp = jp.to_torch()
                    elif type(jp).__module__.startswith("warp"):
                        import warp as wp
                        jp = wp.to_torch(jp)
                    step_state["robot/joint_pos"] = jp.detach().cpu().numpy()
                except Exception:
                    pass

                # Base velocity command (post-ramp, what the policy / reward see).
                try:
                    base_vel = unwrapped.command_manager.get_command("base_velocity")
                    step_state["cmd/base_velocity"] = base_vel.detach().cpu().numpy()
                except Exception:
                    pass

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
            metadata: dict = {
                "step_dt": float(unwrapped.step_dt),
                "cmd_term_name": cmd_term_name,
            }
            # Best-effort labels for the diff-mode plots.
            for attr in ("ordered_pos_output_names", "ordered_vel_output_names",
                          "ref_frames"):
                vals = getattr(cmd, attr, None)
                if isinstance(vals, list):
                    metadata[attr] = list(vals)
            try:
                robot = unwrapped.scene.articulations["robot"]
                metadata["joint_names"] = list(robot.data.joint_names)
            except Exception:
                pass

            trace = {
                "commit": args_cli.commit_label,
                "task": args_cli.task,
                "checkpoint": args_cli.checkpoint,
                "seed": args_cli.seed,
                "num_envs": args_cli.num_envs,
                "n_steps": len(steps),
                "steps": steps,
                "metadata": metadata,
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
