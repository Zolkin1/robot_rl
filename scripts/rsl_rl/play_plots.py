"""Modular plot functions for play.py trajectory visualization.

Each plot function has the signature::

    def plot_<name>(data, metadata, save_dir, env_ids) -> None

where *data* is a dict of numpy arrays with shape ``[T, ...]`` and *metadata*
contains constant info (axis names, joint names, dt).
"""

from __future__ import annotations

import csv
import os
from typing import Any, Callable

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_joint_name(joint_name: str) -> str:
    """Format joint name for display."""
    return joint_name.replace("_joint", "").replace("_", " ").title()


def _unit_for_name(name: str) -> str:
    """Infer a unit string from an axis name like 'com:pos_x' or 'joint:knee'."""
    if ":pos_" in name or "com:" in name and "ori" not in name:
        return "m"
    elif ":ori_" in name:
        return "rad"
    elif "joint:" in name:
        return "rad"
    return ""


def _grid_plot(
    data: dict[str, np.ndarray],
    key_a: str,
    key_b: str | None,
    names: list[str],
    title_fmt: str,
    y_label_fmt: str,
    filename_fmt: str,
    save_dir: str,
    env_ids: list[int],
    label_a: str = "Reference",
    label_b: str = "Actual",
    n_cols: int = 4,
    dt: float | None = None,
) -> None:
    """Shared helper for grid-of-subplots plots (positions, velocities, joints, torques).

    When ``dt`` is provided the x-axis is rendered in seconds; otherwise it
    falls back to integer time steps.
    """
    arr_a = data[key_a]
    n_dims = arr_a.shape[2] if arr_a.ndim >= 3 else arr_a.shape[1]
    n_rows = max(1, (n_dims + n_cols - 1) // n_cols)
    if dt is not None:
        x_axis = np.arange(arr_a.shape[0]) * dt
        x_label = "Time (s)"
    else:
        x_axis = np.arange(arr_a.shape[0])
        x_label = "Time Steps"

    for env_id in env_ids:
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
        fig.suptitle(title_fmt.format(env_id=env_id), fontsize=16)
        axs_flat = np.array(axs).flatten()

        for i in range(n_dims):
            ax = axs_flat[i]
            y_a = arr_a[:, env_id, i] if arr_a.ndim >= 3 else arr_a[:, env_id]
            ax.plot(x_axis, y_a, label=label_a, linewidth=2)
            if key_b is not None:
                y_b = data[key_b][:, env_id, i] if data[key_b].ndim >= 3 else data[key_b][:, env_id]
                ax.plot(x_axis, y_b, label=label_b, linestyle="--", linewidth=2)

            name = names[i] if i < len(names) else f"Dim {i}"
            unit = _unit_for_name(name)
            ax.set_title(name, fontsize=10)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label_fmt.format(unit=unit) if unit else y_label_fmt.format(unit=""))
            ax.grid(True, alpha=0.3)
            if i == 0 and key_b is not None:
                ax.legend()

        # Hide unused subplots
        for i in range(n_dims, len(axs_flat)):
            axs_flat[i].set_visible(False)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, filename_fmt.format(env_id=env_id)),
                     dpi=300, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_positions(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_dir: str,
    env_ids: list[int],
) -> None:
    """Reference vs actual positions (y_des vs y_act), one subplot per dimension."""
    if "y_des" not in data or "y_act" not in data:
        print("[WARN plot_positions] Missing y_des or y_act data, skipping.")
        return
    names = metadata.get("pos_names", [f"Dim {i}" for i in range(data["y_des"].shape[2])])
    _grid_plot(
        data, "y_des", "y_act", names,
        title_fmt="Reference vs Actual Positions (Env {env_id})",
        y_label_fmt="Position ({unit})",
        filename_fmt="positions_env{env_id}.png",
        save_dir=save_dir, env_ids=env_ids,
        dt=metadata.get("dt"),
    )


def plot_velocities(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_dir: str,
    env_ids: list[int],
) -> None:
    """Reference vs actual velocities (dy_des vs dy_act), one subplot per dimension."""
    if "dy_des" not in data or "dy_act" not in data:
        print("[WARN plot_velocities] Missing dy_des or dy_act data, skipping.")
        return
    names = metadata.get("vel_names", [f"Dim {i}" for i in range(data["dy_des"].shape[2])])
    _grid_plot(
        data, "dy_des", "dy_act", names,
        title_fmt="Reference vs Actual Velocities (Env {env_id})",
        y_label_fmt="Velocity ({unit}/s)",
        filename_fmt="velocities_env{env_id}.png",
        save_dir=save_dir, env_ids=env_ids,
        dt=metadata.get("dt"),
    )


def plot_joint_targets(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_dir: str,
    env_ids: list[int],
) -> None:
    """Action targets vs actual joint positions."""
    if "action_targets" not in data or "joint_pos" not in data:
        print("[WARN plot_joint_targets] Missing action_targets or joint_pos data, skipping.")
        return
    raw_names = metadata.get("joint_names", [])
    names = [_format_joint_name(n) for n in raw_names]
    _grid_plot(
        data, "action_targets", "joint_pos", names,
        title_fmt="Joint Angle Targets vs Actual (Env {env_id})",
        y_label_fmt="Angle (rad)",
        filename_fmt="joint_targets_env{env_id}.png",
        save_dir=save_dir, env_ids=env_ids,
        label_a="Target", label_b="Actual",
    )


def plot_teacher_vs_student_actions(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_dir: str,
    env_ids: list[int],
) -> None:
    """Teacher vs. student raw policy outputs, one subplot per action dim."""
    if "teacher_actions" not in data or "student_actions" not in data:
        print("[WARN plot_teacher_vs_student_actions] Missing teacher/student actions, skipping.")
        return
    raw_names = metadata.get("joint_names", [])
    names = [_format_joint_name(n) for n in raw_names]
    _grid_plot(
        data, "teacher_actions", "student_actions", names,
        title_fmt="Teacher vs Student Actions (Env {env_id})",
        y_label_fmt="Action (rad)",
        filename_fmt="teacher_vs_student_env{env_id}.png",
        save_dir=save_dir, env_ids=env_ids,
        label_a="Teacher", label_b="Student",
    )


def plot_torques(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_dir: str,
    env_ids: list[int],
) -> None:
    """Applied joint torques over time."""
    if "applied_torque" not in data:
        print("[WARN plot_torques] Missing applied_torque data, skipping.")
        return
    raw_names = metadata.get("joint_names", [])
    names = [_format_joint_name(n) for n in raw_names]
    _grid_plot(
        data, "applied_torque", None, names,
        title_fmt="Joint Torques (Env {env_id})",
        y_label_fmt="Torque (Nm)",
        filename_fmt="joint_torques_env{env_id}.png",
        save_dir=save_dir, env_ids=env_ids,
        label_a="Torque",
    )


def plot_base_velocity(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_dir: str,
    env_ids: list[int],
) -> None:
    """Commanded base velocity components (lin_x, lin_y, ang_z)."""
    if "base_velocity" not in data:
        print("[WARN plot_base_velocity] Missing base_velocity data, skipping.")
        return

    bv = data["base_velocity"]
    dt = metadata.get("dt", 1.0)
    time_s = np.arange(bv.shape[0]) * dt
    labels = ["Linear X", "Linear Y", "Angular Z"]
    units = ["m/s", "m/s", "rad/s"]
    n_dims = min(bv.shape[2], 3)

    for env_id in env_ids:
        fig, axs = plt.subplots(1, n_dims, figsize=(5 * n_dims, 3))
        fig.suptitle(f"Base Velocity (Env {env_id})", fontsize=16)
        if n_dims == 1:
            axs = [axs]
        for i in range(n_dims):
            axs[i].plot(time_s, bv[:, env_id, i], linewidth=2)
            axs[i].set_title(labels[i])
            axs[i].set_xlabel("Time (s)")
            axs[i].set_ylabel(f"Velocity ({units[i]})")
            axs[i].grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f"base_velocity_env{env_id}.png"),
                     dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_gate_offset(
    ax: "plt.Axes",
    time_s: np.ndarray,
    gate_rel_phi: np.ndarray,
    next_gate_idx: np.ndarray | None,
    early_window: np.ndarray | None,
) -> None:
    """Render gate-relative phi over time with snap-event markers.

    ``gate_rel_phi`` is the manager's monotonic signed phi-distance from
    the currently armed gate (negative = before gate, positive = past
    gate).  Discontinuities mark contact-gate snaps; the value just
    before each discontinuity is the *miss amount* — how early (negative)
    or late (positive) the contact landed relative to the expected gate
    phi.  Reset boundaries also change ``next_gate_idx`` and will appear
    as markers; their offset value reflects whatever phase was sampled
    at reset rather than a true contact miss.

    When ``early_window`` is supplied (per-step ``-W * pre_size`` of the
    armed gate), it's drawn as a step line and the early-fire region
    between the threshold and 0 is shaded — making it visually obvious
    when the window opens (gate_rel_phi crosses the threshold going up)
    and closes (a snap fires).
    """
    ax.plot(time_s, gate_rel_phi, color="gray", alpha=0.55, linewidth=1.0,
            label="gate_rel_phi", zorder=2)
    ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.6)

    if early_window is not None:
        # Early-fire window: contacts in [early_window, 0] snap forward.
        # Use ``step="post"`` so transitions on gate-advance look like
        # piecewise-constant boundaries rather than slanted lines.
        ax.fill_between(
            time_s, early_window, 0.0,
            step="post", color="tab:green", alpha=0.12, zorder=0,
            label="early-fire window",
        )
        ax.plot(time_s, early_window, color="tab:green",
                drawstyle="steps-post", linewidth=1.0, alpha=0.8, zorder=1)

    if next_gate_idx is not None:
        # A snap (or reset) is the only way next_gate_idx changes.
        # ``np.diff`` returns nonzero at index t-1 when next_gate_idx
        # changes between steps t-1 and t, so the fire happened at step
        # ``t = change_steps + 1``.  ``gate_rel_phi`` is logged as the
        # post-update, pre-snap snapshot, so ``gate_rel_phi[t]`` is the
        # value that triggered the fire decision — exactly what we want
        # to display on the marker.
        change_steps = np.where(np.diff(next_gate_idx) != 0)[0]
        fire_steps = change_steps + 1
        if fire_steps.size > 0:
            offsets = gate_rel_phi[fire_steps]
            times = time_s[fire_steps]
            late_mask = offsets > 0
            early_mask = ~late_mask
            if late_mask.any():
                ax.scatter(times[late_mask], offsets[late_mask],
                           color="tab:red", s=28, zorder=3,
                           label=f"late snap (n={int(late_mask.sum())})")
            if early_mask.any():
                ax.scatter(times[early_mask], offsets[early_mask],
                           color="tab:green", s=28, zorder=3,
                           label=f"early snap (n={int(early_mask.sum())})")

    ax.set_title("Gate Offset (negative = early, positive = late)", fontsize=10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("phi from armed gate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def plot_domain_info(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_dir: str,
    env_ids: list[int],
) -> None:
    """Phasing variable, current domain, and (optionally) gate offset + phase obs.

    Layout (rows added only when the corresponding data is present):
      row 0: phasing_var | current_domain
      row 1 (if gate_rel_phi logged): gate offset (full width)
      row 2 (if phase_obs logged):    sin phase obs | cos phase obs
    """
    if "phasing_var" not in data:
        print("[WARN plot_domain_info] Missing phasing_var data, skipping.")
        return

    import matplotlib.gridspec as gridspec

    dt = metadata.get("dt", 1.0)
    time_s = np.arange(data["phasing_var"].shape[0]) * dt

    has_phase_obs = "phase_obs" in data
    has_gate_data = "gate_rel_phi" in data
    phase_labels = metadata.get("phase_obs_labels", [])

    for env_id in env_ids:
        n_rows = 1 + (1 if has_gate_data else 0) + (1 if has_phase_obs else 0)
        fig = plt.figure(figsize=(10, 3 * n_rows))
        gs = gridspec.GridSpec(n_rows, 2, figure=fig)
        fig.suptitle(f"Domain Info (Env {env_id})", fontsize=16)

        row = 0

        # Row 0: phasing var | current domain
        ax_phi = fig.add_subplot(gs[row, 0])
        ax_phi.plot(time_s, data["phasing_var"][:, env_id], linewidth=2)
        ax_phi.set_title("Phasing Var")
        ax_phi.set_xlabel("Time (s)")
        ax_phi.grid(True, alpha=0.3)

        ax_dom = fig.add_subplot(gs[row, 1])
        if "current_domain" in data:
            ax_dom.plot(time_s, data["current_domain"][:, env_id], linewidth=2)
            ax_dom.set_title("Current Domain")
            ax_dom.set_xlabel("Time (s)")
            ax_dom.grid(True, alpha=0.3)
        else:
            ax_dom.set_visible(False)
        row += 1

        # Row 1 (optional): gate offset spans both columns
        if has_gate_data:
            ax_gate = fig.add_subplot(gs[row, :])
            ngi = data["next_gate_idx"][:, env_id] if "next_gate_idx" in data else None
            ew = data["gate_early_window"][:, env_id] if "gate_early_window" in data else None
            _plot_gate_offset(
                ax_gate,
                time_s,
                data["gate_rel_phi"][:, env_id],
                ngi,
                ew,
            )
            row += 1

        # Row 2 (optional): split phase obs into sin and cos halves
        if has_phase_obs:
            phase_data = data["phase_obs"][:, env_id, :]  # [T, D]
            n_dims = phase_data.shape[1]
            half = n_dims // 2

            ax_sin = fig.add_subplot(gs[row, 0])
            for d in range(half):
                label = phase_labels[d] if d < len(phase_labels) else f"sin dim {d}"
                ax_sin.plot(time_s, phase_data[:, d], linewidth=1.5, label=label)
            ax_sin.set_title("Sin Phase Obs")
            ax_sin.set_xlabel("Time (s)")
            ax_sin.legend(fontsize=8)
            ax_sin.grid(True, alpha=0.3)

            ax_cos = fig.add_subplot(gs[row, 1])
            for d in range(half, n_dims):
                label = phase_labels[d] if d < len(phase_labels) else f"cos dim {d}"
                ax_cos.plot(time_s, phase_data[:, d], linewidth=1.5, label=label)
            ax_cos.set_title("Cos Phase Obs")
            ax_cos.set_xlabel("Time (s)")
            ax_cos.legend(fontsize=8)
            ax_cos.grid(True, alpha=0.3)
            row += 1

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f"domain_info_env{env_id}.png"),
                     dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_reference_frame(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_dir: str,
    env_ids: list[int],
) -> None:
    """Reference-frame position and identity over time, for debugging skill transitions.

    Produces a 4-row figure per env:
      1-3. World-frame x/y/z of the current reference frame.
      4. Which reference frame is in use (step plot, y-ticks labelled with frame names).

    Background spans in the top three rows are colour-coded by the active
    frame, so transitions show up as a colour change at a stance boundary.
    This makes it easy to spot weird switches (e.g. left-foot ref becoming
    right-foot mid-swing after a skill change).
    """
    if "ref_poses" not in data or "cur_ref_frame_idx" not in data:
        print("[WARN plot_reference_frame] Missing ref_poses or cur_ref_frame_idx, skipping.")
        return

    ref_poses = data["ref_poses"]            # [T, N_envs, 7]
    ref_idx = data["cur_ref_frame_idx"]      # [T, N_envs]
    dt = metadata.get("dt", 1.0)
    time_s = np.arange(ref_poses.shape[0]) * dt
    frame_names = metadata.get("ref_frames", [])
    n_frames = max(int(ref_idx.max()) + 1, len(frame_names), 1)
    axis_labels = ("X", "Y", "Z")
    cmap = plt.cm.tab10

    for env_id in env_ids:
        idx_env = ref_idx[:, env_id].astype(int)
        pos_env = ref_poses[:, env_id, :3]

        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Reference Frame (Env {env_id})", fontsize=16)

        # Identify frame-change boundaries for axvspan shading
        changes = np.where(np.diff(idx_env) != 0)[0] + 1
        segment_starts = np.concatenate([[0], changes])
        segment_ends = np.concatenate([changes, [len(idx_env)]])

        # Top 3 rows: x, y, z over time with colour-shaded spans per segment
        for i in range(3):
            ax = axs[i]
            for s, e in zip(segment_starts, segment_ends):
                ax.axvspan(time_s[s], time_s[min(e, len(time_s) - 1)],
                           color=cmap(idx_env[s] % 10), alpha=0.15, linewidth=0)
            ax.plot(time_s, pos_env[:, i], color="black", linewidth=1.5)
            ax.set_ylabel(f"{axis_labels[i]} (m)")
            ax.set_title(f"Ref Frame Position — {axis_labels[i]}", fontsize=10)
            ax.grid(True, alpha=0.3)

        # Bottom row: ref frame identity step plot
        ax = axs[3]
        ax.step(time_s, idx_env, where="post", color="black", linewidth=1.5)
        # Colour each segment
        for s, e in zip(segment_starts, segment_ends):
            ax.axvspan(time_s[s], time_s[min(e, len(time_s) - 1)],
                       color=cmap(idx_env[s] % 10), alpha=0.3, linewidth=0)
        ax.set_yticks(range(n_frames))
        if frame_names:
            ax.set_yticklabels(frame_names[:n_frames])
        ax.set_ylim(-0.5, n_frames - 0.5)
        ax.set_ylabel("Ref Frame")
        ax.set_xlabel("Time (s)")
        ax.set_title("Reference Frame In Use", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(save_dir, f"reference_frame_env{env_id}.png"),
                     dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_clf(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_dir: str,
    env_ids: list[int],
) -> None:
    """CLF Lyapunov function: v, vdot, and decay (v + alpha*vdot)."""
    if "v" not in data or "vdot" not in data:
        print("[WARN plot_clf] Missing v or vdot data, skipping.")
        return

    v = data["v"]
    vdot = data["vdot"]
    time_steps = np.arange(v.shape[0])

    for env_id in env_ids:
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axs[0].plot(time_steps, v[:, env_id], label="CLF v", linewidth=2)
        axs[0].set_title("CLF (v)")
        axs[0].set_ylabel("V")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()

        axs[1].plot(time_steps, vdot[:, env_id], label="CLF vdot", linewidth=2)
        axs[1].set_title("CLF (v\u0307)")
        axs[1].set_ylabel("dV/dt")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend()

        alpha = 1.0
        decay = alpha * v[:, env_id] + vdot[:, env_id]
        axs[2].plot(time_steps, decay, label="CLF Decay", linewidth=2)
        axs[2].set_title("CLF Decay (v + \u03b1v\u0307)")
        axs[2].set_xlabel("Time Steps")
        axs[2].set_ylabel("Decay Rate")
        axs[2].grid(True, alpha=0.3)
        axs[2].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"clf_env{env_id}.png"),
                     dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_moe_weights(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_dir: str,
    env_ids: list[int],
) -> None:
    """Mean MoE gate weights averaged over all environments and timesteps."""
    if "gate_weights" not in data:
        print("[WARN plot_moe_weights] Missing gate_weights data, skipping.")
        return

    gate_w = data["gate_weights"]  # [T, N_envs, num_experts]
    # Flatten T and N_envs into a single sample dimension
    flat = gate_w.reshape(-1, gate_w.shape[2])  # [T*N, num_experts]
    mean_weights = flat.mean(axis=0)  # [num_experts]
    std_weights = flat.std(axis=0)    # [num_experts]
    num_experts = mean_weights.shape[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(num_experts), mean_weights, yerr=std_weights,
                  capsize=4, color=plt.cm.tab10.colors[:num_experts], edgecolor="black")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Mean Gate Weight")
    ax.set_title("MoE Expert Utilization (mean over all envs & time)")
    ax.set_xticks(range(num_experts))
    ax.set_xticklabels([f"Expert {i}" for i in range(num_experts)])
    ax.set_ylim(0, max((mean_weights + std_weights).max() * 1.15, 0.1))
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, m, s in zip(bars, mean_weights, std_weights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.008,
                f"{m:.3f}\u00b1{s:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "moe_weights.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_moe_weights_vs_velocity(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_dir: str,
    env_ids: list[int],
) -> None:
    """MoE gate weights as a function of commanded linear-x velocity."""
    if "gate_weights" not in data or "base_velocity" not in data:
        print("[WARN plot_moe_weights_vs_velocity] Missing gate_weights or base_velocity, skipping.")
        return

    gate_w = data["gate_weights"]       # [T, N_envs, num_experts]
    base_vel = data["base_velocity"]     # [T, N_envs, 3]
    num_experts = gate_w.shape[2]

    # Flatten time and env dimensions
    vel_x = base_vel[:, :, 0].flatten()          # [T*N]
    weights = gate_w.reshape(-1, num_experts)     # [T*N, num_experts]

    # Bin by commanded lin_vel_x
    n_bins = 10
    bin_edges = np.linspace(vel_x.min(), vel_x.max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_idx = np.digitize(vel_x, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Mean and std weight per expert per bin
    bin_means = np.zeros((n_bins, num_experts))
    bin_stds = np.zeros((n_bins, num_experts))
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            bin_means[b] = weights[mask].mean(axis=0)
            bin_stds[b] = weights[mask].std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    for e in range(num_experts):
        ax.plot(bin_centers, bin_means[:, e], marker="o", linewidth=2, label=f"Expert {e}")
        ax.fill_between(bin_centers,
                        bin_means[:, e] - bin_stds[:, e],
                        bin_means[:, e] + bin_stds[:, e],
                        alpha=0.15)

    ax.set_xlabel("Commanded Linear X Velocity (m/s)")
    ax.set_ylabel("Mean Gate Weight")
    ax.set_title("MoE Expert Weights vs Commanded Velocity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "moe_weights_vs_velocity.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_moe_tsne(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_dir: str,
    env_ids: list[int],
) -> None:
    """t-SNE embedding of MoE gate weights, colored by locomotion regime.

    Each point represents one (timestep, environment) pair. Points are colored
    by the commanded forward velocity into five regimes: Backward Running,
    Backward Walking, Standing, Walking, and Running.
    """
    if "gate_weights" not in data or "base_velocity" not in data:
        print("[WARN plot_moe_tsne] Missing gate_weights or base_velocity data, skipping.")
        return

    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("[WARN plot_moe_tsne] scikit-learn not installed, skipping t-SNE plot.")
        return

    gate_w = data["gate_weights"]       # [T, num_envs, num_experts]
    base_vel = data["base_velocity"]    # [T, num_envs, 3]

    # Flatten time and env dimensions
    weights = gate_w.reshape(-1, gate_w.shape[2])   # [T*N, num_experts]
    vel_x = base_vel[:, :, 0].flatten()              # [T*N]

    # Subsample to keep t-SNE tractable
    n_total = weights.shape[0]
    max_points = 8000
    rng = np.random.RandomState(42)
    if n_total > max_points:
        idx = rng.choice(n_total, size=max_points, replace=False)
        weights = weights[idx]
        vel_x = vel_x[idx]

    # Assign locomotion regime labels based on signed forward velocity
    labels = np.empty(vel_x.shape[0], dtype=object)
    labels[vel_x < -1.5] = "Backward Running"
    labels[(vel_x >= -1.5) & (vel_x < -0.1)] = "Backward Walking"
    labels[(vel_x >= -0.1) & (vel_x <= 0.1)] = "Standing"
    labels[(vel_x > 0.1) & (vel_x <= 1.5)] = "Walking"
    labels[vel_x > 1.5] = "Running"

    # Run t-SNE
    print("[INFO plot_moe_tsne] Running t-SNE (this may take ~15s)...")
    tsne = TSNE(n_components=2, perplexity=50, random_state=42, learning_rate="auto")
    coords = tsne.fit_transform(weights)  # [N, 2]

    # Plot with one color per regime
    regime_colors = {
        "Backward Running": plt.cm.tab10.colors[4],   # purple
        "Backward Walking": plt.cm.tab10.colors[9],    # cyan
        "Standing":         plt.cm.tab10.colors[0],    # blue
        "Walking":          plt.cm.tab10.colors[1],    # orange
        "Running":          plt.cm.tab10.colors[3],    # red
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for regime_name, color in regime_colors.items():
        mask = labels == regime_name
        if mask.any():
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=[color], label=f"{regime_name} ({mask.sum()})",
                s=8, alpha=0.6, edgecolors="none",
            )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE of MoE Gate Weights by Locomotion Regime")
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "moe_tsne.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save coordinates to CSV for later analysis
    csv_path = os.path.join(save_dir, "moe_tsne_coords.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tsne_1", "tsne_2", "regime", "vel_x"])
        for i in range(coords.shape[0]):
            writer.writerow([
                f"{coords[i, 0]:.6f}", f"{coords[i, 1]:.6f}",
                labels[i], f"{vel_x[i]:.4f}",
            ])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PLOT_REGISTRY: dict[str, Callable] = {
    "positions": plot_positions,
    "velocities": plot_velocities,
    "joint_targets": plot_joint_targets,
    "teacher_vs_student": plot_teacher_vs_student_actions,
    "torques": plot_torques,
    "base_velocity": plot_base_velocity,
    "domain_info": plot_domain_info,
    "reference_frame": plot_reference_frame,
    "clf": plot_clf,
    "moe_weights": plot_moe_weights,
    "moe_weights_vs_velocity": plot_moe_weights_vs_velocity,
    "moe_tsne": plot_moe_tsne,
}

DEFAULT_PLOTS = ["positions", "velocities"]


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def compute_and_save_stats(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_dir: str,
) -> None:
    """Compute tracking statistics and write stats.txt.

    Args:
        data: Stacked numpy data from DataLogger.finalize().
        metadata: Constant metadata (pos_names, vel_names, joint_names, dt).
        save_dir: Directory to write stats.txt into.
    """
    lines: list[str] = []
    csv_rows: list[dict[str, str]] = []

    def _log(text: str = "") -> None:
        print(text)
        lines.append(text)

    def _csv(metric: str, value: float) -> None:
        csv_rows.append({"metric": metric, "value": f"{value:.6f}"})

    _log("=" * 60)
    _log("  Play Statistics")
    _log("=" * 60)

    # Number of envs
    n_envs = None
    for arr in data.values():
        if arr.ndim >= 2:
            n_envs = arr.shape[1]
            break
    n_timesteps = data[next(iter(data))].shape[0]
    _log(f"Timesteps: {n_timesteps}")
    if n_envs is not None:
        _log(f"Number of envs: {n_envs}")

    # Commanded velocity ranges
    if "base_velocity" in data:
        bv = data["base_velocity"]
        vel_labels = ["Linear X", "Linear Y", "Angular Z"]
        vel_units = ["m/s", "m/s", "rad/s"]
        _log("")
        _log("Commanded Velocity Ranges:")
        csv_vel_keys = ["cmd_vel_lin_x", "cmd_vel_lin_y", "cmd_vel_ang_z"]
        for i in range(min(bv.shape[2], 3)):
            v_min, v_max = bv[:, :, i].min(), bv[:, :, i].max()
            _log(f"  {vel_labels[i]:>12s}: [{v_min:+.4f}, {v_max:+.4f}] {vel_units[i]}")
            _csv(f"{csv_vel_keys[i]}_min", float(v_min))
            _csv(f"{csv_vel_keys[i]}_max", float(v_max))

    # Mean Lyapunov V
    if "v" in data:
        v_data = data["v"]
        if v_data.ndim == 3:
            v_data = v_data.squeeze(-1)
        per_env_mean = v_data.mean(axis=0)
        _log("")
        _log("Lyapunov Function (V):")
        _log(f"  Mean across envs: {per_env_mean.mean():.6f}")
        _log(f"  Std  across envs: {per_env_mean.std():.6f}")
        _csv("lyapunov_v_mean", float(per_env_mean.mean()))
        _csv("lyapunov_v_std", float(per_env_mean.std()))

    # Norm squared error
    if all(k in data for k in ("y_des", "y_act", "dy_des", "dy_act")):
        e_pos = data["y_des"] - data["y_act"]
        e_vel = data["dy_des"] - data["dy_act"]
        e = np.concatenate([e_pos, e_vel], axis=2)
        norm_sq = (e ** 2).sum(axis=2)
        per_env_mean = norm_sq.mean(axis=0)
        _log("")
        _log("Norm Squared Error (dot(e,e)):")
        _log(f"  Mean across envs: {per_env_mean.mean():.6f}")
        _log(f"  Std  across envs: {per_env_mean.std():.6f}")
        _csv("norm_sq_error_mean", float(per_env_mean.mean()))
        _csv("norm_sq_error_std", float(per_env_mean.std()))

    # Position errors
    if "y_des" in data and "y_act" in data:
        pos_err = (data["y_des"] - data["y_act"]) ** 2
        n_dims = pos_err.shape[2]
        pos_names = metadata.get("pos_names", [f"Dim {i}" for i in range(n_dims)])

        per_env_mean_err = pos_err.mean(axis=0)
        mean_over_envs = per_env_mean_err.mean(axis=0)
        std_over_envs = per_env_mean_err.std(axis=0)

        _log("")
        _log("Position Errors (MSE per dimension):")
        _log(f"  {'Name':<40s} {'Mean':>12s} {'Std':>12s}")
        _log(f"  {'-'*40} {'-'*12} {'-'*12}")
        for i in range(n_dims):
            name = pos_names[i] if i < len(pos_names) else f"Dim {i}"
            _log(f"  {name:<40s} {mean_over_envs[i]:12.6f} {std_over_envs[i]:12.6f}")

        _print_group_summaries(_log, _csv, pos_names, mean_over_envs, std_over_envs, "pos_error")

    # Velocity errors
    if "dy_des" in data and "dy_act" in data:
        vel_err = (data["dy_des"] - data["dy_act"]) ** 2
        n_dims = vel_err.shape[2]
        vel_names = metadata.get("vel_names", [f"Dim {i}" for i in range(n_dims)])

        per_env_mean_err = vel_err.mean(axis=0)
        mean_over_envs = per_env_mean_err.mean(axis=0)
        std_over_envs = per_env_mean_err.std(axis=0)

        _log("")
        _log("Velocity Errors (MSE per dimension):")
        _log(f"  {'Name':<40s} {'Mean':>12s} {'Std':>12s}")
        _log(f"  {'-'*40} {'-'*12} {'-'*12}")
        for i in range(n_dims):
            name = vel_names[i] if i < len(vel_names) else f"Dim {i}"
            _log(f"  {name:<40s} {mean_over_envs[i]:12.6f} {std_over_envs[i]:12.6f}")

        _print_group_summaries(_log, _csv, vel_names, mean_over_envs, std_over_envs, "vel_error")

    _log("")
    _log("=" * 60)

    # Write files
    os.makedirs(save_dir, exist_ok=True)

    stats_path = os.path.join(save_dir, "stats.txt")
    with open(stats_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nStats saved to: {stats_path}")

    if csv_rows:
        csv_path = os.path.join(save_dir, "tracking_stats.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV saved to: {csv_path}")


def _print_group_summaries(
    _log: Callable,
    _csv: Callable,
    names: list[str],
    mean_over_envs: np.ndarray,
    std_over_envs: np.ndarray,
    csv_prefix: str,
) -> None:
    """Print group summaries (Positions, Orientations, Joints) for error stats."""
    groups = {"Positions": ":pos_", "Orientations": ":ori_", "Joints": "joint:"}
    _log("")
    _log(f"  Group Summaries:")
    _log(f"  {'Group':<40s} {'Mean':>12s} {'Std':>12s}")
    _log(f"  {'-'*40} {'-'*12} {'-'*12}")
    for group_name, pattern in groups.items():
        idxs = [i for i, name in enumerate(names) if pattern in name]
        if idxs:
            g_mean = mean_over_envs[idxs].mean()
            g_std = std_over_envs[idxs].mean()
            _log(f"  {group_name:<40s} {g_mean:12.6f} {g_std:12.6f}")
            _csv(f"{csv_prefix}_{group_name.lower()}_mean", float(g_mean))
            _csv(f"{csv_prefix}_{group_name.lower()}_std", float(g_std))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_plots(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    save_dir: str,
    plot_names: list[str],
    env_ids: list[int],
) -> None:
    """Execute the requested plot functions and save stats.

    Args:
        data: Stacked numpy data from DataLogger.finalize().
        metadata: Constant metadata (pos_names, vel_names, joint_names, dt).
        save_dir: Directory to write PNGs and stats into.
        plot_names: List of plot names from PLOT_REGISTRY, ``"default"``, or ``"all"``.
        env_ids: Which env indices to generate per-env plots for.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Resolve special names
    resolved: list[str] = []
    for name in plot_names:
        if name == "default":
            resolved.extend(DEFAULT_PLOTS)
        elif name == "all":
            resolved.extend(PLOT_REGISTRY.keys())
        else:
            resolved.append(name)

    available = list(PLOT_REGISTRY.keys())
    for name in resolved:
        fn = PLOT_REGISTRY.get(name)
        if fn is None:
            print(f"[WARN] Unknown plot name: {name!r}. Available: {available}")
            continue
        try:
            fn(data, metadata, save_dir, env_ids)
        except Exception as e:
            print(f"[WARN] Plot {name!r} failed: {e}")

    # Always compute stats
    compute_and_save_stats(data, metadata, save_dir)
