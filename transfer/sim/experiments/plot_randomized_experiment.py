# Copyright 2026 Zachary Olkin. All rights reserved.

"""Standalone plotting utility for randomized parameter experiments.

Loads simulation logs from experiment directories and produces:
- Velocity tracking plot (mean +/- 1 std per policy vs commanded)
- Per-joint torque plots (mean +/- 1 std per policy)

Can be used as a standalone script or imported for its plotting functions.
"""

import argparse
import csv
import math
import os
import sys
from collections import OrderedDict

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerBase

from sim.log_utils import extract_data

START_IDX = 150
MA_WINDOW = 30

# Environment experiment names mapping (same as in g1_runner.py / train_policy.py)
EXPERIMENT_NAMES = {
    "vanilla": "vanilla",
    "vanilla_ec": "vanilla",
    "basic": "baseline",
    "lip_clf": "lip",
    "lip_clf_ec": "lip",
    "lip_ref_play": "lip",
    "walking_clf": "walking_clf",
    "walking_clf_sym": "walking-clf-symmetric",
    "walking_clf_ec": "walking_clf",
    "running_clf": "running_clf",
    "running_clf_sym": "running-clf-symmetric",
    "waving_clf": "waving_clf",
    "bow_forward_clf": "bow_forward_clf",
    "bow_forward_clf_sym": "bow_forward-clf-symmetric",
    "bend_up_clf_sym": "bend_up-clf-symmetric",
}


def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """Compute moving average of a 1D array.

    Args:
        data: 1D input array.
        window_size: Number of samples in the averaging window.

    Returns:
        Smoothed array of length ``len(data) - window_size + 1``.
    """
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')


class HandlerOverlay(HandlerBase):
    """Custom legend handler that overlays a line on a shaded patch."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        """Create the legend artists."""
        line, patch = orig_handle
        p = plt.Rectangle(
            (xdescent, ydescent), width, height,
            facecolor=patch.get_facecolor(),
            edgecolor='none',
            transform=trans,
        )
        margin = width * 0.05
        l = Line2D(
            [xdescent + margin, xdescent + width - margin],
            [ydescent + height / 2] * 2,
            color=line.get_color(), lw=line.get_linewidth(),
            transform=trans,
        )
        return [p, l]


def load_experiment_data(experiment_dir: str) -> dict:
    """Load all simulation runs from an experiment directory.

    Args:
        experiment_dir: Path to the experiment folder containing timestamped run subdirs.

    Returns:
        Dict with keys: "runs" (list of per-run dicts with time, commanded_vel,
        actual_vel, torque arrays), "joint_names", "torque_limits".
    """
    runs = []
    successes = []
    force_mags = []
    joint_names = None
    torque_limits = None

    for subdir in sorted(os.listdir(experiment_dir)):
        run_dir = os.path.join(experiment_dir, subdir)
        if not os.path.isdir(run_dir):
            continue

        config_path = os.path.join(run_dir, "sim_config.yaml")
        log_path = os.path.join(run_dir, "sim_log.csv")
        if not os.path.exists(config_path) or not os.path.exists(log_path):
            continue

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            data = extract_data(log_path, config)

            if joint_names is None:
                joint_names = config.get("joint_names", [])
                torque_limits = config.get("torque_limits", [])

            run_entry = {
                "time": np.squeeze(data["time"]),
                "commanded_vel": np.squeeze(data["commanded_vel"]),
                "actual_vel": np.squeeze(data["qvel"]),
                "torque": np.squeeze(data["torque"]),
                "success": None,
            }

            # Load success and force_mag from robustness_data.yaml if present
            robustness_path = os.path.join(run_dir, "robustness_data.yaml")
            if os.path.exists(robustness_path):
                with open(robustness_path) as f:
                    robustness_data = yaml.safe_load(f)
                if robustness_data and "success" in robustness_data:
                    run_entry["success"] = robustness_data["success"]
                    successes.append(robustness_data["success"])
                if robustness_data and "force_mag" in robustness_data:
                    force_mags.append(robustness_data["force_mag"])

            runs.append(run_entry)
        except Exception as e:
            print(f"[Warning] Skipping {run_dir}: {e}")

    # Load tracking_stats.csv if present at the experiment directory root
    tracking_stats: dict[str, float] = {}
    tracking_csv_path = os.path.join(experiment_dir, "tracking_stats.csv")
    if os.path.exists(tracking_csv_path):
        with open(tracking_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                tracking_stats[row["metric"]] = float(row["value"])
        print(f"  Loaded tracking_stats.csv with {len(tracking_stats)} metrics")

    return {
        "runs": runs,
        "successes": successes,
        "force_mags": force_mags,
        "joint_names": joint_names or [],
        "torque_limits": torque_limits or [],
        "tracking_stats": tracking_stats,
    }


def _filter_successful_runs(grouped_data: OrderedDict) -> OrderedDict:
    """Return a copy of grouped_data with only successful runs.

    Runs whose ``success`` field is ``False`` are excluded. Runs with
    ``success`` equal to ``True`` or ``None`` (no robustness data) are kept.
    The ``successes`` and ``force_mags`` lists are preserved unchanged so
    that success-rate reporting remains accurate.

    Args:
        grouped_data: OrderedDict mapping display_name -> experiment data dict.

    Returns:
        New OrderedDict with the same structure but filtered runs.
    """
    filtered = OrderedDict()
    for name, exp_data in grouped_data.items():
        successful_runs = [
            r for r in exp_data["runs"] if r.get("success") is not False
        ]
        filtered[name] = {
            **exp_data,
            "runs": successful_runs,
        }
    return filtered


def _setup_plot_style():
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        'font.size': 18,
        "text.usetex": True,
        "font.family": "serif",
    })


def _interpolate_to_common_grid(runs: list, key: str, time_key: str = "time"):
    """Interpolate variable-length runs to a common time grid.

    Args:
        runs: List of run dicts.
        key: Data key to interpolate (e.g., "actual_vel", "torque").
        time_key: Key for the time array.

    Returns:
        Tuple of (common_time, stacked_data) where stacked_data has shape
        (num_runs, num_timesteps, num_columns).
    """
    if not runs:
        return None, None

    # Use the shortest run's time as the common grid
    min_len = min(len(r[time_key]) for r in runs)
    common_time = runs[0][time_key][:min_len]

    interpolated = []
    for run in runs:
        data = run[key]
        run_time = run[time_key]

        if data.ndim == 1:
            interp = np.interp(common_time, run_time, data)
            interpolated.append(interp)
        else:
            cols = []
            for col_idx in range(data.shape[1]):
                cols.append(np.interp(common_time, run_time, data[:, col_idx]))
            interpolated.append(np.stack(cols, axis=1))

    return common_time, np.stack(interpolated, axis=0)


def _interpolate_and_smooth(
    runs: list,
    key: str,
    window_size: int = MA_WINDOW,
    time_key: str = "time",
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Interpolate runs to a common grid and apply per-run moving average.

    Args:
        runs: List of run dicts.
        key: Data key to interpolate and smooth.
        window_size: Moving average window size.
        time_key: Key for the time array.

    Returns:
        Tuple of (smoothed_time, raw_stacked, smoothed_stacked).
        ``smoothed_time`` is trimmed to match the moving-average output length.
        ``raw_stacked`` has shape (num_runs, num_timesteps, num_columns).
        ``smoothed_stacked`` has shape (num_runs, num_timesteps - window + 1, num_columns).
    """
    common_time, raw_stack = _interpolate_to_common_grid(runs, key, time_key)
    if common_time is None or raw_stack is None:
        return None, None, None

    # Apply moving average per-run, per-column
    smoothed_runs = []
    for run_idx in range(raw_stack.shape[0]):
        run_data = raw_stack[run_idx]
        if run_data.ndim == 1:
            smoothed_runs.append(moving_average(run_data, window_size))
        else:
            cols = []
            for col_idx in range(run_data.shape[1]):
                cols.append(moving_average(run_data[:, col_idx], window_size))
            smoothed_runs.append(np.stack(cols, axis=1))

    smoothed_time = common_time[window_size - 1:]
    return smoothed_time, raw_stack, np.stack(smoothed_runs, axis=0)


def plot_velocity_comparison(
    grouped_data: OrderedDict,
    save_path: str | None = None,
    start_idx: int = START_IDX,
) -> None:
    """Plot velocity tracking comparison across policies (vx, vy, vyaw).

    Args:
        grouped_data: OrderedDict mapping display_name -> experiment data dict
            (as returned by load_experiment_data).
        save_path: If provided, save the plot to this path (without extension).
        start_idx: Index to start plotting from (skip initial transient).
    """
    _setup_plot_style()

    vel_labels = [r'$v_x$ (m/s)', r'$v_y$ (m/s)', r'$\omega_z$ (rad/s)']
    cmd_labels = [r'$v_x^d$', r'$v_y^d$', r'$\omega_z^d$']

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    colors = plt.cm.tab10.colors
    dummy_patch = Patch(facecolor='none', alpha=0.0)
    custom_handles = []
    custom_labels = []

    # Indices into qvel for [vx, vy, wz] and into commanded_vel for [vx, vy, vyaw]
    actual_vel_dims = [0, 1, 5]
    cmd_vel_dims = [0, 1, 2]

    for i, (name, exp_data) in enumerate(grouped_data.items()):
        runs = exp_data["runs"]
        if not runs:
            continue

        color = colors[i % len(colors)]
        smooth_time, raw_stack, smooth_stack = _interpolate_and_smooth(
            runs, "actual_vel"
        )
        if smooth_time is None:
            continue

        # Also get the raw common_time for the raw overlay
        common_time, _ = _interpolate_to_common_grid(runs, "actual_vel")

        for dim in range(3):
            ax = axes[dim]
            a_dim = actual_vel_dims[dim]

            # Raw mean (light overlay)
            raw_dim = raw_stack[:, :, a_dim]
            raw_mean = np.mean(raw_dim, axis=0)
            ax.plot(common_time[start_idx:], raw_mean[start_idx:],
                    color=color, alpha=0.3, linewidth=1.0)

            # Smoothed mean +/- std
            sm_dim = smooth_stack[:, :, a_dim]
            sm_mean = np.mean(sm_dim, axis=0)
            sm_std = np.std(sm_dim, axis=0)

            # Adjust start index for the shorter smoothed array
            sm_start = max(start_idx - (MA_WINDOW - 1), 0)

            ax.plot(smooth_time[sm_start:], sm_mean[sm_start:],
                    color=color, linewidth=2.5)
            ax.fill_between(
                smooth_time[sm_start:],
                sm_mean[sm_start:] - sm_std[sm_start:],
                sm_mean[sm_start:] + sm_std[sm_start:],
                color=color, alpha=0.2,
            )

        line = Line2D([0], [0], color=color, lw=2.5)
        patch = Patch(facecolor=color, alpha=0.15)
        custom_handles.append((line, patch))
        custom_labels.append(name)

    # Plot commanded velocity from first policy's first run
    first_data = next(iter(grouped_data.values()))
    if first_data["runs"]:
        cmd = first_data["runs"][0]["commanded_vel"]
        cmd_time = first_data["runs"][0]["time"]
        for dim in range(3):
            axes[dim].plot(cmd_time, cmd[:, cmd_vel_dims[dim]], 'k--', linewidth=2)

        custom_handles.append((Line2D([0], [0], color='k', linestyle='--'), dummy_patch))
        custom_labels.append(r"Commanded")

    for dim in range(3):
        axes[dim].set_ylabel(vel_labels[dim], fontsize=20)
        axes[dim].tick_params(labelsize=16)
        axes[dim].grid(True)

    axes[2].set_xlabel('Time (s)', fontsize=20)

    axes[0].legend(
        custom_handles, custom_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(custom_labels),
        framealpha=0.0,
        columnspacing=0.8,
        handler_map={tuple: HandlerOverlay()},
        fontsize=16,
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path + ".png", bbox_inches='tight', transparent=False)
        print(f"Saved velocity plot to {save_path}.png")

    plt.close(fig)


def plot_torque_comparison(
    grouped_data: OrderedDict,
    save_path: str | None = None,
    start_idx: int = START_IDX,
) -> None:
    """Plot per-joint torque comparison across policies.

    Args:
        grouped_data: OrderedDict mapping display_name -> experiment data dict.
        save_path: If provided, save the plot to this path (without extension).
        start_idx: Index to start plotting from (skip initial transient).
    """
    _setup_plot_style()

    # Get joint names from first policy
    first_data = next(iter(grouped_data.values()))
    joint_names = first_data["joint_names"]
    num_joints = len(joint_names)

    if num_joints == 0:
        print("[Warning] No joint names found, skipping torque plot.")
        return

    # Create grid of subplots
    ncols = 3
    nrows = math.ceil(num_joints / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3 * nrows), sharex=True)
    axes = axes.flatten()

    colors = plt.cm.tab10.colors

    for joint_idx in range(num_joints):
        ax = axes[joint_idx]

        for i, (name, exp_data) in enumerate(grouped_data.items()):
            runs = exp_data["runs"]
            if not runs:
                continue

            color = colors[i % len(colors)]
            common_time, torque_stack = _interpolate_to_common_grid(runs, "torque")
            if common_time is None:
                continue

            joint_torque = torque_stack[:, :, joint_idx]
            mean_torque = np.mean(joint_torque, axis=0)
            std_torque = np.std(joint_torque, axis=0)

            ax.plot(common_time[start_idx:], mean_torque[start_idx:],
                    color=color, linewidth=1.5, label=name)
            ax.fill_between(
                common_time[start_idx:],
                mean_torque[start_idx:] - std_torque[start_idx:],
                mean_torque[start_idx:] + std_torque[start_idx:],
                color=color, alpha=0.2,
            )

        ax.set_title(joint_names[joint_idx].replace("_", r"\_"), fontsize=12)
        ax.grid(True)
        if joint_idx >= (nrows - 1) * ncols:
            ax.set_xlabel('Time (s)', fontsize=12)
        if joint_idx % ncols == 0:
            ax.set_ylabel('Torque (Nm)', fontsize=12)

    # Hide unused subplots
    for idx in range(num_joints, len(axes)):
        axes[idx].set_visible(False)

    # Add shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(0.5, 0.98), ncol=len(labels),
                   framealpha=0.0, fontsize=14)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path + ".png", bbox_inches='tight', transparent=False)
        print(f"Saved torque plot to {save_path}.png")

    plt.close(fig)


def plot_force_success_histogram(
    grouped_data: OrderedDict,
    save_path: str | None = None,
    force_range: tuple[float, float] = (50.0, 300.0),
) -> None:
    """Plot a grouped bar chart of success rate vs force magnitude bins.

    Args:
        grouped_data: OrderedDict mapping display_name -> experiment data dict
            (must contain "force_mags" and "successes" lists of equal length).
        save_path: If provided, save the plot to this path (without extension).
        num_bins: Number of equal-width bins across the force range.
        force_range: (min, max) force magnitude range in Newtons.
    """
    _setup_plot_style()

    num_bins = int(250 / 50 + 1)

    bin_centers = np.linspace(force_range[0], force_range[1], num_bins)
    bin_edges = np.linspace(force_range[0], force_range[1] + 50., num_bins + 1)
    bin_width = 50

    policy_names = list(grouped_data.keys())
    num_policies = len(policy_names)
    colors = plt.cm.tab10.colors

    # Width of each individual bar within a group
    bar_width = bin_width / (num_policies + 1)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (name, exp_data) in enumerate(grouped_data.items()):
        force_mags = exp_data.get("force_mags", [])
        successes = exp_data.get("successes", [])

        if not force_mags or len(force_mags) != len(successes):
            continue

        mags = np.array(force_mags)
        succs = np.array(successes, dtype=float)

        # Compute success rate per bin
        rates = []
        for b in range(num_bins):
            mask = (mags >= bin_edges[b]) & (mags < bin_edges[b + 1])
            # Include right edge in last bin
            if b == num_bins - 1:
                mask = (mags >= bin_edges[b]) & (mags <= bin_edges[b + 1])
            total = mask.sum()
            if total > 0:
                rates.append(succs[mask].mean())
            else:
                rates.append(0.0)

        offset = (i - (num_policies - 1) / 2) * bar_width
        ax.bar(
            bin_centers + offset,
            rates,
            width=bar_width * 0.9,
            color=colors[i % len(colors)],
            label=name,
            alpha=0.85,
        )

    ax.set_xlabel("Force Magnitude (N)", fontsize=20)
    ax.set_ylabel("Success Rate", fontsize=20)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(force_range[0] - bin_width, force_range[1] + bin_width)
    ax.tick_params(labelsize=16)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=16, framealpha=0.0)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path + ".png", bbox_inches="tight", transparent=False)
        print(f"Saved force success histogram to {save_path}.png")

    plt.close(fig)


def plot_radar_comparison(
    grouped_data: OrderedDict,
    save_path: str | None = None,
    start_idx: int = START_IDX,
) -> None:
    """Plot a radar/spider chart of MA velocity tracking errors across policies.

    Metrics plotted: Mean vx err MA, Mean vy err MA, Mean vyaw err MA.
    Values are normalized so the best (smallest) error per metric equals 1.

    Args:
        grouped_data: OrderedDict mapping display_name -> experiment data dict
            (as returned by load_experiment_data).
        save_path: If provided, save the plot to this path (without extension).
        start_idx: Index to start computing stats from (skip initial transient).
    """
    _setup_plot_style()

    actual_vel_dims = [0, 1, 5]
    cmd_vel_dims = [0, 1, 2]
    dim_labels = [r"$v_x$ err", r"$v_y$ err", r"$\omega_z$ err"]

    # Compute MA errors per policy
    policy_errors: OrderedDict[str, list[float]] = OrderedDict()
    for name, exp_data in grouped_data.items():
        runs = exp_data["runs"]
        if not runs:
            continue

        _, cmd_stack = _interpolate_to_common_grid(runs, "commanded_vel")
        smooth_time, _, smooth_vel = _interpolate_and_smooth(runs, "actual_vel")
        if smooth_time is None or smooth_vel is None or cmd_stack is None:
            continue

        sm_start = max(start_idx - (MA_WINDOW - 1), 0)
        smooth_vel_trimmed = smooth_vel[:, sm_start:, :]
        smooth_cmd_trimmed = cmd_stack[:, (MA_WINDOW - 1) + sm_start:, :]

        errors = []
        for dim in range(3):
            sm_actual = smooth_vel_trimmed[:, :, actual_vel_dims[dim]]
            sm_commanded = smooth_cmd_trimmed[:, :, cmd_vel_dims[dim]]
            errors.append(float(np.mean(np.abs(sm_actual - sm_commanded))))
        policy_errors[name] = errors

    if not policy_errors:
        print("[Warning] No data for radar plot.")
        return

    # Add norm squared error from tracking_stats.csv if all policies have it
    all_have_norm_sq = all(
        "norm_sq_error_mean" in grouped_data[name].get("tracking_stats", {})
        for name in policy_errors
    )
    if all_have_norm_sq:
        dim_labels.append(r"$\|e\|^2$")
        for name in policy_errors:
            policy_errors[name].append(
                grouped_data[name]["tracking_stats"]["norm_sq_error_mean"]
            )

    # Normalize: best (min) per metric = 1.0
    errors_array = np.array(list(policy_errors.values()))  # (num_policies, num_dims)
    min_per_metric = errors_array.min(axis=0)
    min_per_metric[min_per_metric == 0] = 1e-9  # avoid division by zero
    normalized = min_per_metric / errors_array

    # Radar plot setup
    num_dims = len(dim_labels)
    angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    colors = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})

    for i, (name, norm_vals) in enumerate(zip(policy_errors.keys(), normalized)):
        values = norm_vals.tolist() + [norm_vals[0]]  # close the polygon
        color = colors[i % len(colors)]
        ax.plot(angles, values, color=color, linewidth=2.5, label=name)
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), dim_labels, fontsize=18)
    ax.set_rlabel_position(30)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_title("Velocity Tracking Error (normalized)", y=1.08, fontsize=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=14, framealpha=0.0)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path + ".png", bbox_inches="tight", transparent=False)
        print(f"Saved radar plot to {save_path}.png")

    plt.close(fig)


def print_stats_table(
    grouped_data: OrderedDict,
    start_idx: int = START_IDX,
) -> str:
    """Pretty-print a statistics table comparing all policies.

    Args:
        grouped_data: OrderedDict mapping display_name -> experiment data dict.
        start_idx: Index to start computing stats from (skip initial transient).

    Returns:
        The formatted table as a string.
    """
    vel_dim_names = ["vx", "vy", "vyaw"]
    vel_units = ["m/s", "m/s", "rad/s"]
    actual_vel_dims = [0, 1, 5]
    cmd_vel_dims = [0, 1, 2]

    # Compute stats for each policy
    policy_stats = OrderedDict()
    for name, exp_data in grouped_data.items():
        runs = exp_data["runs"]
        if not runs:
            continue

        common_time, vel_stack = _interpolate_to_common_grid(runs, "actual_vel")
        _, cmd_stack = _interpolate_to_common_grid(runs, "commanded_vel")
        _, torque_stack = _interpolate_to_common_grid(runs, "torque")

        if common_time is None:
            continue

        # Slice to skip transient
        vel = vel_stack[:, start_idx:, :]
        cmd = cmd_stack[:, start_idx:, :]
        torque = torque_stack[:, start_idx:, :]

        # Smoothed velocity and commanded stacks
        smooth_time, _, smooth_vel = _interpolate_and_smooth(
            runs, "actual_vel"
        )
        sm_start = max(start_idx - (MA_WINDOW - 1), 0)
        smooth_vel_trimmed = smooth_vel[:, sm_start:, :] if smooth_vel is not None else None
        smooth_cmd_trimmed = cmd_stack[:, (MA_WINDOW - 1) + sm_start:, :] if cmd_stack is not None else None

        stats = {}
        for dim, (dim_name, unit) in enumerate(zip(vel_dim_names, vel_units)):
            actual = vel[:, :, actual_vel_dims[dim]]
            commanded = cmd[:, :, cmd_vel_dims[dim]]
            error = np.abs(actual - commanded)

            stats[f"Mean {dim_name} ({unit})"] = np.mean(actual)
            stats[f"Mean {dim_name} err ({unit})"] = np.mean(error)
            stats[f"Std {dim_name} err ({unit})"] = np.std(error)

            # MA-smoothed stats
            if smooth_vel_trimmed is not None and smooth_cmd_trimmed is not None:
                sm_actual = smooth_vel_trimmed[:, :, actual_vel_dims[dim]]
                sm_commanded = smooth_cmd_trimmed[:, :, cmd_vel_dims[dim]]
                sm_error = np.abs(sm_actual - sm_commanded)

                stats[f"Mean {dim_name} MA ({unit})"] = np.mean(sm_actual)
                stats[f"Mean {dim_name} err MA ({unit})"] = np.mean(sm_error)
                stats[f"Std {dim_name} err MA ({unit})"] = np.std(sm_error)

        # Torque stats: mean of absolute values across all joints and runs
        abs_torque = np.abs(torque)
        stats["Mean |torque| (Nm)"] = np.mean(abs_torque)
        stats["Std |torque| (Nm)"] = np.std(abs_torque)

        policy_stats[name] = stats

    if not policy_stats:
        print("[Warning] No data to compute stats.")
        return ""

    # Build table
    metric_names = list(next(iter(policy_stats.values())).keys())
    policy_names = list(policy_stats.keys())

    # Column widths
    metric_col_w = max(len(m) for m in metric_names) + 2
    val_col_w = max(max(len(n) for n in policy_names) + 2, 12)

    # Header
    header_line = f"{'Metric':<{metric_col_w}}"
    for pname in policy_names:
        header_line += f" | {pname:^{val_col_w}}"
    sep_line = "-" * metric_col_w
    for _ in policy_names:
        sep_line += "-+-" + "-" * val_col_w

    lines = []
    lines.append(f"\n{'=' * len(sep_line)}")
    lines.append("EXPERIMENT STATISTICS")
    lines.append(f"{'=' * len(sep_line)}")
    lines.append(header_line)
    lines.append(sep_line)

    for metric in metric_names:
        row = f"{metric:<{metric_col_w}}"
        for pname in policy_names:
            val = policy_stats[pname][metric]
            row += f" | {val:^{val_col_w}.4f}"
        lines.append(row)

    lines.append(f"{'=' * len(sep_line)}\n")

    result = "\n".join(lines)
    print(result)
    return result


def print_torque_stats_table(
    grouped_data: OrderedDict,
    start_idx: int = START_IDX,
) -> str:
    """Pretty-print a per-motor torque statistics table comparing all policies.

    Args:
        grouped_data: OrderedDict mapping display_name -> experiment data dict.
        start_idx: Index to start computing stats from (skip initial transient).

    Returns:
        The formatted table as a string.
    """
    # Get joint names from first policy
    first_data = next(iter(grouped_data.values()))
    joint_names = first_data.get("joint_names", [])
    if not joint_names:
        print("[Warning] No joint names found, skipping per-motor torque stats.")
        return ""

    policy_names = list(grouped_data.keys())

    # Compute per-joint stats for each policy
    # policy_joint_stats[policy_name] = {"mean": array, "std": array} per joint
    policy_joint_stats: OrderedDict[str, dict] = OrderedDict()
    for name, exp_data in grouped_data.items():
        runs = exp_data["runs"]
        if not runs:
            continue

        _, torque_stack = _interpolate_to_common_grid(runs, "torque")
        if torque_stack is None:
            continue

        torque = torque_stack[:, start_idx:, :]
        abs_torque = np.abs(torque)

        # Per-joint: mean and std across all runs and timesteps
        # Shape: (num_runs, num_timesteps, num_joints) -> per joint scalar
        mean_per_joint = np.mean(abs_torque, axis=(0, 1))
        std_per_joint = np.std(abs_torque, axis=(0, 1))

        policy_joint_stats[name] = {
            "mean": mean_per_joint,
            "std": std_per_joint,
        }

    if not policy_joint_stats:
        print("[Warning] No data to compute per-motor torque stats.")
        return ""

    # Build table
    joint_col_w = max(len(j) for j in joint_names) + 2
    # Two columns per policy: mean and std, width adapts to policy name
    sub_col_w = 10
    min_policy_w = 2 * sub_col_w + 1  # "mean" + space + "std"
    # Per-policy header width must fit both sub-columns and the policy name
    policy_header_ws = [max(min_policy_w, len(pname) + 2) for pname in policy_names]

    # Header
    header_line = f"{'Joint':<{joint_col_w}}"
    sub_header_line = f"{'':<{joint_col_w}}"
    for pname, phw in zip(policy_names, policy_header_ws):
        header_line += f" | {pname:^{phw}}"
        # Distribute sub-column widths within the policy header width
        left_w = phw // 2
        right_w = phw - left_w
        sub_header_line += f" | {'mean':^{left_w}}{'std':^{right_w}}"

    sep_line = "-" * joint_col_w
    for phw in policy_header_ws:
        sep_line += "-+-" + "-" * phw

    lines = []
    lines.append(f"\n{'=' * len(sep_line)}")
    lines.append("PER-MOTOR TORQUE STATISTICS (|torque| in Nm)")
    lines.append(f"{'=' * len(sep_line)}")
    lines.append(header_line)
    lines.append(sub_header_line)
    lines.append(sep_line)

    for j_idx, j_name in enumerate(joint_names):
        row = f"{j_name:<{joint_col_w}}"
        for pname, phw in zip(policy_names, policy_header_ws):
            stats = policy_joint_stats[pname]
            m = stats["mean"][j_idx]
            s = stats["std"][j_idx]
            left_w = phw // 2
            right_w = phw - left_w
            row += f" | {m:^{left_w}.4f}{s:^{right_w}.4f}"
        lines.append(row)

    lines.append(f"{'=' * len(sep_line)}\n")

    result = "\n".join(lines)
    print(result)
    return result


def print_success_table(grouped_data: OrderedDict) -> str:
    """Pretty-print a success rate table comparing all policies.

    Args:
        grouped_data: OrderedDict mapping display_name -> experiment data dict
            (as returned by load_experiment_data, must contain "successes" key).

    Returns:
        The formatted table as a string.
    """
    policy_names = list(grouped_data.keys())

    # Column widths
    name_col_w = max(max(len(n) for n in policy_names), len("Policy")) + 2
    num_col_w = 10

    header = (
        f"{'Policy':<{name_col_w}}"
        f" | {'Runs':^{num_col_w}}"
        f" | {'Successes':^{num_col_w}}"
        f" | {'Rate (%)':^{num_col_w}}"
    )
    sep = "-" * len(header)

    lines = []
    lines.append(f"\n{'=' * len(sep)}")
    lines.append("SUCCESS STATISTICS")
    lines.append(f"{'=' * len(sep)}")
    lines.append(header)
    lines.append(sep)

    for name in policy_names:
        successes = grouped_data[name].get("successes", [])
        total = len(successes)
        if total > 0:
            num_success = sum(1 for s in successes if s)
            rate = 100.0 * num_success / total
        else:
            num_success = 0
            rate = 0.0

        lines.append(
            f"{name:<{name_col_w}}"
            f" | {total:^{num_col_w}}"
            f" | {num_success:^{num_col_w}}"
            f" | {rate:^{num_col_w}.1f}"
        )

    lines.append(f"{'=' * len(sep)}\n")

    result = "\n".join(lines)
    print(result)
    return result


def print_combined_error_table(
    grouped_data: OrderedDict,
    start_idx: int = START_IDX,
) -> str:
    """Pretty-print a combined velocity tracking error table.

    Computes the MA-smoothed mean errors for vx, vy, vyaw, then reports
    each individually plus a total (sum) and combined std dev
    (sqrt of sum of squared per-dim std devs).

    Args:
        grouped_data: OrderedDict mapping display_name -> experiment data dict.
        start_idx: Index to start computing stats from (skip initial transient).

    Returns:
        The formatted table as a string.
    """
    actual_vel_dims = [0, 1, 5]
    cmd_vel_dims = [0, 1, 2]
    dim_names = ["vx", "vy", "vyaw"]

    policy_names = list(grouped_data.keys())

    # Compute per-policy errors
    policy_results: OrderedDict[str, dict] = OrderedDict()
    for name, exp_data in grouped_data.items():
        runs = exp_data["runs"]
        if not runs:
            continue

        _, cmd_stack = _interpolate_to_common_grid(runs, "commanded_vel")
        smooth_time, _, smooth_vel = _interpolate_and_smooth(runs, "actual_vel")
        if smooth_time is None or smooth_vel is None or cmd_stack is None:
            continue

        sm_start = max(start_idx - (MA_WINDOW - 1), 0)
        smooth_vel_trimmed = smooth_vel[:, sm_start:, :]
        smooth_cmd_trimmed = cmd_stack[:, (MA_WINDOW - 1) + sm_start:, :]

        mean_errors = []
        std_errors = []
        for dim in range(3):
            sm_actual = smooth_vel_trimmed[:, :, actual_vel_dims[dim]]
            sm_commanded = smooth_cmd_trimmed[:, :, cmd_vel_dims[dim]]
            err = np.abs(sm_actual - sm_commanded)
            mean_errors.append(float(np.mean(err)))
            std_errors.append(float(np.std(err)))

        total_error = sum(mean_errors)
        combined_std = math.sqrt(sum(s ** 2 for s in std_errors))

        policy_results[name] = {
            "mean_errors": mean_errors,
            "total_error": total_error,
            "combined_std": combined_std,
        }

    if not policy_results:
        print("[Warning] No data for combined error table.")
        return ""

    # Build table
    name_col_w = max(max(len(n) for n in policy_names), len("Policy")) + 2
    val_col_w = 14

    col_headers = [f"Mean {d} err MA" for d in dim_names] + ["Total Error", "Combined Std"]
    header = f"{'Policy':<{name_col_w}}"
    for h in col_headers:
        header += f" | {h:^{val_col_w}}"
    sep = "-" * len(header)

    lines = []
    lines.append(f"\n{'=' * len(sep)}")
    lines.append("COMBINED VELOCITY TRACKING ERROR (MA-smoothed)")
    lines.append(f"{'=' * len(sep)}")
    lines.append(header)
    lines.append(sep)

    for name, res in policy_results.items():
        row = f"{name:<{name_col_w}}"
        for m in res["mean_errors"]:
            row += f" | {m:^{val_col_w}.4f}"
        row += f" | {res['total_error']:^{val_col_w}.4f}"
        row += f" | {res['combined_std']:^{val_col_w}.4f}"
        lines.append(row)

    lines.append(f"{'=' * len(sep)}\n")

    result = "\n".join(lines)
    print(result)
    return result


def _resolve_experiment_dirs(
    env_type: str,
    run_dirs: list[str],
    experiment_name: str,
) -> list[str]:
    """Resolve run directory names to full experiment folder paths.

    Args:
        env_type: Environment type (e.g., running_clf_sym).
        run_dirs: Run directory names or absolute paths.
        experiment_name: Experiment name suffix to match in experiments/ folder.

    Returns:
        List of resolved experiment directory paths.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.dirname(script_dir)
    root_dir = os.path.dirname(os.path.dirname(sim_dir))
    exp_name = EXPERIMENT_NAMES[env_type]
    log_root_path = os.path.join(
        root_dir, "logs", "g1_policies", exp_name, env_type
    )
    print(f"[INFO] Looking for policies in: {log_root_path}")

    resolved = []
    for d in run_dirs:
        if os.path.isabs(d):
            run_dir = d
        else:
            run_dir = os.path.join(log_root_path, d)

        experiments_dir = os.path.join(run_dir, "experiments")
        if not os.path.isdir(experiments_dir):
            print(f"[Warning] No experiments/ folder in {run_dir}")
            continue

        # Find folder ending with _{experiment_name}
        matches = [
            f for f in sorted(os.listdir(experiments_dir))
            if f.endswith(f"_{experiment_name}")
            and os.path.isdir(os.path.join(experiments_dir, f))
        ]
        if not matches:
            print(f"[Warning] No experiment matching '*_{experiment_name}' in {experiments_dir}")
            continue

        # Use the most recent match (last alphabetically = latest timestamp)
        exp_folder = os.path.join(experiments_dir, matches[-1])
        print(f"  Found experiment: {exp_folder}")
        resolved.append(exp_folder)

    return resolved


def main():
    """Run plotting from the command line."""
    parser = argparse.ArgumentParser(
        description="Plot results from randomized parameter experiments."
    )
    parser.add_argument(
        "--env_type", type=str, required=True,
        choices=list(EXPERIMENT_NAMES.keys()),
        help="Environment type (e.g., running_clf_sym, walking_clf_sym).",
    )
    parser.add_argument(
        "--run_dirs", type=str, nargs="+", required=True,
        help="Run directory names (resolved under logs/g1_policies/...).",
    )
    parser.add_argument(
        "--names", type=str, nargs="+", required=True,
        help="Display names for each experiment directory.",
    )
    parser.add_argument(
        "--experiment_name", type=str, required=True,
        help="Experiment name suffix to match (e.g., 'TEST_torque_ablation').",
    )
    parser.add_argument(
        "--save_to_dirs", action="store_true", default=True,
        help="Save plots into each experiment directory.",
    )
    args = parser.parse_args()

    if len(args.run_dirs) != len(args.names):
        raise ValueError("Number of run_dirs must match number of names.")

    # Resolve experiment directories
    experiment_dirs_list = _resolve_experiment_dirs(
        args.env_type, args.run_dirs, args.experiment_name
    )

    if len(experiment_dirs_list) != len(args.names):
        print(f"[Warning] Found {len(experiment_dirs_list)} experiment dirs "
              f"but {len(args.names)} names were given.")

    # Load data
    grouped_data = OrderedDict()
    experiment_dirs = {}
    for exp_dir, name in zip(experiment_dirs_list, args.names):
        print(f"\nLoading data from: {exp_dir}")
        grouped_data[name] = load_experiment_data(exp_dir)
        experiment_dirs[name] = exp_dir
        print(f"  Loaded {len(grouped_data[name]['runs'])} runs for '{name}'")

    # Filter to only successful runs for plots and stats
    filtered_data = _filter_successful_runs(grouped_data)
    for name in filtered_data:
        n_total = len(grouped_data[name]["runs"])
        n_success = len(filtered_data[name]["runs"])
        print(f"  '{name}': using {n_success}/{n_total} successful runs for plots/stats")

    # Save plots to each experiment directory (using successful runs only)
    for name, exp_dir in experiment_dirs.items():
        # Per-policy plots
        single_data = OrderedDict({name: filtered_data[name]})
        vel_path = os.path.join(exp_dir, "velocity_tracking")
        torque_path = os.path.join(exp_dir, "joint_torques")
        plot_velocity_comparison(single_data, save_path=vel_path)
        plot_torque_comparison(single_data, save_path=torque_path)

        # Comparison plots (saved to each experiment dir)
        comparison_vel_path = os.path.join(exp_dir, "comparison_velocity")
        comparison_torque_path = os.path.join(exp_dir, "comparison_torques")
        plot_velocity_comparison(filtered_data, save_path=comparison_vel_path)
        plot_torque_comparison(filtered_data, save_path=comparison_torque_path)
        plot_radar_comparison(
            filtered_data,
            save_path=os.path.join(exp_dir, "radar_comparison"),
        )

    # Print stats tables and save to file
    # Use filtered data for velocity/torque stats, unfiltered for success rate
    stats_text = print_stats_table(filtered_data)
    stats_text += print_torque_stats_table(filtered_data)
    stats_text += print_success_table(grouped_data)
    stats_text += print_combined_error_table(filtered_data)

    for name, exp_dir in experiment_dirs.items():
        stats_path = os.path.join(exp_dir, "experiment_stats.txt")
        with open(stats_path, "w") as f:
            f.write(stats_text)
        print(f"Saved stats to {stats_path}")

    # Generate force-magnitude success histogram if any policy has force data
    has_force = any(
        len(d.get("force_mags", [])) > 0 for d in grouped_data.values()
    )
    if has_force:
        for name, exp_dir in experiment_dirs.items():
            plot_force_success_histogram(
                grouped_data,
                save_path=os.path.join(exp_dir, "force_success_histogram"),
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
