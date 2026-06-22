"""Double integrator comparison experiment.

Loads trajectory data (npz) from multiple play runs (different checkpoints or
policies) and generates overlay comparison plots.  Each run directory receives
a copy of the plots under ``experiments/<name>_<timestamp>/``.

Usage:
    python experiments/di_comparison.py \
        --run_dirs RUN_A RUN_A RUN_B \
        --checkpoints model_15 model_100 model_50 \
        --names "Iter 15" "Iter 100" "Run B" \
        --experiment_name training_progress \
        --gamma 0.99 --horizon 24 --alpha 0.02
"""

import argparse
import os
import sys
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_run_dir(run_dir: str) -> str:
    """Resolve a run directory to an absolute path.

    If ``run_dir`` is already absolute, return it as-is.  Otherwise resolve
    relative to ``logs/g1_policies/double_integrator/double_integrator/``.
    """
    if os.path.isabs(run_dir):
        return run_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    return os.path.join(
        root_dir, "logs", "g1_policies", "double_integrator",
        "double_integrator", run_dir,
    )


def _load_npz(run_dir: str, checkpoint: str) -> dict[str, np.ndarray]:
    """Load ``trajectory_data.npz`` for a given run and checkpoint.

    Args:
        run_dir: Absolute path to the training run directory.
        checkpoint: Checkpoint name (e.g. ``model_15``).

    Returns:
        Dictionary of numpy arrays loaded from the npz file.
    """
    npz_path = os.path.join(run_dir, checkpoint, "plots", "trajectory_data.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Trajectory data not found: {npz_path}\n"
            f"Run play_policy.py with --checkpoint={checkpoint} --log_data first."
        )
    return dict(np.load(npz_path))


# ---------------------------------------------------------------------------
# Discounted return helper
# ---------------------------------------------------------------------------

def _discounted_return(
    reward: np.ndarray,
    gamma: float,
    horizon: int,
) -> np.ndarray:
    """Compute (optionally finite-horizon) discounted return.

    For each starting time t, sums discounted rewards over the horizon.
    If the horizon extends past the trajectory, uses the last available
    reward value.

    Args:
        reward: Array of shape ``[T, N]``.
        gamma: Discount factor.
        horizon: Positive for finite horizon, negative/zero for infinite.

    Returns:
        Discounted return array of same shape as ``reward``.
    """
    n_steps = reward.shape[0]
    H = horizon if horizon > 0 else n_steps
    disc = np.zeros_like(reward)
    for t in range(n_steps):
        for k in range(H):
            idx = min(t + k, n_steps - 1)
            disc[t] += (gamma ** k) * reward[idx]
    return disc


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]


def _plot_comparison(
    datasets: OrderedDict[str, dict[str, np.ndarray]],
    key: str,
    ylabel: str,
    title: str,
    save_path: str,
    ref_line: float | None = None,
) -> None:
    """Plot a single variable across multiple datasets.

    Args:
        datasets: Ordered mapping ``{name: npz_dict}``.
        key: Key to plot from each npz dict (shape ``[T, N]``).
        ylabel: Y-axis label.
        title: Plot title.
        save_path: Path to save the figure (without extension).
        ref_line: Optional horizontal reference line value.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for i, (name, d) in enumerate(datasets.items()):
        if key not in d:
            continue
        arr = d[key]  # [T, N]
        time = d["time"]
        color = COLORS[i % len(COLORS)]
        mean = arr.mean(axis=1)
        std = arr.std(axis=1)
        ax.plot(time, mean, color=color, linewidth=2, label=name)
        ax.fill_between(time, mean - std, mean + std, color=color, alpha=0.2)

    if ref_line is not None:
        ax.axhline(ref_line, color="k", linestyle="--", linewidth=0.8)

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {save_path}.png")


def _plot_discounted_comparison(
    datasets: OrderedDict[str, dict[str, np.ndarray]],
    gamma: float,
    horizon: int,
    save_path: str,
) -> None:
    """Plot discounted CLF reward comparison.

    Args:
        datasets: Ordered mapping ``{name: npz_dict}``.
        gamma: Discount factor.
        horizon: Finite horizon length (<=0 for infinite).
        save_path: Path to save the figure (without extension).
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for i, (name, d) in enumerate(datasets.items()):
        if "clf_reward" not in d:
            continue
        disc = _discounted_return(d["clf_reward"], gamma, horizon)
        time = d["time"]
        color = COLORS[i % len(COLORS)]
        mean = disc.mean(axis=1)
        std = disc.std(axis=1)

        ax.plot(time, mean, color=color, linewidth=2, label=name)
        ax.fill_between(time, mean - std, mean + std,
                        color=color, alpha=0.2)

    title_suffix = f"H={horizon}" if horizon > 0 else "infinite"
    ax.set_title(f"Discounted CLF Reward (\u03b3={gamma}, {title_suffix})")
    ax.set_ylabel("Discounted CLF Reward")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path + ".png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved {save_path}.png")


def _plot_discounted_min_comparison(
    datasets: OrderedDict[str, dict[str, np.ndarray]],
    gamma: float,
    horizon: int,
    save_path: str,
) -> None:
    """Plot minimum discounted CLF reward comparison.

    Instead of mean +/- std, this plots the minimum discounted reward
    across all environments at each timestep.

    Args:
        datasets: Ordered mapping ``{name: npz_dict}``.
        gamma: Discount factor.
        horizon: Finite horizon length (<=0 for infinite).
        save_path: Path to save the figure (without extension).
    """
    rc = {
        "text.usetex": True,
        "font.family": "serif",
        "axes.titlesize": 28,
        "axes.labelsize": 32,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.fontsize": 24,
        "lines.linewidth": 6,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(10, 4))

        for i, (name, d) in enumerate(datasets.items()):
            if "clf_reward" not in d:
                continue
            disc = _discounted_return(d["clf_reward"], gamma, horizon)
            time = d["time"]
            color = COLORS[i % len(COLORS)]
            k = max(1, int(np.ceil(0.4 * disc.shape[1])))
            bottom_40 = np.partition(disc, k, axis=1)[:, :k]
            avg_bottom_40 = bottom_40.mean(axis=1)

            ax.plot(time, avg_bottom_40, color=color, label=name)

        ax.set_ylabel(r"Discounted CLF Reward")
        ax.set_xlabel(r"Time (s)")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(save_path + ".png", dpi=150, bbox_inches="tight")
        fig.savefig(save_path + ".svg", bbox_inches="tight")
        plt.close(fig)
    print(f"[INFO] Saved {save_path}.png and {save_path}.svg")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the double integrator comparison experiment."""
    parser = argparse.ArgumentParser(
        description="Compare double integrator performance across checkpoints / runs."
    )
    parser.add_argument(
        "--run_dirs", type=str, nargs="+", required=True,
        help="Run directory names (relative to log root, or absolute paths).",
    )
    parser.add_argument(
        "--checkpoints", type=str, nargs="+", required=True,
        help="Checkpoint names (e.g. model_15 model_100). Must match --run_dirs length.",
    )
    parser.add_argument(
        "--names", type=str, nargs="+", required=True,
        help="Display names for each entry (used in plot legends).",
    )
    parser.add_argument(
        "--experiment_name", type=str, default="di_comparison",
        help="Name for the experiment (used in folder naming).",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor for discounted CLF reward (default: 0.99).",
    )
    parser.add_argument(
        "--horizon", type=int, default=24,
        help="Finite horizon for discounted return (default: 24). Use <=0 for infinite.",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.02,
        help="CLF decay rate for the decay condition plot (default: 0.02).",
    )
    args = parser.parse_args()

    n = len(args.run_dirs)
    if len(args.checkpoints) != n or len(args.names) != n:
        raise ValueError(
            f"--run_dirs ({n}), --checkpoints ({len(args.checkpoints)}), "
            f"and --names ({len(args.names)}) must all have the same length."
        )

    # Resolve paths and load data
    datasets = OrderedDict()
    resolved_dirs = []
    for run_dir, ckpt, name in zip(args.run_dirs, args.checkpoints, args.names):
        abs_dir = _resolve_run_dir(run_dir)
        resolved_dirs.append(abs_dir)
        print(f"Loading: {name} <- {abs_dir}/{ckpt}/plots/")
        datasets[name] = _load_npz(abs_dir, ckpt)

    # Pre-compute derived quantities
    for name, d in datasets.items():
        if "v" in d and "vdot" in d:
            d["clf_decay"] = d["vdot"] + args.alpha * d["v"]

    # Create experiment folder timestamp
    experiment_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_folder_name = f"{args.experiment_name}_{experiment_timestamp}"

    # Determine unique run dirs for saving plots
    unique_dirs = list(OrderedDict.fromkeys(resolved_dirs))
    experiment_folders = []
    for d in unique_dirs:
        folder = os.path.join(d, "experiments", experiment_folder_name)
        os.makedirs(folder, exist_ok=True)
        experiment_folders.append(folder)

    # Generate all plots, saving a copy to each unique run directory
    plot_specs = [
        ("pos", "Position (m)", "Position Comparison", "position_comparison"),
        ("vel", "Velocity (m/s)", "Velocity Comparison", "velocity_comparison"),
        ("force", "Force (N)", "Force Comparison", "force_comparison"),
        ("v", "V", "CLF (V) Comparison", "v_comparison"),
        ("vdot", r"$\Delta V$", r"CLF ($\Delta V$) Comparison", "delta_v_comparison"),
        ("clf_decay", "Decay Rate",
         rf"CLF Decay ($\Delta V + \alpha V$, $\alpha$={args.alpha})",
         "clf_decay_comparison"),
    ]

    for exp_folder in experiment_folders:
        # Standard comparison plots
        for key, ylabel, title, filename in plot_specs:
            ref = 0.0 if key in ("pos", "vel", "force", "clf_decay") else None
            _plot_comparison(datasets, key, ylabel, title,
                             os.path.join(exp_folder, filename), ref_line=ref)

        # Discounted CLF reward
        _plot_discounted_comparison(
            datasets, args.gamma, args.horizon,
            os.path.join(exp_folder, "discounted_clf_reward_comparison"),
        )

        # Min discounted CLF reward
        _plot_discounted_min_comparison(
            datasets, args.gamma, args.horizon,
            os.path.join(exp_folder, "min_discounted_clf_reward_comparison"),
        )

    # Summary
    print(f"\n{'=' * 60}")
    print("Experiment complete!")
    print(f"{'=' * 60}")
    print(f"Experiment name: {experiment_folder_name}")
    print(f"Entries compared: {n}")
    print(f"\nPlots saved to:")
    for folder in experiment_folders:
        print(f"  {folder}")


if __name__ == "__main__":
    main()
