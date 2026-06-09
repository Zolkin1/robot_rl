# Copyright 2026 Zachary Olkin. All rights reserved.

"""Play stats comparison across multiple policies.

Reads pre-existing plots/play_stats.txt files from multiple policy run
directories, parses the Lyapunov (V) values and group error summaries,
and produces comparison histograms and LaTeX tables.

Outputs (saved to experiments/{timestamp}_{experiment_name}/ in each run dir):
- V value bar chart with error bars (one bar per policy)
- Position error subgroup bar chart (Positions/Orientations/Joints per policy)
- Velocity error subgroup bar chart (same subgroups)
- LaTeX table with all V and group error values

Usage:
    python experiments/play_stats_comparison.py \
        --env_type running_clf_sym \
        --run_dirs dir1 dir2 dir3 \
        --names "Policy A" "Policy B" "Policy C" \
        --experiment_name ablation_v_comparison
"""

import argparse
import os
import re
import sys
import textwrap
from collections import OrderedDict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

# Muted color palette (softer Tableau shades, suitable for publications)
# Uses the first shade from each of the tab20c color groups
PLOT_COLORS = [plt.cm.tab20c.colors[i] for i in range(0, 20, 4)]

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Environment experiment names mapping (same as in randomized_params_experiment.py)
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


def _setup_plot_style() -> None:
    """Configure matplotlib for publication-quality plots."""
    plt.rcParams.update({
        'font.size': 18,
        "text.usetex": True,
        "font.family": "serif",
    })


def _wrap_label(name: str, width: int = 15) -> str:
    """Wrap a label string into a LaTeX shortstack for multi-line x-tick labels.

    Args:
        name: The label text to wrap.
        width: Maximum characters per line before wrapping.

    Returns:
        LaTeX shortstack string with wrapped lines.
    """
    lines = textwrap.wrap(name, width=width)
    if len(lines) <= 1:
        return name
    return r"\shortstack{" + r"\\".join(lines) + "}"


def parse_play_stats(filepath: str) -> dict:
    """Parse a play_stats.txt file and extract V values and group error summaries.

    Args:
        filepath: Absolute path to the play_stats.txt file.

    Returns:
        Dict with keys:
            - v_mean (float): Mean Lyapunov V across envs.
            - v_std (float): Std of V across envs.
            - pos_error_groups (OrderedDict): group_name -> (mean, std) for position errors.
            - vel_error_groups (OrderedDict): group_name -> (mean, std) for velocity errors.
    """
    with open(filepath, "r") as f:
        text = f.read()

    result = {
        "v_mean": None,
        "v_std": None,
        "norm_sq_mean": None,
        "norm_sq_std": None,
        "pos_error_groups": OrderedDict(),
        "vel_error_groups": OrderedDict(),
    }

    # Parse V values
    v_mean_match = re.search(r"Mean across envs:\s+([\d.]+)", text)
    v_std_match = re.search(r"Std\s+across envs:\s+([\d.]+)", text)
    if v_mean_match:
        result["v_mean"] = float(v_mean_match.group(1))
    if v_std_match:
        result["v_std"] = float(v_std_match.group(1))

    # Parse norm squared error
    norm_sq_section = re.search(r"Norm Squared Error.*?Mean across envs:\s+([\d.]+).*?Std\s+across envs:\s+([\d.]+)", text, re.DOTALL)
    if norm_sq_section:
        result["norm_sq_mean"] = float(norm_sq_section.group(1))
        result["norm_sq_std"] = float(norm_sq_section.group(2))

    # Parse group summaries using section-aware parsing
    def _parse_group_summaries(section_text: str) -> OrderedDict:
        """Parse group summary lines from a section of text."""
        groups = OrderedDict()
        # Match lines like: "  Positions                                    0.159887     0.005850"
        pattern = re.compile(r"^\s+(Positions|Orientations|Joints)\s+([\d.]+)\s+([\d.]+)", re.MULTILINE)
        for m in pattern.finditer(section_text):
            groups[m.group(1)] = (float(m.group(2)), float(m.group(3)))
        return groups

    # Split into position and velocity sections
    pos_section_match = re.search(
        r"Position Error Group Summaries:(.*?)(?=Velocity Errors|$)", text, re.DOTALL
    )
    vel_section_match = re.search(
        r"Velocity Error Group Summaries:(.*?)(?====|$)", text, re.DOTALL
    )

    if pos_section_match:
        result["pos_error_groups"] = _parse_group_summaries(pos_section_match.group(1))
    if vel_section_match:
        result["vel_error_groups"] = _parse_group_summaries(vel_section_match.group(1))

    return result


def plot_v_bar_chart(
    policy_data: OrderedDict,
    save_path: str | None = None,
) -> None:
    """Plot a bar chart of V (Lyapunov) values with error bars across policies.

    Args:
        policy_data: OrderedDict mapping policy name -> parsed play_stats dict.
        save_path: If provided, save plot to this path (without extension).
    """
    _setup_plot_style()
    colors = PLOT_COLORS

    names = list(policy_data.keys())
    means = [policy_data[n]["v_mean"] for n in names]
    stds = [policy_data[n]["v_std"] for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(names)), 6))

    bars = ax.bar(
        x, means, yerr=stds,
        width=0.6,
        color=[colors[i % len(colors)] for i in range(len(names))],
        capsize=6,
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_ylabel("Tracking Error ($V = e^TPe)$", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels([_wrap_label(n) for n in names], fontsize=16)
    ax.tick_params(labelsize=16)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path + ".png", bbox_inches="tight", transparent=False)
        fig.savefig(save_path + ".pdf", bbox_inches="tight", transparent=True)
        print(f"Saved V bar chart to {save_path}.png")

    plt.close(fig)


def plot_norm_sq_error_bar_chart(
    policy_data: OrderedDict,
    save_path: str | None = None,
    groups: list[str] | None = None,
    member_names: list[str] | None = None,
) -> None:
    """Plot a bar chart of norm squared error (dot(e,e)) with error bars across policies.

    Args:
        policy_data: OrderedDict mapping policy name -> parsed play_stats dict.
        save_path: If provided, save plot to this path (without extension).
        groups: Optional list of group labels, one per policy. Policies sharing a
            group label are clustered together on the x-axis. If None, each policy
            gets its own bar (original behavior).
        member_names: Optional list of legend labels, one per position index within
            each group. The i-th member in every group shares the same color and
            legend entry. If None, falls back to individual policy names.
    """
    _setup_plot_style()
    colors = PLOT_COLORS

    names = list(policy_data.keys())
    means = [policy_data[n]["norm_sq_mean"] for n in names]
    stds = [policy_data[n]["norm_sq_std"] for n in names]

    if any(m is None for m in means):
        print("Norm squared error data not available for all policies, skipping plot.")
        return

    if groups is not None:
        # Grouped bar layout: cluster policies by group
        unique_groups = list(OrderedDict.fromkeys(groups))
        num_groups = len(unique_groups)

        # Build mapping: group_label -> list of (policy_name, mean, std) tuples
        group_members: OrderedDict[str, list[tuple[str, float, float]]] = OrderedDict()
        for g in unique_groups:
            group_members[g] = []
        for name, g, m, s in zip(names, groups, means, stds):
            group_members[g].append((name, m, s))

        max_members = max(len(v) for v in group_members.values())
        x = np.arange(num_groups)
        bar_width = 0.8 / max_members

        fig, ax = plt.subplots(figsize=(max(8, 3 * num_groups), 6))

        # Track which position indices have been added to legend
        legend_added: set[int] = set()

        for group_idx, group_label in enumerate(unique_groups):
            members = group_members[group_label]
            for j, (pname, m, s) in enumerate(members):
                offset = (j - (len(members) - 1) / 2) * bar_width
                # Color by position index so the i-th member in every group matches
                bar_color = colors[j % len(colors)]
                # Legend label: use member_names if provided, else policy name
                if j not in legend_added:
                    label = member_names[j] if member_names is not None else pname
                else:
                    label = None
                ax.bar(
                    x[group_idx] + offset, m, yerr=s,
                    width=bar_width * 0.9,
                    color=bar_color,
                    capsize=4,
                    alpha=0.85,
                    label=label,
                    edgecolor="black",
                    linewidth=0.5,
                )
                legend_added.add(j)

        ax.set_ylabel(r"Reference Tracking Squared Error ($e^T e$)", fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels([_wrap_label(g) for g in unique_groups], fontsize=20)
        ax.tick_params(labelsize=20)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=20, framealpha=0.0)
    else:
        # Original ungrouped layout: one bar per policy
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(names)), 6))

        ax.bar(
            x, means, yerr=stds,
            width=0.6,
            color=[colors[i % len(colors)] for i in range(len(names))],
            capsize=6,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_ylabel(r"Reference Tracking Squared Error ($e^T e$)", fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels([_wrap_label(n) for n in names], fontsize=16)
        ax.tick_params(labelsize=16)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path + ".png", bbox_inches="tight", transparent=False)
        fig.savefig(save_path + ".pdf", bbox_inches="tight", transparent=True)
        print(f"Saved norm squared error bar chart to {save_path}.png")

    plt.close(fig)


def plot_subgroup_error_bar_chart(
    policy_data: OrderedDict,
    error_type: str = "pos",
    save_path: str | None = None,
) -> None:
    """Plot a grouped bar chart of subgroup errors across policies.

    Args:
        policy_data: OrderedDict mapping policy name -> parsed play_stats dict.
        error_type: "pos" for position errors, "vel" for velocity errors.
        save_path: If provided, save plot to this path (without extension).
    """
    _setup_plot_style()
    colors = PLOT_COLORS

    key = "pos_error_groups" if error_type == "pos" else "vel_error_groups"
    title_type = "Position" if error_type == "pos" else "Velocity"

    names = list(policy_data.keys())
    num_policies = len(names)

    # Get subgroup names from the first policy that has data
    subgroup_names = []
    for n in names:
        groups = policy_data[n][key]
        if groups:
            subgroup_names = list(groups.keys())
            break

    if not subgroup_names:
        print(f"No {error_type} error group data found, skipping plot.")
        return

    num_groups = len(subgroup_names)
    x = np.arange(num_groups)
    bar_width = 0.8 / num_policies

    fig, ax = plt.subplots(figsize=(max(8, 3 * num_groups), 6))

    for i, name in enumerate(names):
        groups = policy_data[name][key]
        means = [groups.get(g, (0.0, 0.0))[0] for g in subgroup_names]
        stds = [groups.get(g, (0.0, 0.0))[1] for g in subgroup_names]

        offset = (i - (num_policies - 1) / 2) * bar_width
        ax.bar(
            x + offset, means, yerr=stds,
            width=bar_width * 0.9,
            color=colors[i % len(colors)],
            capsize=4,
            alpha=0.85,
            label=name,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_ylabel(f"{title_type} Error", fontsize=20)
    ax.set_xticks(x)
    ax.set_xticklabels(subgroup_names, fontsize=16)
    ax.tick_params(labelsize=16)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=14, framealpha=0.0)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path + ".png", bbox_inches="tight", transparent=False)
        fig.savefig(save_path + ".pdf", bbox_inches="tight", transparent=True)
        print(f"Saved {error_type} error bar chart to {save_path}.png")

    plt.close(fig)


def generate_latex_table(policy_data: OrderedDict) -> str:
    """Generate LaTeX table code for V values and group errors.

    Args:
        policy_data: OrderedDict mapping policy name -> parsed play_stats dict.

    Returns:
        String containing LaTeX tabular code.
    """
    names = list(policy_data.keys())

    lines = []
    lines.append("% Auto-generated by play_stats_comparison.py")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Lyapunov function and tracking error comparison.}")
    lines.append(r"\label{tab:play_stats}")

    # Column spec: Policy | V | dot(e,e) | Pos groups | Vel groups
    lines.append(r"\begin{tabular}{l c c c c c c c c}")
    lines.append(r"\toprule")
    lines.append(
        r"Policy & $V$ & $\|e\|^2$ & "
        r"\multicolumn{3}{c}{Position Error} & "
        r"\multicolumn{3}{c}{Velocity Error} \\"
    )
    lines.append(
        r" & (mean $\pm$ std) & (mean $\pm$ std) & "
        r"Pos. & Ori. & Joints & "
        r"Pos. & Ori. & Joints \\"
    )
    lines.append(r"\midrule")

    for name in names:
        d = policy_data[name]

        # V value
        v_str = f"${d['v_mean']:.2f} \\pm {d['v_std']:.2f}$"

        # Norm squared error
        if d["norm_sq_mean"] is not None:
            norm_sq_str = f"${d['norm_sq_mean']:.4f} \\pm {d['norm_sq_std']:.4f}$"
        else:
            norm_sq_str = "--"

        # Position error groups
        pos_groups = d["pos_error_groups"]
        pos_strs = []
        for g in ["Positions", "Orientations", "Joints"]:
            if g in pos_groups:
                m, s = pos_groups[g]
                pos_strs.append(f"${m:.4f} \\pm {s:.4f}$")
            else:
                pos_strs.append("--")

        # Velocity error groups
        vel_groups = d["vel_error_groups"]
        vel_strs = []
        for g in ["Positions", "Orientations", "Joints"]:
            if g in vel_groups:
                m, s = vel_groups[g]
                vel_strs.append(f"${m:.4f} \\pm {s:.4f}$")
            else:
                vel_strs.append("--")

        # Escape underscores in policy name for LaTeX
        escaped_name = name.replace("_", r"\_")
        row = f"{escaped_name} & {v_str} & {norm_sq_str} & " + " & ".join(pos_strs) + " & " + " & ".join(vel_strs) + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main() -> None:
    """Run the play stats comparison."""
    parser = argparse.ArgumentParser(
        description="Compare play_stats.txt across multiple policies."
    )
    parser.add_argument(
        "--env_type", type=str, required=True,
        choices=list(EXPERIMENT_NAMES.keys()),
        help="Environment type (e.g., running_clf_sym, walking_clf_sym).",
    )
    parser.add_argument(
        "--run_dirs", type=str, nargs="+", required=True,
        help="Run directory names (relative to logs/g1_policies/{experiment}/{env_type}/) "
             "or absolute paths.",
    )
    parser.add_argument(
        "--names", type=str, nargs="+", required=True,
        help="Display names for each policy (used in plots and tables).",
    )
    parser.add_argument(
        "--experiment_name", type=str, required=True,
        help="Name for the experiment (used in folder naming).",
    )
    parser.add_argument(
        "--groups", type=str, nargs="+", default=None,
        help="Group labels for each policy (same length as --names). "
             "Policies sharing a group label are clustered together in the "
             "norm squared error bar chart. If omitted, no grouping is applied.",
    )
    parser.add_argument(
        "--member_names", type=str, nargs="+", default=None,
        help="Legend labels for each position index within groups. "
             "Length must equal the number of members per group. "
             "The i-th member in every group shares the same color and legend entry.",
    )
    args = parser.parse_args()

    if len(args.run_dirs) != len(args.names):
        raise ValueError("Number of --run_dirs must match number of --names.")
    if args.groups is not None and len(args.groups) != len(args.names):
        raise ValueError("Number of --groups must match number of --names.")
    if args.member_names is not None and args.groups is None:
        raise ValueError("--member_names requires --groups to be specified.")

    # Resolve run dirs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.dirname(script_dir)
    root_dir = os.path.dirname(os.path.dirname(sim_dir))
    experiment_name = EXPERIMENT_NAMES[args.env_type]
    log_root_path = os.path.join(
        root_dir, "logs", "g1_policies", experiment_name, args.env_type
    )
    print(f"[INFO] Looking for policies in: {log_root_path}")

    run_dirs = []
    for d in args.run_dirs:
        if os.path.isabs(d):
            run_dirs.append(d)
        else:
            run_dirs.append(os.path.join(log_root_path, d))

    # Parse play_stats.txt from each run
    policy_data = OrderedDict()
    for run_dir, name in zip(run_dirs, args.names):
        stats_path = os.path.join(run_dir, "plots", "play_stats.txt")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"play_stats.txt not found: {stats_path}")

        print(f"Parsing: {stats_path}")
        policy_data[name] = parse_play_stats(stats_path)

        d = policy_data[name]
        print(f"  {name}: V = {d['v_mean']:.4f} +/- {d['v_std']:.4f}")

    # Generate experiment folder name
    experiment_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_folder_name = f"{experiment_timestamp}_{args.experiment_name}"
    print(f"\nExperiment folder: {experiment_folder_name}")

    # Create experiment folders and generate outputs
    experiment_folders = {}
    for run_dir, name in zip(run_dirs, args.names):
        experiment_folder = os.path.join(run_dir, "experiments", experiment_folder_name)
        os.makedirs(experiment_folder, exist_ok=True)
        experiment_folders[name] = experiment_folder

    # Generate plots (save to each experiment folder)
    for name, exp_folder in experiment_folders.items():
        plot_v_bar_chart(
            policy_data,
            save_path=os.path.join(exp_folder, "v_bar_chart"),
        )
        plot_subgroup_error_bar_chart(
            policy_data,
            error_type="pos",
            save_path=os.path.join(exp_folder, "position_error_bar_chart"),
        )
        plot_subgroup_error_bar_chart(
            policy_data,
            error_type="vel",
            save_path=os.path.join(exp_folder, "velocity_error_bar_chart"),
        )
        plot_norm_sq_error_bar_chart(
            policy_data,
            save_path=os.path.join(exp_folder, "norm_sq_error_bar_chart"),
            groups=args.groups,
            member_names=args.member_names,
        )

    # Generate LaTeX table
    latex_code = generate_latex_table(policy_data)
    print(f"\n{'='*60}")
    print("LaTeX Table:")
    print(f"{'='*60}")
    print(latex_code)

    for name, exp_folder in experiment_folders.items():
        table_path = os.path.join(exp_folder, "latex_table.txt")
        with open(table_path, "w") as f:
            f.write(latex_code + "\n")
        print(f"Saved LaTeX table to {table_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("Play stats comparison complete!")
    print(f"{'='*60}")
    print(f"Experiment name: {experiment_folder_name}")
    print(f"Policies compared: {len(args.names)}")
    print(f"\nExperiment folders:")
    for name, folder in experiment_folders.items():
        print(f"  {name}: {folder}")


if __name__ == "__main__":
    main()
