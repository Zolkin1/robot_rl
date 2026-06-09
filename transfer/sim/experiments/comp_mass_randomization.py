# Copyright 2026 Zachary Olkin. All rights reserved.

import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sim.log_utils import extract_data
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerBase

class HandlerOverlay(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        line, patch = orig_handle
        # Create patch (shade) first
        p = plt.Rectangle((xdescent, ydescent), width, height,
                  facecolor=patch.get_facecolor(),
                  edgecolor='none',
                  transform=trans)

        # Then line on top
        margin = width*0.05
        l = Line2D([xdescent +margin, xdescent + width -margin],
                   [ydescent + height / 2] * 2,
                   color=line.get_color(), lw=line.get_linewidth(),
                   transform=trans)
        return [p, l]

def load_multiple_runs_from_root(root_dir_pattern="mass_randomization_"):
    
    experiment_dirs = [
        os.path.join(root_dir, d)
        for root_dir, _, _ in os.walk(".")
        for d in os.listdir(root_dir)
        if d.startswith(root_dir_pattern) and os.path.isdir(os.path.join(root_dir, d))
    ]

    grouped_data = defaultdict(list)
    for exp_dir in experiment_dirs:
        label = os.path.basename(exp_dir)
        for subdir in sorted(os.listdir(exp_dir)):
            run_dir = os.path.join(exp_dir, subdir)
            if not os.path.isdir(run_dir):
                continue
            try:
                with open(os.path.join(run_dir, "sim_config.yaml")) as f:
                    config = yaml.safe_load(f)
                data = extract_data(os.path.join(run_dir, "sim_log.csv"), config)
                grouped_data[label].append({
                    "run_dir": run_dir,
                    "config": config,
                    "time": np.squeeze(data["time"]),
                    "commanded_vel": np.squeeze(data["commanded_vel"]),
                    "actual_vel": np.squeeze(data["qvel"])
                })
            except Exception as e:
                print(f"[Warning] Skipping {run_dir}: {e}")

    return grouped_data

def plot_combined_velocity(grouped_data, save_path=None, label_override=None):
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
    })

    fig, ax = plt.subplots(1, 1, figsize=(10, 5),sharex=True)
    colors = plt.cm.tab10.colors

    first_label = next(iter(grouped_data))
    time = grouped_data[first_label][0]["time"]


    start_idx = 150

    dummy_patch = Patch(facecolor='none', alpha=0.0)  # Transparent dummy patch for handler
    custom_handles = []
    custom_labels = []

    for i, (label, runs) in enumerate(grouped_data.items()):
        display_label = label_override[label] if label_override and label in label_override else label
        color = colors[i % len(colors)]
        time = runs[0]["time"]

        # Handle variable-length runs by interpolating to common time grid
        actual_vels = []
        for run in runs:
            # Interpolate each run to the common time grid
            if len(run["actual_vel"]) > 0 and len(time) > 0:
                interp_vel = np.interp(time, run["time"], run["actual_vel"][:, 0])
                actual_vels.append(interp_vel)
        
        if actual_vels:
            actual_stack = np.stack(actual_vels)
            mean_actual = np.mean(actual_stack, axis=0)
            std_actual = np.std(actual_stack, axis=0)
        else:
            # Fallback if no valid data
            mean_actual = np.zeros_like(time)
            std_actual = np.zeros_like(time)

        ax.plot(time[start_idx:], mean_actual[start_idx:], color=color, linewidth=2.5)
        ax.fill_between(time[start_idx:], mean_actual[start_idx:] - std_actual[start_idx:], mean_actual[start_idx:] + std_actual[start_idx:],
                        color=color, alpha=0.2)

        line = Line2D([0], [0], color=color, lw=2.5)
        patch = Patch(facecolor=color, alpha=0.15)
        custom_handles.append((line, patch))
        custom_labels.append(fr"{display_label}")

        print(f"[{display_label}] Average std: {np.mean(std_actual[start_idx:])} \n Average mean: {np.mean(mean_actual[start_idx:])}")


    commanded = grouped_data[first_label][0]["commanded_vel"][:, 0]
    ax.plot(time, commanded, 'k--', linewidth=2)
    custom_handles.append((Line2D([0], [0], color='k', linestyle='--'), dummy_patch))
    custom_labels.append(r"$v_x^d$")

    ax.set_ylabel(r'$v_x$ (m/s)',fontsize=20)
    ax.set_xlabel('Time (s)',fontsize=20)
    ax.set_ylim(0.0, 3.05)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax.grid(True)

    ax.legend(
     custom_handles,
     custom_labels,
     loc="lower center",
     bbox_to_anchor=(0.5, 0.96),
     ncol=len(custom_labels),  # Puts all legend entries in one row
     framealpha=0.0,
     columnspacing=0.8,
     handler_map={tuple: HandlerOverlay()},
     fontsize=20
     )


    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path + ".png", bbox_inches='tight', transparent=True)
        fig.savefig(save_path + ".pdf", bbox_inches='tight', transparent=True)
        fig.savefig(save_path + ".svg", bbox_inches='tight', transparent=True)

    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="mass_randomization_",
                        help="Prefix of experiment folders to aggregate")
    parser.add_argument("--save_name", type=str, default="experiments/plots/mass_randomization_comparison")
    args = parser.parse_args()

    grouped_data = load_multiple_runs_from_root(args.prefix)

    default_labels = list(grouped_data.keys())
    default_legends = [label.replace("mass_randomization_", "") for label in default_labels]
    label_override = {k: v for k, v in zip(default_labels, default_legends)}

    # Optionally manually override here
    label_override.update({"mass_randomization_g1_21j_config_baseline": "Baseline"})
    label_override.update({"mass_randomization_g1_21j_config_lip": "LIP-CLF"})
    label_override.update({"mass_randomization_g1_21j_config_hzd": "HZD-CLF"})
    label_override.update({"mass_randomization_g1_21j_config_running_clf_15": "CLF Weight 1.5"})
    label_override.update({"mass_randomization_g1_21j_config_running_no_clf": "No CLF"})
    label_override.update({"mass_randomization_g1_21j_config_running": "CLF Weight 1"})

    if len(grouped_data) == 0:
        print("No data found. Check experiment folder structure and naming.")
    else:
        plot_combined_velocity(grouped_data, save_path=args.save_name, label_override=label_override)
