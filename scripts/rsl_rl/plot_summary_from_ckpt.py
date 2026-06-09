# Copyright 2026 Zachary Olkin. All rights reserved.

import os
import glob,re
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

units = {'v': ['m/s'], 'vdot': ['m/s²']}
env_ids = 0

def extract_step(folder_name):
    match = re.search(r"play_step_(\d+)", folder_name)
    return int(match.group(1)) if match else -1

def aggregate_checkpoints(setup_path):
    ckpt_dirs = sorted(
        glob.glob(os.path.join(setup_path, "play_step_*")),
        key=extract_step
    )
    step_list = [extract_step(p) for p in ckpt_dirs]

    all_data = {}
    for ckpt in ckpt_dirs:
        print(f"loading {ckpt}")
        data = load_data(ckpt, keys=("v", "vdot", "y_act", "y_out"))
        for k, v in data.items():
            all_data.setdefault(k, []).append(v)

    for k in all_data:
        all_data[k] = np.stack(all_data[k], axis=0)  # shape: [N_ckpt, T, ...]

    if 'y_act' in all_data and 'y_out' in all_data:
        all_data['error_y'] = all_data['y_act'] - all_data['y_out']

    all_data["step_list"] = np.array(step_list)
    return all_data


def load_data(log_dir, keys=("v", "vdot", "y_act", "y_out")):
    data = {}
    for key in keys:
        path = os.path.join(log_dir, f"{key}.pkl")
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            val = pickle.load(f)
            if isinstance(val[0], torch.Tensor):
                val = np.array([v.cpu().numpy() for v in val])
            else:
                val = np.array(val)
            data[key] = val
    return data




def plot_comparison(setup_dict, save_dir=None):
    keys_to_plot = ["v", "vdot", "error_y"]
    colors = ["g", "m", "r"]

    # Optional per-key y-axis limits
    ylims = {
        "v": (0, 150),
        "vdot": (-100, 100),
        "error_y": (0, 1.2),
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Step 1: Aggregate once per setup
    aggregated_data = {}
    for name, path in setup_dict.items():
        aggregated_data[name] = aggregate_checkpoints(path)

    # Step 2: Plot each key
    for key, color in zip(keys_to_plot, colors):
        plt.figure(figsize=(10, 5))

        for name, data in aggregated_data.items():
            if key not in data:
                continue
            val = data[key]  # [N_ckpt, T, B] or [N_ckpt, T, B, D]

            if val.ndim == 4:
                val = np.linalg.norm(val, axis=-1)  # → [N_ckpt, T, B]
            val = val.mean(axis=-1)  # average over B → [N_ckpt, T]

            mean = val.mean(axis=1)  # average over T → [N_ckpt]
            std = val.std(axis=1)

            chkpt = data["step_list"]

            plt.plot(chkpt, mean, label=f'{name} mean')
            plt.fill_between(chkpt, mean - std, mean + std, alpha=0.3)

        plt.title(f'{key} vs Training Iteration')
        plt.xlabel("Training Iteration")
        plt.ylabel(units.get(key, [''])[0])

        # Apply ylim if specified
        if key in ylims:
            plt.ylim(*ylims[key])

        plt.grid(True)
        plt.legend()

        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{key}_vs_training_iter.png"), dpi=300)

        plt.show()




if __name__ == "__main__":
    SETUPS = {
        "clf": "intermediate_log/clf",
        # "clf_vdot": "intermediate_log/clf_vdot",
        "ref_tracking": "intermediate_log/ref_tracking"
    }
    plot_comparison(SETUPS, save_dir="intermediate_log/plots")
