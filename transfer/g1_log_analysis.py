# Copyright 2026 Zachary Olkin. All rights reserved.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import glob
import os
import re
import ast
from typing import Tuple, List

def parse_g1_log(file_path: str, config_path: str, fields_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    num_obs = None
    num_actions = None
    gl_flat = False
    control_entries = config.get("onboard", {}).get("control", [])
    for entry in control_entries:
        if entry.get("pkg") == "g1_control":
            params = entry.get("params", {})
            num_obs = params.get("num_obs")
            num_actions = params.get("num_actions")
            gl_flat = params.get("gl_flag", False)
            break

    if num_obs is None or num_actions is None:
        raise ValueError("Missing 'num_obs' or 'num_actions' under g1_control in config")

    timestamps, observations, actions = parse_malformed_csv(file_path, fields_path, num_obs, num_actions)

    obs_labels = generate_gl_flat_obs_labels(num_obs) if gl_flat else [f"obs[{i}]" for i in range(num_obs)]
    return timestamps, observations, actions, obs_labels



def parse_malformed_csv(file_path: str, fields_path: str, num_obs: int, num_actions: int):
    """
    Field-aware temporary parser for malformed CSV logs.
    Handles arbitrary ordering of ["time", "observation", "action"] in fields.csv.
    Assumes one of the fields (obs or act) is in quoted string format.
    """
    time_list = []
    obs_list = []
    act_list = []

    # Read column order from fields.csv (should be just header with no rows)
    fields_df = pd.read_csv(fields_path, nrows=0)
    column_order = list(fields_df.columns)

    expected_fields = {"time", "observation", "action"}
    if set(column_order) != expected_fields or len(column_order) != 3:
        raise ValueError(f"Expected fields: {expected_fields}, got: {column_order}")

    pattern = r'^([0-9\.eE+-]+),"(.*?)",(.*)$'

    for line in open(file_path, 'r'):
        match = re.match(pattern, line.strip())
        if not match:
            print(f"Skipping malformed line: {line.strip()}")
            continue

        # Always parse time, quoted array, and rest
        time_str, quoted_str, trailing_str = match.groups()
        try:
            timestamp = float(time_str)
            quoted_data = ast.literal_eval(f"{quoted_str}")
            trailing_data = [float(x) for x in trailing_str.strip().split(',')]
        except Exception as e:
            print(f"Failed to parse line: {e}")
            continue

        # Field positions: field[0] = "time", field[1] = "quoted", field[2] = "rest"
        field_map = {
            column_order[1]: quoted_data,
            column_order[2]: trailing_data
        }

        obs = field_map.get("observation")
        act = field_map.get("action")

        if len(obs) != num_obs:
            print(f"Observation size mismatch: got {len(obs)}, expected {num_obs}")
            continue
        if len(act) != num_actions:
            print(f"Action size mismatch: got {len(act)}, expected {num_actions}")
            continue

        time_list.append(timestamp)
        obs_list.append(obs)
        act_list.append(act)

    return np.array(time_list), np.array(obs_list), np.array(act_list)



def generate_gl_flat_obs_labels(num_obs: int) -> List[str]:
    """Generate observation labels assuming GL flat layout."""
    labels = []
    labels += ["omega_x", "omega_y", "omega_z"]
    labels += ["g_proj_x", "g_proj_y", "g_proj_z"]
    labels += ["cmd_vx", "cmd_vy", "cmd_wz"]

    joint_vel_labels = [f"qvel_{i}" for i in range(21)]
    joint_pos_labels = [f"qpos_{i}" for i in range(21)]
    past_action_labels = [f"a_prev_{i}" for i in range(21)]
    phase_labels = ["sin_phase", "cos_phase"]

    labels += joint_vel_labels + joint_pos_labels + past_action_labels + phase_labels
    return labels[:num_obs]  # truncate if fewer

def plot_observation_and_action(timestamps: np.ndarray, obs: np.ndarray, acts: np.ndarray,
                                 obs_labels: list = None, act_labels: list = None,
                                 indices: dict = None, obs_keywords: list = None):
    if obs_keywords and obs_labels:
        obs_indices = [i for i, label in enumerate(obs_labels) if any(kw in label for kw in obs_keywords)]
    else:
        obs_indices = indices.get("obs", list(range(obs.shape[1]))) if indices else list(range(obs.shape[1]))

    act_indices = indices.get("acts", list(range(acts.shape[1]))) if indices else list(range(acts.shape[1]))

    plt.figure(figsize=(12, 5))
    for i in obs_indices:
        label = obs_labels[i] if obs_labels and i < len(obs_labels) else f"obs[{i}]"
        plt.plot(timestamps, obs[:, i], label=label)
    plt.xlabel("Time [s]")
    plt.ylabel("Observation")
    plt.title("Selected Observations Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 5))
    for i in act_indices:
        label = act_labels[i] if act_labels and i < len(act_labels) else f"act[{i}]"
        plt.plot(timestamps, acts[:, i], label=label)
    plt.xlabel("Time [s]")
    plt.ylabel("Action")
    plt.title("Selected Actions Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    log_dir = "/home/amy/gitrepo/robot_rl/g1_hardware_log/2025-07-21_16-15-32"
    config_files = glob.glob(os.path.join(log_dir, "*.yaml"))
    if not config_files:
        raise FileNotFoundError("No YAML config file found in the log directory")
    config_path = config_files[0]

    fields_path = os.path.join(log_dir, "fields.csv")
    file_path = os.path.join(log_dir, "g1_control.csv")

    timestamps, obs, acts, obs_labels = parse_g1_log(file_path, config_path, fields_path)
    plot_observation_and_action(
        timestamps,
        obs,
        acts,
        obs_labels=obs_labels,
        indices={"acts": [0, 1, 2]},         # Only plot specific action indices
        obs_keywords=["cmd_"]      # Plot angular velocity + joint positions
    )
