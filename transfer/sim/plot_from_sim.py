# Copyright 2026 Zachary Olkin. All rights reserved.

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sim.log_utils import extract_data, find_most_recent_timestamped_folder

# Environment experiment names mapping (same as in g1_runner.py)
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


def find_latest_run(log_root_path):
    """Find the latest run directory in the given path."""
    run_dirs = glob.glob(os.path.join(log_root_path, "*"))
    if not run_dirs:
        return None

    # Get the latest run directory
    latest_run = max(run_dirs, key=os.path.getmtime)
    run_name = os.path.basename(latest_run)

    return run_name


def find_most_recent_mujoco_log(mujoco_logs_dir):
    """Find the most recent timestamped folder in the MuJoCo logs directory."""
    return find_most_recent_timestamped_folder(mujoco_logs_dir)


# Make plots
def plot_joints_and_actions(data, save_dir, joint_names=None):
    fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(18, 16))

    FLOATING_BASE = 7

    for i in range(7):
        for j in range(3):
            joint_idx = i + 7 * j
            joint_label = joint_names[joint_idx] if joint_names and joint_idx < len(joint_names) else f"joint {joint_idx}"
            
            axes[i, j].plot(data["time"], data["qpos"][:, joint_idx + FLOATING_BASE], label="qpos")
            axes[i, j].plot(data["time"], data["action"][:, joint_idx], label="action")
            axes[i, j].set_xlabel("time", fontsize=10)
            axes[i, j].set_ylabel(f"{joint_label} (rad)", fontsize=10)
            axes[i, j].grid()
            axes[i, j].legend(fontsize=8)
            axes[i, j].tick_params(axis='both', which='major', labelsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "joints_and_actions.png"), dpi=150, bbox_inches='tight')


def plot_torques(data, save_dir, joint_names=None, torque_limits=None):
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(16, 14))

    for i in range(6):
        for j in range(2):
            joint_idx = i + 6 * j
            joint_label = joint_names[joint_idx] if joint_names and joint_idx < len(joint_names) else f"joint {joint_idx}"
            
            # Plot actual torque
            axes[i, j].plot(data["time"], data["torque"][:, joint_idx], label="Actual")
            
            # Plot torque limits as dashed lines if available
            if torque_limits and joint_idx < len(torque_limits):
                torque_min, torque_max = torque_limits[joint_idx]
                axes[i, j].axhline(y=torque_max, color='r', linestyle='--', alpha=0.7, label=f'Max ({torque_max:.1f})')
                axes[i, j].axhline(y=torque_min, color='r', linestyle='--', alpha=0.7, label=f'Min ({torque_min:.1f})')
            
            axes[i, j].set_xlabel("time", fontsize=10)
            axes[i, j].set_ylabel(f"{joint_label} torque (Nm)", fontsize=10)
            axes[i, j].grid()
            axes[i, j].legend(fontsize=8)
            axes[i, j].tick_params(axis='both', which='major', labelsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "torques.png"), dpi=150, bbox_inches='tight')


def plot_vels(data, save_dir, joint_names=None):
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 10))

    FLOATING_BASE = 6

    for i in range(6):
        for j in range(2):
            joint_idx = i + 6 * j
            joint_label = joint_names[joint_idx] if joint_names and joint_idx < len(joint_names) else f"joint {joint_idx}"
            
            axes[i, j].plot(data["time"], data["qvel"][:, joint_idx + FLOATING_BASE])
            axes[i, j].set_xlabel("time")
            axes[i, j].set_ylabel(f"{joint_label} vel (rad/s)")
            axes[i, j].grid()

    plt.savefig(os.path.join(save_dir, "vels.png"))


def plot_base(data, save_dir):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))

    for i in range(3):
        for j in range(2):
            if j == 0:
                axes[i, j].plot(data["time"], data["qpos"][:, i])
                axes[i, j].set_xlabel("time")
                axes[i, j].set_ylabel(f"qpos {i} (m)")
                axes[i, j].grid()
            else:
                axes[i, j].plot(data["time"], data["qvel"][:, i])
                axes[i, j].set_xlabel("time")
                axes[i, j].set_ylabel(f"qvel {i} (m/s)")
                axes[i, j].grid()

    plt.savefig(os.path.join(save_dir, "floating_base.png"))


def plot_ankles(data):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))

    axes[0, 0].plot(data["time"], data["left_ankle_pos"][:, 0])
    axes[0, 1].plot(data["time"], data["left_ankle_pos"][:, 1])
    axes[0, 2].plot(data["time"], data["left_ankle_pos"][:, 2])
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("left_ankle_pos x (m)")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("left_ankle_pos y (m)")
    axes[0, 2].set_xlabel("time")
    axes[0, 2].set_ylabel("left_ankle_pos z (m)")

    axes[1, 0].plot(data["time"], data["right_ankle_pos"][:, 0])
    axes[1, 1].plot(data["time"], data["right_ankle_pos"][:, 1])
    axes[1, 2].plot(data["time"], data["right_ankle_pos"][:, 2])
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel("right_ankle_pos x (m)")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel("right_ankle_pos y (m)")
    axes[1, 2].set_xlabel("time")
    axes[1, 2].set_ylabel("right_ankle_pos z (m)")


def moving_average(data, window_size):
    """Compute moving average of 1D array."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='valid')


def plot_velocity_comparison(data, save_dir):
    """Plot comparison between commanded and actual velocities."""
    time = data["time"]
    qvel = data["qvel"]
    commanded_vel = data["commanded_vel"]

    # Extract base velocities (first 3 elements of qvel)
    base_vel = qvel[:, :3]

    # Moving average parameters
    window_size = 30
    # Adjusted time array for moving average (shorter due to 'valid' mode convolution)
    time_ma = time[window_size - 1:]

    # Create figure with 3 subplots for x, y, and angular velocities
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("Commanded vs Actual Velocities")

    # Plot x velocity (round commanded to nearest 0.2)
    commanded_x_vel_rounded = np.round(commanded_vel[:, 0] / 0.2) * 0.2
    axes[0].plot(time, commanded_x_vel_rounded, "r--", label="Commanded")
    axes[0].plot(time, base_vel[:, 0], "b-", alpha=0.3, label="Actual")
    axes[0].plot(time_ma, moving_average(base_vel[:, 0], window_size), "b-", label="Actual (avg)")
    axes[0].set_ylabel("X Velocity (m/s)")
    axes[0].legend()
    axes[0].grid(True)

    # Plot y velocity
    axes[1].plot(time, commanded_vel[:, 1], "r--", label="Commanded")
    axes[1].plot(time, base_vel[:, 1], "b-", alpha=0.3, label="Actual")
    axes[1].plot(time_ma, moving_average(base_vel[:, 1], window_size), "b-", label="Actual (avg)")
    axes[1].set_ylabel("Y Velocity (m/s)")
    axes[1].legend()
    axes[1].grid(True)

    # Plot angular velocity
    axes[2].plot(time, commanded_vel[:, 2], "r--", label="Commanded")
    axes[2].plot(time, qvel[:, 5], "b-", alpha=0.3, label="Actual")
    axes[2].plot(time_ma, moving_average(qvel[:, 5], window_size), "b-", label="Actual (avg)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Angular Velocity (rad/s)")
    axes[2].legend()
    axes[2].grid(True)

    plt.savefig(os.path.join(save_dir, "velocity_comparison.png"))


def plot_position_comparison(data, save_dir):
    """Plot comparison between desired and actual positions."""
    time = data["time"]
    qpos = data["qpos"]
    commanded_vel = data["commanded_vel"]

    # Extract base position (first 3 elements of qpos)
    actual_pos = qpos[:, :3]
    # need to extract the initial yaw
    quat = qpos[:, 4:8]
    yaw = 2 * np.arctan2(quat[:, 2], quat[:, 3])
    actual_pos[:, 2] = yaw
    actual_yaw = yaw

    # Calculate desired position by integrating commanded velocity
    dt = time[1] - time[0]  # Assuming constant time step
    desired_pos = np.zeros_like(actual_pos)
    desired_pos[0] = actual_pos[0]  # Start from actual position

    for i in range(1, len(time)):
        desired_pos[i] = desired_pos[i - 1] + commanded_vel[i - 1] * dt

    # Create figure with 3 subplots for x, y, and angular positions
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle("Desired vs Actual Positions")

    # Plot x position
    axes[0].plot(time, desired_pos[:, 0], "r--", label="Desired")
    axes[0].plot(time, actual_pos[:, 0], "b-", label="Actual")
    axes[0].set_ylabel("X Position (m)")
    axes[0].legend()
    axes[0].grid(True)

    # Plot y position
    axes[1].plot(time, desired_pos[:, 1], "r--", label="Desired")
    axes[1].plot(time, actual_pos[:, 1], "b-", label="Actual")
    axes[1].set_ylabel("Y Position (m)")
    axes[1].legend()
    axes[1].grid(True)

    # Plot angular position
    axes[2].plot(time, desired_pos[:, 2], "r--", label="Desired")
    axes[2].plot(time, actual_yaw, "b-", label="Actual")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Angular Position (rad)")
    axes[2].legend()
    axes[2].grid(True)

    plt.savefig(os.path.join(save_dir, "position_comparison.png"))


def create_plots(log_dir):
    """Create plots from the specified log directory.

    Args:
        log_dir: Path to the directory containing sim_config.yaml and sim_log.csv
    """
    print(f"Loading data from {log_dir}.")

    # Parse the config file
    config_path = os.path.join(log_dir, "sim_config.yaml")
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load data
    data_path = os.path.join(log_dir, "sim_log.csv")
    if not os.path.exists(data_path):
        print(f"[ERROR] Data file not found: {data_path}")
        sys.exit(1)

    data = extract_data(data_path, config)

    print("============== Data generated using " + config["simulator"] + " ===============")

    print(f"time shape: {data['time'].shape}")
    print(f"qpos shape: {data['qpos'].shape}")
    print(f"qvel shape: {data['qvel'].shape}")
    print(f"torque shape: {data['torque'].shape}")
    print(f"action shape: {data['action'].shape}")
    print(f"left_ankle_pos shape: {data['left_ankle_pos'].shape}")
    print(f"right_ankle_pos shape: {data['right_ankle_pos'].shape}")
    print(f"commanded_vel shape: {data['commanded_vel'].shape}")

    # Get joint names and torque limits from config if available
    joint_names = config.get("joint_names", None)
    torque_limits = config.get("torque_limits", None)
    if joint_names:
        print(f"Using joint names: {joint_names}")
    else:
        print("No joint names found in config, using default labels")

    if torque_limits:
        print(f"Using torque limits: {torque_limits}")
    else:
        print("No torque limits found in config")

    # Create plots directory
    plots_dir = os.path.join(log_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to {plots_dir}")

    # Make plots
    plot_joints_and_actions(data, plots_dir, joint_names)
    plot_torques(data, plots_dir, joint_names, torque_limits)
    plot_base(data, plots_dir)
    plot_velocity_comparison(data, plots_dir)
    plot_position_comparison(data, plots_dir)

    print(f"[INFO] Plots saved to {plots_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create plots from MuJoCo simulation logs")
    parser.add_argument("--env_type", type=str, required=True,
                       choices=list(EXPERIMENT_NAMES.keys()),
                       help="Type of environment (e.g., walking_clf, running_clf)")
    parser.add_argument("--load_run", type=str, required=False, default=None,
                       help="Specific run directory to load. If not specified, uses latest run.")
    parser.add_argument("--log_session", type=str, required=False, default=None,
                       help="Specific log session (timestamp folder) to plot. If not specified, uses latest.")
    args = parser.parse_args()

    # Construct path to logs directory (same structure as g1_runner.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))

    # Build path: logs/g1_policies/{experiment_name}/{env_type}
    experiment_name = EXPERIMENT_NAMES[args.env_type]
    base_log_path = os.path.join(root_dir, "logs", "g1_policies", experiment_name)
    log_root_path = os.path.join(base_log_path, args.env_type)

    print(f"[INFO] Looking for logs in: {log_root_path}")

    # Find the run directory
    if args.load_run:
        run_name = args.load_run
        run_dir = os.path.join(log_root_path, run_name)
        if not os.path.exists(run_dir):
            print(f"[ERROR] Specified run directory not found: {run_dir}")
            sys.exit(1)
    else:
        print("[INFO] No run specified, finding latest run...")
        run_name = find_latest_run(log_root_path)
        if not run_name:
            print(f"[ERROR] No runs found in {log_root_path}")
            sys.exit(1)
        run_dir = os.path.join(log_root_path, run_name)

    print(f"[INFO] Using run: {run_name}")

    # Find the mujoco_logs directory
    mujoco_logs_dir = os.path.join(run_dir, "mujoco_logs")
    if not os.path.exists(mujoco_logs_dir):
        print(f"[ERROR] MuJoCo logs directory not found: {mujoco_logs_dir}")
        print(f"[INFO] Make sure to run g1_runner.py with --log first")
        sys.exit(1)

    # Find the specific log session
    if args.log_session:
        log_dir = os.path.join(mujoco_logs_dir, args.log_session)
        if not os.path.exists(log_dir):
            print(f"[ERROR] Specified log session not found: {log_dir}")
            sys.exit(1)
    else:
        print("[INFO] No log session specified, finding latest...")
        log_dir = find_most_recent_mujoco_log(mujoco_logs_dir)
        if not log_dir:
            print(f"[ERROR] No log sessions found in {mujoco_logs_dir}")
            sys.exit(1)

    print(f"[INFO] Using log session: {os.path.basename(log_dir)}")

    # Create the plots
    create_plots(log_dir)


if __name__ == "__main__":
    main()
