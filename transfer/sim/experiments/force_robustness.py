# Copyright 2026 Zachary Olkin. All rights reserved.

from typing import Literal
import argparse
import yaml
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from transfer.sim.simulation import Simulation
from transfer.sim.robot import Robot
from transfer.sim.rl_policy_wrapper import RLPolicy
from transfer.sim.plot_from_sim import create_plots_for_newest

from performance_statistics import compute_stats
from velocity_commands import speed_steps, smooth_ramp

from transfer.sim.log_utils import find_most_recent_timestamped_folder, extract_data

FORCE_START = 3.0
FORCE_STOP = 3.125
START_ADDED_MASS = 0. #6.0
DELTA_MASS = 0. #1.5

def force_robustness(sim_time, force_vec):
    """Provide a force to be applied to the base based on the time."""
    if sim_time > 3 and sim_time < 3.125:   #1/8th second force
        return force_vec
    else:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--simulator", type=str, help="Choice of simulator to run (isaac_sim or mujoco)")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Parse the config file with default values
    required_fields = {
        "checkpoint_path": str,
        "dt": float,
        "num_obs": int,
        "num_action": int,
        "period": int,
        "robot_name": str,
        "action_scale": float,
        "default_angles": list,
        "qvel_scale": float,
        "ang_vel_scale": float,
        "command_scale": float,
        "policy_type": Literal["mlp", "cnn"],
    }

    # Check for required fields
    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ValueError(f"Missing required fields in config file: {', '.join(missing_fields)}")

    config_file_name = os.path.splitext(os.path.basename(args.config_file))[0]

    # Make the RL policy
    policy = RLPolicy(
        dt=config["dt"],
        checkpoint_path=config["checkpoint_path"],
        num_obs=config["num_obs"],
        num_action=config["num_action"],
        period=config["period"],
        cmd_scale=config["command_scale"],
        action_scale=config["action_scale"],
        default_angles=config["default_angles"],
        qvel_scale=config["qvel_scale"],
        ang_vel_scale=config["ang_vel_scale"],
        height_map_scale=config.get("height_map_scale", None),
        policy_type=config["policy_type"]
    )

    # Create robot instance
    robot_instance = Robot(robot_name=config["robot_name"], scene_name=config.get("scene", "basic_scene"),
                           input_function=smooth_ramp)

    # Run all the simulations
    run_logs = []

    NUM_FORCES = 24
    NUM_ANGLES = 24

    successes = np.empty((NUM_FORCES, NUM_ANGLES))
    force_mags = np.linspace(100, 300, NUM_FORCES)
    angles = np.linspace(0, 2*np.pi, NUM_ANGLES)

    for i in range(NUM_FORCES):
        for j in range(NUM_ANGLES):
            # Create and run simulation
            sim = Simulation(policy, robot_instance, log=True,
                             log_dir=config.get("log_dir", os.path.join(os.getcwd(), "logs")),
                             use_height_sensor=config.get("height_map_scale") is not None, tracking_body_name="torso_link")

            # Compute the force
            force_vec = np.array([force_mags[i]*np.cos(angles[j]), force_mags[i]*np.sin(angles[j]), 0, 0, 0, 0])

            force_partial = partial(force_robustness, force_vec=force_vec)
            successes[i, j] = int(sim.run_headless(total_time=12, force_disturbance=force_partial))

            run_logs.append(sim.get_logging_folder())

            # Make plots and statistics
            stats = compute_stats(0)

            # Log the robustness constants
            robustness_data = {
                'force_start': FORCE_START,
                'force_stop': FORCE_STOP,
                'force_mag': force_mags[i],
                'force_angle': angles[j],
            }

            with open(os.path.join(sim.get_logging_folder(), 'robustness_data.yaml'), 'w') as f:
                yaml.dump(robustness_data, f)

            robot_instance.reset_robot()

    mean_success = np.mean(successes, axis=1)
    std_success = np.std(successes, axis=1)

    fig, axes = plt.subplots(1, 1, figsize=(10, 12))
    fig.suptitle('Mean Success vs Force Magnitude')

    # x Axis
    axes.plot(force_mags, mean_success, 'b-', marker='o', label='Success Rate')
    # TODO: Make actual error bars
    axes.fill_between(force_mags, np.maximum(mean_success - std_success, 0), np.minimum(mean_success + std_success, 1), color='blue', alpha=0.3, label='±1 Std Dev')
    axes.grid(True)
    axes.set_ylabel('Success Rate')
    axes.set_xlabel('Force Magnitude (N)')
    axes.legend()


    plt.savefig(f"experiments/plots/success_rate_force_test_{config_file_name}.png")


if __name__ == "__main__":
    main()




