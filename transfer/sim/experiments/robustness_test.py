# Copyright 2026 Zachary Olkin. All rights reserved.

from typing import Literal
import argparse
import yaml
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from sim.experiments.velocity_commands import smooth_ramp_running

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
FORCE_VEC = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
DELTA_FORCE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
START_ADDED_MASS = 8.0 #6.0
DELTA_MASS = 1.0 #0. #1.5

METRIC = False

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
                           input_function=smooth_ramp_running, use_pd=config["use_pd"])

    # Run all the simulations
    run_logs = []
    mean_error = []
    std_error = []
    added_masses = []

    NUM_RUNS = 3

    for i in range(NUM_RUNS):
        # Added mass robustness
        robot_instance.add_base_mass(START_ADDED_MASS + i*DELTA_MASS)

        # Create and run simulation
        sim = Simulation(policy, robot_instance, log=True,
                         log_dir=config.get("log_dir", os.path.join(os.getcwd(), "logs")),
                         use_height_sensor=config.get("height_map_scale") is not None, tracking_body_name="torso_link")
        force_partial = partial(force_robustness, force_vec=FORCE_VEC + i*DELTA_FORCE)
        sim.run_headless(total_time=8, force_disturbance=force_partial)

        run_logs.append(sim.get_logging_folder())

        # Make plots and statistics
        create_plots_for_newest()
        stats = compute_stats(0)
        mean_error.append(stats['mean_velocity_error'])
        std_error.append(stats['std_dev_velocity_error'])

        # Log the robustness constants
        robustness_data = {
            'added_mass': START_ADDED_MASS + i*DELTA_MASS,
            'force_start': FORCE_START,
            'force_stop': FORCE_STOP,
            'force_vector': FORCE_VEC + i*DELTA_FORCE,
        }
        added_masses.append(robustness_data['added_mass'])

        with open(os.path.join(sim.get_logging_folder(), 'robustness_data.yaml'), 'w') as f:
            yaml.dump(robustness_data, f)

        robot_instance.reset_robot()

    if METRIC:
        mean_error = np.array(mean_error)
        print("Mean error: ", mean_error)
        std_error = np.array(std_error)

        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle('Mean Error vs Added Mass')

        # x Axis
        axes[0].plot(added_masses, mean_error[:, 0], 'b-', marker='o', label='Mean Error')
        if NUM_RUNS > 1:
            axes[0].fill_between(added_masses, np.maximum(mean_error[:, 0] - std_error[:, 0], 0), mean_error[:, 0] + std_error[:, 0], color='blue', alpha=0.3, label='±1 Std Dev')
        axes[0].grid(True)
        axes[0].set_ylabel('X Velocity Mean Squared Error (m/s)')
        axes[0].legend()

        # y Axis
        axes[1].plot(added_masses, mean_error[:, 1], 'b-', marker='o', label='Mean Error')
        if NUM_RUNS > 1:
            axes[1].fill_between(added_masses, np.maximum(mean_error[:, 1] - std_error[:, 1], 0), mean_error[:, 1] + std_error[:, 1], color='blue', alpha=0.3, label='±1 Std Dev')
        axes[1].grid(True)
        axes[1].set_ylabel('Y Velocity Mean Squared Error (m/s)')
        axes[1].legend()

        # w Axis
        axes[2].plot(added_masses, mean_error[:, 2], 'b-', marker='o', label='Mean Error')
        if NUM_RUNS > 1:
            axes[2].fill_between(added_masses, np.maximum(mean_error[:, 2] - std_error[:, 2], 0), mean_error[:, 2] + std_error[:, 2], color='blue', alpha=0.3, label='±1 Std Dev')
        axes[2].grid(True)
        axes[2].set_ylabel('Angular Velocity Mean Squared Error (m/s)')
        axes[2].legend()

        plt.savefig(f"experiments/plots/metric_robustness_test_{config_file_name}.png")
    else:
        plt.close()  # TODO: Is this necessary?

        # Load back in all the data
        actual_vel_list = []
        commanded_vel_list = []
        time_list = []
        for i in range(NUM_RUNS):
            # Parse the data
            log_dir = os.path.join(os.getcwd(), run_logs[i])

            # Load in the data
            with open(os.path.join(log_dir, "sim_config.yaml")) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
                data = extract_data(os.path.join(log_dir, "sim_log.csv"), config)

                actual_vel_list.append(data["qvel"])
                commanded_vel_list.append(data["commanded_vel"])
                time_list.append(data["time"])

        time_np = np.stack(time_list)
        commanded_vel_np = np.stack(commanded_vel_list)
        actual_vel_np = np.stack(actual_vel_list)

        mean_actual = np.mean(actual_vel_np, axis=0)
        std_actual = np.std(actual_vel_np, axis=0)

        # Make some plots
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        fig.suptitle('Commanded vs Actual Velocities')

        axes[0].plot(time_np[0, :], commanded_vel_np[0, :, 0], 'r--', label='Commanded')
        axes[1].plot(time_np[0, :], commanded_vel_np[0, :, 1], 'r--', label='Commanded')
        axes[2].plot(time_np[0, :], commanded_vel_np[0, :, 2], 'r--', label='Commanded')

        # Plot x velocity
        axes[0].plot(time_np[0], mean_actual[:, 0], 'b-', label=f'Actual_{i}')
        axes[0].fill_between(np.squeeze(time_np[0, :]), mean_actual[:, 0] - std_actual[:, 0],
                             mean_actual[:, 0] + std_actual[:, 0], color='blue', alpha=0.3, label='±1 Std Dev')
        axes[0].set_ylabel('X Velocity (m/s)')
        axes[0].legend()
        axes[0].grid(True)

        # Plot y velocity
        axes[1].plot(time_np[0, :], mean_actual[:, 1], 'b-', label=f'Actual_{i}')
        axes[1].fill_between(np.squeeze(time_np[0, :]), mean_actual[:, 1] - std_actual[:, 1],
                             mean_actual[:, 1] + std_actual[:, 1], color='blue', alpha=0.3, label='±1 Std Dev')
        axes[1].set_ylabel('Y Velocity (m/s)')
        axes[1].legend()
        axes[1].grid(True)

        # Plot angular velocity
        axes[2].plot(time_np[0, :], mean_actual[:, 2], 'b-', label=f'Actual_{i}')
        axes[2].fill_between(np.squeeze(time_np[0, :]), mean_actual[:, 2] - std_actual[:, 2],
                             mean_actual[:, 2] + std_actual[:, 2], color='blue', alpha=0.3, label='±1 Std Dev')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Angular Velocity (rad/s)')
        axes[2].legend()
        axes[2].grid(True)

        plt.savefig(f"experiments/plots/velocity_robustness_test_{config_file_name}.png")
        print(f"saving the figure to 'experiments/plots/velocity_robustness_test_{config_file_name}.png'")

if __name__ == "__main__":
    main()




