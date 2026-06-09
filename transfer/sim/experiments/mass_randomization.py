# Copyright 2026 Zachary Olkin. All rights reserved.

from typing import Literal
import argparse
import yaml
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sim.simulation import Simulation
from sim.robot import Robot
from sim.rl_policy_wrapper import RLPolicy
from sim.plot_from_sim import create_plots_for_newest
from sim.log_utils import find_most_recent_timestamped_folder, extract_data

from performance_statistics import compute_stats
from velocity_commands import speed_steps, smooth_ramp, ramped_speed_steps, smooth_ramp_running


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--simulator", type=str, help="Choice of simulator to run (isaac_sim or mujoco)")
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)

    config_file_name = os.path.splitext(os.path.basename(args.config_file))[0]

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

    seed = 42
    rng = np.random.default_rng(seed)

    # Create robot instance
    robot_instance = Robot(robot_name=config["robot_name"], scene_name=config.get("scene", "basic_scene"),
                           input_function=smooth_ramp_running, use_pd=config["use_pd"], rng=rng)

    # run_logs = []
    base_log_dir = os.path.join("experiments", f"mass_randomization_{config_file_name}")
    os.makedirs(base_log_dir, exist_ok=True)


    NUM_RUNS = 50

    for i in range(NUM_RUNS):
        # Adjust the torso mass position
        max_movement = np.array([0.05,0.05,0.01])
        pos_movement = robot_instance.randomize_torso_mass_pos(max_movement)

        # Create and run simulation
        sim = Simulation(policy, robot_instance, log=True,
                         log_dir=base_log_dir,
                         use_height_sensor=config.get("height_map_scale") is not None, tracking_body_name="torso_link")
        sim.run_headless(total_time=10, force_disturbance=None)

        # run_logs.append(sim.get_logging_folder())

        # Make plots and statistics
        # create_plots_for_newest()
        # compute_stats(0)

        # Log the robustness constants
        robustness_data = {
            'torso_mass_pos': pos_movement.tolist(),
        }

        with open(os.path.join(sim.get_logging_folder(), 'robustness_data.yaml'), 'w') as f:
            yaml.dump(robustness_data, f)

        robot_instance.reset_robot()
        robot_instance.reset_torso_mass_pos()

    # Load back in all the data
    # actual_vel_list = []
    # commanded_vel_list = []
    # time_list = []
    # for i in range(NUM_RUNS):
    #     # Parse the data
    #     log_dir = os.path.join(os.getcwd(), run_logs[i])

    #     # Load in the data
    #     with open(os.path.join(log_dir, "sim_config.yaml")) as f:
    #         config = yaml.load(f, Loader=yaml.FullLoader)
    #         data = extract_data(os.path.join(log_dir, "sim_log.csv"), config)

    #         actual_vel_list.append(data["qvel"])
    #         commanded_vel_list.append(data["commanded_vel"])
    #         time_list.append(data["time"])

    # time_np = np.stack(time_list)
    # commanded_vel_np = np.stack(commanded_vel_list)
    # actual_vel_np = np.stack(actual_vel_list)

    # mean_actual = np.mean(actual_vel_np, axis=0)
    # std_actual = np.std(actual_vel_np, axis=0)

    # # Make some plots
    # plt.rcParams.update({'font.size': 18})
    # plt.rcParams.update({
    #     "text.usetex": True,  # Use LaTeX for all text
    #     "font.family": "serif",  # Use serif font (default LaTeX style)
    #     "font.serif": ["Computer Modern Roman"],  # LaTeX default font
    # })
    # fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    # # fig.suptitle('Commanded vs Actual Velocities')

    # axes.plot(time_np[0, :], commanded_vel_np[0, :, 0], 'k--', linewidth="2", label='Commanded')
    # # axes[1].plot(time_np[0, :], commanded_vel_np[0, :, 1], 'r--', label='Commanded')
    # # axes[2].plot(time_np[0, :], commanded_vel_np[0, :, 2], 'r--', label='Commanded')

    # # Plot x velocity
    # axes.plot(time_np[0], mean_actual[:, 0], 'tab:blue', label=f'Actual_{i}')
    # axes.fill_between(np.squeeze(time_np[0, :]), mean_actual[:, 0] - std_actual[:, 0], mean_actual[:, 0] + std_actual[:, 0], color='blue', alpha=0.3, label='±1 Std Dev')
    # axes.set_ylabel(r'$v_x$ (m/s)')
    # axes.set_xlabel('Time (s)')
    # axes.legend()
    # axes.grid(True)
    # axes.set_ylim(-0.6, 1.05)

    # # Plot y velocity
    # axes[1].plot(time_np[0, :], mean_actual[:, 1], 'b-', label=f'Actual_{i}')
    # axes[1].fill_between(np.squeeze(time_np[0, :]), mean_actual[:, 1] - std_actual[:, 1], mean_actual[:, 1] + std_actual[:, 1], color='blue', alpha=0.3, label='±1 Std Dev')
    # axes[1].set_ylabel('Y Velocity (m/s)')
    # axes[1].legend()
    # axes[1].grid(True)
    #
    # # Plot angular velocity
    # axes[2].plot(time_np[0, :], mean_actual[:, 2], 'b-', label=f'Actual_{i}')
    # axes[2].fill_between(np.squeeze(time_np[0, :]), mean_actual[:, 2] - std_actual[:, 2], mean_actual[:, 2] + std_actual[:, 2], color='blue', alpha=0.3, label='±1 Std Dev')
    # axes[2].set_xlabel('Time (s)')
    # axes[2].set_ylabel('Angular Velocity (rad/s)')
    # axes[2].legend()
    # axes[2].grid(True)
    # os.makedirs("experiments/plots", exist_ok=True)
    # plt.savefig(f"experiments/plots/mass_randomization_{config_file_name}.png")
    # plt.savefig(f"experiments/plots/mass_randomization_{config_file_name}.svg", transparent=True)

if __name__ == "__main__":
    main()
    # plt.show()




