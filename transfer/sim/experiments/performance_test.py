# Copyright 2026 Zachary Olkin. All rights reserved.

from typing import Literal
import argparse
import yaml
import os
from pathlib import Path

# Import from the transfer.sim package
from sim.simulation import Simulation
from sim.robot import Robot
from sim.rl_policy_wrapper import RLPolicy
from sim.plot_from_sim import create_plots_for_newest

from sim.simulation import Simulation
from sim.robot import Robot
from sim.rl_policy_wrapper import RLPolicy
from sim.plot_from_sim import create_plots_for_newest

from performance_statistics import compute_stats
from velocity_commands import speed_steps, smooth_ramp, speed_steps_running, smooth_ramp_running

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

    policy_base_dir = Path.cwd().parent.parent / "logs"

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
                           input_function=speed_steps_running, use_pd=config["use_pd"])

    # Create and run simulation
    sim = Simulation(policy, robot_instance, log=config.get("log", False),
                     log_dir=config.get("log_dir", os.path.join(os.getcwd(), "logs")),
                     use_height_sensor=config.get("height_map_scale") is not None, tracking_body_name="torso_link")
    sim.run_headless(total_time=21)
    # sim.run(total_time=24)
    # Make plots and statistics
    create_plots_for_newest()
    compute_stats(0)


if __name__ == "__main__":
    main()




