import argparse
import os
import sys
import glob
from typing import Literal
import numpy as np

import yaml

from plot_from_sim import create_plots

# Add the project root to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rl_policy import RLPolicy

from robot import Robot
from simulation import Simulation

from experiments.velocity_commands import speed_steps, smooth_ramp_running


def find_latest_run(log_root_path):
    """Find the latest run directory in the given path."""
    run_dirs = glob.glob(os.path.join(log_root_path, "*"))
    if not run_dirs:
        return None

    # Get the latest run directory
    latest_run = max(run_dirs, key=os.path.getmtime)
    run_name = os.path.basename(latest_run)

    return run_name


def main():
    parser = argparse.ArgumentParser(description="Run G1 robot in MuJoCo simulation with RL policy")
    parser.add_argument("--experiment", type=str, default="g1",
                       help="Experiment name matching the folder under logs/rsl_rl/ (e.g., 'g1')")
    parser.add_argument("--load_run", type=str, required=False, default=None,
                       help="Specific run directory to load (e.g., '2025-11-17_16-15-56_running_testA'). If not specified, uses latest run.")
    parser.add_argument("--scene", type=str, default="basic_scene",
                       help="Scene name for MuJoCo simulation")
    parser.add_argument("--robot_name", type=str, default="g1_21j",
                       help="Robot name")
    parser.add_argument("--use_ic", action="store_true", default=False,
                       help="Use the valid initial condition from the policy YAML")
    parser.add_argument("--log", action="store_true", default=False,
                       help="Enable logging")
    parser.add_argument("--log_dir", type=str, default=None,
                       help="Directory for logging")
    args = parser.parse_args()

    # Construct path to logs directory (same structure as play.py)
    # Get the script directory (transfer/sim) and go up two levels to root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(script_dir))

    # Build path: logs/rsl_rl/{experiment}
    log_root_path = os.path.join(root_dir, "logs", "rsl_rl", args.experiment)

    print(f"[INFO] Looking for policies in: {log_root_path}")

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

    # Paths to policy and parameters
    exported_dir = os.path.join(run_dir, "exported")
    policy_path = os.path.join(exported_dir, "policy.pt")
    param_path = os.path.join(exported_dir, "policy_parameters.yaml")

    # Check if files exist
    if not os.path.exists(policy_path):
        print(f"[ERROR] Policy file not found: {policy_path}")
        print(f"[INFO] Make sure to run play.py first to export the policy")
        sys.exit(1)

    if not os.path.exists(param_path):
        print(f"[ERROR] Parameters file not found: {param_path}")
        print(f"[INFO] Make sure to run play.py first to export parameters")
        sys.exit(1)

    print(f"[INFO] Loading policy from: {policy_path}")
    print(f"[INFO] Loading parameters from: {param_path}")

    # Make the RL policy
    policy = RLPolicy(param_path, policy_path)
    policy.load()

    max_acc = policy.get_max_acc()
    if max_acc is not None:
        print(f"[INFO] Ramping velocity command with max_acc={max_acc} m/s^2")
    else:
        print("[INFO] No max_acc set in policy params; passing velocity commands without ramping")

    # Create robot instance
    gains = {"kp_y": 1.5, "kd_y": 0.3, "kp_yaw": 0.8, "kd_yaw": 0.3}
    robot_instance = Robot(
        robot_name=args.robot_name,
        scene_name=args.scene,
        joystick_scaling=np.array([1,1,1]),
        input_function=None,
        use_pd=False,
        gains=gains,
    )

    # Set initial condition from YAML if requested
    if args.use_ic:
        robot_instance.set_initial_condition(policy)

    # Set up log directory
    if args.log_dir is None:
        log_dir = os.path.join(run_dir, "mujoco_logs")
    else:
        log_dir = args.log_dir

    # Create and run simulation
    sim = Simulation(
        policy,
        robot_instance,
        log=args.log,
        log_dir=log_dir,
        use_height_sensor=False,  # Not using height sensor for now
        tracking_body_name="torso_link",
    )
    sim.run(-1)  # Run forever

    # Generate plots after simulation if logging was enabled
    if args.log and sim.new_log_folder:
        print(f"[INFO] Generating plots from: {sim.new_log_folder}")
        create_plots(sim.new_log_folder)


if __name__ == "__main__":
    main()
