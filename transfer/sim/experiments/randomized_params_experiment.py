# Copyright 2026 Zachary Olkin. All rights reserved.

"""Randomized physical parameter experiment.

Runs M mujoco simulations for each of N policies, with pre-computed randomized
physical parameters (torso mass scaling, torso COM offset, joint damping kd scaling).
The same parameter sets are used across all policies for fair comparison.

Each randomization and force push disturbance is opt-in via CLI flags.
With no flags, all parameters use nominal values (no randomization, no force).

After all simulations, generates velocity tracking and per-joint torque plots.

Usage:
    python experiments/randomized_params_experiment.py \
        --env_type running_clf_sym \
        --run_dirs 2026-02-15_11-25-13_T1_running_... 2026-02-14_09-51-15_T0_running_... \
        --names "Policy A" "Policy B" \
        --num_runs 50 \
        --experiment_name ablation \
        --seed 42 \
        --total_time 10 \
        --randomize_mass --randomize_com --randomize_kd --force_push
"""

import argparse
import os
import sys
from collections import OrderedDict
from datetime import datetime
from functools import partial

import numpy as np
import yaml

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sim.rl_policy import RLPolicy
from sim.robot import Robot
from sim.simulation import Simulation

from velocity_commands import (
    step_to_max,
    smooth_ramp_running,
    smooth_ramp,
    speed_steps_running,
    speed_steps,
    ramped_speed_steps,
)
from plot_randomized_experiment import (
    load_experiment_data,
    plot_velocity_comparison,
    plot_torque_comparison,
    plot_force_success_histogram,
    plot_radar_comparison,
    print_combined_error_table,
    print_stats_table,
    print_torque_stats_table,
    print_success_table,
    _filter_successful_runs,
)

VELOCITY_COMMANDS = {
    "step_to_max": step_to_max,
    "smooth_ramp_running": smooth_ramp_running,
    "smooth_ramp": smooth_ramp,
    "speed_steps_running": speed_steps_running,
    "speed_steps": speed_steps,
    "ramped_speed_steps": ramped_speed_steps,
}

# Environment experiment names mapping (same as in g1_runner.py / train_policy.py)
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

# --- Randomization ranges ---
TORSO_MASS_SCALE_RANGE = (0.9, 1.1)
COM_OFFSET_XY_RANGE = 0.05  # meters
COM_OFFSET_Z_RANGE = 0.01   # meters
KD_SCALE_RANGE = (0.95, 1.05)

# --- Force push ranges ---
FORCE_MAG_MIN = 50.0   # Newtons
FORCE_MAG_MAX = 300.0  # Newtons
FORCE_MAG_STEP = 50.0  # Newtons
FORCE_START_TIME = 4.0  # seconds
FORCE_DURATION = 0.125  # seconds


def _force_disturbance(sim_time: float, force_vec: np.ndarray) -> np.ndarray:
    """Apply a brief force impulse at a fixed time window.

    Args:
        sim_time: Current simulation time in seconds.
        force_vec: 6D force/torque vector to apply during the window.

    Returns:
        6D force/torque vector (zeros outside the window).
    """
    if FORCE_START_TIME <= sim_time < FORCE_START_TIME + FORCE_DURATION:
        return force_vec
    return np.zeros(6)


def pre_compute_params(
    num_runs: int,
    seed: int,
    randomize_mass: bool = False,
    randomize_com: bool = False,
    randomize_kd: bool = False,
    force_push: bool = False,
) -> list[dict]:
    """Pre-compute M sets of randomized physical parameters.

    Args:
        num_runs: Number of parameter sets to generate.
        seed: RNG seed for reproducibility.
        randomize_mass: Whether to randomize torso mass scaling.
        randomize_com: Whether to randomize torso COM offset.
        randomize_kd: Whether to randomize kd gain scaling.
        force_push: Whether to add a force push disturbance.

    Returns:
        List of dicts, each with keys: torso_mass_scale, com_offset, kd_scale,
        and optionally force_mag, force_angle when force_push is enabled.
    """
    rng = np.random.default_rng(seed)

    params = []
    for _ in range(num_runs):
        if randomize_mass:
            torso_mass_scale = rng.uniform(*TORSO_MASS_SCALE_RANGE)
        else:
            torso_mass_scale = 1.0

        if randomize_com:
            com_offset = np.array([
                rng.uniform(-COM_OFFSET_XY_RANGE, COM_OFFSET_XY_RANGE),
                rng.uniform(-COM_OFFSET_XY_RANGE, COM_OFFSET_XY_RANGE),
                rng.uniform(-COM_OFFSET_Z_RANGE, COM_OFFSET_Z_RANGE),
            ])
        else:
            com_offset = np.zeros(3)

        if randomize_kd:
            kd_scale = rng.uniform(*KD_SCALE_RANGE)
        else:
            kd_scale = 1.0

        p = {
            "torso_mass_scale": torso_mass_scale,
            "com_offset": com_offset,
            "kd_scale": kd_scale,
        }

        params.append(p)

    # Assign deterministic force bins if force push is enabled
    if force_push:
        force_bins = np.arange(FORCE_MAG_MIN, FORCE_MAG_MAX + FORCE_MAG_STEP / 2, FORCE_MAG_STEP)
        num_bins = len(force_bins)
        if num_runs % num_bins != 0:
            raise ValueError(
                f"num_runs ({num_runs}) must be divisible by the number of force "
                f"bins ({num_bins}). Force bins: {force_bins.tolist()}"
            )
        runs_per_bin = num_runs // num_bins
        for bin_idx, force_mag in enumerate(force_bins):
            angles = np.linspace(0, 2 * np.pi, runs_per_bin, endpoint=False)
            for j, angle in enumerate(angles):
                param_idx = bin_idx * runs_per_bin + j
                params[param_idx]["force_mag"] = float(force_mag)
                params[param_idx]["force_angle"] = float(angle)

    return params


def run_experiment_for_policy(
    run_dir: str,
    experiment_folder: str,
    param_sets: list[dict],
    total_time: float,
    velocity_command_fn=None,
) -> None:
    """Run all M simulations for a single policy.

    Args:
        run_dir: Path to the RL training run directory.
        experiment_folder: Path to write experiment logs.
        param_sets: Pre-computed randomization parameters.
        total_time: Simulation duration in seconds.
        velocity_command_fn: Velocity command function to use. Defaults to smooth_ramp_running.
    """
    if velocity_command_fn is None:
        velocity_command_fn = smooth_ramp_running
    # Load policy
    exported_dir = os.path.join(run_dir, "exported")
    policy_path = os.path.join(exported_dir, "policy.pt")
    param_path = os.path.join(exported_dir, "policy_parameters.yaml")

    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy not found: {policy_path}")
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Policy params not found: {param_path}")

    policy = RLPolicy(param_path, policy_path)
    policy.load()

    os.makedirs(experiment_folder, exist_ok=True)

    num_runs = len(param_sets)
    for i, params in enumerate(param_sets):
        print(f"\n{'='*60}")
        print(f"Run {i + 1}/{num_runs}")
        print(f"  torso_mass_scale: {params['torso_mass_scale']:.4f}")
        print(f"  com_offset: {params['com_offset']}")
        print(f"  kd_scale: {params['kd_scale']:.4f}")
        if "force_mag" in params:
            print(f"  force_mag: {params['force_mag']:.1f} N")
            print(f"  force_angle: {params['force_angle']:.4f} rad")
        print(f"{'='*60}")

        # Create fresh robot for each run
        robot = Robot(
            robot_name="g1_21j",
            scene_name="basic_scene",
            joystick_scaling=np.array([1, 1, 1]),
            input_function=velocity_command_fn,
            use_pd=False,
        )

        # Apply physical parameter randomization
        robot.scale_torso_mass(params["torso_mass_scale"])
        robot.set_torso_mass_pos_offset(params["com_offset"])

        # Set PD gains from policy, then scale kd
        # robot.set_pd_gains_from_policy(policy)
        robot.scale_kd_gains(params["kd_scale"])

        # Build force disturbance callable if force push is enabled
        force_disturbance = None
        if "force_mag" in params:
            force_vec = np.array([
                params["force_mag"] * np.cos(params["force_angle"]),
                params["force_mag"] * np.sin(params["force_angle"]),
                0.0, 0.0, 0.0, 0.0,
            ])
            force_disturbance = partial(_force_disturbance, force_vec=force_vec)

        # Create and run simulation (skip_gain_setup since we already set gains)
        sim = Simulation(
            policy, robot,
            log=True,
            log_dir=experiment_folder,
            use_height_sensor=False,
            tracking_body_name="torso_link",
        )
        success = sim.run_headless(
            total_time=total_time,
            force_disturbance=force_disturbance,
            # skip_gain_setup=True,
        )

        # Save robustness data to the run's log folder
        robustness_data = {
            "torso_mass_scale": float(params["torso_mass_scale"]),
            "com_offset": params["com_offset"].tolist(),
            "kd_scale": float(params["kd_scale"]),
            "success": success,
        }
        if "force_mag" in params:
            robustness_data["force_mag"] = float(params["force_mag"])
            robustness_data["force_angle"] = float(params["force_angle"])

        robustness_path = os.path.join(sim.get_logging_folder(), "robustness_data.yaml")
        with open(robustness_path, "w") as f:
            yaml.dump(robustness_data, f)

        print(f"Run {i + 1} {'succeeded' if success else 'FAILED'}")


def main():
    """Run the randomized parameter experiment."""
    parser = argparse.ArgumentParser(
        description="Run randomized physical parameter experiment across multiple policies."
    )
    parser.add_argument(
        "--env_type", type=str, required=True,
        choices=list(EXPERIMENT_NAMES.keys()),
        help="Environment type (e.g., running_clf_sym, walking_clf_sym).",
    )
    parser.add_argument(
        "--run_dirs", type=str, nargs="+", required=True,
        help="Run directory names (e.g., '2026-02-15_11-25-13_T1_running_...'). "
             "Resolved under logs/g1_policies/{experiment_name}/{env_type}/. "
             "Absolute paths are also accepted.",
    )
    parser.add_argument(
        "--names", type=str, nargs="+", required=True,
        help="Display names for each policy (used in plots).",
    )
    parser.add_argument(
        "--num_runs", type=int, default=50,
        help="Number of randomized simulation runs per policy (default: 50).",
    )
    parser.add_argument(
        "--experiment_name", type=str, default="randomized_params",
        help="Name for the experiment (used in folder naming).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for parameter generation (default: 42).",
    )
    parser.add_argument(
        "--total_time", type=float, default=10.0,
        help="Total simulation time in seconds (default: 10).",
    )
    parser.add_argument(
        "--randomize_mass", action="store_true", default=False,
        help="Enable torso mass randomization.",
    )
    parser.add_argument(
        "--randomize_com", action="store_true", default=False,
        help="Enable torso COM offset randomization.",
    )
    parser.add_argument(
        "--randomize_kd", action="store_true", default=False,
        help="Enable kd gain scaling randomization.",
    )
    parser.add_argument(
        "--force_push", action="store_true", default=False,
        help="Enable force push disturbance.",
    )
    parser.add_argument(
        "--velocity_command", type=str, default="smooth_ramp_running",
        choices=list(VELOCITY_COMMANDS.keys()),
        help="Velocity command profile to use (default: smooth_ramp_running).",
    )
    args = parser.parse_args()

    if len(args.run_dirs) != len(args.names):
        raise ValueError("Number of --run_dirs must match number of --names.")

    # Resolve run dirs the same way g1_runner.py does
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sim_dir = os.path.dirname(script_dir)  # transfer/sim
    root_dir = os.path.dirname(os.path.dirname(sim_dir))  # project root
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

    # Pre-compute parameter sets (shared across all policies)
    enabled = []
    if args.randomize_mass:
        enabled.append("mass")
    if args.randomize_com:
        enabled.append("com")
    if args.randomize_kd:
        enabled.append("kd")
    if args.force_push:
        enabled.append("force_push")
    print(f"Pre-computing {args.num_runs} parameter sets with seed={args.seed}...")
    print(f"  Enabled randomizations: {enabled if enabled else ['none (nominal)']}")
    if args.force_push:
        force_bins = np.arange(FORCE_MAG_MIN, FORCE_MAG_MAX + FORCE_MAG_STEP / 2, FORCE_MAG_STEP)
        num_bins = len(force_bins)
        runs_per_bin = args.num_runs // num_bins if args.num_runs % num_bins == 0 else "?"
        print(f"  Force bins (N): {force_bins.tolist()}")
        print(f"  Angles per bin: {runs_per_bin}")
    param_sets = pre_compute_params(
        args.num_runs,
        args.seed,
        randomize_mass=args.randomize_mass,
        randomize_com=args.randomize_com,
        randomize_kd=args.randomize_kd,
        force_push=args.force_push,
    )

    # Resolve velocity command function
    velocity_command_fn = VELOCITY_COMMANDS[args.velocity_command]
    print(f"  Velocity command: {args.velocity_command}")

    # Generate shared experiment timestamp
    experiment_timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_folder_name = f"{experiment_timestamp}_{args.experiment_name}"
    print(f"Experiment folder: {experiment_folder_name}")

    # Run experiments for each policy
    experiment_folders = {}
    for run_dir, name in zip(run_dirs, args.names):
        print(f"\n{'#'*60}")
        print(f"Running experiment for: {name}")
        print(f"Run directory: {run_dir}")
        print(f"{'#'*60}")

        experiment_folder = os.path.join(run_dir, "experiments", experiment_folder_name)
        experiment_folders[name] = experiment_folder

        run_experiment_for_policy(
            run_dir=run_dir,
            experiment_folder=experiment_folder,
            param_sets=param_sets,
            total_time=args.total_time,
            velocity_command_fn=velocity_command_fn,
        )

    # Generate plots for each policy individually
    print(f"\n{'#'*60}")
    print("Generating plots...")
    print(f"{'#'*60}")

    for name, exp_folder in experiment_folders.items():
        print(f"\nPlotting for: {name}")
        single_data = OrderedDict({name: load_experiment_data(exp_folder)})
        filtered_single_data = _filter_successful_runs(single_data)

        vel_path = os.path.join(exp_folder, "velocity_tracking")
        torque_path = os.path.join(exp_folder, "joint_torques")
        plot_velocity_comparison(filtered_single_data, save_path=vel_path)
        plot_torque_comparison(filtered_single_data, save_path=torque_path)

    # Generate comparison plots across all policies
    if len(experiment_folders) > 1:
        print("\nGenerating comparison plots across all policies...")
        grouped_data = OrderedDict()
        for name, exp_folder in experiment_folders.items():
            grouped_data[name] = load_experiment_data(exp_folder)

        # Filter to only successful runs for plots and stats
        filtered_data = _filter_successful_runs(grouped_data)
        for name in filtered_data:
            n_total = len(grouped_data[name]["runs"])
            n_success = len(filtered_data[name]["runs"])
            print(f"  '{name}': using {n_success}/{n_total} successful runs for plots/stats")

        # Save comparison plots to each experiment folder
        for name, exp_folder in experiment_folders.items():
            plot_velocity_comparison(
                filtered_data,
                save_path=os.path.join(exp_folder, "comparison_velocity"),
            )
            plot_torque_comparison(
                filtered_data,
                save_path=os.path.join(exp_folder, "comparison_torques"),
            )
            plot_radar_comparison(
                filtered_data,
                save_path=os.path.join(exp_folder, "radar_comparison"),
            )

    # Print stats tables and save to file
    if len(experiment_folders) > 1:
        stats_text = print_stats_table(filtered_data)
        stats_text += print_torque_stats_table(filtered_data)
        stats_text += print_success_table(grouped_data)  # unfiltered for success rate
        stats_text += print_combined_error_table(filtered_data)
    else:
        # Single policy — still print stats
        single_grouped = OrderedDict()
        for name, exp_folder in experiment_folders.items():
            single_grouped[name] = load_experiment_data(exp_folder)
        filtered_single = _filter_successful_runs(single_grouped)
        for name in filtered_single:
            n_total = len(single_grouped[name]["runs"])
            n_success = len(filtered_single[name]["runs"])
            print(f"  '{name}': using {n_success}/{n_total} successful runs for plots/stats")
        stats_text = print_stats_table(filtered_single)
        stats_text += print_torque_stats_table(filtered_single)
        stats_text += print_success_table(single_grouped)  # unfiltered for success rate
        stats_text += print_combined_error_table(filtered_single)

    for name, exp_folder in experiment_folders.items():
        stats_path = os.path.join(exp_folder, "experiment_stats.txt")
        with open(stats_path, "w") as f:
            f.write(stats_text)
        print(f"Saved stats to {stats_path}")

    # Generate force-magnitude success histogram if force push was enabled
    if args.force_push:
        all_data = grouped_data if len(experiment_folders) > 1 else single_grouped
        has_force = any(
            len(d.get("force_mags", [])) > 0 for d in all_data.values()
        )
        if has_force:
            for name, exp_folder in experiment_folders.items():
                plot_force_success_histogram(
                    all_data,
                    save_path=os.path.join(exp_folder, "force_success_histogram"),
                )

    # Print summary
    print(f"\n{'='*60}")
    print("Experiment complete!")
    print(f"{'='*60}")
    print(f"Experiment name: {experiment_folder_name}")
    print(f"Policies tested: {len(args.names)}")
    print(f"Runs per policy: {args.num_runs}")
    print(f"Seed: {args.seed}")
    print(f"\nExperiment folders:")
    for name, folder in experiment_folders.items():
        print(f"  {name}: {folder}")


if __name__ == "__main__":
    main()
