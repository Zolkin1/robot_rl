"""One-time script to generate golden values for regression tests.

Run: env_isaaclab/bin/python source/robot_rl/tests/generate_golden_values.py
"""

import sys
from pathlib import Path

import torch

torch.set_printoptions(precision=10, linewidth=200, sci_mode=False)

sys.path.insert(0, str(Path(__file__).parent))
from conftest import STANDING_YAML, WALKING_YAML, RUNNING_YAML, DEVICE

from robot_rl.tasks.manager_based.robot_rl.mdp.commands.traj_tracking.trajectory_manager import TrajectoryManager


def generate_for(name: str, yaml_path: str, test_times: list[float]):
    """Generate golden values for a trajectory at specific times."""
    m = TrajectoryManager(yaml_path, None, DEVICE)
    t = torch.tensor(test_times)

    print(f"\n{'='*60}")
    print(f"# {name}")
    print(f"# YAML: {yaml_path}")
    print(f"# type: {m.traj_data.trajectory_type}")
    print(f"# total_time: {m.traj_data.total_time}")
    print(f"# num_pos_outputs: {m.traj_data.num_pos_outputs}")
    print(f"# num_vel_outputs: {m.traj_data.num_vel_outputs}")
    print(f"{'='*60}")

    print(f"\n# Test times: {test_times}")

    pos, vel = m.get_output(t)
    print(f"\n# get_output pos (shape {pos.shape}):")
    for i, time in enumerate(test_times):
        print(f"#   t={time}: {pos[i].tolist()}")

    print(f"\n# get_output vel (shape {vel.shape}):")
    for i, time in enumerate(test_times):
        print(f"#   t={time}: {vel[i].tolist()}")

    accel = m.get_acceleration(t)
    print(f"\n# get_acceleration (shape {accel.shape}):")
    for i, time in enumerate(test_times):
        print(f"#   t={time}: {accel[i].tolist()}")

    # For golden value constants, print compact torch tensors
    print(f"\n# --- GOLDEN VALUES FOR COPY-PASTE ---")
    for i, time in enumerate(test_times):
        t_name = f"t{str(time).replace('.', '_')}"
        print(f"\n# t={time}")
        print(f"GOLDEN_{name.upper()}_POS_{t_name} = {pos[i].tolist()}")
        print(f"GOLDEN_{name.upper()}_VEL_{t_name} = {vel[i].tolist()}")
        print(f"GOLDEN_{name.upper()}_ACCEL_{t_name} = {accel[i].tolist()}")


if __name__ == "__main__":
    generate_for("standing", STANDING_YAML, [0.0, 0.2])
    generate_for("running", RUNNING_YAML, [0.0, 0.05, 0.133, 0.15])
    generate_for("walking", WALKING_YAML, [0.0, 0.2])
