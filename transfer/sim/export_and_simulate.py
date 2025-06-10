import os
import glob
import argparse
from transfer.sim.simulation import Simulation
from transfer.sim.robot import Robot
from transfer.sim.rl_policy_wrapper import RLPolicy


def find_policy_files(policy_dir):
    """Find all policy.pt files in the specified directory."""
    return glob.glob(os.path.join(policy_dir, "**/policy.pt"), recursive=True)


def run_simulation(policy_path, log_dir):
    """Run a single simulation with the given policy."""
    # Policy parameters - these should match your training configuration
    policy_params = {
        'dt': 0.02,  # Policy update frequency
        'num_obs': 48,  # Number of observations
        'num_action': 21,  # Number of actions
        'cmd_scale': [1.0, 1.0, 1.0],  # Command velocity scaling
        'period': 1.0,  # Phase period
        'action_scale': 1.0,  # Action scaling
        'default_angles': None,  # Will be set by robot
        'qvel_scale': 1.0,  # Joint velocity scaling
        'ang_vel_scale': 1.0,  # Angular velocity scaling
    }

    # Initialize robot and policy
    robot = Robot()
    policy = RLPolicy(checkpoint_path=policy_path, **policy_params)
    
    # Create simulation instance
    sim = Simulation(policy=policy, robot=robot, log=True, log_dir=log_dir)
    
    # Run simulation
    try:
        sim.run()
    except Exception as e:
        print(f"Error running simulation for policy {policy_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run batch simulations on exported policies')
    parser.add_argument('--policy_dir', type=str, required=True,
                        help='Directory containing exported policy files (policy.pt)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save simulation logs')
    args = parser.parse_args()

    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)

    # Find all policy files
    policy_files = find_policy_files(args.policy_dir)
    if not policy_files:
        print(f"No policy files found in {args.policy_dir}")
        return

    print(f"Found {len(policy_files)} policy files")
    
    # Run simulations for each policy
    for policy_path in policy_files:
        policy_name = os.path.basename(os.path.dirname(policy_path))
        print(f"\nSimulating policy: {policy_name}")
        run_simulation(policy_path, args.log_dir)


if __name__ == "__main__":
    main() 