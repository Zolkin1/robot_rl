# Copyright 2026 Zachary Olkin. All rights reserved.

import os
import numpy as np
import yaml

from sim.log_utils import find_most_recent_timestamped_folder, extract_data

def get_index(time_vec, time: float):
    """Gets the index associated with a given time."""
    closest_idx = np.argmin(np.abs(time_vec - time))

    return closest_idx

def compute_stats(start_time = 0):
    """Compute the statistics and save the information to a file in the given directory."""
    # Load in the data from rerun
    log_dir = os.getcwd() + "/logs"
    print(f"Looking for logs in {log_dir}.")
    newest = find_most_recent_timestamped_folder(log_dir)

    print(f"Loading data from {newest}.")

    # Parse the config file
    with open(os.path.join(newest, "sim_config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        data = extract_data(os.path.join(newest, "sim_log.csv"), config)

        robot = config["robot"]
        policy = config["policy"]
        policy_dt = config["policy_dt"]


        time = data['time']
        commanded_vel = data['commanded_vel']
        act_vel = data['qvel'][:, [0, 1, 5]]

        start_idx = get_index(time, start_time)

        # Floating base velocity mean error
        mean_error = np.mean(np.square(commanded_vel[start_idx:, :] - act_vel[start_idx:, :]), axis=0)

        # Std dev
        std_dev_error = np.std(np.square(commanded_vel[start_idx:, :] - act_vel[start_idx:, :]), axis=0)

        # Save
        stats = {
            'mean_velocity_error': mean_error.tolist(),
            'std_dev_velocity_error': std_dev_error.tolist(),
        }

        with open(os.path.join(newest, 'stats.yaml'), 'w') as f:
            yaml.dump(stats, f)

        return stats



if __name__ == "__main__":
    # Compute statistics and save
    compute_stats(0)