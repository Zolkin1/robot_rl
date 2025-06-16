import os
import re
import yaml
import csv
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def find_most_recent_timestamped_folder(base_path):
  """
  Finds the path of the most recent folder named with a YYYY-MM-DD-HH-MM-SS timestamp
  within a specified base path.

  Args:
    base_path (str): The directory to search within.

  Returns:
    str: The full path to the most recent timestamped folder, or None if none found.
  """
  most_recent_folder = None
  latest_timestamp = None

  # Regular expression to match the YYYY-MM-DD-HH-MM-SS format
  timestamp_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$")

  try:
    # List all entries in the base directory
    entries = os.listdir(base_path)

    for entry in entries:
      entry_path = os.path.join(base_path, entry)

      # Check if the entry is a directory and matches the timestamp pattern
      if os.path.isdir(entry_path) and timestamp_pattern.match(entry):
        try:
          # Parse the timestamp from the folder name
          folder_timestamp = datetime.strptime(entry, "%Y-%m-%d-%H-%M-%S")

          # If this is the first timestamped folder found, or if it's more recent
          if latest_timestamp is None or folder_timestamp > latest_timestamp:
            latest_timestamp = folder_timestamp
            most_recent_folder = entry_path

        except ValueError:
          # This handles cases where a folder name matches the pattern but isn't
          # a valid date/time string (unlikely with the previous script, but good practice)
          print(f"Warning: Directory '{entry}' matches pattern but has invalid timestamp.")
          pass # Skip this directory

  except FileNotFoundError:
    print(f"Error: Base path '{base_path}' not found.")
    return None
  except Exception as e:
    print(f"An unexpected error occurred: {e}")
    return None

  return most_recent_folder

def extract_data(filepath, config):
    data_structure = config.get('data_structure')
    print("\nData structure from config:")
    for item in data_structure:
        print(f"  {item['name']}: length {item['length']}")
    
    extracted_data_lists = {item['name']: [] for item in data_structure if 'name' in item}
    print("\nInitialized data lists:")
    for name in extracted_data_lists:
        print(f"  {name}")

    with open(filepath, "r") as f:
        csv_reader = csv.reader(f)

        row_count = 0
        for row in csv_reader:
            numeric_row = []
            for item in row:
                numeric_row.append(float(item))

            current_index = 0
            for item in data_structure:
                name = item.get('name')
                length = item.get('length')
                component_data = numeric_row[current_index: current_index + length]
                extracted_data_lists[name].append(component_data)
                current_index += length

            row_count += 1

        # Convert lists of data to NumPy arrays
        extracted_data_arrays = {}
        for name, data_list in extracted_data_lists.items():
            if data_list:  # Only create array if there is data
                extracted_data_arrays[name] = np.array(data_list)
                print(f"\nLoaded data for {name}:")
                print(f"  Shape: {extracted_data_arrays[name].shape}")
            else:  # Create empty array if no data was collected for this component
                # Determine the shape based on the config length
                component_length = next((item['length'] for item in data_structure if item.get('name') == name), 0)
                extracted_data_arrays[name] = np.empty((0, component_length))
                print(f"\nNo data found for {name}, created empty array with shape (0, {component_length})")

        return extracted_data_arrays

# Make plots
def plot_joints_and_actions(data):
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 10))

    FLOATING_BASE = 7

    for i in range(6):
        for j in range(2):
            axes[i, j].plot(data["time"], data["qpos"][:, i + 6*j + FLOATING_BASE])
            axes[i, j].plot(data["time"], data["action"][:, i + 6*j])
            axes[i, j].set_xlabel("time")
            axes[i, j].set_ylabel(f"qpos {i + 6*j + FLOATING_BASE} (rad)")
            axes[i, j].grid()

def plot_torques(data):
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 10))

    for i in range(6):
        for j in range(2):
            axes[i, j].plot(data["time"], data["torque"][:, i + 6*j])
            axes[i, j].set_xlabel("time")
            axes[i, j].set_ylabel(f"torque {i + 6*j} (Nm)")
            axes[i, j].grid()

def plot_vels(data):
    fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(10, 10))

    FLOATING_BASE = 6

    for i in range(6):
        for j in range(2):
            axes[i, j].plot(data["time"], data["qvel"][:, i + 6*j + FLOATING_BASE])
            axes[i, j].set_xlabel("time")
            axes[i, j].set_ylabel(f"qvel {i + 6*j + FLOATING_BASE} (rad/s)")
            axes[i, j].grid()

def plot_base(data):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))

    for i in range(3):
        for j in range(2):
            if j == 0:
                axes[i, j].plot(data["time"], data["qpos"][:, i])
                axes[i, j].set_xlabel("time")
                axes[i, j].set_ylabel(f"qpos {i} (m)")
            else:
                axes[i, j].plot(data["time"], data["qvel"][:, i])
                axes[i, j].set_xlabel("time")
                axes[i, j].set_ylabel(f"qvel {i} (m/s)")

def plot_ankles(data):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))

    axes[0, 0].plot(data["time"], data["left_ankle_pos"][:, 0])
    axes[0, 1].plot(data["time"], data["left_ankle_pos"][:, 1])
    axes[0, 2].plot(data["time"], data["left_ankle_pos"][:, 2])
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel(f"left_ankle_pos x (m)")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel(f"left_ankle_pos y (m)")
    axes[0, 2].set_xlabel("time")
    axes[0, 2].set_ylabel(f"left_ankle_pos z (m)")

    axes[1, 0].plot(data["time"], data["right_ankle_pos"][:, 0])
    axes[1, 1].plot(data["time"], data["right_ankle_pos"][:, 1])
    axes[1, 2].plot(data["time"], data["right_ankle_pos"][:, 2])
    axes[1, 0].set_xlabel("time")
    axes[1, 0].set_ylabel(f"right_ankle_pos x (m)")
    axes[1, 1].set_xlabel("time")
    axes[1, 1].set_ylabel(f"right_ankle_pos y (m)")
    axes[1, 2].set_xlabel("time")
    axes[1, 2].set_ylabel(f"right_ankle_pos z (m)")

def plot_velocity_comparison(data):
    """Plot comparison between commanded and actual velocities."""
    time = data['time']
    qvel = data['qvel']
    commanded_vel = data['commanded_vel']
    
    # Extract base velocities (first 3 elements of qvel)
    actual_vel = qvel[:, :3]
    
    # Create figure with 3 subplots for x, y, and angular velocities
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Commanded vs Actual Velocities')
    
    # Plot x velocity
    axes[0].plot(time, commanded_vel[:, 0], 'r--', label='Commanded')
    axes[0].plot(time, actual_vel[:, 0], 'b-', label='Actual')
    axes[0].set_ylabel('X Velocity (m/s)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot y velocity
    axes[1].plot(time, commanded_vel[:, 1], 'r--', label='Commanded')
    axes[1].plot(time, actual_vel[:, 1], 'b-', label='Actual')
    axes[1].set_ylabel('Y Velocity (m/s)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot angular velocity
    axes[2].plot(time, commanded_vel[:, 2], 'r--', label='Commanded')
    axes[2].plot(time, actual_vel[:, 2], 'b-', label='Actual')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Angular Velocity (rad/s)')
    axes[2].legend()
    axes[2].grid(True)

def plot_position_comparison(data):
    """Plot comparison between desired and actual positions."""
    time = data['time']
    qpos = data['qpos']
    commanded_vel = data['commanded_vel']
    
    # Extract base position (first 3 elements of qpos)
    actual_pos = qpos[:, :3]
    #need to extract the initial yaw 
    quat = qpos[:,4:8]
    yaw = 2 * np.arctan2(quat[:,2], quat[:,3])
    actual_pos[:,2] = yaw
    actual_yaw = yaw
    
    # Calculate desired position by integrating commanded velocity
    dt = time[1] - time[0]  # Assuming constant time step
    desired_pos = np.zeros_like(actual_pos)
    desired_pos[0] = actual_pos[0]  # Start from actual position
    
    for i in range(1, len(time)):
        desired_pos[i] = desired_pos[i-1] + commanded_vel[i-1] * dt
    

    # Create figure with 3 subplots for x, y, and angular positions
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle('Desired vs Actual Positions')
    
    # Plot x position
    axes[0].plot(time, desired_pos[:, 0], 'r--', label='Desired')
    axes[0].plot(time, actual_pos[:, 0], 'b-', label='Actual')
    axes[0].set_ylabel('X Position (m)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot y position
    axes[1].plot(time, desired_pos[:, 1], 'r--', label='Desired')
    axes[1].plot(time, actual_pos[:, 1], 'b-', label='Actual')
    axes[1].set_ylabel('Y Position (m)')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot angular position
    axes[2].plot(time, desired_pos[:, 2], 'r--', label='Desired')
    axes[2].plot(time, actual_yaw, 'b-', label='Actual')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Angular Position (rad)')
    axes[2].legend()
    axes[2].grid(True)

if __name__ == "__main__":
    # Load in the data from rerun
    log_dir = os.getcwd() + "/logs"
    print(f"Looking for logs in {log_dir}.")
    newest = find_most_recent_timestamped_folder(log_dir)

    print(f"Loading data from {newest}.")

    # TODO: Load in pkl or csv
    # Parse the config file
    with open(os.path.join(newest, "sim_config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

        data = extract_data(os.path.join(newest, "sim_log.csv"), config)

        robot = config["robot"]
        policy = config["policy"]
        policy_dt = config["policy_dt"]

        # print(data)

    print(f"=============="
          f" Data generated using " + config["simulator"] + " "
          "===============")

    print(f"time shape: {data['time'].shape}")
    print(f"qpos shape: {data['qpos'].shape}")
    print(f"qvel shape: {data['qvel'].shape}")
    print(f"torque shape: {data['torque'].shape}")
    print(f"action shape: {data['action'].shape}")
    print(f"left_ankle_pos shape: {data['left_ankle_pos'].shape}")
    print(f"right_ankle_pos shape: {data['right_ankle_pos'].shape}")
    print(f"commanded_vel shape: {data['commanded_vel'].shape}")

    # Make a plot
    plot_joints_and_actions(data)
    # plot_torques(data)
    # plot_vels(data)
    plot_base(data)
    # import pdb; pdb.set_trace()
    # plot_ankles(data)

    # Plot velocity and position comparisons
    plot_velocity_comparison(data)
    plot_position_comparison(data)

    plt.show()
