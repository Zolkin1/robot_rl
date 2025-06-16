from typing import Tuple
import os
import time
import math
import numpy as np
import csv
import yaml
import mujoco
import mujoco.viewer
from datetime import datetime
import pygame

from transfer.sim.robot import Robot
from transfer.sim.simulation import Simulation

def get_model_data(robot: str, scene: str):
    """Create the mj model and data from the given robot."""
    if robot != "g1_21j":
        raise ValueError("Invalid robot name! Only support g1_21j for now.")

    file_name = robot + "_" + scene + ".xml"
    relative_path = "robots/g1/" + file_name
    path = os.path.join(os.getcwd(), relative_path)
    print(f"Trying to load the xml at {path}")
    mj_model = mujoco.MjModel.from_xml_path(path)
    mj_data = mujoco.MjData(mj_model)

    return mj_model, mj_data

def get_projected_gravity(quat):
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    pg = np.zeros(3)

    pg[0] = 2 * (-qz * qx + qw * qy)
    pg[1] = -2 * (qz * qy + qw * qx)
    pg[2] = 1 - 2 * (qw * qw + qz * qz)

    return pg

def log_row_to_csv(filename, data):
    """
    Appends a single row of data to an existing CSV file.

    Args:
      filename (str): The path to the CSV file.
      data_row (list): A list of data points for the row.
    """
    try:
        # Open in append mode ('a') to add data to the end of the file
        # newline='' is important to prevent extra blank rows
        with open(filename, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(data)
        # print(f"Appended row to {filename}") # Uncomment for verbose logging
    except Exception as e:
        print(f"Error appending row to {filename}: {e}")

def run_simulation(policy, robot: str, scene: str, log: bool, log_dir: str, use_height_sensor: bool = False):
    """Run the simulation.
    
    Args:
        policy: The policy to use for control
        robot: Robot name
        scene: Scene name
        log: Whether to log data
        log_dir: Directory to save logs
        use_height_sensor: Whether to use height sensor (default: False)
    """
    # Create robot instance
    robot_instance = Robot(robot, scene)
    
    # Create and run simulation
    sim = Simulation(policy, robot_instance, log, log_dir, use_height_sensor=use_height_sensor)
    sim.run()

def ray_cast_sensor(model, data, site_name, size: Tuple[float, float], x_y_num_rays: Tuple[int, int], sen_offset: float = 0) -> np.array:
    """Using a grid pattern, create a height map using ray casting."""
    ray_pos_shape = x_y_num_rays
    ray_pos_shape = ray_pos_shape + (3,)
    ray_pos = np.zeros(ray_pos_shape)

    # Get the site location
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    site_pos = data.site_xpos[site_id].copy()

    # Add to the global z
    site_pos[2] = site_pos[2] + 10

    site_pos[0] = site_pos[0] - size[0] / 2.
    site_pos[1] = site_pos[1] - size[1] / 2.

    # Ray information
    direction = np.zeros(3)
    direction[2] = -1
    geom_group = np.zeros(6, dtype=np.int32)
    geom_group[2] = 1  # Only include group 2

    # Ray spacing
    spacing = np.zeros(3)
    spacing[0] = size[0] / (x_y_num_rays[0] - 1)
    spacing[1] = size[1] / (x_y_num_rays[1] - 1)

    # Loop through the rays
    for xray in range(x_y_num_rays[0]):
        for yray in range(x_y_num_rays[1]):
            geom_id = np.zeros(1, dtype=np.int32)
            offset = spacing.copy()
            offset[0] = spacing[0] * xray
            offset[1] = spacing[1] * yray

            ray_origin = offset + site_pos
            ray_pos[xray, yray, 2] = -mujoco.mj_ray(
                model, data,
                ray_origin.astype(np.float64),
                direction.astype(np.float64),
                geom_group, 1, -1, geom_id
            )

            ray_pos[xray, yray, :] = ray_origin + ray_pos[xray, yray, :]
    # print(f"ray_pos: {ray_pos}, shape: {ray_pos.shape}")

    return ray_pos
