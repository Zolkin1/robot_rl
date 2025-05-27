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
from scipy.optimize import direct


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

def run_simulation(policy, robot: str, scene: str, log: bool, log_dir: str):
    """Run the simulation."""

    # Setup mj model and data
    mj_model, mj_data = get_model_data(robot, scene)
    scene = mujoco.MjvScene(mj_model, maxgeom=1000)
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()

    keyframe_index = mj_model.keyframe("standing").id
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, keyframe_index)

    # Compute sim steps per policy update
    sim_steps_per_policy_update = int(policy.dt / mj_model.opt.timestep)
    sim_loop_rate = sim_steps_per_policy_update * mj_model.opt.timestep
    viewer_rate = math.ceil((1/100) / mj_model.opt.timestep)
    print(viewer_rate)

    print(f"Starting mujoco simulation with robot {robot}.\n"
          f"Policy dt set to {policy.dt} s ({sim_steps_per_policy_update} steps per policy update.)\n"
          f"Simulation dt set to {mj_model.opt.timestep} s. Sim loop rate set to {sim_loop_rate} s.\n")
    nu = policy.get_num_actions()

    # Setup joystick
    pygame.init()
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    if joystick_count < 1:
        print("No joystick detected, using initial command from config instead.")
        joystick = None
    else:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(f"Using controller: {joystick.get_name()}")

    if log:
        # Make a new directroy based on the current time
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        new_folder_path = os.path.join(log_dir, timestamp_str)
        try:
            os.makedirs(new_folder_path, exist_ok=True)
            print(f"Successfully created folder: {new_folder_path}")
        except OSError as e:
            print(f"Error creating folder {new_folder_path}: {e}")
        print(f"Saving rerun logs to {new_folder_path}.")
        log_file = os.path.join(new_folder_path, "sim_log.csv")
        sim_config = {
            'simulator': "mujoco",
            'robot': robot,
            'policy': policy.get_chkpt_path(),
            'policy_dt': policy.dt,
            'data_structure' : [
                {'name': 'time', 'length': 1},
                {'name': 'qpos', 'length': mj_data.qpos.shape[0]},
                {'name': 'qvel', 'length': mj_data.qvel.shape[0]},
                {'name': 'obs', 'length': policy.get_num_obs()},
                {'name': 'action', 'length': policy.get_num_actions()},
                {'name': 'torque', 'length': mj_data.qpos.shape[0] - 7},
            ]
        }
        with open(os.path.join(new_folder_path, "sim_config.yaml"), 'w') as f:
            yaml.dump(sim_config, f)

    x_y_num_rays = (5, 5)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

        des_vel = np.zeros(3)

        # Setup geoms
        ray_pos = ray_cast_sensor(mj_model, mj_data, "height_sensor_site", (1, 1), x_y_num_rays, 0.0)
        # Add custom debug spheres
        ii = 0
        for pos in ray_pos.reshape(-1, 3):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[ii],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.05, 0, 0]),
                pos=pos,
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 1]),
            )
            viewer.user_scn.ngeom += 1
            ii += 1

        while viewer.is_running():
            start_time = time.time()

            # Get the commanded action
            # Apply control signal here.
            if joystick is not None:
                for event in pygame.event.get():
                    pass
                # Left stick: control vx, vy (2D plane), right stick X-axis: vyaw
                vy = -(joystick.get_axis(0))
                vx = -(joystick.get_axis(1))
                vyaw = -(joystick.get_axis(3))

                # Clip or zero out small values
                if abs(vx) < 0.1:
                    vx = 0
                else:
                    vx = np.clip(vx, -1, 1)
                if abs(vy) < 0.1:
                    vy = 0
                else:
                    vy = np.clip(vy, -1, 1)
                if abs(vyaw) < 0.1:
                    vyaw = 0
                else:
                    vyaw = np.clip(vyaw, -1.5, 1.5)
                des_vel[0] = vx
                des_vel[1] = vy
                des_vel[2] = vyaw   # TODO: Why does this not seem to work?


                # Extract relevant info
            sim_time = mj_data.time
            qpos = mj_data.qpos
            qvel = mj_data.qvel

            # Compute the control action
            pg = get_projected_gravity(qpos[3:7])
            obs = policy.create_obs(qpos[7:], qvel[3:6], qvel[6:], sim_time, pg, des_vel)
            u = policy.get_action(obs)
            mj_data.ctrl[:nu] = u

            # Step the simulator
            for i in range(sim_steps_per_policy_update):
                ray_pos = ray_cast_sensor(mj_model, mj_data, "height_sensor_site", (1, 1), x_y_num_rays, 0.0)
                ii = 0
                for pos in ray_pos.reshape(-1, 3):
                    viewer.user_scn.geoms[ii].pos = pos
                    ii += 1

                # Update scene
                mujoco.mjv_updateScene(mj_model, mj_data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)

                mujoco.mj_step(mj_model, mj_data)
                if log:
                    torques = []
                    for j in range(mj_model.nu):
                        torques.append(mj_data.actuator_force[j])

                    log_row_to_csv(log_file, [mj_data.time] + mj_data.qpos.tolist() + mj_data.qvel.tolist() + obs[0, :].numpy().tolist() + u.tolist() + torques)
                if i % viewer_rate == 0:
                    viewer.sync()

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < 1*sim_loop_rate:
                time.sleep(1*sim_loop_rate - elapsed)


def ray_cast_sensor(model, data, site_name, size: Tuple[float, float], x_y_num_rays: Tuple[int, int], sen_offset: float = 0) -> np.array:
    """Using a grid pattern, create a height map using ray casting."""

    ray_pos_shape = x_y_num_rays
    ray_pos_shape = ray_pos_shape + (3,)
    ray_pos = np.zeros(ray_pos_shape)

    # Get the site location
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    site_pos = data.site_xpos[site_id]

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
    spacing[0] = size[0]/(x_y_num_rays[0] - 1)
    spacing[1] = size[1]/(x_y_num_rays[1] - 1)

    # Loop through the rays
    for xray in range(x_y_num_rays[0]):
        for yray in range(x_y_num_rays[1]):
            geom_id = np.zeros(1, dtype=np.int32)
            offset = spacing.copy()
            offset[0] = spacing[0] * xray
            offset[1] = spacing[1] * yray

            ray_origin = offset + site_pos
            ray_pos[xray, yray, 2] = -mujoco.mj_ray(model, data,
                          ray_origin.astype(np.float64), direction.astype(np.float64),
                          geom_group, 1, -1, geom_id)

            ray_pos[xray, yray, :] = site_pos + ray_pos[xray, yray, :] + offset

    ray_pos[:, :, 2] = ray_pos[:, :, 2] - sen_offset

    return ray_pos
