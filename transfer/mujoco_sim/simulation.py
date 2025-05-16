import os
import time
import math
import numpy as np
import csv
import yaml
import mujoco
import mujoco.viewer
from datetime import datetime

def get_model_data(robot: str):
    """Create the mj model and data from the given robot."""
    if robot != "g1_21j":
        raise ValueError("Invalid robot name! Only support g1_21j for now.")

    relative_path = "robots/g1/g1_21j_basic_scene.xml"
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

def run_simulation(policy, robot: str, log: bool, log_dir: str):
    """Run the simulation."""

    # Setup mj model and data
    mj_model, mj_data = get_model_data(robot)

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


    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

        des_vel = np.zeros(3)
        des_vel[0] = 0.

        while viewer.is_running():
            start_time = time.time()

            # Extract relevant info
            sim_time = mj_data.time
            qpos = mj_data.qpos
            qvel = mj_data.qvel

            # Compute the control action
            pg = get_projected_gravity(qpos[3:7])
            obs = policy.create_obs(qpos, qvel, sim_time, pg, des_vel)
            u = policy.get_action(obs)
            mj_data.ctrl[:nu] = u

            # Step the simulator
            for i in range(sim_steps_per_policy_update):
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
            if elapsed < 5*sim_loop_rate:
                time.sleep(5*sim_loop_rate - elapsed)

