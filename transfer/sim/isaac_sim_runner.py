# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
parser.add_argument("--config_file", type=str, help="Config file name in the config folder")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import csv
import os
import numpy as np
import yaml
from datetime import datetime

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from rl_policy_wrapper import RLPolicy

##
# Pre-defined configs
##
from robot_rl.assets.robots.g1_21j import G1_MINIMAL_CFG  # isort: skip


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2"
    # Each group will have a robot in it
    origins = [[0.0, 0.0, 0.0]]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])

    # Articulation
    robot_cfg = G1_MINIMAL_CFG.copy()
    robot_cfg.prim_path = "/World/Origin.*/Robot"
    robot = Articulation(cfg=robot_cfg)

    # return the scene information
    scene_entities = {"robot": robot}
    return scene_entities, origins

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

def run_simulator(log_file: str, policy_wrapper):
    """Runs the simulation loop."""
    # Setup the sim
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    entities, origins = design_scene()
    origins = torch.tensor(origins, device=sim.device)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    print(f"Logging to {log_file}.")

    # Desired velocity
    des_vel = np.zeros(3)
    des_vel[0] = 0.5

    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    robot = entities["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
        # -- generate action from policy
        obs = policy_wrapper.create_obs(robot.data.joint_pos[0, :].cpu().numpy(), robot.data.root_ang_vel_b[0, :].cpu().numpy(),
                                        robot.data.joint_vel[0, :].cpu().numpy(), sim.current_time,
                                        robot.data.projected_gravity_b[0, :].cpu().numpy(), des_vel)
        action_mj = policy_wrapper.get_action(obs)
        action_isaac = policy_wrapper.get_action_isaac()

        # TODO: Fix so it walks

        # -- apply action to the robot
        robot.set_joint_position_target(torch.tensor(action_isaac, device=sim.device))
        # -- write data to sim
        robot.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        robot.update(sim_dt)

        # TODO: Get the torques
        torques = np.zeros(21)  # TODO: get this number programatically
        row = ([sim.current_time] + robot.data.root_pos_w[0,:].cpu().numpy().tolist()
               + robot.data.root_quat_w[0, :].cpu().numpy().tolist()
               + policy_wrapper.convert_to_mujoco(robot.data.joint_pos[0, :].cpu().numpy()).tolist()
               + robot.data.root_lin_vel_b[0, :].cpu().numpy().tolist()
               + robot.data.root_ang_vel_b[0, :].cpu().numpy().tolist()
               + policy_wrapper.convert_to_mujoco(robot.data.joint_vel[0,:].cpu().numpy()).tolist()
               + obs[0,:].numpy().tolist()
               + action_mj.tolist()
               + torques.tolist())
        log_row_to_csv(log_file, row)


def main():
    """Main function."""
    # Parse the config file
    config_file = args_cli.config_file

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        checkpoint_path = config["checkpoint_path"]
        dt = config["dt"]
        num_obs = config["num_obs"]
        num_action = config["num_action"]
        period = config["period"]
        robot_name = config["robot_name"]
        action_scale = config["action_scale"]
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        qvel_scale = config["qvel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        command_scale = config["command_scale"]

    # Make the RL policy
    policy = RLPolicy(dt=dt, checkpoint_path=checkpoint_path, num_obs=num_obs, num_action=num_action, period=period,
                      cmd_scale=command_scale, action_scale=action_scale, default_angles=default_angles,
                      qvel_scale=qvel_scale, ang_vel_scale=ang_vel_scale, )

    # Setup the logging
    # Make a new directroy based on the current time
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    new_folder_path = os.path.join(os.getcwd(), "transfer/sim/logs/" + timestamp_str)
    try:
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"Successfully created folder: {new_folder_path}")
    except OSError as e:
        print(f"Error creating folder {new_folder_path}: {e}")
    print(f"Saving rerun logs to {new_folder_path}.")
    log_file = os.path.join(new_folder_path, "sim_log.csv")
    sim_config = {
        'simulator': "isaacsim",
        'robot': robot_name,
        'policy': policy.get_chkpt_path(),
        'policy_dt': policy.dt,
        'data_structure': [
            {'name': 'time', 'length': 1},
            {'name': 'qpos', 'length': 28},
            {'name': 'qvel', 'length': 27},
            {'name': 'obs', 'length': policy.get_num_obs()},
            {'name': 'action', 'length': policy.get_num_actions()},
            {'name': 'torque', 'length': 21},
        ]
    }
    with open(os.path.join(new_folder_path, "sim_config.yaml"), 'w') as f:
        yaml.dump(sim_config, f)

    # Run the simulator
    run_simulator(log_file, policy)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()