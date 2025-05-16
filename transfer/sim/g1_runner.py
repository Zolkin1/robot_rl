import argparse
import os

import yaml
import numpy as np

from mj_simulation import run_simulation
from isaac_sim_runner import run_isaac_sim
from rl_policy_wrapper import RLPolicy

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, help="Config file name in the config folder")
parser.add_argument("--simulator", type=str, help="Choice of simulator to run (isaac_sim or mujoco)")
args = parser.parse_args()

config_file = args.config_file

# Parse the config file
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

sim = args.simulator
if sim == "mujoco":
    # Make the RL policy
    policy = RLPolicy(dt=dt, checkpoint_path=checkpoint_path, num_obs=num_obs, num_action=num_action, period=period,
                      cmd_scale=command_scale, action_scale=action_scale, default_angles=default_angles,
                      qvel_scale=qvel_scale, ang_vel_scale=ang_vel_scale,)

    # Run the simulator
    run_simulation(policy, robot_name, log=True, log_dir=os.getcwd()+"/logs/")

elif sim == "isaac_sim":
    run_isaac_sim(1, "G1-flat-vel")

