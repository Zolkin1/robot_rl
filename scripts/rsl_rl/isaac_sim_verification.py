# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate. Always set to 1.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--config_file", type=str, default=None, help="Config file name in the config folder")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.num_envs = 1
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
from datetime import datetime
import yaml
import numpy as np
import csv

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import robot_rl.tasks  # noqa: F401


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    if hasattr(env_cfg, "__prepare_tensors__") and callable(getattr(env_cfg, "__prepare_tensors__")):
        env_cfg.__prepare_tensors__()
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # Dump info to config file in the transfer/sim log dir
    # Parse the robot config
    checkpoint_path, robot = parse_config_file(cli_args)    # Make a new directroy based on the current time
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    new_folder_path = os.path.join(os.getcwd(), "transfer/sim/log/" + timestamp_str)
    try:
        os.makedirs(new_folder_path, exist_ok=True)
        print(f"Successfully created folder: {new_folder_path}")
    except OSError as e:
        print(f"Error creating folder {new_folder_path}: {e}")
    print(f"Saving rerun logs to {new_folder_path}.")
    log_file = os.path.join(new_folder_path, "sim_log.csv")
    sim_config = {
        'simulator': "isaacsim",
        'robot': robot,
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

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)

            # Log data
            log_row_to_csv(log_file, [env.sim.current_time])
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


def parse_config_file(args):
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

    return checkpoint_path, robot_name
    # return (checkpoint_path, dt, num_obs, num_action, period, robot_name, action_scale,
    #         default_angles, qvel_scale, ang_vel_scale, command_scale)

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

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
