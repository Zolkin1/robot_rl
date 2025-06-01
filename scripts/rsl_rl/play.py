# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import pickle
import time
import os
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

import robot_rl.tasks  # noqa: F401

class DataLogger:
    def __init__(self, enabled=True, log_dir=None, variables=None):
        self.enabled = enabled
        self.data = {}
        self.log_dir = log_dir
        self.variables = variables or []
        
        if enabled and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            print(f"[INFO] Logging data to directory: {log_dir}")
            # Initialize data storage for each variable
            for var in self.variables:
                self.data[var] = []
    
    def log_from_dict(self, data_dict):
        """Log data from a dictionary, only logging variables that were specified in initialization"""
        if not self.enabled:
            return
            
        for var in self.variables:
            if var in data_dict:
                self.data[var].append(data_dict[var])
    
    def save(self):
        """Save all logged data to pickle files"""
        if not self.enabled or not self.log_dir:
            return
            
        for var in self.variables:
            if var in self.data:
                filepath = os.path.join(self.log_dir, f"{var}.pkl")
                with open(filepath, "wb") as f:
                    pickle.dump(self.data[var], f)
                print(f"[INFO] Saved {var} data to {filepath}")

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

    # Define variables to log
    log_vars = [
        'y_out',
        'dy_out',
        'base_velocity',
        'cur_swing_time',
        "stance_foot_pos",
        "stance_foot_ori",
        'y_act',
        'dy_act',
        'v',
        'vdot',
        'stance_foot_pos_0',
        'stance_foot_ori_0',
    ]

    # Setup logging
    log_dir = os.path.join("logs", "play", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    logger = DataLogger(enabled=True, log_dir=log_dir, variables=log_vars)

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    log_terms_list = []

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, reward, _, extra = env.step(actions)
           
            data = extract_reference_trajectory(env, log_vars)
            # Merge extra["log"] if it exists
            # if "log" in extra and isinstance(extra["log"], dict):
            #     # Convert any torch tensors to Python scalars for logging
            #     for k, v in extra["log"].items():
            #         if hasattr(v, "item"):
            #             data[k] = v.item()
            #         else:
            #             data[k] = v
            #         # print(f"logging {k} with value {data[k]}")
            logger.log_from_dict(data)


            # if "log" in extra and isinstance(extra["log"], dict):
            #     # Convert tensors to scalars for logging
            #     log_terms = {}
            #     for k, v in extra["log"].items():
            #         log_terms[k] = v.item() if hasattr(v, "item") else v
            #     log_terms_list.append(log_terms)

        timestep += 1
        if args_cli.video:
            
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        
        if timestep > max(100, args_cli.video_length):
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    
    # Save all logged data
    logger.save()

    # Save all log terms to a single pickle file
    if log_terms_list:
        log_terms_path = os.path.join(log_dir, "log_terms.pkl")
        with open(log_terms_path, "wb") as f:
            pickle.dump(log_terms_list, f)
        print(f"[INFO] Saved all log terms to {log_terms_path}")

def extract_reference_trajectory(env, log_vars):
    # Get the underlying environment by unwrapping
    unwrapped_env = env.unwrapped
    # Get the HLIP reference term from the command manager
    hlip_Ref = unwrapped_env.command_manager.get_term("hlip_ref")
    results = {}

    for var in log_vars:
        if hasattr(hlip_Ref, var):
            results[var] = getattr(hlip_Ref, var)
        elif var == "base_velocity":
            results[var] = unwrapped_env.command_manager.get_command("base_velocity")
        else:
            results[var] = None  # or raise an error/warning if you prefer

    return results

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
