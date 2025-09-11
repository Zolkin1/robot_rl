# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--mass", type=float, default=13.0, help="Mass of torso")

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

import os
import time

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

import robot_rl.tasks  # noqa: F401

import omni.usd
from isaaclab.sim import schemas, schemas_cfg
from pxr import UsdPhysics             # need this to apply the API


def set_torso_mass(mass_kg: float, env_id: int = 0):
    stage      = omni.usd.get_context().get_stage()
    torso_path = f"/World/envs/env_{env_id}/Amber/amber3_PF/torso"

    # Ensure MassAPI exists; modify_mass_properties() returns False otherwise
    prim = stage.GetPrimAtPath(torso_path)
    if not UsdPhysics.MassAPI(prim):
        UsdPhysics.MassAPI.Apply(prim)                          # :contentReference[oaicite:2]{index=2}

    cfg = schemas_cfg.MassPropertiesCfg(mass=mass_kg)           # only this attr is set :contentReference[oaicite:3]{index=3}
    schemas.modify_mass_properties(torso_path, cfg, stage)      # returns True on success :contentReference[oaicite:4]{index=4}

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
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    base_env = env.unwrapped

    # #custom command velocity
    # command_tensor = torch.tensor([[-0.4, 0.0, 0.0]], device=base_env.device).repeat(base_env.num_envs, 1)
    # base_env.command_manager.command = command_tensor
    
    obs_mgr  = base_env.observation_manager
    rew_mgr  = base_env.reward_manager                 # new: access reward manager

    # run the manager once to get the dict   {name: tensor}
    # obs_dict = obs_mgr.compute()            # shapes:  [N, dim_i]


    # convert to single‑agent (if needed)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # now fetch the *final* observation manager
    obs_mgr  = env.unwrapped.observation_manager
    obs_dict = obs_mgr.compute()
    print("=== Observation slice map ===")
    slice_map = {}
    obs_columns = []
    start = 0
    for name, tensor in obs_dict.items():   # preserves cfg order
        dim = tensor.shape[1] if tensor.ndim > 1 else 1
        sl  = slice(start, start + dim)
        slice_map[name] = sl
        width = dim
        if width == 1:
            obs_columns.append(name)
        else:
            obs_columns += [f"{name}[{i}]" for i in range(width)]
        print(f"{name:25s} -> {sl}")
        start += dim
    print("Vector length:", start)
    # Build a flat list of column names for every element of every observation
    obs_columns = []
    for name, sl in slice_map.items():
        width = sl.stop - sl.start
        print("+------------WIDTH:",width)
        if width == 1:
            obs_columns.append(name)                      # e.g. "projected_gravity"
        else:
            # tag individual components: "joint_pos[0]", "joint_pos[1]", …
            obs_columns += [f"{name}[{i}]" for i in range(width)]
    # ------------------------------------------------------------
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
    print("-"*20,dt,"-"*20)
    # run the manager once to get the dict {name: tensor}
    term_names = rew_mgr.active_terms          # <-- names once, after env is ready
    run_stamp = os.path.basename(os.path.dirname(resume_path))          # 2025‑07‑17_10‑20‑13
    ckpt_id   = os.path.basename(resume_path).replace("model_", "").replace(".pt", "")  # 13999
    csv_name  = f"{run_stamp}_{ckpt_id}.csv"                            # 2025‑07‑17_10‑20‑13_13999.csv

    # 2) Define the results directory *explicitly*
    results_dir = "/home/s-ritwik/src/robot_rl/SURF_results"

    # 3) Make sure the directory exists
    os.makedirs(results_dir, exist_ok=True)

    # 4) Final path for the CSV
    csv_path = os.path.join(results_dir, csv_name)
    # ------------------------------------------------------------

    # open the file for writing rewards
    fieldnames = ["step"] + rew_mgr.active_terms + obs_columns

    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    # csv_file = open("reward_terms.csv", "w", newline="")
    # writer   = csv.DictWriter(csv_file, fieldnames=["step"] + rew_mgr.active_terms)
    # writer.writeheader()
    timestep = 0
    step = 0
    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    set_torso_mass(args_cli.mass)
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
        obs, _, _, _ = env.step(actions)
        # print(obs) 
        # for name, sl in slice_map.items():
        #     print(name, obs[:, sl])             # per-env values for that term

        # --- reward logging ---
        rew_mgr.compute(dt)                       # updates _step_reward
        vals = rew_mgr._step_reward.mean(0)       # [num_terms]
        # --- observation logging ---
        obs_mean = obs.mean(0)                       # mean over envs, [obs_dim]

        row = {"step": step}
        #REWARDS
        for name, v in zip(rew_mgr.active_terms, vals):
            row[name] = float(v.item())
        #OBSERVATION    
        for name, sl in slice_map.items():
            vec = obs_mean[sl]                      # tensor view, may be 0‑D or 1‑D
            if vec.numel() == 1:                    # scalar
                row[name] = float(vec.item())
            else:                                   # vector → expand components
                for i, val in enumerate(vec):
                    row[f"{name}[{i}]"] = float(val.item())
        writer.writerow(row)

        step += 1

        timestep += 1
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
    csv_file.close()
    simulation_app.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
