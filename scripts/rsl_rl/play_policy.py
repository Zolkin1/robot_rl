#use the same play config,but use different trained policy
#based on the specific envrionment specifyed, use the corresponding trained policy


task = "custom-Isaac-Velocity-Flat-G1-Play-v0"

#load different trained policy based on the env

# "vanilla","custom","clf"

import argparse
import os
import sys
import time
import pickle

from isaaclab.app import AppLauncher
import cli_args

# Import plot_trajectories functions
from plot_trajectories import plot_trajectories

# Environment names mapping
ENVIRONMENTS = {
    "vanilla": "custom-Isaac-Velocity-Flat-G1-Play-v0",
    "custom": "G1-flat-vel-play",
    "clf": "G1-flat-lip-vel-play",
}

# Experiment names mapping for different environments
EXPERIMENT_NAMES = {
    "vanilla": "g1_isaac",
    "custom": "g1",
    "clf": "g1_clf",
}


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


def parse_args():
    parser = argparse.ArgumentParser(description="Play trained RL policies for different environments.")
    parser.add_argument(
        "--env_type",
        type=str,
        choices=list(ENVIRONMENTS.keys()),
        help="Type of environment to play (vanilla/custom/clf)"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Override the default experiment name for the environment type"
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=True,
        help="Record videos during playback."
    )
    parser.add_argument(
        "--video_length",
        type=int,
        default=200,
        help="Length of the recorded video (in steps)."
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of environments to simulate."
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        default=False,
        help="Run in real-time, if possible."
    )
    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_known_args()


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


def main():
    args_cli, hydra_args = parse_args()
    
    if not args_cli.env_type:
        print("Please specify an environment type using --env_type")
        print("Available options:", list(ENVIRONMENTS.keys()))
        sys.exit(1)

    # Set the task based on environment type
    args_cli.task = ENVIRONMENTS[args_cli.env_type]
    
    # Get experiment name (use override if provided, otherwise use default)
    experiment_name = args_cli.exp_name or EXPERIMENT_NAMES[args_cli.env_type]
    
    # always enable cameras to record video
    if args_cli.video:
        args_cli.enable_cameras = True

    # clear out sys.argv for Hydra
    sys.argv = [sys.argv[0]] + hydra_args

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        # Import necessary modules after app launch
        import gymnasium as gym
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab.envs import (
            DirectMARLEnv,
            multi_agent_to_single_agent,
        )
        from isaaclab.utils.dict import print_dict
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
        import robot_rl.tasks  # noqa: F401

        # Configure PyTorch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

        # parse configuration
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
        )
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "g1_policies", args_cli.env_type, experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        
        # Get checkpoint path from the training directory
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loading checkpoint from: {resume_path}")
        
        # Use the checkpoint directory for saving results
        play_log_dir = os.path.dirname(resume_path)
        print(f"[INFO] Saving results to directory: {play_log_dir}")

        # create isaac environment
        if hasattr(env_cfg, "__prepare_tensors__") and callable(getattr(env_cfg, "__prepare_tensors__")):
            env_cfg.__prepare_tensors__()
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(play_log_dir, "videos"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during playback.")
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
        logger = DataLogger(enabled=True, log_dir=play_log_dir, variables=log_vars)

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
                obs, reward, _, extra = env.step(actions)
                
                # Log data
                data = extract_reference_trajectory(env, log_vars)
                logger.log_from_dict(data)

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

        # Create plots directory and generate plots
        plot_dir = os.path.join(play_log_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        print(f"[INFO] Generating plots in directory: {plot_dir}")
        plot_trajectories(logger.data, save_dir=plot_dir)

    finally:
        # Ensure simulation app is closed
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()