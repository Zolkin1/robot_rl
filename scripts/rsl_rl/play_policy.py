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
import glob

from isaaclab.app import AppLauncher
import cli_args
import torch

# Import plot_trajectories functions
from plot_trajectories import plot_trajectories, plot_hzd_trajectories
from train_policy import ENVIRONMENTS
# Experiment names mapping for different environments
EXPERIMENT_NAMES = {
    "vanilla": "g1_isaac",
    "custom": "g1",
    "clf": "g1",
    "ref_tracking": "g1",
    "stair": "g1",
    "clf_vdot": "g1",
    "height-scan-flat": "g1"
}

SIM_ENVIRONMENTS = {
    "vanilla": "custom-Isaac-Velocity-Flat-G1-Play-v0",
    "custom": "custom-Isaac-Velocity-Flat-G1-Play-v0",
    "clf": "G1-flat-ref-play",
    "ref_tracking": "G1-flat-ref-play",
    "clf_vdot": "G1-flat-ref-play",
    "stair": "G1-stair-play",
    "height-scan-flat": "G1-height-scan-flat-play",
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


def parse_sim_speed(value):
    try:
        # Split the input string by commas and convert each value to float
        return [float(x) for x in value.split(',')]
    except:
        raise argparse.ArgumentTypeError("Sim speed must be comma-separated floats (e.g. '1.0,0.0,0.0')")

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
        default=400,
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

    parser.add_argument(
        "--export_policy",
        action="store_true",
        default=False,
        help="Export the policy to ONNX and JIT formats."
    )

    parser.add_argument(
        "--sim_speed",
        type=parse_sim_speed,
        default=None,
        help="Simulation speed as comma-separated values for x,y,z velocities (e.g. '1.0,0.0,0.0')."
    )

    parser.add_argument(
        "--play_log_dir",
        type=str,
        default = None,
        help="export directory "
    )

    parser.add_argument(
        "--log_data",
        action="store_true",
        default=False,
        help="Log data during playback."
    )
    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_known_args()


def extract_reference_trajectory(env, log_vars):
    # Get the underlying environment by unwrapping
    unwrapped_env = env.unwrapped

    cfg_name = type(env.cfg).__name__

    ref = unwrapped_env.command_manager.get_term("hlip_ref")
    results = {}

    for var in log_vars:
        if hasattr(ref, var):
            val = getattr(ref, var)
            results[var] = val.clone() if isinstance(val, torch.Tensor) else val
        elif var in ref.metrics:
            results[var] = ref.metrics[var]
        elif var == "base_velocity":
            results[var] = unwrapped_env.command_manager.get_command("base_velocity")
        else:
            results[var] = None  # or raise an error/warning if you prefer

    return results


def find_latest_checkpoint(log_root_path):
    """Find the latest checkpoint in the given directory."""
    # Find all run directories
    run_dirs = glob.glob(os.path.join(log_root_path, "*"))
    if not run_dirs:
        return None, None
    
    # Get the latest run directory
    latest_run = max(run_dirs, key=os.path.getmtime)
    
    # Find all checkpoint files in the latest run
    checkpoint_files = glob.glob(os.path.join(latest_run, "model_*.pt"))
    if not checkpoint_files:
        return None, None
    
    # Get the latest checkpoint
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    checkpoint_num = int(os.path.basename(latest_checkpoint).split("_")[1].split(".")[0])
    run_name = os.path.basename(latest_run)
    
    return run_name, checkpoint_num


def main():
    args_cli, hydra_args = parse_args()
    
    if not args_cli.env_type:
        print("Please specify an environment type using --env_type")
        print("Available options:", list(ENVIRONMENTS.keys()))
        sys.exit(1)

    print("[DEBUG] Starting main function")
    # Set the task based on environment type
    args_cli.task = SIM_ENVIRONMENTS[args_cli.env_type]
    print(f"[DEBUG] Using task: {args_cli.task}")
    
    # Get experiment name (use override if provided, otherwise use default)
    experiment_name = args_cli.exp_name or EXPERIMENT_NAMES[args_cli.env_type]
    print(f"[DEBUG] Using experiment name: {experiment_name}")
    
    # always enable cameras to record video
    if args_cli.video:
        args_cli.enable_cameras = True

    # clear out sys.argv for Hydra
    sys.argv = [sys.argv[0]] + hydra_args

    print("[DEBUG] Launching Omniverse app")
    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    print("[DEBUG] Omniverse app launched")

    try:
        print("[DEBUG] Importing required modules")
        # Import necessary modules after app launch
        import gymnasium as gym
        import torch
        # from rsl_rl.runners import OnPolicyRunner
        from robot_rl.network.custom_policy_runner import CustomOnPolicyRunner
    
        from isaaclab.envs import (
            DirectMARLEnv,
            multi_agent_to_single_agent,
        )
        from isaaclab.utils.dict import print_dict
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
        from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
        import robot_rl.tasks  # noqa: F401
        print("[DEBUG] Modules imported successfully")

        # Configure PyTorch
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

        print("[DEBUG] Parsing configurations")
        # parse configuration
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs
        )

        if args_cli.sim_speed is not None:
            env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.sim_speed[0], args_cli.sim_speed[0])
            env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.sim_speed[1], args_cli.sim_speed[1])
            env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.sim_speed[2], args_cli.sim_speed[2])

        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
        print("[DEBUG] Configurations parsed")

        # specify directory for logging experiments
        if args_cli.env_type == "exo_hzd" or args_cli.env_type == "exo_hlip":
            log_root_path = os.path.join("logs", "exo_policies", args_cli.env_type, experiment_name)
        else:
            log_root_path = os.path.join("logs", "g1_policies", args_cli.env_type, experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[DEBUG] Log root path: {log_root_path}")
        
        # If no checkpoint is specified, find the latest one
        if not agent_cfg.load_run or not agent_cfg.load_checkpoint:
            print("[DEBUG] Finding latest checkpoint")
            latest_run, latest_checkpoint = find_latest_checkpoint(log_root_path)
            if latest_run and latest_checkpoint:
                print(f"[DEBUG] Found latest checkpoint: run={latest_run}, checkpoint={latest_checkpoint}")
                agent_cfg.load_run = latest_run
                agent_cfg.load_checkpoint = latest_checkpoint
            else:
                print("[ERROR] No checkpoints found in the specified directory")
                sys.exit(1)
        
        # Get checkpoint path from the training directory
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[DEBUG] Checkpoint path: {resume_path}")
        
        # Use the checkpoint directory for saving results
        if not args_cli.play_log_dir:
            play_log_dir = os.path.dirname(resume_path)
        else:
            play_log_dir = args_cli.play_log_dir
        
        print(f"[DEBUG] Play log directory: {play_log_dir}")

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
            print("[DEBUG] Setting up video recording")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[DEBUG] Loading model checkpoint from: {resume_path}")
        # load previously trained model
        ppo_runner = CustomOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)

        # obtain the trained policy for inference
        policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
  
        # Export policy if requested
        if args_cli.export_policy:
            print("[DEBUG] Exporting policy to ONNX and JIT formats")
            try:
                # version 2.3 onwards
                policy_nn = ppo_runner.alg.policy
            except AttributeError:
                # version 2.2 and below
                policy_nn = ppo_runner.alg.actor_critic

            # export policy to onnx/jit
            export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
            os.makedirs(export_model_dir, exist_ok=True)
            export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(
                policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
            )
            print(f"[DEBUG] Policy exported to {export_model_dir}")

        dt = env.unwrapped.step_dt


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
            "error_sw_x",
            "error_sw_y",
            "error_sw_z",
            "error_sw_roll",
            "error_sw_pitch",
            "error_sw_yaw",
            "error_com_x",
            "error_com_y",
            "error_com_z",
            "error_pelvis_roll",
            "error_pelvis_pitch",
            "error_pelvis_yaw" 
        ]
        
        # Setup logging
        logger = DataLogger(enabled=True, log_dir=play_log_dir, variables=log_vars)

        # reset environment
        obs, _ = env.get_observations()
        timestep = 0
        print("[DEBUG] Starting simulation loop")

        # viewer = env.unwrapped.scene.viewer

        # Choose the robot's prim path
        # robot_prim_path = "/World/robot"  # Or your robot's actual path

        # Set the camera to follow the robot
        # viewer.set_camera_follow(robot_prim_path)
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
                if args_cli.log_data:
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

        print("[DEBUG] Simulation loop ended")
        # close the simulator
        env.close()

        # Save all logged data
        if args_cli.log_data:
            logger.save()

            # Create plots directory and generate plots
            plot_dir = os.path.join(play_log_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            print(f"[DEBUG] Generating plots in directory: {plot_dir}")
            
            
            plot_trajectories(logger.data, save_dir=plot_dir)

    except Exception as e:
        print(f"[ERROR] An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure simulation app is closed
        if simulation_app is not None:
            simulation_app.close()
            print("[DEBUG] Simulation app closed")


if __name__ == "__main__":
    main()