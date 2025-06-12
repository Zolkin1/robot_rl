import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher
import cli_args

# Environment names
ENVIRONMENTS = {
    "vanilla": "custom-Isaac-Velocity-Flat-G1-v0",
    "custom": "G1-flat-vel",
    "ref_tracking": "G1-flat-ref-tracking",
    "clf": "G1-flat-clf",
    "stair": "G1-stair",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train RL policies for different environments.")
    parser.add_argument("--env_type", type=str, choices=list(ENVIRONMENTS.keys()), 
                       help="Type of environment to train on (vanilla/custom/clf)")
    parser.add_argument("--video", action="store_true", default=False, 
                       help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, 
                       help="Length of the recorded video (in steps).")
    parser.add_argument("--video_interval", type=int, default=2000, 
                       help="Interval between video recordings (in steps).")
    parser.add_argument("--num_envs", type=int, default=None, 
                       help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, 
                       help="Seed used for the environment")
    parser.add_argument("--max_iterations", type=int, default=None, 
                       help="RL Policy training iterations.")
    parser.add_argument("--distributed", action="store_true", default=False, 
                       help="Run training with multiple GPUs or nodes.")
    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_known_args()

def main():
    args_cli, hydra_args = parse_args()
    
    if not args_cli.env_type:
        print("Please specify an environment type using --env_type")
        print("Available options:", list(ENVIRONMENTS.keys()))
        sys.exit(1)

    # Set the task based on environment type
    args_cli.task = ENVIRONMENTS[args_cli.env_type]
    
    # always enable cameras to record video
    if args_cli.video:
        args_cli.enable_cameras = True

    sys.argv = [sys.argv[0]] + hydra_args
    
    # Configure livestream properly
    # args_cli.livestream = 2
    args_cli.headless = True 
    # clear out sys.argv for Hydra
    

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import necessary modules after app launch
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_pickle, dump_yaml
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config
    import robot_rl.tasks  # noqa: F401

    # Configure PyTorch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def train(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, 
             agent_cfg: RslRlOnPolicyRunnerCfg):
        """Train with RSL-RL agent."""
        # Override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
        )

        # Set the environment seed
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # Multi-gpu training configuration
        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
            agent_cfg.device = f"cuda:{app_launcher.local_rank}"
            seed = agent_cfg.seed + app_launcher.local_rank
            env_cfg.seed = seed
            agent_cfg.seed = seed

        # Create organized directory structure for logging

        base_log_path = os.path.join("logs", "g1_policies", args_cli.env_type)
        log_root_path = os.path.join(base_log_path, agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        
        # Create timestamp-based run directory
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg.run_name:
            log_dir += f"_{agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        # Create environment
        if hasattr(env_cfg, "__prepare_tensors__") and callable(getattr(env_cfg, "__prepare_tensors__")):
            env_cfg.__prepare_tensors__()
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # Convert to single-agent if needed
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)


        # Handle resume path
        # agent_cfg.resume = True
        # resume_path = "/home/amy/gitrepo/robot_rl/logs/g1_policies/clf/g1/2025-06-10_15-50-55/model_4400.pt"
        # agent_cfg
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        # Setup video recording if enabled
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # Wrap environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        # Create and configure runner
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
        runner.add_git_repo_to_log(__file__)

        # Load checkpoint if resuming
        if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)

        # Save configurations
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

        # Run training
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

        # Cleanup
        env.close()

    # Run training
    train()
    # Close sim app
    simulation_app.close()

if __name__ == "__main__":
    main()









