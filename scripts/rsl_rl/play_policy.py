import argparse
import os
import sys
import time
import pickle
import glob
import numpy as np

from isaaclab.app import AppLauncher
import cli_args
import torch

# Import plot_trajectories functions
from plot_trajectories import plot_trajectories, compute_and_save_stats
from train_policy import ENVIRONMENTS, EXPERIMENT_NAMES
# Experiment names mapping for different environments

SIM_ENVIRONMENTS = {
    "vanilla": "G1-vanilla-walking",
    "lip_clf": "G1-lip-ref-play",

    "walking_clf": "G1-walking-clf-play",
    "walking_clf_sym": "G1-walking-clf-play",

    "running_clf": "G1-running-clf-play",
    "running_clf_sym": "G1-running-clf-play",
    "running_clf_sym_exp": "G1-running-clf-experiment",

    "waving_clf": "G1-waving-clf-play",

    "bow_forward_clf": "G1-bow_forward-clf-play",
    "bow_forward_clf_sym": "G1-bow_forward-clf-symmetric",    # TODO: make this a play

    "bend_up_clf_sym": "G1-bend_up-clf-play",

    "double_integrator": "double-integrator",
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
        default=False,
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
        default=2,
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

    parser.add_argument(
        "--paper",
        action="store_true",
        default=False,
        help="Generate paper-ready figures with LaTeX fonts."
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for computing discounted return plots (default: 0.99)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility. Use -1 for a random seed.",
    )
    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_known_args()


def extract_reference_trajectory(env, log_vars, command_name):
    # Get the underlying environment by unwrapping
    unwrapped_env = env.unwrapped

    cfg_name = type(env.cfg).__name__

    ref = unwrapped_env.command_manager.get_term(command_name)
    results = {}

    for var in log_vars:
        if hasattr(ref, var):
            val = getattr(ref, var)
            results[var] = val.clone() if isinstance(val, torch.Tensor) else val
        elif var in ref.metrics:
            results[var] = ref.metrics[var]
        elif var == "base_velocity":
            results[var] = unwrapped_env.command_manager.get_command("base_velocity").clone()
        elif var == "axis_names":
            # Extract axis names from end effector config
            if hasattr(ref, 'ee_config') and hasattr(ref.ee_config, 'axis_names'):
                results[var] = ref.ee_config.axis_names
            elif hasattr(ref, 'gait_config') and hasattr(ref.gait_config, '_gait_cache'):
                results[var] = ref.gait_config._gait_cache[list(ref.gait_config._gait_cache.keys())[0]].axis_names
            else:
                raise ValueError("[Extract Reference] Could not find the axis name!")
        elif var in ("v_log", "phi_keys"):
            # Get v_log and phi_keys from manager's get_v_log() method
            if "v_log" not in results:  # Only call once for both variables
                v_log, phi_keys = ref.manager.get_v_log()
                results["v_log"] = v_log.clone() if isinstance(v_log, torch.Tensor) else v_log
                results["phi_keys"] = phi_keys.clone() if isinstance(phi_keys, torch.Tensor) else phi_keys
        else:
            raise ValueError(f"[Extract Reference] No variable matching the given name [{var}] found in the command!")
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

# Before main
args_cli, hydra_args = parse_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

print("[DEBUG] Launching Omniverse app")
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

def main():
    # args_cli, hydra_args = parse_args()
    
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

    # Import necessary modules after app launch
    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner, DistillationRunner
    # from robot_rl.network.custom_policy_runner import CustomOnPolicyRunner

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

    if args_cli.sim_speed is not None and env_cfg.commands is not None:
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (args_cli.sim_speed[0], args_cli.sim_speed[0])
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (args_cli.sim_speed[1], args_cli.sim_speed[1])
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (args_cli.sim_speed[2], args_cli.sim_speed[2])

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = agent_cfg.seed
    print("[DEBUG] Configurations parsed")

    # specify directory for logging experiments
   
    base_log_path = os.path.join("logs", "g1_policies", EXPERIMENT_NAMES[args_cli.env_type])
    env_type_base = args_cli.env_type.split("_exp")[0] if "_exp" in args_cli.env_type else args_cli.env_type
    log_root_path = os.path.join(base_log_path, env_type_base)
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
    
    # Anchor the checkpoint regex to avoid partial matches (e.g. model_15 matching model_150)
    if isinstance(agent_cfg.load_checkpoint, int):
        agent_cfg.load_checkpoint = rf"model_{agent_cfg.load_checkpoint}\.pt$"
    elif isinstance(agent_cfg.load_checkpoint, str) and not agent_cfg.load_checkpoint.endswith("$"):
        if not agent_cfg.load_checkpoint.endswith(".pt"):
            agent_cfg.load_checkpoint = agent_cfg.load_checkpoint + r"\.pt$"
        else:
            agent_cfg.load_checkpoint = agent_cfg.load_checkpoint + r"$"

    # Get checkpoint path from the training directory
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[DEBUG] Checkpoint path: {resume_path}")
    
    # Use the checkpoint directory for saving results
    if not args_cli.play_log_dir:
        ckpt_name = os.path.basename(resume_path).replace('.pt', '')
        if args_cli.env_type == "double_integrator":
            play_log_dir = os.path.join(os.path.dirname(resume_path), ckpt_name)
        else:
            play_log_dir = os.path.dirname(resume_path)
    else:
        play_log_dir = args_cli.play_log_dir
    
    print(f"[DEBUG] Play log directory: {play_log_dir}")

    # create isaac environment
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
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # TODO: Delete
    # ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Export policy if requested
    if args_cli.export_policy:
        if runner.alg.policy.__class__.__name__ == "ActorCriticCNN":
            from robot_rl.network.exporter import export_policy_as_jit, export_policy_as_onnx
          

        print("[DEBUG] Exporting policy to ONNX and JIT formats")
        try:
            # version 2.3 onwards
            policy_nn = runner.alg.policy
        except AttributeError:
            # version 2.2 and below
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export policy to onnx/jit
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        os.makedirs(export_model_dir, exist_ok=True)
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx"
        )
        print(f"[DEBUG] Policy exported to {export_model_dir}")

    dt = env.unwrapped.step_dt

    # Dynamically generate log variables based on the command type
    if args_cli.env_type == "double_integrator":
        log_vars = ['v', 'vdot']
    else:
        log_vars = [
            'y_des',
            'dy_des',
            'base_velocity',
            'y_act',
            'dy_act',
            'v',
            'vdot',
            'ordered_pos_output_names', # TODO: Use these
            'ordered_vel_output_names',
            'current_domain',
            'phasing_var',
            # 'v_log',
            # 'phi_keys',
            # 'CLF_EMA_0',
            # 'CLF_EMA_1',
            # 'CLF_EMA_2',
            # 'CLF_EMA_3',
            # 'CLF_EMA_4',
            # 'CLF_EMA_5',
            # 'CLF_EMA_6',
            # 'CLF_EMA_7',
            # 'CLF_EMA_8',
            # 'CLF_EMA_9',
            # 'CLF_EMA_10',
            # 'domain_durations',
            # 'gait_indices',
        ]

    # Get the command term to determine what type of trajectory we're using
    if args_cli.env_type == "double_integrator":
        command_name = "traj_ref"
    elif "lip" in args_cli.env_type:
        command_name = "hlip_ref"
    elif "clf" in args_cli.env_type:
        command_name = "traj_ref"
    elif args_cli.env_type == "vanilla":
        raise ValueError(f"Need to implement vanilla in the play policy!")
    else:
        raise ValueError(f"No valid command name for {args_cli.env_type}")

    # Setup logging (include actions/joint_names separately since they don't come from extract_reference_trajectory)
    extra_vars = ['reward', 'action_targets', 'joint_pos', 'applied_torque', 'joint_names']
    if args_cli.env_type == "double_integrator":
        extra_vars.extend(['joint_vel', 'clf_reward'])
    logger = DataLogger(enabled=True, log_dir=play_log_dir, variables=log_vars + extra_vars)

    # reset environment
    obs = env.get_observations()
    action = policy(obs)

    # Export information to a yaml
    from export_parameters import export_policy_parameters
    export_policy_parameters(env, obs, action, play_log_dir)

    timestep = 0
    print("[DEBUG] Starting simulation loop")

    obs = env.get_observations()
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
                data = extract_reference_trajectory(env, log_vars, command_name)
                robot = env.unwrapped.scene.articulations["robot"]
                data['reward'] = reward.clone()
                # Use the appropriate action term name
                action_term_name = "force" if args_cli.env_type == "double_integrator" else "joint_pos"
                action_term = env.unwrapped.action_manager.get_term(action_term_name)
                data['action_targets'] = action_term._processed_actions.clone()
                data['joint_pos'] = robot.data.joint_pos.clone()
                data['applied_torque'] = robot.data.applied_torque.clone()
                data['joint_names'] = list(robot.data.joint_names)
                if args_cli.env_type == "double_integrator":
                    data['joint_vel'] = robot.data.joint_vel.clone()
                    reward_mgr = env.unwrapped.reward_manager
                    clf_idx = reward_mgr.active_terms.index("clf_reward")
                    data['clf_reward'] = reward_mgr._step_reward[:, clf_idx].clone()
                logger.log_from_dict(data)

        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
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
        
        # Determine trajectory type based on command type
        if args_cli.env_type == "double_integrator":
            from plot_double_integrator import plot_double_integrator
            eta = 10
            delta = 0.99
            lam = 0.02
            c2 = 2.732050657272339
            c1 = 0.7320508360862732
            creg = 0
            sigma = 0.6830136775970459
            cbar = 0.1    # TODO: Fix
            zetap = eta/(sigma**2)
            zetam = np.exp(-cbar/sigma**2) * eta/sigma**2
            print(f"[DEBUG] zetam = {zetam}")
            b = np.sqrt( (zetap + creg)*c2/(zetam * c1 * (1 - delta*(1-lam))) )
            qc = (1/delta)*(1 - (1 - delta * (1-lam))*zetam/(zetap + creg) )
            print(f"[DEBUG] qc = {qc}")
            plot_double_integrator(logger.data, save_dir=plot_dir, dt=dt, gamma=args_cli.gamma, a=qc, b=b)
        else:
            plot_trajectories(logger.data, save_dir=plot_dir, trajectory_type="end_effector", paper=args_cli.paper)
            compute_and_save_stats(logger.data, save_dir=plot_dir)

    # Ensure simulation app is closed
    if simulation_app is not None:
        simulation_app.close()
        print("[DEBUG] Simulation app closed")


if __name__ == "__main__":
    main()