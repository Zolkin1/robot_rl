# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
import robot_rl  # noqa: F401 — register gym environments
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip
from data_logger import DataLogger  # isort: skip
from play_plots import run_plots, PLOT_REGISTRY  # isort: skip
from export_parameters import export_policy_parameters  # isort: skip
from robot_rl.network.moe_network import MixtureOfExperts  # isort: skip

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_resolution", type=str, default="1920x1080",
    help='Recording resolution as "WxH" (e.g. "1920x1080", "2560x1440"). '
         "Overrides env_cfg.viewer.resolution.  Higher = sharper but slower.",
)
parser.add_argument(
    "--video_follow_robot", action="store_true", default=True,
    help="Lock the recording camera to the robot (origin_type=asset_root). "
         "Default on; pass --no-video_follow_robot to keep the static world camera.",
)
parser.add_argument("--no-video_follow_robot", dest="video_follow_robot", action="store_false")
parser.add_argument(
    "--video_eye", type=float, nargs=3, default=(0.0, 5.0, 0.8), metavar=("X", "Y", "Z"),
    help="Camera eye offset from the robot root when --video_follow_robot is set (m). "
         "Default is a side view: 5 m off the robot's +y axis at ~hip+0.5 height. "
         "Interpreted in world frame by default; pass --video_track_yaw to rotate the "
         "offset into the robot's heading frame (camera stays on the robot's left).",
)
parser.add_argument(
    "--video_lookat", type=float, nargs=3, default=(0.0, 0.0, 0.3), metavar=("X", "Y", "Z"),
    help="Camera lookat offset from the robot root when --video_follow_robot is set (m). "
         "Default aims at root_pos + 0.2 m (the robot's centroid for a G1 / humanoid). "
         "If the robot is missing from frame, this is the first knob to adjust.",
)
parser.add_argument(
    "--video_focal_length", type=float, default=24.0,
    help="USD camera focal length (mm) — controls zoom / FOV.  Smaller = wider angle. "
         "Default 24 mm (~74° horizontal FOV with the standard 35 mm aperture); "
         "drop to ~18 for very wide, raise to 35–50 for telephoto.",
)
parser.add_argument(
    "--video_env_index", type=int, default=0,
    help="Which env to follow when --video_follow_robot is set (default 0).",
)
parser.add_argument(
    "--video_track_yaw", action="store_true", default=False,
    help="Rotate the camera eye/lookat offsets into the robot's heading frame "
         "so the relative view (e.g. side view) stays consistent across turns. "
         "Default off — offsets are world-aligned.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--plots", type=str, default="default",
    help='Comma-separated plot names, "default", "all", or "none". '
         f"Available: {', '.join(['default', 'all', 'none', *PLOT_REGISTRY.keys()])}",
)
parser.add_argument("--plot_envs", type=int, default=1, help="Number of envs to generate plots for.")
parser.add_argument(
    "--plot_time_range", type=str, default=None,
    help='Restrict plots to a time window in seconds. Format: "start,end" — '
         'either end may be empty (e.g. "5,", ",10"). Requires logging.',
)
parser.add_argument("--max_steps", type=int, default=None, help="Exit after this many steps (None = run forever).")
parser.add_argument(
    "--train_mode",
    action="store_true",
    default=False,
    help="Run the policy as it behaves during training: actor/critic kept in torch .train() "
         "mode (dropout/batchnorm active) AND stochastic action sampling enabled. "
         "Default play is .eval() mode with deterministic actions.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# Check for installed RSL-RL version
installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # Patch activate_contact_sensors for recursive traversal (must be after SimulationApp)
        import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401

        # grab task name for checkpoint path
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        # handle deprecated configurations
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

        # set the environment seed
        # note: certain randomizations occur in the environment initialization so we set the seed here
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            # For distillation, allow overriding the experiment via CLI
            if agent_cfg.class_name == "DistillationRunner" and getattr(args_cli, "teacher_experiment", None):
                play_log_path = os.path.abspath(
                    os.path.join("logs", "rsl_rl", args_cli.teacher_experiment)
                )
            else:
                play_log_path = log_root_path
            resume_path = get_checkpoint_path(play_log_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # Apply video-recording knobs to the viewer cfg before env construction.
        # ``env_cfg.viewer.resolution`` is the offscreen-render product size the
        # ``rgb_array`` render path reads from (see manager_based_rl_env.py:286).
        # We also create a dedicated USD camera prim and point the render
        # product at it so the recording is decoupled from the GUI viewport
        # ``/OmniverseKit_Persp``.  The viewer cfg's built-in tracking
        # (``origin_type=asset_root``) drives the persp cam via
        # ``sim.set_camera_view`` → ``visualizer._set_viewport_camera``, which
        # writes to the visualizer's pinned camera path, *not* whatever
        # ``cam_prim_path`` we point the render product at.  So we set
        # ``origin_type="world"`` to disable that tracking and instead drive
        # the recording camera ourselves from the main loop using
        # ``isaacsim.core.utils.viewports.set_camera_view`` (which writes the
        # USD transform on the prim path we give it).  No render products are
        # ever created during training, so this whole branch is a no-op there.
        record_cam_state = None  # populated below if --video; consumed in the loop
        if args_cli.video:
            try:
                w_str, h_str = args_cli.video_resolution.lower().split("x")
                env_cfg.viewer.resolution = (int(w_str), int(h_str))
            except (ValueError, AttributeError) as exc:
                raise SystemExit(
                    f"--video_resolution must be 'WxH', got '{args_cli.video_resolution}': {exc}"
                )

            from pxr import UsdGeom
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            recording_cam_path = "/World/RecordingCam"
            cam_prim = stage.GetPrimAtPath(recording_cam_path)
            if not cam_prim.IsValid():
                cam_prim = UsdGeom.Camera.Define(stage, recording_cam_path).GetPrim()
            # Dial the focal length on the USD camera so the framing isn't
            # the default telephoto.  ``horizontalAperture`` stays at the
            # USD default (20.955 mm, 35 mm-equivalent); focal length is
            # the per-shot zoom knob.
            UsdGeom.Camera(cam_prim).GetFocalLengthAttr().Set(args_cli.video_focal_length)
            env_cfg.viewer.cam_prim_path = recording_cam_path
            # Disable the built-in asset tracker — it would move /OmniverseKit_Persp,
            # not our recording cam.  We drive the recording cam ourselves below.
            env_cfg.viewer.origin_type = "world"

            record_cam_state = {
                "prim_path": recording_cam_path,
                "eye_offset": np.asarray(args_cli.video_eye, dtype=float),
                "lookat_offset": np.asarray(args_cli.video_lookat, dtype=float),
                "env_index": args_cli.video_env_index,
                "follow_robot": args_cli.video_follow_robot,
                "track_yaw": args_cli.video_track_yaw,
            }
            print(
                f"[INFO] Video recording: {env_cfg.viewer.resolution[0]}x{env_cfg.viewer.resolution[1]}"
                f" via dedicated USD camera '{recording_cam_path}'"
                + (f", following robot in env {args_cli.video_env_index}"
                   if args_cli.video_follow_robot else ", static world camera")
            )

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # --- DEBUG: contact sensor ---
        for name, sensor in env.unwrapped.scene.sensors.items():
            if hasattr(sensor, 'body_names'):
                print(f"[{name}] bodies: {sensor.body_names}, num_bodies: {sensor.num_sensors}")
        # --- END DEBUG ---

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

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
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)

        # print parameter counts
        if isinstance(runner, DistillationRunner):
            student_params = sum(p.numel() for p in runner.alg.student.parameters())
            teacher_params = sum(p.numel() for p in runner.alg.teacher.parameters())
            print(f"Student Parameters: {student_params:,}")
            print(f"Teacher Parameters: {teacher_params:,}")
        else:
            actor_params = sum(p.numel() for p in runner.alg.actor.parameters())
            critic_params = sum(p.numel() for p in runner.alg.critic.parameters())
            print(f"Actor Parameters: {actor_params:,}")
            print(f"Critic Parameters: {critic_params:,}")
            print(f"Total Parameters: {actor_params + critic_params:,}")

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        if args_cli.train_mode:
            runner.alg.train_mode()
            print("[INFO] Running policy in torch .train() mode with stochastic actions.")

        # export the trained policy to JIT and ONNX formats
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        if version.parse(installed_version) >= version.parse("4.0.0"):
            # use the new export functions for rsl-rl >= 4.0.0
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None  # Not needed for rsl-rl >= 4.0.0
        else:
            # extract the neural network for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            # extract the normalizer
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            # export to JIT and ONNX
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        # Detect MoE architecture and set up gate weight capture
        _policy_model = runner.alg.student if isinstance(runner, DistillationRunner) else runner.alg.actor
        teacher_model = runner.alg.teacher if isinstance(runner, DistillationRunner) else None
        _moe_net = getattr(_policy_model, "mlp", None)
        is_moe = isinstance(_moe_net, MixtureOfExperts)
        if is_moe:
            print(f"[INFO] MoE detected: {_moe_net.num_experts} experts. Gate weights will be logged.")
            _last_gate_w: dict[str, torch.Tensor] = {}

            def _gate_hook(module, input, output):
                _last_gate_w["val"] = output.detach().clone()

            _moe_net.gate.register_forward_hook(_gate_hook)
            gate_weights_log: list[torch.Tensor] = []

        dt = env.unwrapped.step_dt

        # Set up data logging
        plot_names = [s.strip() for s in args_cli.plots.split(",")]
        logging_enabled = plot_names != ["none"]
        if logging_enabled:
            logger = DataLogger(env)
        env_ids = list(range(min(args_cli.plot_envs, env_cfg.scene.num_envs)))

        # reset environment
        obs = env.get_observations()

        # Export policy parameters (YAML with joint info, gains, obs terms, etc.)
        with torch.inference_mode():
            actions = policy(obs)
        export_policy_parameters(env, obs, actions, log_dir)

        timestep = 0
        # simulate environment
        try:
            while True:
                start_time = time.time()
                # Drive the dedicated recording camera before the step so
                # the next render (triggered inside env.step via
                # RecordVideo) reads a fresh transform.  Writes directly to
                # the USD prim via Isaac Sim's viewport util — render
                # products read the prim transform, no viewport_api needed.
                if record_cam_state is not None and record_cam_state["follow_robot"]:
                    robot = env.unwrapped.scene.articulations["robot"]
                    import warp as wp
                    env_idx = record_cam_state["env_index"]
                    root_pos_w = wp.to_torch(robot.data.root_pos_w)[
                        env_idx, :3
                    ].detach().cpu().numpy()
                    eye_off = record_cam_state["eye_offset"]
                    lookat_off = record_cam_state["lookat_offset"]
                    if record_cam_state["track_yaw"]:
                        # Rotate the offsets by the robot's yaw so e.g. a
                        # side-view (+y) stays on the robot's left even as
                        # the robot turns.
                        yaw = float(wp.to_torch(robot.data.heading_w)[env_idx].item())
                        c, s = np.cos(yaw), np.sin(yaw)
                        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                        eye_off = rot @ eye_off
                        lookat_off = rot @ lookat_off
                    eye_w = (root_pos_w + eye_off).tolist()
                    target_w = (root_pos_w + lookat_off).tolist()
                    import isaacsim.core.utils.viewports as _isaacsim_viewports
                    _isaacsim_viewports.set_camera_view(
                        eye=eye_w, target=target_w,
                        camera_prim_path=record_cam_state["prim_path"],
                    )

                # run everything in inference mode
                with torch.inference_mode():
                    # agent stepping
                    actions = policy(obs, stochastic_output=args_cli.train_mode)
                    teacher_actions = (
                        teacher_model(obs, stochastic_output=args_cli.train_mode)
                        if teacher_model is not None else None
                    )
                    # env stepping
                    obs, _, dones, _ = env.step(actions)
                    # reset recurrent states for episodes that have terminated
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)

                if logging_enabled:
                    logger.collect_step(env)
                    if teacher_model is not None:
                        logger.collect_distillation_actions(
                            student_actions=actions, teacher_actions=teacher_actions
                        )
                if is_moe and "val" in _last_gate_w:
                    gate_weights_log.append(_last_gate_w["val"])

                timestep += 1
                if args_cli.video and timestep >= args_cli.video_length:
                    break
                if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
                    break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            pass
        finally:
            # Generate plots from collected data
            if logging_enabled and logger.num_steps > 0:
                data, metadata = logger.finalize()
                # Inject MoE gate weights into data for plotting
                if is_moe and gate_weights_log:
                    data["gate_weights"] = torch.stack(gate_weights_log).cpu().numpy()
                    metadata["num_experts"] = _moe_net.num_experts
                plots_dir = os.path.join(log_dir, "plots")
                time_range = None
                if args_cli.plot_time_range:
                    parts = args_cli.plot_time_range.split(",")
                    if len(parts) != 2:
                        raise ValueError(
                            f"--plot_time_range expects 'start,end'; got {args_cli.plot_time_range!r}"
                        )
                    start = float(parts[0]) if parts[0].strip() else None
                    end = float(parts[1]) if parts[1].strip() else None
                    time_range = (start, end)
                run_plots(data, metadata, plots_dir, plot_names, env_ids, time_range=time_range)

            # close the simulator
            env.close()


if __name__ == "__main__":
    main()