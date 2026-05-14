# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Inspection script: place the robot at the end-of-previous-domain of a
"start skill", pre-load the velocity command's pending queue with a
"target skill", and let the trained policy play forward in slow motion
so the cross-fade transition is visible.  Loops the seed → play →
settle cycle so the operator can re-watch the same transition.

Single-file inspection tool; touches no library code.  See
:func:`seed_transition` for the bookkeeping that places the env in the
canonical "gate would fire on the next step" state.
"""

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
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

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(
    description=(
        "Inspect the learned skill-transition cross-fade.  Seeds the env "
        "at the end-of-previous-domain pose of --start_skill, pre-loads "
        "the pending queue with --target_skill, plays forward slowly, "
        "and loops."
    )
)
parser.add_argument("--task", type=str, default=None, required=True, help="Name of the task.")
parser.add_argument("--start_skill", type=str, default=None, required=True, help="Skill the env is in before the gate fires.")
parser.add_argument("--target_skill", type=str, default=None, required=True, help="Skill the pending queue commits to at the next gate fire.")
parser.add_argument("--slowmo", type=float, default=4.0, help="Wall-clock multiplier of sim_dt for slow-motion playback (1.0 = real time).")
parser.add_argument("--settle_seconds", type=float, default=2.0, help="After cross-fade saturation, hold this many wall-clock seconds before re-seeding.")
parser.add_argument("--gate_idx", type=int, default=0, help="Which gate's end-of-previous-domain phase to spawn at (mod num_gates of the start trajectory).")
parser.add_argument("--max_loops", type=int, default=20, help="Hard cap on loop iterations (default 20).")
parser.add_argument(
    "--plots", type=str, default="default",
    help='Comma-separated plot names, "default", "all", or "none". '
         f"Available: {', '.join(['default', 'all', 'none', *PLOT_REGISTRY.keys()])}",
)
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

installed_version = metadata.version("rsl-rl-lib")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Spawn a single inspection env and loop a forced skill transition."""
    with launch_simulation(env_cfg, args_cli):
        # Match play.py's contact-sensor patch.
        import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401

        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # Apply non-hydra CLI overrides; force single-env inspection mode.
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = 1
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # Cross-fade prerequisites — fail fast with a helpful message.
        if env_cfg.commands.traj_ref.contact_gate_window_frac is None:
            raise SystemExit(
                "This script requires the contact gate enabled.  Set "
                "env_cfg.commands.traj_ref.contact_gate_window_frac to a "
                "non-None value (e.g. 0.1) on the cfg you're playing."
            )
        if env_cfg.commands.traj_ref.transition_blend_end_phi <= 0.0:
            raise SystemExit(
                "This script needs the cross-fade enabled.  Set "
                "env_cfg.commands.traj_ref.transition_blend_end_phi > 0 "
                "(e.g. 0.25)."
            )
        if not env_cfg.commands.traj_ref.gate_skill_change_on_contact:
            raise SystemExit(
                "This script needs the pending-skill mechanism enabled.  Set "
                "env_cfg.commands.traj_ref.gate_skill_change_on_contact = True."
            )

        # Make sure trajectory-ref debug viz is on so the operator sees
        # the per-frame trajectory cylinders during the transition.
        env_cfg.commands.traj_ref.debug_vis = True

        # Disable the velocity cmd's normal mid-episode resampling so the
        # only skill change comes from this script's seeded pending queue.
        # Otherwise the play cfg's ``resampling_time_range=(2, 2)`` +
        # ``skill_transition_prob=1.0`` would force a new skill every
        # 2 s of settle, clobbering the target we just transitioned to.
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)

        # Resolve checkpoint exactly like play.py.
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
            if agent_cfg.class_name == "DistillationRunner" and getattr(args_cli, "teacher_experiment", None):
                play_log_path = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.teacher_experiment))
            else:
                play_log_path = log_root_path
            resume_path = get_checkpoint_path(play_log_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        log_dir = os.path.dirname(resume_path)
        env_cfg.log_dir = log_dir

        # Construct the env + policy.
        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            raise SystemExit("DirectMARL envs aren't supported by this inspection script.")
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise SystemExit(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # --- Resolve the multiskill cmd handles. -----------------------
        cmd = env.unwrapped.command_manager.get_term("traj_ref")
        cond_term = env.unwrapped.command_manager.get_term("base_velocity")
        mgr = cmd.manager
        device = env.unwrapped.device
        asset = env.unwrapped.scene[cmd.cfg.asset_name]

        # --- Skill-name → index lookups with friendly errors. ----------
        # Two index spaces are in play and they no longer have to match:
        #   * ``mgr.skill_name_to_idx`` indexes ``mgr.skills`` (the loaded
        #     trajectory subfolders — may be a superset of what the env
        #     actually samples, e.g. a ``stair_up/`` folder on a flat env).
        #   * ``cond_term._skill_list`` (== ``terrain.skill_list``) defines
        #     the values ``cond_term.skill_id`` can take.  We must write
        #     into that index space when seeding the velocity cmd.
        cond_skills = list(cond_term._skill_list)
        if args_cli.start_skill not in cond_skills:
            raise SystemExit(
                f"--start_skill '{args_cli.start_skill}' not in env's sampled skills: {cond_skills}"
            )
        if args_cli.target_skill not in cond_skills:
            raise SystemExit(
                f"--target_skill '{args_cli.target_skill}' not in env's sampled skills: {cond_skills}"
            )
        if args_cli.start_skill == args_cli.target_skill:
            raise SystemExit("--start_skill and --target_skill must differ.")

        start_skill_mgr_idx = mgr.skill_name_to_idx[args_cli.start_skill]
        target_skill_mgr_idx = mgr.skill_name_to_idx[args_cli.target_skill]
        start_skill_cond_idx = cond_skills.index(args_cli.start_skill)
        target_skill_cond_idx = cond_skills.index(args_cli.target_skill)
        start_skill = mgr.skills[start_skill_mgr_idx]
        target_skill = mgr.skills[target_skill_mgr_idx]

        # Pick the closest-velocity pair across the two skills.
        start_vel_cond = start_skill.conditioning_tensor[:, :3]   # [Ns, 3]
        target_vel_cond = target_skill.conditioning_tensor[:, :3]  # [Nt, 3]
        dist = torch.cdist(start_vel_cond, target_vel_cond)
        flat = int(dist.argmin())
        si, ti = flat // dist.shape[1], flat % dist.shape[1]
        start_traj_idx = start_skill.traj_indices[si]
        target_traj_idx = target_skill.traj_indices[ti]
        start_vel = start_vel_cond[si].to(device)
        target_vel = target_vel_cond[ti].to(device)
        print(
            f"[INFO] start_skill='{args_cli.start_skill}' traj={start_traj_idx} vel={tuple(start_vel.tolist())} "
            f"→ target_skill='{args_cli.target_skill}' traj={target_traj_idx} vel={tuple(target_vel.tolist())} "
            f"(L2 vel distance {float(dist.min()):.4f})"
        )

        # End-of-previous-domain phase for the start traj.
        num_gates = int(mgr._num_gates_per_traj[start_traj_idx].item())
        if num_gates == 0:
            raise SystemExit(
                f"Trajectory {start_traj_idx} for skill '{args_cli.start_skill}' has no contact gates; "
                "there's no end-of-previous-domain to seed at."
            )
        gate_idx = args_cli.gate_idx % num_gates
        gate_phi = float(mgr._gate_phi_table[start_traj_idx, gate_idx].item())
        end_phi = (gate_phi - mgr._EPS_PHI) % 1.0
        print(
            f"[INFO] seeding at end-of-previous-domain phase={end_phi:.4f} "
            f"(gate {gate_idx} of {num_gates}, gate_phi={gate_phi:.4f})"
        )

        # --- Trajectory base/joint index maps (precompute once). -------
        # Base frame indices: the trajectory cmd's ``ref_frames`` list
        # holds the ref frames the active contact body cycles through.
        # For a seed-only purpose we just need the *first* ref frame's
        # pos/ori in the trajectory output layout; the active ref frame
        # at phase=end_phi will be the contact body of the domain about
        # to end, which the trajectory generates outputs for.  Use the
        # ``base_frame_name`` from the reset-event cfg if available;
        # otherwise default to "pelvis_link" (matches the G1 cfg).
        base_frame_name = "pelvis_link"
        pos_indices = [
            cmd.ordered_pos_output_names.index(f"{base_frame_name}:pos_{a}") for a in ("x", "y", "z")
        ]
        ori_indices = [
            cmd.ordered_pos_output_names.index(f"{base_frame_name}:ori_{a}") for a in ("x", "y", "z", "w")
        ]
        lin_vel_indices = [
            cmd.ordered_vel_output_names.index(f"{base_frame_name}:pos_{a}") for a in ("x", "y", "z")
        ]
        ang_vel_indices = [
            cmd.ordered_vel_output_names.index(f"{base_frame_name}:ori_{a}") for a in ("x", "y", "z")
        ]
        joint_pos_idx_in_traj = [
            cmd.ordered_pos_output_names.index(f"joint:{j}") for j in asset.joint_names
        ]
        joint_vel_idx_in_traj = [
            cmd.ordered_vel_output_names.index(f"joint:{j}") for j in asset.joint_names
        ]

        env_ids = torch.tensor([0], device=device, dtype=torch.long)

        def seed_transition() -> None:
            """Place env 0 at end-of-previous-domain of start_skill and
            prime the pending queue with target_skill.

            Under the new bucket-driven skill model, the trajectory cmd's
            per-step state machine reads ``vel_target_b`` and re-derives
            the desired skill each tick.  To make the seeded pending
            "stick" we must place ``vel_target_b`` inside the *target*
            bucket — that way the state machine sees
            ``desired == pending`` and is idempotent.  Semantically this
            represents the instant just *after* the velocity command's
            ramp crossed the bucket boundary, end-of-previous-domain of
            the old skill, gate about to fire.
            """
            # 1. Velocity cmd: sampled bucket = target (informational);
            #    ``vel_target_b`` snapped into the target bucket so the
            #    state machine sees the post-crossing state.
            cond_term.sampled_skill_id[env_ids] = target_skill_cond_idx
            cond_term.vel_target_sampled_b[env_ids, :] = target_vel.unsqueeze(0)
            cond_term.vel_target_b[env_ids, :] = target_vel.unsqueeze(0)
            cond_term.heading_target[env_ids] = 0.0

            # 2. Trajectory cmd: active skill is still the *start* skill
            #    (we're at end-of-previous-domain of the old trajectory);
            #    pending = target (queued, awaiting gate fire).
            cmd.skill_id[env_ids] = start_skill_cond_idx
            cmd.pending.enqueue(
                env_ids,
                torch.tensor([target_skill_cond_idx], device=device, dtype=torch.long),
            )

            # 3. Refresh the conditioner ``vel_command_b`` from the
            #    just-set ``vel_target_b`` — the trajectory manager's
            #    ``_select_trajectories`` reads ``cond_term.command``
            #    (alias for ``vel_command_b``), so without this the cache
            #    would rebuild against a stale velocity and pick the
            #    wrong start trajectory.
            cond_term._update_command()

            # 4. Rebuild the manager's trajectory cache under the freshly-set
            #    ``cmd.skill_id`` + fresh conditioner; eagerly resolve so
            #    the assignment is valid for the ``get_desired_outputs``
            #    call below.
            mgr.invalidate_cache()
            mgr.get_current_trajectory_indices()

            # 4. Set the trajectory phase deterministically.
            phase = torch.tensor([end_phi], device=device)
            mgr.set_phase(phase, env_ids)

            # 5. Read the desired trajectory output at this phase.
            cmd.get_desired_outputs(phase, env_ids=env_ids)
            y_des = cmd.y_des[env_ids]           # [1, P]
            dy_des = cmd.dy_des[env_ids]         # [1, V]

            # 6. Compose the world-frame robot pose.  For flat-terrain
            #    skills the ref_frame_offset / stair_offset are zero,
            #    but apply them anyway for robustness.
            domain_idx = mgr.get_current_domains(phase, env_ids=env_ids)
            traj_idx = mgr.get_current_trajectory_indices()[env_ids]
            ref_offset = mgr.data["ref_frame_offset"][traj_idx, domain_idx]      # [1, 3]
            stair_offset = mgr.data["origin_relative_to_stair_center"][traj_idx]  # [1, 3]
            spawn_offset = ref_offset + stair_offset

            base_pos_rel = y_des[:, pos_indices]
            base_ori_quat_w = y_des[:, ori_indices]
            base_pos_w = base_pos_rel + env.unwrapped.scene.env_origins[env_ids] + spawn_offset
            base_pose = torch.cat([base_pos_w, base_ori_quat_w], dim=-1)

            base_vel = torch.cat(
                [dy_des[:, lin_vel_indices], dy_des[:, ang_vel_indices]], dim=-1
            )

            joint_pos = y_des[:, joint_pos_idx_in_traj]
            joint_vel = dy_des[:, joint_vel_idx_in_traj]

            # 7. Anchor the reference-frame pose at env_origin + spawn_offset
            #    (identity quat) so the trajectory cmd's next compute reads
            #    the right anchor.
            cmd.ref_poses[env_ids, :3] = env.unwrapped.scene.env_origins[env_ids] + spawn_offset
            cmd.ref_poses[env_ids, 3:6] = 0.0
            cmd.ref_poses[env_ids, 6] = 1.0

            # 8. Write robot state to sim.
            asset.write_root_pose_to_sim_index(root_pose=base_pose, env_ids=env_ids)
            asset.write_root_link_velocity_to_sim_index(root_velocity=base_vel, env_ids=env_ids)
            asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
            asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

            # 9. Clear cross-fade state + grace timer so the loop starts
            #    from a clean slate.
            cmd.transition.clear(env_ids)
            cmd.time_since_traj_change_s[env_ids] = 0.0

            # 10. Make sure the debug viz is on (cfg may or may not have set it).
            cmd.set_debug_vis(True)

        sim_dt = env.unwrapped.step_dt
        target_step_wallclock = args_cli.slowmo * sim_dt
        obs = env.get_observations()

        # --- Data logging (mirrors play.py). ----------------------------
        # Reuses the same DataLogger / run_plots pipeline so the standard
        # play plots (joint pos, contact, cmd vs. actual, etc.) come out
        # the same way regardless of whether the env was driven by random
        # commands (play.py) or by this script's forced transitions.
        plot_names = [s.strip() for s in args_cli.plots.split(",")]
        logging_enabled = plot_names != ["none"]
        logger = DataLogger(env) if logging_enabled else None
        plot_env_ids = [0]  # single inspection env

        loops_completed = 0
        try:
            # All seed + step work runs inside ``torch.inference_mode``.
            # ``env.step`` (which the play loop runs under inference mode)
            # marks the command-cmd's per-env buffers as inference
            # tensors; subsequent seed-time in-place writes outside
            # inference mode raise.  Keeping the whole loop in this
            # context avoids that without re-cloning the buffers.
            with torch.inference_mode():
                while loops_completed < args_cli.max_loops:
                    print(
                        f"\n[INFO] === loop {loops_completed + 1}/{args_cli.max_loops}: "
                        f"{args_cli.start_skill} → {args_cli.target_skill} ==="
                    )
                    seed_transition()
                    # Force an observation read so the policy sees the
                    # seeded state on its first step.
                    obs = env.get_observations()

                    settle_started = None
                    while True:
                        step_start = time.time()
                        actions = policy(obs)
                        obs, _, _, _ = env.step(actions)

                        if logger is not None:
                            logger.collect_step(env)

                        # Throttle wall-clock so each step takes
                        # ``slowmo * sim_dt`` seconds.
                        elapsed = time.time() - step_start
                        remaining = target_step_wallclock - elapsed
                        if remaining > 0:
                            time.sleep(remaining)

                        if not bool(cmd.transition.active[0].item()):
                            if settle_started is None:
                                settle_started = time.time()
                            elif time.time() - settle_started >= args_cli.settle_seconds:
                                break
                        else:
                            settle_started = None

                    loops_completed += 1
        except KeyboardInterrupt:
            pass
        finally:
            if logger is not None and logger.num_steps > 0:
                data, metadata = logger.finalize()
                plots_dir = os.path.join(log_dir, "plots_skill_transition")
                run_plots(data, metadata, plots_dir, plot_names, plot_env_ids)
            env.close()


if __name__ == "__main__":
    main()
