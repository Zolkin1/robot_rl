# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import csv
from pathlib import Path
from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--csv_out", type=Path, default=Path("amber_joint_log.csv"),
                    help="Path to the CSV file that will store joint positions.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR , ISAACLAB_NUCLEUS_DIR
# print("________________________________________________________")
# print(ISAACLAB_NUCLEUS_DIR)

import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg

import omni.physics.tensors as physx
import omni.usd
from pxr import Gf, UsdPhysics, UsdGeom

STIFFNESS = 500
DAMPING = 25
# --- AMBER ROBOT CONFIGURATION ---
ROBOT_ASSETS_AMBER = "/home/s-ritwik/src/robot_rl/robot_assets/amber5/amber"

AMBER_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_ASSETS_AMBER}/amber_test.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,

        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.7),
        joint_pos={
            "q1_left": 0.0,
            "q2_left": 0.0,
            "q1_right": 0.0,
            "q2_right": 0.0,
            # "base_link_to_base_link2": 0,
            # "base_link2_to_base_link3": 0,
            # "base_link3_to_torso": 0,
            # "fixed":0,

        },
        joint_vel={
            "q1_left":   0.0,
            "q2_left":   0.0,
            "q1_right":  0.0,
            "q2_right":  0.0,
            # "base_link_to_base_link2": 0,
            # "base_link2_to_base_link3": 0,
            # "base_link3_to_torso": 0,
            # "fixed":0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "left_thigh_act": ImplicitActuatorCfg(
            joint_names_expr=["q1_left"],
            effort_limit_sim=400.0,
            velocity_limit_sim=4.0,
            stiffness=STIFFNESS,
            damping=DAMPING,
        ),
        "left_shin_act": ImplicitActuatorCfg(
            joint_names_expr=["q2_left"],
            effort_limit_sim=400.0,
            velocity_limit_sim=4.0,
            stiffness=STIFFNESS,
            damping=DAMPING,
        ),
        "right_thigh_act": ImplicitActuatorCfg(
            joint_names_expr=["q1_right"],
            effort_limit_sim=400.0,
            velocity_limit_sim=4.0,
            stiffness=STIFFNESS,
            damping=DAMPING,
        ),
        "right_shin_act": ImplicitActuatorCfg(
            joint_names_expr=["q2_right"],
            effort_limit_sim=400.0,
            velocity_limit_sim=5.0,
            stiffness=STIFFNESS,
            damping=DAMPING,
        ),
    },
)
# ---------------------------------

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    # robot
    # Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    # Dofbot = DOFBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Dofbot")
    Amber = AMBER_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Amber")
    contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Amber/amber3_PF/torso",
            update_period=0.0,       # every sim step
            history_length=1,        # only current step
            debug_vis=False,
        )

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    amber = scene["Amber"]
    device = amber.data.default_root_state.device
    n_envs = args_cli.num_envs
    

    # ──────────────── CSV SET-UP ────────────────
    joint_names = amber.data.joint_names          # list[str]  length = n_joints
    csv_path    = args_cli.csv_out.expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    csv_fh  = open(csv_path, "w", newline="")
    writer  = csv.writer(csv_fh)
    writer.writerow(["step", "sim_time", "env_id", *joint_names])

    # make sure we always close the file, even on Ctrl-C
    try:
        last_reset_step = torch.full(
        (n_envs,), -1_000_000, dtype=torch.int32, device=device
        )
        COOLDOWN = 10
        while simulation_app.is_running():
            # reset every 500 steps
            # if count % 200 == 0:
            #     count = 0
            #     scene.reset()

            #     # Reset Amber
            #     root = scene["Amber"].data.default_root_state.clone()
            #     root[:, :3] += scene.env_origins
            #     scene["Amber"].write_root_pose_to_sim(root[:, :7])
            #     scene["Amber"].write_root_velocity_to_sim(root[:, 7:])
            #     scene["Amber"].write_joint_state_to_sim(
            #         scene["Amber"].data.default_joint_pos.clone(),
            #         scene["Amber"].data.default_joint_vel.clone(),
            #     )
            #     scene.write_data_to_sim()
            #     sim.step()
            #     scene.update(sim_dt)

            #     print("[INFO]: Resetting all robots state...")

            # 1) Per-joint randomness: set to 0.0 if you want that joint fixed at default.
            random_scales = {
                # "base_link_to_base_link2": 0.0,   # torso yaw
                # "base_link2_to_base_link3": 0.0,  # prismatic Z (usually left unconstrained here)
                # "base_link3_to_torso":      0.0,  # prismatic X
                "q1_left":   1.3,
                "q2_left":   1.4,
                "q1_right":  1.2,
                "q2_right":  0.6,
                # "prism_z": 1.0,
            }

            amber = scene["Amber"]

            # 2) Grab defaults: shape (n_envs, n_joints)
            default_all = amber.data.default_joint_pos.clone()  
            joint_names = amber.data.joint_names     # list of length n_joints
            n_envs, n_joints = default_all.shape

            # 3) Build a zero-tensor of the same shape, then fill in the joints you want
            random_offsets = torch.zeros_like(default_all)
            for joint_name, scale in random_scales.items():
                if scale > 0.0:
                    try:
                        idx = joint_names.index(joint_name)
                    except ValueError:
                        continue  # joint not found—skip
                    # for each env, one random sample
                    random_offsets[:, idx] = scale * torch.randn(n_envs)

            # 4) Sum defaults + offsets → your target
            amber_target = default_all + random_offsets

            # 5) Send to sim
            amber.set_joint_position_target(amber_target)

            # ─────────────── LOG JOINT POSITIONS ───────────────
            # (n_envs, n_joints) → iterate so each env gets its own CSV row
            cur_pos = amber.data.joint_pos.cpu().numpy()
            for env_id in range(n_envs):
                writer.writerow([count, sim_time, env_id, *cur_pos[env_id]])
            # flush every ~100 steps so OS buffer is persisted
            if count % 100 == 0:
                csv_fh.flush()


            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            sim_time += sim_dt
            count += 1
            # ---- check for torso–ground contact ----
            # amber = scene["Amber"]
            # contact_forces is (n_envs, n_sensors), ours has one sensor

            forces = scene["contact_forces"].data.net_forces_w          # if any env has non-zero contact force:
            fallen = (forces.abs().sum(dim=(1,2)) > 0.0)
            to_reset = fallen & ((count - last_reset_step) > COOLDOWN)

            if to_reset.any():
                default_root = amber.data.default_root_state.clone()
                default_root[:, :3] += scene.env_origins

                # root pose / velocity
                root_state = amber.data.root_state_w.clone()
                root_state[to_reset] = default_root[to_reset]
                amber.write_root_pose_to_sim(root_state[:, :7])
                amber.write_root_velocity_to_sim(root_state[:, 7:])

                # joint state (masked)
                cur_jpos = amber.data.joint_pos.clone()
                cur_jvel = amber.data.joint_vel.clone()
                cur_jpos[to_reset] = amber.data.default_joint_pos[to_reset]
                cur_jvel[to_reset] = amber.data.default_joint_vel[to_reset]
                amber.write_joint_state_to_sim(cur_jpos, cur_jvel)

                scene.write_data_to_sim()
                sim.step(); scene.update(sim_dt)     # flush one frame

                last_reset_step[to_reset] = count
                continue
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C detected – finishing up, CSV saved at:", csv_path)
    finally:
        csv_fh.close()



# def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     sim_dt = sim.get_physics_dt()
#     count = 0

#     amber      = scene["Amber"]
#     device     = amber.data.default_root_state.device
#     n_envs     = args_cli.num_envs

#     # ----- per-env book-keeping ------------------------------------------------
#     last_reset_step   = torch.full((n_envs,), -1_000_000, dtype=torch.int32, device=device)
#     COOLDOWN_FRAMES   = 20                       # after a reset, wait before another is allowed

#     fall_ctr          = torch.zeros(n_envs, dtype=torch.int32, device=device)
#     WAIT_FRAMES       = 5                        # must lie on ground this many frames

#     GRACE_FRAMES      = 30                        # frames with zero PD error & no random kicks
#     skip_random_until = torch.zeros(n_envs, dtype=torch.int32, device=device)

#     # --------------------------------------------------------------------------
#     while simulation_app.is_running():

#         # ───────────────────── generate joint targets ────────────────────────
#         random_scales = {"q1_left": 1.3, "q2_left": 1.4, "q1_right": 1.2, "q2_right": 0.6}
#         # random_scales = {"q1_left": 0.0, "q2_left": 0.0, "q1_right": 0.0, "q2_right": 0.0}

#         default_all = amber.data.default_joint_pos.clone()          # (n_envs, n_joints)
#         joint_names = amber.data.joint_names
#         random_offsets = torch.zeros_like(default_all)

#         for jname, scale in random_scales.items():
#             if scale <= 0.0:
#                 continue
#             try:
#                 jidx = joint_names.index(jname)
#             except ValueError:
#                 continue
#             random_offsets[:, jidx] = scale * torch.randn(n_envs, device=device)

#         # ▶ suppress offsets during grace window
#         grace_mask = (count < skip_random_until)                    # (n_envs,) bool
#         random_offsets[grace_mask] = 0.0

#         amber_target = default_all + random_offsets
#         amber.set_joint_position_target(amber_target)

#         # ───────────────────── physics step ──────────────────────────────────
#         scene.write_data_to_sim()
#         sim.step()
#         scene.update(sim_dt)
#         count += 1

#         # ───────────────────── fall detection ────────────────────────────────
#         forces   = scene["contact_forces"].data.net_forces_w        # (n_envs, n_sensors, 3)
#         fallen   = (forces.abs().sum(dim=(1, 2)) > 0.0)             # bool (n_envs,)

#         # update per-env fall counter
#         fall_ctr = torch.where(fallen, fall_ctr + 1,
#                                torch.zeros_like(fall_ctr))

#         to_reset = (fall_ctr >= WAIT_FRAMES)                        # waited long enough
#         to_reset &= ((count - last_reset_step) > COOLDOWN_FRAMES)   # cool-down satisfied

#         if not to_reset.any():
#             continue

#         # ───────────────────── perform masked reset ──────────────────────────
#         default_root = amber.data.default_root_state.clone()
#         default_root[:, 2] += 0.5                 # +5 cm clearance
#         default_root[:, :3] += scene.env_origins

#         root_state = amber.data.root_state_w.clone()  # (n_envs, 13)
#         root_state[to_reset] = default_root[to_reset]
#         root_state[to_reset, 7:] = 0.0                # zero lin & ang vel

#         amber.write_root_pose_to_sim(root_state[:, :7])
#         amber.write_root_velocity_to_sim(root_state[:, 7:])

#         # joint state
#         cur_jpos = amber.data.joint_pos.clone()
#         cur_jvel = amber.data.joint_vel.clone()
#         cur_jpos[to_reset] = amber.data.default_joint_pos[to_reset]
#         cur_jvel[to_reset] = 0.0

#         amber.write_joint_state_to_sim(cur_jpos, cur_jvel)

#         # ▶ zero PD error for GRACE_FRAMES
#         amber.set_joint_position_target(cur_jpos)
#         skip_random_until[to_reset] = count + GRACE_FRAMES

#         # push writes & flush one step so sensors clear
#         scene.write_data_to_sim()
#         sim.step(); scene.update(sim_dt)

#         # bookkeeping
#         last_reset_step[to_reset] = count
#         fall_ctr[to_reset]        = 0


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,              # 0.5 ms physics step
        render_interval=5,     # (optional) render every 2 physics steps ≈ 250 Hz video
        device=args_cli.device
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    # ─────────────── ADD PLANAR CONSTRAINT ───────────────
    # _constrain_amber_to_xz(scene, args_cli.num_envs)

    # ────────────────────────────────────────────────────────
    # Play the simulator
    sim.reset()
    scene.reset() 
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()