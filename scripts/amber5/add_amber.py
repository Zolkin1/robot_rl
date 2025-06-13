# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
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

import numpy as np
import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg

STIFFNESS = 1000
DAMPING = 50
# --- AMBER ROBOT CONFIGURATION ---
ROBOT_ASSETS_AMBER = "/home/s-ritwik/src/robot_rl/robot_assets/amber5/amber"

AMBER_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_ASSETS_AMBER}/amber4.usd",
        activate_contact_sensors=True,
        # path_in_usd="/Amber",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(   
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # lift Amber up so you can see it, and offset to the side
        pos=(0., 0., 1.3),
        joint_pos={
            # world → base_link is fixed (handled in USD), so we start at:
            # "base_link_to_base_link2": 0.0,   # revolute (yaw about Y)
            # "base_link2_to_base_link3": 0.0,  # prismatic (Z translation)
            # "base_link3_to_torso": 0.0,
            # "revy":      0.0,  # prismatic (X translation)
            # then the five leg joints:
            "q1_left":   0.0,  # torso → left_thigh
            "q2_left":   0.0,  # left_thigh → left_shin
            "q1_right":  0.0,  # torso → right_thigh
            "q2_right":  0.0,  # right_thigh → right_shin
        },
        joint_vel={
            # "base_link_to_base_link2": 0.0,
            # "base_link2_to_base_link3": 0.0,
            # "base_link3_to_torso": 0.0,
            # "revy":      0.0,
            "q1_left":   0.0,
            "q2_left":   0.0,
            "q1_right":  0.0,
            "q2_right":  0.0,
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
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Amber/torso",
        update_period=0.0,       # every sim step
        history_length=1,        # only current step
        debug_vis=False,
    )
    # robot
    # Jetbot = JETBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Jetbot")
    # Dofbot = DOFBOT_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Dofbot")
    Amber = AMBER_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Amber")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # reset every 500 steps
        if count % 300 == 0:
            count = 0

            # Reset Amber
            root = scene["Amber"].data.default_root_state.clone()
            root[:, :3] += scene.env_origins
            scene["Amber"].write_root_pose_to_sim(root[:, :7])
            scene["Amber"].write_root_velocity_to_sim(root[:, 7:])
            scene["Amber"].write_joint_state_to_sim(
                scene["Amber"].data.default_joint_pos.clone(),
                scene["Amber"].data.default_joint_vel.clone(),
            )

            scene.reset()
            print("[INFO]: Resetting all robots state...")

        # 1) Per-joint randomness: set to 0.0 if you want that joint fixed at default.
        random_scales = {
            # "base_link_to_base_link2": 0.0,   # torso yaw
            # "base_link2_to_base_link3": 0.0,  # prismatic Z (usually left unconstrained here)
            # "base_link3_to_torso":      0.0,  # prismatic X
            "q1_left":   0.2,
            "q2_left":   0.2,
            "q1_right":  0.4,
            "q2_right":  0.4,
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

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        sim_time += sim_dt
        count += 1
        # ---- check for torso–ground contact ----
        amber = scene["Amber"]
        # contact_forces is (n_envs, n_sensors), ours has one sensor
        forces = scene["contact_forces"].data.net_forces_w          # if any env has non-zero contact force:
        if (forces.abs() > 0.0).any():
                print("[INFO] Torso hit the ground—resetting robot…")
                # reset exactly as you do on your 300-step timer:
                root = scene["Amber"].data.default_root_state.clone()
                root[:, :3] += scene.env_origins
                scene["Amber"].write_root_pose_to_sim(root[:, :7])
                scene["Amber"].write_root_velocity_to_sim(root[:, 7:])
                scene["Amber"].write_joint_state_to_sim(
                scene["Amber"].data.default_joint_pos.clone(),
                    scene["Amber"].data.default_joint_vel.clone(),
                )
                scene.reset()
                count = 0
                continue
        
from pxr import UsdPhysics, Gf, Sdf
import omni.usd

def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # ─────────────── ADD PLANAR CONSTRAINT ───────────────
    # _constrain_amber_to_xz(scene, args_cli.num_envs)

    # ────────────────────────────────────────────────────────
    # Play the simulator
    sim.reset()
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
