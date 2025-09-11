#!/usr/bin/env python3
# Copyright (c) 2025, The Isaac-Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
#
#   ./isaaclab.sh -p scripts/create_amber_base_env.py --num_envs 100

"""Multi-env AMBER planar biped with contact-based reset."""

# ───────────────────────────────── 1.  Launch Omniverse first ─────────────────
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

simulation_app = AppLauncher(args_cli).app   # Isaac Sim handle

# ───────────────────────────────── 2.  Standard stuff ─────────────────────────
import torch

# ───────────────────────────────── 3.  Isaac-Lab imports ──────────────────────
import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
import isaaclab.envs.mdp as mdp

# ───────────────────────────────── 4.  Robot cfg ──────────────────────────────
ROBOT_ASSETS_AMBER = "/home/s-ritwik/src/robot_rl/robot_assets/amber5/amber/amber4.usd"

AMBER_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Amber",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_ASSETS_AMBER,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=200.0,
            max_angular_velocity=200.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            enabled_self_collisions=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos=dict(q1_left=0.0, q2_left=0.0, q1_right=0.0, q2_right=0.0),
    ),
     actuators={
        "left_hip": ImplicitActuatorCfg(
            joint_names_expr=["q1_left"],
            effort_limit=300.0,
            velocity_limit=40.0,
            stiffness=0.0,          # ←── add
            damping=4.0,            # ←── add
        ),
        "left_knee": ImplicitActuatorCfg(
            joint_names_expr=["q2_left"],
            effort_limit=300.0,
            velocity_limit=40.0,
            stiffness=0.0,
            damping=4.0,
        ),
        "right_hip": ImplicitActuatorCfg(
            joint_names_expr=["q1_right"],
            effort_limit=300.0,
            velocity_limit=40.0,
            stiffness=0.0,
            damping=4.0,
        ),
        "right_knee": ImplicitActuatorCfg(
            joint_names_expr=["q2_right"],
            effort_limit=300.0,
            velocity_limit=40.0,
            stiffness=0.0,
            damping=4.0,
        ),
    },
)

# ───────────────────────────────── 5.  A/O cfg ────────────────────────────────
JOINT_NAMES = ["q1_left", "q2_left", "q1_right", "q2_right"]

@configclass
class ActionsCfg:
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot",
                                             joint_names=JOINT_NAMES, scale=30.0)

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        def __post_init__(self):
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

# ───────────────────────────────── 6.  Scene cfg ──────────────────────────────
@configclass
class AmberSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(200.0, 200.0)),
    )
    robot = AMBER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=600.0),
    )
    # per-env contact sensor so we can read net_forces_w
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso", update_period=0.0, debug_vis=False
    )

# ───────────────────────────────── 7.  Env cfg ────────────────────────────────
@configclass
class AmberEnvCfg(ManagerBasedEnvCfg):
    scene = AmberSceneCfg(num_envs=1024, env_spacing=2.5)
    observations = ObservationsCfg()
    actions = ActionsCfg()
    def __post_init__(self):
        self.viewer.eye = [6.0, 0.0, 3.5]
        self.viewer.lookat = [0.0, 0.0, 1.0]
        self.decimation = 2
        self.sim.dt = 1 / 240
        self.sim.enable_contact_forces = True
        self.sim.physx.contact_collection = "last_substep"

# ───────────────────────────────── 8.  Contact-reset ──────────────────────────
def contact_based_reset(env, cooldown_steps, last_reset, sim_frame):
    scene  = env.scene
    amber  = scene["robot"]

    forces = scene["contact_forces"].data.net_forces_w
    if forces is None:
        return

    fallen   = forces.abs().sum(dim=(1, 2)) > 0.0
    to_reset = fallen & ((sim_frame - last_reset) > cooldown_steps)
    if not to_reset.any():
        return

    # ――― size-safe defaults ―――
    N = amber.data.joint_pos.shape[0]              # actual envs spawned
    default_root  = amber.data.default_root_state[:N].clone()
    default_root[:, :3] += scene.env_origins[:N]
    default_joint = amber.data.default_joint_pos[:N]
    default_vel   = amber.data.default_joint_vel[:N]

    # root pose / velocity
    root_state = amber.data.root_state_w.clone()
    root_state[to_reset] = default_root[to_reset]
    amber.write_root_pose_to_sim(root_state[:, :7], env_ids=to_reset.nonzero().squeeze(-1))
    amber.write_root_velocity_to_sim(root_state[:, 7:], env_ids=to_reset.nonzero().squeeze(-1))

    # joint state
    amber.write_joint_state_to_sim(
        default_joint[to_reset], default_vel[to_reset],
        env_ids=to_reset.nonzero().squeeze(-1)
    )

    # flush
    scene.write_data_to_sim()
    env.sim.step(); scene.update(env.sim_dt)
    last_reset[to_reset] = sim_frame

# ───────────────────────────────── 9.  Main loop ──────────────────────────────
def main() -> None:
    cfg = AmberEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device

    env = ManagerBasedEnv(cfg)
    cooldown = int(1.0 / cfg.sim.dt)        # 1 second
    last_reset = torch.zeros(cfg.scene.num_envs,
                             dtype=torch.long, device=env.device)

    sim_frame = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            torques = torch.randn_like(env.action_manager.action)
            env.step(torques)                      # forces updated here
            contact_based_reset(env, cooldown, last_reset, sim_frame)

            if sim_frame % (cfg.decimation * 200) == 0:
                env.reset()

            sim_frame += 1

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
