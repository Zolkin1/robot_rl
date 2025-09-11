import csv
import numpy as np
import torch
from pxr import Gf, UsdGeom
from isaaclab.utils.math import subtract_frame_transforms, euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv
import omni.usd
import math,time

from source.robot_rl.robot_rl.tasks.manager_based.robot_rl.amber import mdp
from transfer.Model_based.Amber.amber_cfg import PERIOD, WDES
from transfer.Model_based.Amber.amber_utils import draw_foot_trajectory, draw_step_and_com, get_projected_gravity, debug_print_joints, compute_step_location_local, quat_apply_inverse
PERIOD=0.8
Swing_ht=0.2
import casadi as ca
reference_step = ca.Function.load("transfer/Model_based/Amber/amber_reference_step.casadi")
from pathlib import Path
import math
from isaaclab.managers import ObservationTermCfg as ObsTerm
from source.robot_rl.robot_rl.tasks.manager_based.robot_rl.amber.amber_rough_env_lip_cfg import Nom_ht,half_cycle,visualise_flag
from isaaclab.managers import SceneEntityCfg
from typing import Sequence

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

# persistent storage for each scene instance
_desired_targets = {}
_prev_sign      = {}

def desired_foot_targets_obs(
    scene,
    sim_time: float,
    n_envs: int,
    desired_vel: Sequence[float],
    Ts: float = 0.4,
    nom_height: float = 0.8,
    wdes: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("Amber", body_names=["left_shin","right_shin"]),
    half_cycle: bool = True,
    visualize: bool = False,
) -> torch.Tensor:
    """
    Continuous or half‐cycle‐gated LIP–ICP foothold generator.

    Returns: [n_envs,4] = [ΔLx,ΔLz, ΔRx,ΔRz] – the x,z offset from current shin positions
    """
    # grab the Amber asset
    amber = scene[asset_cfg.name]
    device = amber.data.default_root_state.device

    # init per‐scene storage
    key = id(scene)
    if key not in _desired_targets:
        pos0 = amber.data.body_pos_w   # [n_envs, bodies, 3]
        B    = pos0.shape[1]
        init_L = pos0[:, B-2, :].clone()
        init_R = pos0[:, B-1, :].clone()
        _desired_targets[key] = torch.stack((init_L, init_R), dim=1).to(device)  # [n_envs,2,3]
        _prev_sign[key]      = torch.zeros(n_envs, dtype=torch.long, device=device)

    targets   = _desired_targets[key]   # [n_envs,2,3]
    prev_sign = _prev_sign[key]         # [n_envs]

    # compute phase / half‐cycle flip
    phi_cycle = (sim_time % (2 * Ts)) / (2 * Ts)  # [0,1)
    sign_now  = 1 if phi_cycle > 0.5 else -1
    sign_tensor = torch.full((n_envs,), sign_now, dtype=torch.long, device=device)
    changed = sign_tensor != prev_sign

    # update targets
    swing = 1 if sign_now > 0 else 0
    if half_cycle:
        if changed.any():
            new_steps = compute_step_location_local_policy(
                scene, sim_time, n_envs, desired_vel,
                nom_height, Ts, wdes, visualize=False
            )  # [n_envs,3]
            targets[changed, swing, :] = new_steps[changed]
            _prev_sign[key] = sign_tensor.clone()
    else:
        new_steps = compute_step_location_local_policy(
            scene, sim_time, n_envs, desired_vel,
            nom_height, Ts, wdes, visualize=False
        )
        targets[:, swing, :] = new_steps
        _prev_sign[key] = sign_tensor.clone()

    # optional USD viz
    if visualize:
        stage = omni.usd.get_context().get_stage()
        for i in range(n_envs):
            for side, col in [(0, (1.0,0.0,1.0)), (1, (1.0,1.0,0.0))]:
                path = f"/World/debug/future_foot_{side}_{i}"
                if not stage.GetPrimAtPath(path):
                    sph = UsdGeom.Sphere.Define(stage, path)
                    sph.GetRadiusAttr().Set(0.03)
                    sph.CreateDisplayColorAttr().Set([Gf.Vec3f(*col)])
                else:
                    sph = UsdGeom.Sphere(stage.GetPrimAtPath(path))
                UsdGeom.XformCommonAPI(sph).SetTranslate(
                    Gf.Vec3d(*targets[i, side, :].cpu().tolist())
                )
        # 1) COM as big red sphere
        com = (13*amber.data.body_com_pos_w[:, 3, :]+3.4261*amber.data.body_com_pos_w[:, 4, :]
            +1.1526*amber.data.body_com_pos_w[:, 5, :]+3.4261*amber.data.body_com_pos_w[:, 6, :]
            +1.1526*amber.data.body_com_pos_w[:, 7, :] )/(13+2*3.4261+2*1.1526
        )  # [N,3]
        for i in range(n_envs):
            path = f"/World/debug/com_sphere_{i}"
            if not stage.GetPrimAtPath(path):
                sph = UsdGeom.Sphere.Define(stage, path)
                sph.GetRadiusAttr().Set(0.1)
                sph.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
            else:
                sph = UsdGeom.Sphere(stage.GetPrimAtPath(path))
            UsdGeom.XformCommonAPI(sph).SetTranslate(Gf.Vec3d(*com[i].cpu().tolist()))
    # compute relative x,z to current shin positions
    # pos = amber.data.body_pos_w[:, [asset_cfg.body_ids[0], asset_cfg.body_ids[1]], :]  # [n_envs,2,3]
    B   = amber.data.body_pos_w.shape[1]
    pos = amber.data.body_pos_w[:, [B-2, B-1], :]  # [n_envs,2,3]
    rel = targets - pos                        # [n_envs,2,3]
    rel_xz = rel[:, :, [0, 2]]                 # [n_envs,2,2]
    return rel_xz.reshape(n_envs, 4)


def compute_step_location_local_policy(
    scene,
    sim_time: float,
    n_envs: int,
    desired_vel: Sequence[float],
    nom_height: float,
    Tswing: float,
    wdes: float,
    visualize: bool = True
) -> torch.Tensor:
    """
    Single‐call LIP–ICP foothold solver.  Returns [n_envs,3] world–frame p = next swing‐foot target.
    """
    amber  = scene["Amber"]
    device = amber.data.default_root_state.device

    # 1) commanded planar velocity
    cmd_np  = np.array(desired_vel, dtype=np.float32)
    command = torch.from_numpy(cmd_np[:2]).to(device).unsqueeze(0).repeat(n_envs,1)  # [n_envs,2]

    # 2) COM estimate
    r = (
        13*amber.data.body_com_pos_w[:,3,:]
      + 3.4261*amber.data.body_com_pos_w[:,4,:]
      + 1.1526*amber.data.body_com_pos_w[:,5,:]
      + 3.4261*amber.data.body_com_pos_w[:,6,:]
      + 1.1526*amber.data.body_com_pos_w[:,7,:]
    ) / (13 + 2*3.4261 + 2*1.1526)  # [n_envs,3]

    # 3) natural freq
    omega = math.sqrt(9.81 / nom_height)
    icp_0 = torch.zeros((n_envs,3), device=device)
    icp_0[:,:2] = command / omega

    # 4) last two foot positions
    pos      = amber.data.body_pos_w
    B        = pos.shape[1]
    foot_pos = pos[:, [B-1, B-2], :].clone()  # [n_envs,2,3]

    # 5) stance foot index
    tp     = (sim_time % (2*Tswing)) / (2*Tswing)
    phi_c  = math.sin(2*math.pi*tp)
    stance_idx = int(0.5 - 0.5 * np.sign(phi_c))
    stance_foot = foot_pos[:, stance_idx, :].clone()
    stance_foot[:,2] = 0.0

    # 6) to‐local / to‐global helpers
    def to_local(v,q):  return quat_rotate(yaw_quat(quat_inv(q)), v)
    def to_global(v,q): return quat_rotate(yaw_quat(q), v)

    # 7) final ICP
    expT = math.exp(omega * Tswing)
    icp_f = expT*icp_0 + (1-expT)*to_local(r - stance_foot, amber.data.body_quat_w[:,3,:])
    icp_f[:,2] = 0.0

    # 8) bias b
    bx = (torch.abs(command[:,0]) * Tswing) / (expT - 1.0)
    by = (wdes * torch.ones(n_envs,device=device)) * np.sign(phi_c) / (expT + 1.0)
    b  = torch.stack((bx, by, torch.zeros_like(bx)), dim=1)

    # 9) clamp local
    p_loc = icp_f.clone()
    p_loc[:,0] = torch.clamp(icp_f[:,0] - b[:,0], -0.5, 0.5)
    p_loc[:,1] = torch.clamp(icp_f[:,1] - b[:,1], -0.3, 0.3)

    # 10) back to world
    p = to_global(p_loc, amber.data.body_quat_w[:,3,:]) + r
    p[:,2] = 0.0

    # lateral‐axis override: keep original y
    swing_idx = 1 - stance_idx
    body_index = (B-1) if swing_idx==0 else (B-2)
    p[:,1] = amber.data.body_pos_w[:, body_index, 1]

    # USD viz of the next step
    if visualize:
        stage = omni.usd.get_context().get_stage()
        for i in range(n_envs):
            path = f"/World/debug/future_step_{i}"
            if not stage.GetPrimAtPath(path):
                sph = UsdGeom.Sphere.Define(stage, path)
                sph.GetRadiusAttr().Set(0.02)
            else:
                sph = UsdGeom.Sphere(stage.GetPrimAtPath(path))
            UsdGeom.XformCommonAPI(sph).SetTranslate(Gf.Vec3d(*p[i].cpu().tolist()))

    return p


def run_simulator(sim, scene, policy, simulation_app, args_cli):
    sim_dt   = sim.get_physics_dt()            # 0.005s → 200 Hz
    print("Sim dt:",sim_dt)
    sim_time = 0.0
    count    = 0
    video_active   = getattr(args_cli, "video", False)
    video_max_step = getattr(args_cli, "video_length", 0)
    act_q1_left = None
    act_q2_left = None
    act_q1_right = None
    act_q2_right = None
    USE_CASADI_IK = args_cli.use_casadi_ik
    if args_cli.video:
        from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file  # :contentReference[oaicite:0]{index=0}

    if USE_CASADI_IK:
        print("Using custom pin IK")
    amber    = scene["Amber"]
    device   = amber.data.default_root_state.device
    n_envs   = args_cli.num_envs
    # =============================================================================
    # 2)  Optional import & controller setup  (only built when needed)
    # =============================================================================
    if not USE_CASADI_IK:
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        # --- build once, after scene is created ----------------------------------
        def build_diff_ik(scene, num_envs, device):
            # we treat each foot as its own end-effector; choose one per half-cycle
            diff_cfg = DifferentialIKControllerCfg(
                command_type      = "pose",
                use_relative_mode = False,
                ik_method         = "dls",    # damped-least-squares
            )
            return DifferentialIKController(diff_cfg, num_envs=num_envs, device=device)

        # call right after you create the scene:
        diff_ik = build_diff_ik(scene, n_envs, device)

        # cache some IDs for quick access
        names = list(amber.data.joint_names)
        actuated_ids = [names.index(j) for j in ["q1_left", "q2_left", "q1_right", "q2_right"]]

        # body IDs for toes: B-2 (left)  and  B-1 (right)
        B_total         = amber.num_bodies
        toe_body_id_L   = B_total - 2
        toe_body_id_R   = B_total - 1
    # =============================================================================

    
    debug = args_cli.debug
    # ─── CSV SET-UP ────────────────────────────────────────────────────────────
    # ─── CSV SET-UP ─── (replace your existing header)
    csv_path = args_cli.csv_out.expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fh = open(csv_path, "w", newline="")
    writer = csv.writer(csv_fh)
    # extend header with action, foot and COM columns
    header = [
        "step", "sim_time", "env_id",
        *amber.data.joint_names,
        "act_q1_left", "act_q2_left", "act_q1_right", "act_q2_right",
        # current foot positions
        "cur_foot_l_x", "cur_foot_l_y", "cur_foot_l_z",
        "cur_foot_r_x", "cur_foot_r_y", "cur_foot_r_z",
        # target foot positions (foot_w)
        "tgt_foot_l_x", "tgt_foot_l_y", "tgt_foot_l_z",
        "tgt_foot_r_x", "tgt_foot_r_y", "tgt_foot_r_z",
        "com_x", "com_y", "com_z"
    ]
    writer.writerow(header)

    # ─── Masked‐reset bookkeeping ───────────────────────────────────────────────
    last_reset_step = torch.full((n_envs,), -1_000_000, dtype=torch.int32, device=device)
    COOLDOWN        = 10  # frames

   


    # ─── Buffers for inline swing trajectory + IK ─────────────────────────────
    N                 = 80
    pfoot_l_buffer    = None   # shape (n_envs, N, 3)
    pfoot_r_buffer    = None
    warmup_done   = False          # first full half-cycle = both feet glued
    foot_w_stack  = None           # will hold [6×N] DM for IK tracking
    buffer_idx        = 0
    swing_left_flag   = False
    foot_l_init       = None   # world‐frame pos at start of swing
    foot_r_init       = None
    com_init          = None   # world‐frame COM at start
    q_guess           = ca.DM([0.,0.,0.,0.])
    next_foot = torch.empty((n_envs, 3), device=device)    # will be overwritten later
    foot_w_stack  = None                            # empty


    # ─── Frequency settings ────────────────────────────────────────────────────
    policy_rate = 4  # every 4 steps → 200/4 = 50 Hz
    lip_rate    = max(1, int((PERIOD/2.0) / sim_dt))  # half‐cycle interval
    print("Lip rate:",200/lip_rate,"Hz")
    # number of physics steps in one half-cycle
    half_steps = int((PERIOD/2.0) / sim_dt)     # e.g. 0.4 / 0.005 = 80
    lip_rate   = half_steps                     # call LIP & spline gen once per half-cycle
    print("Lip rate:",200/lip_rate,"Hz")

    ik_rate    = max(1, half_steps // N)        # call IK N times per half-cycle
    print("IK Rate:",200/ik_rate,"Hz")

    # ────────────────────────────────────────────────────────────────────────
    q_vals=[0,0,0,0]
    x_array = ca.GenDM_zeros(n_envs,N)
    y_array = ca.GenDM_zeros(n_envs,N)
    policy_flag = args_cli.policy
    # set_torso_mass(10)

    try:
        while simulation_app.is_running():
            simulation_app.update()

            # ────────────────────── Global reset every 500 steps ───────────────────
            # if count % 500 == 0:
            #     count = 0
            #     scene.reset()
            #     sim.reset()
            #     root = amber.data.default_root_state.clone()
            #     root[:, :3] += scene.env_origins
            #     amber.write_root_pose_to_sim(root[:, :7])
            #     amber.write_root_velocity_to_sim(root[:, 7:])
            #     amber.write_joint_state_to_sim(
            #         amber.data.default_joint_pos.clone(),
            #         amber.data.default_joint_vel.clone(),
            #     )
            #     scene.write_data_to_sim()
            #     sim.step()
            #     scene.update(sim_dt)
            #     print("[INFO]: Resetting all robots state...")
            #     warmup_done=False


            # ────────────── Policy (or tuning) at lower frequency ─────────────────
            # if count % policy_rate == 0:
                # Gather sensor data
            qpos = amber.data.joint_pos.cpu().numpy()[0] - amber.data.default_joint_pos.cpu().numpy()      # (7,)
            qvel = amber.data.joint_vel.cpu().numpy()[0]- amber.data.default_joint_vel .cpu().numpy()    # (7,)
            ori  = amber.data.body_quat_w[0, 3, :].cpu().numpy()   # (4,)  qw,qx,qy,qz
            quat = np.array([ori[0], ori[1], ori[2], ori[3]], dtype=np.float32)  # same order
            body_ang_vel = amber.data.body_link_ang_vel_w[0, 3, :].cpu().numpy()
            des_vel = np.array(args_cli.desired_vel, dtype=np.float32)
            _G_VEC_W = torch.tensor([0.0, 0.0, -1.0])
            # projected gravity
            quat  = amber.data.body_link_quat_w[:, 3, :]          # [N,4]  (w,x,y,z) of link 3
            # ensure gravity vector on correct device & dtype
            g_vec = _G_VEC_W.to(quat)
            # broadcast g_vec to match envs
            g = g_vec.expand(quat.shape[0], 3)
            # rotate into body frame via inverse quaternion
            g_body = quat_apply_inverse(quat, g).cpu().numpy()       # [N,3]
            ## LIP OBS
            future_feet = desired_foot_targets_obs(
                scene,
                sim_time,
                args_cli.num_envs,
                args_cli.desired_vel,
                Ts=PERIOD/2,
                nom_height=Nom_ht,
                wdes=WDES,
                visualize=True,      # or False
                half_cycle=half_cycle,
            ).cpu().numpy()
            # print("loading future feet",future_feet)
            # ─── Build observation & run policy ───
            obs = policy.create_obs(
                qjoints=     qpos,
                body_ang_vel=body_ang_vel,
                qvel=        qvel,
                time=        sim.current_time,
                projected_gravity=g_body,
                des_vel=     des_vel,
                future_feet= future_feet,
            )
            if policy_flag == 1:
                _ = policy.get_action(obs.to(device))   # updates policy.action_isaac
                action_isaac = policy.get_action_isaac()
                default_all   = amber.data.default_joint_pos.clone()  # (1,7)
                target_tensor = torch.from_numpy(action_isaac).to(device).unsqueeze(0)  # (1,4)
                joint_targets = default_all.clone()

                # scatter into exactly those 4 actuated joints
                actuated_names = actuated_names = ["q1_left","q1_right","q2_left","q2_right"]

                all_names      = list(amber.data.joint_names)
                # print(all_names)
                for i, name in enumerate(actuated_names):
                    idx = all_names.index(name)
                    joint_targets[:, idx] = target_tensor[0, i]
                # print(joint_targets)

                amber.set_joint_position_target(joint_targets)
            scene.write_data_to_sim()
            if policy_flag == 0:
                if not warmup_done and (foot_w_stack is None or buffer_idx >= N):
                    # Capture current foot positions in world
                    pos_world   = amber.data.body_pos_w.cpu().numpy()[0]   # (B,3)
                    B           = pos_world.shape[0]
                    foot_l_init = pos_world[B - 2].copy()                  # left toe link
                    foot_r_init = pos_world[B - 1].copy()                  # right toe link

                    # Build a constant foot_w_stack for one half-cycle (T = PERIOD/2)
                    foot_w_stack = ca.DM.zeros(6, N)
                    for i in range(N):
                        # Row order: Lx Rx Ly Ry Lz Rz
                        foot_w_stack[0, i] = foot_l_init[0]
                        foot_w_stack[1, i] = foot_r_init[0]
                        foot_w_stack[2, i] = foot_l_init[1]
                        foot_w_stack[3, i] = foot_r_init[1]
                        # Lz, Rz already zero  → both feet on ground

                    buffer_idx = 0                       # start reading from column 0
                    print("[BOOT] Warm-up cycle: both feet planted")
                    # IK-follow block below will consume this stack
                    # ───────────────────────────────────────────────────────────────────
                # ─────────── LIP‐ICP + inline Bézier trajectory every half‐cycle ────────
                if warmup_done and count % lip_rate == 0:
                    # Compute next foot target
                    next_foot = compute_step_location_local(
                        sim_time=   sim_time,
                        scene=      scene,
                        args_cli=   args_cli,
                        nom_height= 0.8,
                        Tswing=     PERIOD/2.0,
                        wdes=       WDES,
                        visualize=  False
                    )
                    # print("-----------------------------",next_foot,"-----------")
                    # Capture initial foot & COM positions
                    # Capture initial foot & COM positions
                    pos_world   = amber.data.body_pos_w.cpu().numpy()[0]   # (B,3)
                    B           = pos_world.shape[0]
                    foot_l_init = pos_world[B-2].copy()
                    foot_r_init = pos_world[B-1].copy()
                    # com_init =(13*amber.data.body_com_pos_w.cpu().numpy()[:, 3, :]+3.4261*amber.data.body_com_pos_w.cpu().numpy()[:, 4, :]
                    #     +1.1526*amber.data.body_com_pos_w.cpu().numpy()[:, 5, :]+3.4261*amber.data.body_com_pos_w.cpu().numpy()[:, 6, :]
                    #     +1.1526*amber.data.body_com_pos_w.cpu().numpy()[:, 7, :] )/(13+2*3.4261+2*1.1526
                    # )
                    # com_init= com_init[0]
                    com_init    = pos_world[3].copy()
                    # print("--Foot com pos--",com_init)
                    swing_left_flag = float(next_foot[0,1].cpu()) < com_init[1]
                    if swing_left_flag:
                        print("--LEFT FOOT IN SWING ", foot_l_init, next_foot)
                        foot_z=foot_l_init[2]
                        foot_z_stance=foot_r_init[2]
                    else:
                        print("-- RIGHT FOOT IN SWING", foot_r_init, next_foot)
                        foot_z=foot_r_init[2]
                        foot_z_stance=foot_l_init[2]

                    # Bézier parameters
                    T       = PERIOD / 2.0                     # half-cycle
                    ts_np   = np.linspace(0.0, T, N)           # time vector
                    ts      = ca.DM(ts_np)
                    phase_np = np.clip((ts_np / T - 0.25) * 2.0, 0.0, 1.0)   # ∈ [0,1]
                    phase    = ca.DM(phase_np)

                    def cubic_bezier_interpolation(z0, z1, tau):
                        """scalar or DM → scalar/DM (0≤tau≤1)"""
                        tau = ca.fmax(0, ca.fmin(1, tau))
                        return z0 + (z1 - z0) * (tau**3 + 3 * tau**2 * (1 - tau))
                    stance_z = -0.05
                    mask   = phase_np <= 0.5
                    z0_arr = cubic_bezier_interpolation(foot_z,        Swing_ht, 2 * phase)
                    z1_arr = cubic_bezier_interpolation(Swing_ht, stance_z,        2 * phase - 1)
                    z_array = ca.if_else(mask, z0_arr, z1_arr)       # DM(N,1)
                    # z0_arr_s = cubic_bezier_interpolation(foot_z_stance, stance_z, 2 * phase)
                    # z1_arr_s = cubic_bezier_interpolation(stance_z, stance_z, 2 * phase - 1)
                    # z_array_s = ca.if_else(mask, z0_arr_s, z1_arr_s)       # DM(N,1)
                    # ----------------------------------------------
                    # stance-foot vertical trajectory (DM(N,1))
                    # ----------------------------------------------
                    ramp_factors = ca.DM([i/9.0 for i in range(10)]).reshape((10, 1))      # 0 … 1
                    ramp         = foot_z_stance * (1.0 - ramp_factors)                  # DM(10,1)

                    press        = -0.03 * ca.DM.ones(N-10, 1)                           # DM(N-10,1)

                    z_array_s    = ca.vertcat(ramp, press)
                    # print(f"swing: {z_array};------stance{z_array_s}")
                    # ────────────────────────────────────────────────────────────────────
                    # 3)  COM path (x moves linearly, y const) ───────────────────────────
                    # ────────────────────────────────────────────────────────────────────
                    v_x     = float(args_cli.desired_vel[0])
                    x_array = com_init[0] + v_x * ts                   # DM(N,1)
                    y_array = com_init[1] + ca.DM.zeros(N, 1)

                    # ────────────────────────────────────────────────────────────────────
                    # 4)  Build foot_w_stack (6×N) with stance/swing logic ──────────────
                    #     Row indexing: 0=Lx, 1=Rx, 2=Ly, 3=Ry, 4=Lz, 5=Rz
                    # ────────────────────────────────────────────────────────────────────
                    stance_idx = 1 if swing_left_flag else 0   # 0=L, 1=R (for x & y rows)
                    swing_idx  = 0 if swing_left_flag else 1
                    foot_w_stack = ca.DM.zeros(6, N)

                    Lx0, Ly0, Lz0 = foot_l_init        # copy for readability
                    Rx0, Ry0, Rz0 = foot_r_init

                    # target X of swing foot (final position)  – shift by nominal step
                    swing_dx = next_foot[0,0].cpu().item() - \
                            (Lx0 if swing_left_flag else Rx0)
                    # print(swing_dx)
                    # print("swing_dx |")
                    for i in range(N):
                        # stance foot: stays glued
                        foot_w_stack[stance_idx,   i] = Lx0 if stance_idx == 0 else Rx0
                        foot_w_stack[stance_idx+2, i] = Ly0 if stance_idx == 0 else Ry0
                        foot_w_stack[stance_idx+4, i] = z_array_s[i]                               # z=0

                        # swing foot: X interpolation, same Y, Bézier Z
                        tau = phase[i]
                        # print("Phase:",tau)
                        swing_x = (Lx0 if swing_idx == 0 else Rx0) + swing_dx * tau
                        foot_w_stack[swing_idx,   i] = swing_x
                        foot_w_stack[swing_idx+2, i] = Ly0 if swing_idx == 0 else Ry0
                        foot_w_stack[swing_idx+4, i] = z_array[i]
                    # ────────────────────────────────────────────────────────────────────
                    buffer_idx = 0   # reset buffer pointer
                    # print("___________",foot_w_stack[:,0])
                    # print("lx:",foot_w_stack[0,:])
                    # print("rx",foot_w_stack[1,:])
                    # print("lz:",foot_w_stack[4,:])
                    # print("rz:",foot_w_stack[5,:])
                    # print("xl:",foot_w_stack[0,:])
        
                    # Now `foot_w_stack` (6×N DM) holds at column k the full [Lx, Rx,Ly,Ry,Lz,Rz]
                    # and can be stepped through in your IK‐loop immediately afterward.
                

                # ─────────────── IK follow for buffered trajectory ────────────────────
                if foot_w_stack is not None and buffer_idx < N and count % ik_rate == 0:
                    # Build foot targets for CasADi from the precomputed DM stack
                    t_buff_start = time.time()

                    # each column of foot_w_stack is [Lx, Ly, Lz, Rx, Ry, Rz]
                    foot_w = foot_w_stack[:, buffer_idx]
                    # print(foot_w)

                    # phase + COM
                    phase  = ca.DM((buffer_idx + 1) / N)
                    com_x_curr  = amber.data.body_pos_w.cpu().numpy()[0, 3, 0]
                    com_y_curr  = amber.data.body_pos_w.cpu().numpy()[0, 3, 1]
                    com_x  = x_array[buffer_idx]
                    com_y  = y_array[buffer_idx]
                    # com_z  = amber.data.body_com_pos_w.cpu().numpy()[0, 3, 2]   # NEW
                    com_z  = amber.data.body_pos_w.cpu().numpy()[0, 3, 2]   # NEW
                    com_z = 1.29
                    # print(f"Qguess:{q_guess}")
                    # q_guess =ca.DM([0.0,0.0,0.0,0.0])
                    q1_lim = 70 * math.pi / 180      # ±70° → ±1.22173 rad
                    q2_lim = 85 * math.pi / 180     # ±90° → ±1.57080 rad

                    # convert DM → NumPy for element-wise test
                    q_np = np.array(q_guess).astype(float)

                    # indices: 0 and 2 are q1 (hips) ; 1 and 3 are q2 (knees)
                    for i in (0, 2):                               # q1 joints
                        if abs(q_np[i]) > q1_lim:
                            q_np[i] = 0.0
                    for i in (1, 3):                               # q2 joints
                        if abs(q_np[i]) > q2_lim:
                            q_np[i] = 0.0

                    # back to CasADi DM
                    q_guess = ca.DM(q_np)
                    q_guess = [ 0.85542953, -1.3705312 ,  0.85542953 ,-1.3705312 ]
                    if USE_CASADI_IK:
                        # IK solve (new signature: no z_swing, but needs com_z)
                        q_cas, _ = reference_step(
                            phase, foot_w, com_x, com_y, com_z, q_guess
                        )
                        q_guess = q_cas
                        q_vals   = np.array(q_cas).astype(float).flatten()   # already in rad

                    else:
                        # -------- Isaac-Lab Differential IK -----------------------------------
                        # 1) Pick which foot is EE this half-cycle
                        ee_body_id = toe_body_id_L if swing_left_flag else toe_body_id_R
                        ee_jac_id  = ee_body_id - 1 if amber.is_fixed_base else ee_body_id

                        # 2) Build the pose goal for that foot  (xyz + quaternion)
                        foot_xyz  = foot_w[0:3] if swing_left_flag else foot_w[3:6]       # DM(3)
                        goal_pos  = torch.tensor(foot_xyz.full().flatten(), device=device)
                        goal_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)      # keep upright
                        diff_goal = torch.cat((goal_pos, goal_quat)).repeat(n_envs, 1)

                        diff_ik.set_command(diff_goal)

                        # 3) Grab sim data
                        J     = amber.root_physx_view.get_jacobians()[:, ee_jac_id, :, actuated_ids]
                        q_act = amber.data.joint_pos[:, actuated_ids]
                        ee_p  = amber.data.body_pos_w[:, ee_body_id]
                        ee_q  = amber.data.body_quat_w[:, ee_body_id]
                        root_p, root_q = amber.data.root_pos_w, amber.data.root_quat_w
                        # transform EE into root frame
                        ee_pb, ee_qb = subtract_frame_transforms(root_p, root_q, ee_p, ee_q)

                        # 4) Run diff-IK → desired joint angles
                        q_des = diff_ik.compute(ee_pb, ee_qb, J, q_act)    # (N_envs, 4)

                        # 5) Pack into the same variable names the rest of your code expects
                        q_vals = ca.DM(q_des[0].cpu().numpy().reshape(-1, 1))   # 4×1 DM in radians
                    # ------------------------------------------------------------------------- #


                    pos_world_1   = amber.data.body_pos_w.cpu().numpy()[0]   # (B,3)
                    B           = pos_world.shape[0]
                    foot_l_init_curr = pos_world_1[B-2].copy()
                    foot_r_init_curr = pos_world_1[B-1].copy()
                    if debug:
                        #       ------------------------------------------------------------------------
                        print(f"COM USING INTERPOLATION||x:{com_x}; y:{com_y}; z_com:{amber.data.body_pos_w.cpu().numpy()[0, 3, 2]}||; |IK angles:{q_cas*180/math.pi}|")
                        print(f"CURRENT COM POSITIONS :||x:{com_x_curr:.4f}; y:{com_y_curr:.4f}; z_com:{com_z}||; ")
                        # print((13*amber.data.body_com_pos_w[:, 3, :]+3.4261*amber.data.body_com_pos_w[:, 4, :]
                        #     +1.1526*amber.data.body_com_pos_w[:, 5, :]+3.4261*amber.data.body_com_pos_w[:, 6, :]
                        #     +1.1526*amber.data.body_com_pos_w[:, 7, :] )/(13+2*3.4261+2*1.1526
                        # ))
                        print(f"footpos for IK:|{foot_w}|")
                        print(f"------------------------------Curr left foot:|{foot_l_init_curr}|; right foot:|{foot_r_init_curr}|")
                        # print(f"Body com:{amber.data.root_com_pos_w.cpu().numpy()[0] }")
                        for eid in range(amber.data.joint_pos.shape[0]):
                            debug_print_joints(amber, env_id=eid)

                    # Scatter into full joint‐target
                    # joint_targets = default_all.copy()
                    names = list(amber.data.joint_names)
                    joint_targets = amber.data.default_joint_pos.clone().cpu().numpy()
                    for i, jn in enumerate(["q1_left","q2_left","q1_right","q2_right"]):
                        joint_targets[0, names.index(jn)] = q_vals[i]

                    amber.set_joint_position_target(torch.from_numpy(joint_targets).to(device))
                    # print("time for IK sovle:",time.time()-t_buff_start)
                    # print(f"IK for {buffer_idx}/{N}")
                    draw_step_and_com(scene, next_foot)
                    draw_foot_trajectory(scene, foot_w_stack)

                    buffer_idx += 1
                if not warmup_done and buffer_idx >= N:
                    warmup_done = True
                    print("[BOOT] Warm-up finished → switching to LIP planner")
            # ───────────────────────────────────────────────────────────────────
            # ─────────────────────────── Logging ──────────────────────────────────
            cur_q = amber.data.joint_pos.cpu().numpy()            # (n_envs, n_joints)
            # make sure action_isaac is always available here:
            # action_isaac = policy.action_isaac  # shape (4,)
            for env_id in range(n_envs):
                # 1) current joint positions
                cur_q_row = cur_q[env_id].tolist()
                # 2) commanded targets
                # act_row = action_isaac.tolist()
                act_row = [float(q_vals[i]) for i in range(4)]

                # 3) foot positions: last two bodies are [B-2]=left, [B-1]=right
                bodies = amber.data.body_pos_w.cpu().numpy()
                foot_l = bodies[env_id, -2, :].tolist()
                foot_r = bodies[env_id, -1, :].tolist()
                # 4) COM: use body_com_pos_w at index 3 (torso)
                com = amber.data.body_com_pos_w.cpu().numpy()[env_id, 3, :].tolist()
                if policy_flag == 1:
                    fw= [0,0,0,0,0,0]
                    tgt_l = [0,0,0]
                    tgt_r = [0,0,0]
                else:
                    fw = np.array(foot_w).astype(float).flatten()  # length 6
                    tgt_l = [fw[0], fw[2], fw[4]]
                    tgt_r = [fw[1], fw[3], fw[5]]

                # current foot positions from sim:
                bodies = amber.data.body_pos_w.cpu().numpy()  # shape (n_envs, n_bodies, 3)
                cur_l  = bodies[env_id, -2, :].tolist()
                cur_r  = bodies[env_id, -1, :].tolist()

                # write one combined row
                writer.writerow(
                    [count, sim_time, env_id]
                    + cur_q_row
                    + act_row
                    + cur_l + cur_r
                    + tgt_l + tgt_r
                    + com
                )
            if count % 100 == 0:
                csv_fh.flush()

            # ─── Contact‐based masked resets ───────────────────────────────────────
            forces  = scene["contact_forces"].data.net_forces_w
            fallen  = (forces.abs().sum(dim=(1,2)) > 0.05)
            to_reset = fallen & ((count - last_reset_step) > COOLDOWN)
            if to_reset.any():
                default_root = amber.data.default_root_state.clone()
                default_root[:, :3] += scene.env_origins

                root_state = amber.data.root_state_w.clone()
                root_state[to_reset] = default_root[to_reset]
                amber.write_root_pose_to_sim(root_state[:, :7])
                amber.write_root_velocity_to_sim(root_state[:, 7:])

                cur_jpos = amber.data.joint_pos.clone()
                cur_jvel = amber.data.joint_vel.clone()
                cur_jpos[to_reset] = amber.data.default_joint_pos[to_reset]
                cur_jvel[to_reset] = amber.data.default_joint_vel[to_reset]
                amber.write_joint_state_to_sim(cur_jpos, cur_jvel)
                warmup_done = False
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)
                last_reset_step[to_reset] = count
                # print("--------------------Contact-----------------------")
            # ───────────────────────── Physics step ────────────────────────────────
            sim.step()
            scene.update(sim_dt)
            # ───────────────────────── Video frames ────────────────────────────────

            if getattr(args_cli, "video", False):
                max_frames = args_cli.video_length
                vp = get_active_viewport()
                if vp is None:
                    raise RuntimeError("Could not find an active viewport for recording!")
                frame_dir = Path("videos/frames")
                frame_dir.mkdir(parents=True, exist_ok=True)
                if count < max_frames:
                    # write a png for this physics step
                    capture_viewport_to_file(vp, str(frame_dir / f"frame_{count:06d}.png"))
                elif count == max_frames:
                    print("[INFO] Reached --video_length; stopping frame capture.")
            # sim_time = sim.current_time
            sim_time += sim_dt

            count   += 1
            
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C – CSV saved at:", csv_path)
    finally:
        csv_fh.close()

