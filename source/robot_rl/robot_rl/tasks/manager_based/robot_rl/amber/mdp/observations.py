from __future__ import annotations

import torch
from typing import TYPE_CHECKING
import math, time
import numpy as np
import csv
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_rotate_inverse, yaw_quat, quat_rotate, quat_inv
import omni.usd  
import torch
from pxr import Gf, UsdGeom

_G = 9.81

def sin_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    phase = torch.tensor(2*torch.pi * (env.sim.current_time / period))
    sphase = torch.sin(phase)

    sphase = torch.ones((env.num_envs, 1), device=env.device) * sphase

    return sphase

def cos_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    phase = torch.tensor(2*torch.pi * (env.sim.current_time / period))
    cphase = torch.cos(phase)

    cphase = torch.ones((env.num_envs, 1), device=env.device) * cphase

    return cphase

def is_ground_phase(env: ManagerBasedRLEnv, period: float) -> torch.Tensor:
    sp = sin_phase(env, period)
    cp = cos_phase(env, period)

    return torch.tensor([(sp < 0.0), (cp < 0.0)])

# gravity constant
_G = 9.81


def _phi_contact_scalar(phi: float) -> float:
    """Eq.(18): sin(2πφ) / sqrt(sin²(2πφ)+0.04)."""
    s = math.sin(2 * math.pi * phi)
    return s / math.sqrt(s * s + 0.04)

def compute_step_location_local(
    sim_time: float,
    scene,
    num_envs: int,
    desired_vel: Sequence[float],
    nom_height: float,
    Tswing: float,
    wdes: float,
    visualize: bool = True
) -> torch.Tensor:
    """
    Compute next foothold in world‐frame using a local‐frame LIP ICP method.
    Always recompute on each call (no half‐cycle guard). Returns [num_envs×3].
    """
    amber  = scene["robot"]
    device = amber.data.default_root_state.device
    n_envs = num_envs

    # --- static storage of original lateral foot‐Y offsets ---
    if not hasattr(compute_step_location_local, "_orig_foot_y"):
        pos0   = amber.data.body_pos_w               # (n_envs, n_bodies, 3)
        B      = pos0.shape[1]
        feet0  = pos0[:, [B-1, B-2], :]               # (n_envs, 2, 3)
        compute_step_location_local._orig_foot_y = feet0[:, :, 1].clone()

    # 1) commanded velocity in local frame [N,2]
    cmd_np  = np.array(desired_vel, dtype=np.float32)
    command = torch.from_numpy(cmd_np[:2]).to(device).unsqueeze(0).repeat(n_envs, 1)

    # 2) COM position in world from body index 3 [N,3]
    # r = amber.data.body_pos_w[:, 3, :]
    r = (13*amber.data.body_com_pos_w[:, 3, :]+3.4261*amber.data.body_com_pos_w[:, 4, :]
        +1.1526*amber.data.body_com_pos_w[:, 5, :]+3.4261*amber.data.body_com_pos_w[:, 6, :]
        +1.1526*amber.data.body_com_pos_w[:, 7, :] )/(13+2*3.4261+2*1.1526
    )
    # 3) build ICP base
    omega = math.sqrt(9.81 / nom_height)
    icp_0 = torch.zeros((n_envs, 3), device=device)
    icp_0[:, :2] = command[:, :2] / omega

    # 4) last two foot positions [N,2,3]
    pos      = amber.data.body_pos_w
    B        = pos.shape[1]
    foot_pos = pos[:, [B-1, B-2], :]

    # 5) phase clock → stance foot
    tp    = (sim_time % (2*Tswing)) / (2*Tswing)
    phi_c = torch.tensor(
        math.sin(2*math.pi*tp) / math.sqrt(math.sin(2*math.pi*tp)**2 + Tswing),
        device=device
    )
    stance_idx  = int(0.5 - 0.5 * torch.sign(phi_c).item())
    stance_foot = foot_pos[:, stance_idx, :].clone()
    stance_foot[:, 2] = 0.0

    # 6) transforms
    def to_local(v, quat):
        return quat_rotate(yaw_quat(quat_inv(quat)), v)
    def to_global(v, quat):
        return quat_rotate(yaw_quat(quat), v)

    # 7) final ICP in local frame
    exp_omT = math.exp(omega * Tswing)
    icp_f = (
        exp_omT * icp_0
        + (1 - exp_omT) * to_local(r - stance_foot, amber.data.body_quat_w[:,3,:])
    )
    icp_f[:, 2] = 0.0

    # 8) compute bias b
    sd = torch.abs(command[:, 0]) * Tswing
    wd = wdes * torch.ones(n_envs, device=device)
    bx = sd / (exp_omT - 1.0)
    by = torch.sign(phi_c) * wd / (exp_omT + 1.0)
    b  = torch.stack((bx, by, torch.zeros_like(bx)), dim=1)

    # 9) clip in local
    p_local = icp_f.clone()
    p_local[:, 0] = torch.clamp(icp_f[:, 0] - b[:, 0], -0.5, 0.5)
    p_local[:, 1] = torch.clamp(icp_f[:, 1] - b[:, 1], -0.3, 0.3)

    # 10) back to world, zero Z
    p = to_global(p_local, amber.data.body_quat_w[:,3,:]) + r
    p[:, 2] = 0.0

    # override lateral step to original Y
    swing_idx = 1 - stance_idx
    # --- dynamic override: use the swing foot's current y each call ---
    # swing_idx already computed above (0 or 1)
    B = amber.data.body_pos_w.shape[1]
    # foot_pos was pos[:, [B-1, B-2], :]  → index 0→body B-1, 1→body B-2
    body_index = (B-1) if swing_idx == 0 else (B-2)
    # grab the world‐frame y‐coords of that foot
    current_y = amber.data.body_pos_w[:, body_index, 1]   # [n_envs]
    # override the lateral coordinate
    p[:, 1] = current_y

    # USD visualization (same as before)
    if visualize:
        stage = omni.usd.get_context().get_stage()
        # future‐step spheres
        for i in range(n_envs):
            path = f"/World/debug/future_step_{i}"
            if not stage.GetPrimAtPath(path):
                sph = UsdGeom.Sphere.Define(stage, path)
                sph.GetRadiusAttr().Set(0.02)
            else:
                sph = UsdGeom.Sphere(stage.GetPrimAtPath(path))
            UsdGeom.XformCommonAPI(sph).SetTranslate(Gf.Vec3d(*p[i].cpu().tolist()))

        # COM as big red sphere
        com = amber.data.body_pos_w[:, 3, :]
        for i in range(n_envs):
            path = f"/World/debug/com_sphere_{i}"
            if not stage.GetPrimAtPath(path):
                com_sph = UsdGeom.Sphere.Define(stage, path)
                com_sph.GetRadiusAttr().Set(0.1)
                com_sph.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
            else:
                com_sph = UsdGeom.Sphere(stage.GetPrimAtPath(path))
            UsdGeom.XformCommonAPI(com_sph).SetTranslate(Gf.Vec3d(*com[i].cpu().tolist()))

    # stash into scene for downstream use
    if not hasattr(scene, "current_des_step"):
        scene.current_des_step = torch.zeros((n_envs, 3), device=device)
    scene.current_des_step[:] = p

    return p


def debug_plot_and_print(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    Ts: float,
    nom_height: float,
    wdes: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    debug: bool,
    visualize: bool,
) -> torch.Tensor:
    # call your obs fn with visualize=True → draws spheres if a USD stage exists
    _ = desired_foot_targets_obs(
        env,
        Ts=Ts,
        nom_height=nom_height,
        wdes=wdes,
        command_name=command_name,
        asset_cfg=asset_cfg,
        debug=debug,
        visualize=visualize,
    )
    # return dummy zeros so the EventTerm API is happy
    return torch.zeros(env_ids.shape[0], device=env.device)


@torch.no_grad()
def desired_foot_targets_obs(
    env: ManagerBasedRLEnv,
    Ts: float                    = 0.4,                   # half‐cycle duration
    nom_height: float            = 0.8,                  # pendulum height
    wdes: float                  = 0.1,                   # desired half‐width
    command_name: str            = "base_velocity",
    asset_cfg: SceneEntityCfg    = SceneEntityCfg("robot"),
    debug: bool                  = False,
    visualize: bool              = False,
    half_cycle: bool             = True,
) -> torch.Tensor:
    """
    Returns [N,6] = [Lx,Ly,Lz,  Rx,Ry,Rz]:
      • runs LIP–ICP once per half‐period to update swing‐foot target
      • holds previous target for stance foot
      • debug=True prints COM, last_L, last_R, new_L, new_R for env-0
      • visualize=True draws spheres for the next swing target & COM
    """
    amber  = env.scene[asset_cfg.name]
    device = env.device
    N      = env.num_envs

    # 0) persistent targets, one per leg
    fn = desired_foot_targets_obs
    if not hasattr(fn, "_init"):
        pos0       = amber.data.body_pos_w          # [N, bodies, 3]
        B          = pos0.shape[1]
        init_L     = pos0[:, B-2, :].clone()
        init_R     = pos0[:, B-1, :].clone()
        fn._targets   = torch.stack((init_L, init_R), dim=1).to(device)  # [N,2,3]
        fn._prev_sign = torch.zeros(N, dtype=torch.long, device=device)
        fn._init      = True

    targets   = fn._targets       # [N,2,3]
    prev_sign = fn._prev_sign

    # snapshot for debug
    last_L = targets[:, 0, :].clone()
    last_R = targets[:, 1, :].clone()

    # 1) phase → swing vs stance
    t         = env.sim.current_time
    phi_cycle = (t % (2*Ts)) / (2*Ts)           # ∈ [0,1)
    sign_now  = 1 if phi_cycle > 0.5 else -1
    sign_tensor  = torch.full((N,), sign_now, dtype=torch.long, device=device)
    changed_mask = sign_tensor != prev_sign

    # 2) update only on half‐cycle flips
    if half_cycle== True :
        if changed_mask.any():
            swing = 1 if sign_now > 0 else 0
            new_steps = compute_step_location_local(
                sim_time   = t,
                scene      = env.scene,
                num_envs   = env.num_envs,
                desired_vel= env.command_manager.get_command("base_velocity")[0].cpu().tolist(),
                nom_height = nom_height,
                Tswing     = Ts,
                wdes       = wdes,
                visualize  = False,
            ) # [N,3]
            #for half cycle
            targets[changed_mask, swing, :] = new_steps[changed_mask]
            #for continious
            # targets[:, swing, :] = new_steps
            prev_sign.copy_(sign_tensor)
    else:
        swing = 1 if sign_now > 0 else 0
        new_steps = compute_step_location_local(
            sim_time   = t,
            scene      = env.scene,
            num_envs   = env.num_envs,
            desired_vel= env.command_manager.get_command("base_velocity")[0].cpu().tolist(),
            nom_height = nom_height,
            Tswing     = Ts,
            wdes       = wdes,
            visualize  = False,
        ) # [N,3]
        #for half cycle
        # targets[changed_mask, swing, :] = new_steps[changed_mask]
        #for continious
        targets[:, swing, :] = new_steps
        prev_sign.copy_(sign_tensor)
    # 3) debug print
    if debug and N > 0:
        # COM
        com0 = amber.data.body_pos_w[0, 3, :].cpu().numpy()
        print(f"[t={t:.3f}] COM={com0}")

        # current foot positions
        pos0      = amber.data.body_pos_w[0]           # [bodies,3]
        B         = pos0.shape[0]
        curr_L0   = pos0[B-2, :].cpu().numpy()
        curr_R0   = pos0[B-1, :].cpu().numpy()
        print(f" curr_L={curr_L0}  curr_R={curr_R0}")

        # last targets
        print(f" last_L={last_L[0].cpu().numpy()}  last_R={last_R[0].cpu().numpy()}")

        # new targets
        print(f" new_L ={targets[0,0,:].cpu().numpy()}  new_R ={targets[0,1,:].cpu().numpy()}")


    # 4) visualize in USD
    if visualize:
        stage = omni.usd.get_context().get_stage()

        # 1) COM as big red sphere
        com = (13*amber.data.body_com_pos_w[:, 3, :]+3.4261*amber.data.body_com_pos_w[:, 4, :]
            +1.1526*amber.data.body_com_pos_w[:, 5, :]+3.4261*amber.data.body_com_pos_w[:, 6, :]
            +1.1526*amber.data.body_com_pos_w[:, 7, :] )/(13+2*3.4261+2*1.1526
        )  # [N,3]
        for i in range(N):
            path = f"/World/debug/com_sphere_{i}"
            if not stage.GetPrimAtPath(path):
                sph = UsdGeom.Sphere.Define(stage, path)
                sph.GetRadiusAttr().Set(0.1)
                sph.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
            else:
                sph = UsdGeom.Sphere(stage.GetPrimAtPath(path))
            UsdGeom.XformCommonAPI(sph).SetTranslate(Gf.Vec3d(*com[i].cpu().tolist()))

        # 2) Last left & right foot contacts
        # for i in range(N):
        #     # last left foot (blue)
        #     ll_path = f"/World/debug/last_left_foot_{i}"
        #     if not stage.GetPrimAtPath(ll_path):
        #         sph = UsdGeom.Sphere.Define(stage, ll_path)
        #         sph.GetRadiusAttr().Set(0.04)
        #         sph.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 0.0, 1.0)])
        #     else:
        #         sph = UsdGeom.Sphere(stage.GetPrimAtPath(ll_path))
        #     UsdGeom.XformCommonAPI(sph).SetTranslate(Gf.Vec3d(*last_L[i].cpu().tolist()))

        #     # last right foot (green)
        #     lr_path = f"/World/debug/last_right_foot_{i}"
        #     if not stage.GetPrimAtPath(lr_path):
        #         sph = UsdGeom.Sphere.Define(stage, lr_path)
        #         sph.GetRadiusAttr().Set(0.04)
        #         sph.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])
        #     else:
        #         sph = UsdGeom.Sphere(stage.GetPrimAtPath(lr_path))
        #     UsdGeom.XformCommonAPI(sph).SetTranslate(Gf.Vec3d(*last_R[i].cpu().tolist()))

        # 3) Future left & right foot targets

        for i in range(N):
            # future left foot (magenta)
            fl_path = f"/World/debug/future_left_foot_{i}"
            if not stage.GetPrimAtPath(fl_path):
                sph = UsdGeom.Sphere.Define(stage, fl_path)
                sph.GetRadiusAttr().Set(0.03)
                sph.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 1.0)])
            else:
                sph = UsdGeom.Sphere(stage.GetPrimAtPath(fl_path))
            UsdGeom.XformCommonAPI(sph).SetTranslate(Gf.Vec3d(*targets[i, 0, :].cpu().tolist()))

            # future right foot (yellow)
            fr_path = f"/World/debug/future_right_foot_{i}"
            if not stage.GetPrimAtPath(fr_path):
                sph = UsdGeom.Sphere.Define(stage, fr_path)
                sph.GetRadiusAttr().Set(0.03)
                sph.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 1.0, 0.0)])
            else:
                sph = UsdGeom.Sphere(stage.GetPrimAtPath(fr_path))
            UsdGeom.XformCommonAPI(sph).SetTranslate(Gf.Vec3d(*targets[i, 1, :].cpu().tolist()))

    # 5) flatten and return
    # print(targets.reshape(N, 6))
    pos0       = amber.data.body_pos_w          # [N, bodies, 3]
    B          = pos0.shape[1]
    curr_pos = amber.data.body_pos_w[:, [B-2,B-1], :]  # [N,2,3]

    # relative = desired_world − current_world
    rel_targets = targets - curr_pos                            # [N,2,3]
    rel_xz      = rel_targets[:, :, [0, 2]]                           # [N,2,2]
    # print(f"right target:{targets[:,1].cpu().tolist()} left target:{targets[:,0].cpu().tolist()}")
    # print(f"right pos   :{curr_pos[:,1].cpu().tolist()} left pos   :{curr_pos[:,0].cpu().tolist()}")
    # reshape to [N,4] in order [ΔLx, ΔLz, ΔRx, ΔRz]
    return rel_xz.view(N, 4)





@torch.no_grad()
def current_foot_positions(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Observation term returning the **current** left‐ and right‐foot positions
    in world‐frame.

    Outputs a tensor of shape [num_envs, 6]:
      [x_L, y_L, z_L,  x_R, y_R, z_R]
    where “left” is body index B-2 and “right” is B-1.
    """
    asset = env.scene[asset_cfg.name]
    # all body positions: [N, num_bodies, 3]
    pos_w = asset.data.body_pos_w
    B     = pos_w.shape[1]

    # left foot = index B-2, right foot = index B-1
    left  = pos_w[:, B-2, :]   # [N,3]
    right = pos_w[:, B-1, :]   # [N,3]

    # concatenate → [N,6]
    return torch.cat((left, right), dim=1)