import numpy as np
import torch
from pxr import Gf, UsdGeom
from isaaclab.utils.math import yaw_quat, quat_rotate, quat_inv
import omni.usd
import math




def compute_step_location_direct(
    sim_time: float,
    scene,
    args_cli,
    des_vel: np.ndarray,      # shape (3,) → [vx, vy, vyaw]
    nom_height: float,
    Tswing: float,
    wdes: float,
    visualize: bool = True
) -> torch.Tensor:
    """
    Compute Amber’s next footstep in world‐frame, given raw sim_time, scene, and a direct
    base-velocity command (no Hydra env needed).
    Returns a tensor [1,3] of the desired foothold.
    """
    amber = scene["Amber"]
    device = amber.data.default_root_state.device
    stage = omni.usd.get_context().get_stage()
    
    def _update_arrow_x(prim_base: str,
                        origin: np.ndarray,
                        vec: np.ndarray,
                        color: tuple):
        """
        Draw / update a single arrow (shaft + cone) whose length, orientation and
        colour are set from `vec`, but **only the x-component** of vec is used.
        Uses a fixed prim path so we overwrite the same geometry each frame.
        """
        stage = omni.usd.get_context().get_stage()

        # keep only vx, sign preserves direction along world-X
        vx = float(vec[0])
        vec = np.array([vx, 0.0, 0.0], dtype=np.float64)

        length = abs(vx)
        if length < 1e-4:
            length = 1e-4                                            # non-zero

        # unit +X or -X
        dir_n = np.array([np.sign(vx) if vx != 0 else 1.0, 0.0, 0.0])

        shaft_path = prim_base + "/shaft"
        head_path  = prim_base + "/head"

        # create the prims once
        if not stage.GetPrimAtPath(shaft_path):
            UsdGeom.Cylinder.Define(stage, shaft_path).GetRadiusAttr().Set(0.01)
            cone = UsdGeom.Cone.Define(stage, head_path)
            cone.GetRadiusAttr().Set(0.02)
            cone.GetHeightAttr().Set(0.04)

        # fetch prims as Xformables
        shaft_xf = UsdGeom.Xformable(stage.GetPrimAtPath(shaft_path))
        head_xf  = UsdGeom.Xformable(stage.GetPrimAtPath(head_path))

        # clear previous ops (avoids stacking new ops each frame)
        shaft_xf.ClearXformOpOrder()
        head_xf.ClearXformOpOrder()

        # shaft: scale Z to 0.8*length, translate to mid-point
        shaft_xf.AddScaleOp().Set(Gf.Vec3f(1, 1, length * 0.4))
        mid = origin + dir_n * (length * 0.2)
        shaft_xf.AddTranslateOp().Set(Gf.Vec3d(*mid.tolist()))
        UsdGeom.Cylinder(shaft_xf.GetPrim()).GetDisplayColorAttr().Set([color])

        # head: translate to (origin + 0.8*vec + 0.02 along dir), no need to orient
        # tip = origin + dir_n * (length * 0.8 + 0.02)
        # head_xf.AddTranslateOp().Set(Gf.Vec3d(*tip.tolist()))
        # UsdGeom.Cone(head_xf.GetPrim()).GetDisplayColorAttr().Set([color])

    # device = torch.device(scene._sim.device)  # grab the torch device from your sim
    # 1) current COM velocity (green)
    com_pos   = amber.data.body_pos_w.cpu().numpy()[0,3,:]
    com_vel   = amber.data.body_link_lin_vel_w.cpu().numpy()[0,3,:]
    # _update_arrow_x("/World/debug/vel_arrow", com_pos, com_vel, (0.0, 1.0, 0.0))

    # 2) commanded base velocity (blue)
    # rotate local [vx,vy,0] into world
    root_quat = amber.data.root_quat_w[0]                    # => Tensor on device
    cmd_local = torch.tensor([*args_cli.desired_vel[:2], 0.0],
                         dtype=torch.float32, device=device)
    # use your quat_rotate / yaw_quat helpers:
    cmd_world = quat_rotate(yaw_quat(root_quat), cmd_local)
    cmd_world_np = cmd_world.cpu().numpy()
    # _update_arrow_x("/World/debug/cmd_arrow", com_pos, cmd_world_np, (0.0, 0.0, 1.0))

    # 1) “command” is just your desired_vel XY
    vx = torch.tensor([[des_vel[0]]], device=device)  # shape (1,1)    # pad to 3D so we can reuse the same code
    # cmd3 = torch.cat([cmd, torch.zeros((1,1),device=device)], dim=1)  # [1,3]

    # 2) COM position from Amber’s body_pos_w slot #3
    r = amber.data.body_pos_w[:, 3, :]    # [1,3]
    print(r)
    # 3) ICP‐initial
    g     = 9.81
    omega = math.sqrt(g / nom_height)
    icp0 = torch.zeros((1,3), device=device)
    icp0[0,0] = vx / omega    # x-axis only

    # 4) stance foot pos (zero Z)
    pos = amber.data.body_pos_w           # [1, bodies, 3]
    B   = pos.shape[1]
    feet = pos[:, [B-1, B-2], :]          # [1,2,3]
    # print(feet)
    # phase clock
    tp    = (sim_time % (2*Tswing)) / (2*Tswing)
    phi_c = math.sin(2*math.pi*tp) / math.sqrt(math.sin(2*math.pi*tp)**2 + Tswing)
    idx   = int((0.5 - 0.5 * np.sign(phi_c)))
    stance = feet[:, idx, :].clone()
    stance[0,2] = 0.0
    # print("stance:",stance)
    # print("phase",phi_c)
    # 5) local/global transforms
    def to_local(v):
        return quat_rotate(
            yaw_quat(quat_inv(amber.data.root_quat_w)), v
        )
    def to_global(v):
        return quat_rotate(
            yaw_quat(amber.data.root_quat_w), v
        )

    # 6) capture‐point final in local
    expT = math.exp(omega * Tswing)
    icpf = expT*icp0 + (1-expT)*to_local(r - stance)
    icpf[0,2] = 0.0

    # 7) offset b
    sd = abs(des_vel[0])*Tswing
    bx = sd / (expT - 1.0)
    # y-bias removed
    b  = torch.tensor([[bx, 0.0, 0.0]], device=device)

    # 8) clip
    p_local = icpf.clone()
    p_local[0,0] = torch.clamp(icpf[0,0]-b[0,0], -0.5, 0.5)
    p_local[0,1] = 0.0   # Y always zero for planar x-motion
    # 9) back to world
    p = to_global(p_local) + r
    p[0,1] = 0.0  # force Y in world to 0
    p[0,2] = 0.0

    # 10) optional USD viz
    if visualize:
        path = "/World/debug/future_step_0"
        if not stage.GetPrimAtPath(path):
            sph = UsdGeom.Sphere.Define(stage, path)
            sph.GetRadiusAttr().Set(0.02)
        else:
            sph = UsdGeom.Sphere(stage.GetPrimAtPath(path))
        coords = p[0].cpu().numpy().tolist()    # -> [float, float, float]
        UsdGeom.XformCommonAPI(sph).SetTranslate(
            Gf.Vec3d(*coords)
        )
        com_path = "/World/debug/com_sphere"
        if not stage.GetPrimAtPath(com_path):
            com_sph = UsdGeom.Sphere.Define(stage, com_path)
            com_sph.GetRadiusAttr().Set(0.02)
        else:
            com_sph = UsdGeom.Sphere(stage.GetPrimAtPath(com_path))
        # grab COM world‐position from body_pos_w index 3
        com_xyz = amber.data.body_pos_w.cpu().numpy()[0, 3, :].tolist()  # [x,y,z]
        UsdGeom.XformCommonAPI(com_sph).SetTranslate(
            Gf.Vec3d(*com_xyz)
        )
        # tint it red
        com_sph.GetDisplayColorAttr().Set([(1.0, 0.0, 0.0)])
    return p  # a [1,3] tensor