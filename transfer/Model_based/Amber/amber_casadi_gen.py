#!/usr/bin/env python3
import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin
import numpy as np
import os

# ---------------------------------------------------------------------
# 1) User‐specified constants
# ---------------------------------------------------------------------
URDF      = "/home/s-ritwik/src/cusadi/Amber/amber_free.urdf"
FOOT_FRAMES = ["left_toe", "right_toe"]
DAMPING   = 1e-4
IK_ITERS  = 20
IK_TOL    = 1e-6

# ---------------------------------------------------------------------
# 2) Build numeric Pinocchio model (for frame IDs)
# ---------------------------------------------------------------------
model_num = pin.buildModelFromUrdf(URDF)
data_num  = model_num.createData()
frame_ids = [model_num.getFrameId(f) for f in FOOT_FRAMES]

# ---------------------------------------------------------------------
# 3) Build CasADi‐compatible model + data
# ---------------------------------------------------------------------
cmodel = cpin.Model(model_num)
cdata  = cmodel.createData()

# ---------------------------------------------------------------------
# 4) Declare SX symbols for inputs (now a full 6‐vector of foot world pos)
# ---------------------------------------------------------------------
phase   = ca.SX.sym("phase")
foot_w  = ca.SX.sym("foot_w", 6)            # [Lx Rx Ly Ry Lz Rz]
x_com   = ca.SX.sym("com_x")
y_com   = ca.SX.sym("com_y")
com_z   = ca.SX.sym("com_z")                # NEW
q_cur   = ca.SX.sym("q_cur", 4)
p_com = ca.vertcat(x_com, y_com, com_z)     # uses com_z now

# ---------------------------------------------------------------------
# 5) Reshape & build body‐frame targets
# ---------------------------------------------------------------------
# 5.1) reshape into a (2×3) SX matrix of world‐foot positions
foot_pos_world = ca.reshape(foot_w, 2, 3)

# 5.2) subtract COM and add vertical swing
foot_body = ca.SX.zeros(2, 3)
for i in range(2):
    pw  = foot_pos_world[i, :].T                 # SX(3×1)
    com = ca.vertcat(x_com, y_com, com_z)        # SX(3×1)   ← NEW
    pb  = pw - com                               # SX(3×1)
    foot_body[i, :] = pb.T                       # store row

# Flatten to a 6×1 vector (optional second output, unchanged)
foot_body_flat = ca.reshape(foot_body, 6, 1)

# ---------------------------------------------------------------------
# 6) IK: damped‐least‐squares per foot (identical to your version)
# ---------------------------------------------------------------------
q = q_cur  # SX(4×1)
for k, fid in enumerate(frame_ids):
    tgt_k = foot_body[k, :].T  # SX(3×1)
    for _ in range(IK_ITERS):
        # forward kinematics
        cpin.forwardKinematics(cmodel, cdata, q)
        cpin.updateFramePlacements(cmodel, cdata)
        # position error
        p_cur   = cdata.oMf[fid].translation   # SX(3×1)
        err     = tgt_k - p_cur                # SX(3×1)
        norm_e  = ca.norm_2(err)               # SX scalar
        # Jacobian & DLS step
        J6      = cpin.computeFrameJacobian(
                     cmodel, cdata, q, fid,
                     pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                  )                            # SX(6×4)
        Jpos    = J6[0:3, :]                   # SX(3×4)
        λI      = DAMPING * ca.SX.eye(3)       # SX(3×3)
        Jdls    = Jpos.T @ ca.inv(Jpos@Jpos.T + λI)  # SX(4×3)
        dq      = Jdls @ err                   # SX(4×1)
        # masked update (stop if below tol)
        q       = ca.if_else(norm_e < IK_TOL, q, q + dq)

q_ref = q  # final SX(4×1)

# ---------------------------------------------------------------------
# 7) Build & save the CasADi function “amber_reference_step”
# ---------------------------------------------------------------------
F = ca.Function(
    "amber_reference_step",
    [phase, foot_w, x_com, y_com, com_z, q_cur],
    [q_ref, foot_body_flat],
)
out_name = "amber_reference_step.casadi"
F.save(out_name)
print(f"[+] Wrote “{out_name}” at {os.path.abspath(out_name)}")
