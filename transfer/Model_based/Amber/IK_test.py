#!/usr/bin/env python3
"""
quick_ik_test.py  – run Amber IK once with hard-coded data
"""

import casadi as ca
import numpy as np

# 1)  Load the CasADi IK function  (make sure path is correct!)
ik_fun = ca.Function.load("transfer/Model_based/Amber/amber_reference_step.casadi")

# ---------------------------------------------------------------------
# 2)  Hard-coded inputs  (taken from your debug print)
# ---------------------------------------------------------------------
phase   = ca.DM(0.5)  # halfway through the half-cycle; tweak if you like

# foot_w order  = [Lx, Rx, Ly, Ry, Lz, Rz]
foot_w  = ca.DM([
    -3.77379e-08,     # Lx
    -2.98198e-08,     # Rx
    -0.0968508,       # Ly
     0.100716,        # Ry
     0.2,             # Lz
     0.0              # Rz
])
# foot_w  = ca.DM([
#     0.,     # Lx
#     0.,     # Rx
#     -0.0968508,       # Ly
#      0.100716,        # Ry
#      0.0,             # Lz
#      0.0              # Rz
# ])
com_x   = 0#3.552713678800501e-15
com_y   = 0#5.05611404187088e-24
com_z   = 1.3963213682174683

# current joint guess (4×1)   – here just zeros
q_cur   = ca.DM.zeros(4, 1)

# ---------------------------------------------------------------------
# 3)  Call IK and print results
# ---------------------------------------------------------------------
q_ref, p_body = ik_fun(phase, foot_w, com_x, com_y, com_z, q_cur)

print("IK angles (deg):",       np.array(q_ref).flatten())
print("foot positions in body:", np.array(p_body).flatten())  # optional
