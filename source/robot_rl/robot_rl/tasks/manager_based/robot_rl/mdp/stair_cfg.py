from isaaclab.utils import configclass
from .cmd_cfg import HLIPCommandCfg
from .stair_cmd import StairCmd

Q_weights = [
    25.0,   200.0,    # com_x pos, vel
    300.0,   50.0,   # com_y pos, vel
    600.0,  20.0,  # com_z pos, vel
    420.0,    20.0,    # pelvis_roll pos, vel
    200.0,    10.0,    # pelvis_pitch pos, vel
    300.0,    10.0,    # pelvis_yaw pos, vel
    1500.0, 125.0,  # swing_x pos, vel
    1700.0,  125.0,  # swing_y pos, vel
    3500.0, 100.0,   # swing_z pos, vel
    30.0,    1.0,    # swing_ori_roll pos, vel
    10.0,    1.0,    # swing_ori_pitch pos, vel
    400.0,    10.0,    # swing_ori_yaw pos, vel
    500.0,    10.0,    # waist_yaw pos, vel
    40.0,1.0, #left sholder pitch
    40.0,1.0, #right sholder pitch
    100,1.0, #left sholder roll
    100,1.0, #right sholder roll
    50,1.0, #left sholder yaw
    50,1.0, #right sholder yaw
    30.0,1.0, #left elbow 
    30.0,1.0, #right elbow 
]


R_weights = [
        0.1, 0.1, 0.1,    # CoM inputs: allow moderate effort
        0.05,0.05,0.05,   # pelvis inputs: lower torque priority
        0.05,0.05,0.05,   # swing foot linear inputs
        0.02,0.02,0.02,    # swing foot orientation inputs: small adjustments
        0.1,0.01,0.01,
        0.01,0.01,0.01,
        0.01,0.01,0.01,
    ]

@configclass
class StairHLIPCommandCfg(HLIPCommandCfg):
    """Commands for the G1 Stair environment."""
    class_type: type = StairCmd
    Q_weights = Q_weights
    R_weights = R_weights
    debug_vis: bool = False    # enable debug visualization