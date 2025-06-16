from isaaclab.utils import configclass
from .cmd_cfg import HLIPCommandCfg
from .stair_cmd import StairCmd

Q_weights = [
    1.0,   300.0,    # com_x pos, vel
    100.0,   1.0,   # com_y pos, vel
    1.0,  1.0,  # com_z pos, vel
    1.0,    1.0,    # pelvis_roll pos, vel
    1.0,    1.0,    # pelvis_pitch pos, vel
    101.0,    11.0,    # pelvis_yaw pos, vel
    1000.0, 250.0,  # swing_x pos, vel
    1700.0,  125.0,  # swing_y pos, vel
    3500.0, 200.0,   # swing_z pos, vel
    100.0,    1.0,    # swing_ori_roll pos, vel
    100.0,    1.0,    # swing_ori_pitch pos, vel
    400.0,    10.0,    # swing_ori_yaw pos, vel
    10.0,    1.0,    # waist_yaw pos, vel
    10.0,1.0, #left sholder pitch
    10.0,1.0, #right sholder pitch
    10.0,1.0, #left sholder roll
    10.0,1.0, #right sholder roll
    10.0,1.0, #left sholder yaw
    10.0,1.0, #right sholder yaw
    10.0,1.0, #left elbow 
    10.0,1.0, #right elbow 
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
    z_sw_max: float = 0.15
    debug_vis: bool = False    # enable debug visualization