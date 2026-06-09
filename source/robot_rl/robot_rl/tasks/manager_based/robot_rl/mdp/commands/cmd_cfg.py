# Copyright 2026 Zachary Olkin. All rights reserved.

from .hlip_cmd import HLIPCommandTerm
from isaaclab.utils import configclass
from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils


Q_weights = [
    25.0,   200.0,    # com_x pos, vel
    300.0,   50.0,   # com_y pos, vel
    400.0,  10.0,  # com_z pos, vel
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
class HLIPCommandCfg(CommandTermCfg):
    """
    Configuration for the HLIPCommandTerm.
    """
    class_type: type = HLIPCommandTerm
    asset_name: str = "robot"
    yaw_idx: list[int] = [5,11]
    T_ds: float = 0.0          # double support duration (s)
    z0: float = 0.67           # CoM height (m)
    y_nom: float = 0.25        # nominal lateral foot offset (m)
    gait_period: float = 0.8   # gait cycle period (s)
    debug_vis: bool = False    # enable debug visualization
    z_sw_max: float = 0.1    # max swing foot z height (m); this is ankle height so different from actual foot position
    z_sw_min: float = 0.0
    v_history_len: int = 5
    pelv_pitch_ref: float = 0.0
    waist_yaw_ref: float = 0.0
    shoulder_ref: list[float] = [0.16, 0.0, 0.0]
    elbow_ref: float = 0.1
    foot_target_range_y: list[float] = [0.1,0.5]
    resampling_time_range: tuple[float, float] = (5.0, 15.0)  # Resampling time range in seconds

    # Visualization configurations
    footprint_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/footprint",
        markers={
            "foot": sim_utils.CuboidCfg(
                size=(0.1, 0.065, 0.018),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
            )
        }
    )

    foot_body_name: str = ".*_ankle_roll_link"
    upper_body_joint_name: list[str] = [
                    "waist_yaw_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                ]

    Q_weights = Q_weights
    R_weights = R_weights





