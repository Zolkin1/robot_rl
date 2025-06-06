from .hlip_cmd import HLIPCommandTerm
from .hzd_cmd import HZDCommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
import torch

Q_weights = [
    25.0,   200.0,    # com_x pos, vel
    200.0,   50.0,   # com_y pos, vel
    300.0,  10.0,  # com_z pos, vel
    400.0,    100.0,    # pelvis_roll pos, vel
    60.0,    10.0,    # pelvis_pitch pos, vel
    200.0,    100.0,    # pelvis_yaw pos, vel
    1500.0, 125.0,  # swing_x pos, vel
    1500.0,  125.0,  # swing_y pos, vel
    2500.0, 25.0,   # swing_z pos, vel
    20.0,    1.0,    # swing_ori_roll pos, vel
    5.0,    1.0,    # swing_ori_pitch pos, vel
    200.0,    1.0,    # swing_ori_yaw pos, vel
    500.0,    10.0,    # waist_yaw pos, vel
    15.0,1.0, #left sholder pitch
    15.0,1.0, #right sholder pitch
    100,1.0, #left sholder roll
    100,1.0, #right sholder roll
    100,1.0, #left sholder yaw
    100,1.0, #right sholder yaw
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
    T_ds: float = 0.0          # double support duration (s)
    z0: float = 0.65           # CoM height (m)
    y_nom: float = 0.2        # nominal lateral foot offset (m)
    gait_period: float = 0.8   # gait cycle period (s)
    debug_vis: bool = False    # enable debug visualization
    z_sw_max: float = 0.14    # max swing foot z height (m); this is ankle height so different from actual foot position
    z_sw_min: float = 0.0
    v_history_len: int = 5
    pelv_pitch_ref: float = 0.1
    resampling_time_range: tuple[float, float] = (5.0, 15.0)  # Resampling time range in seconds
    # Command sampling ranges
    ranges: dict = {
        "pos_x": (-0.25, 0.25),
        "pos_y": (0.2, 0.3),
        "pos_z": (0.0, 0.5),
        "yaw": (-0.7, 0.7),
        "timing": (0.5, 1.5),
    }

    # Visualization configurations
    footprint_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/footprint",
        markers={
            "foot": sim_utils.CuboidCfg(
                size=(0.1, 0.065, 0.018),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
            )
        }
    )

    foot_body_name: str = ".*_ankle_roll_link"
    const_joint_name: list[str] = [
                    "waist_yaw_joint",
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                ]

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_pose",
        markers={
            "goal": sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
            )
        }
    )
    # scale marker
    goal_pose_visualizer_cfg.markers["goal"].scale = (0.1, 0.1, 0.1)

    current_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/current_pose",
        markers={
            "current": sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))
            )
        }
    )
    current_pose_visualizer_cfg.markers["current"].scale = (0.1, 0.1, 0.1)

    Q_weights = Q_weights
    R_weights = R_weights



HZD_Q_weights = [
    1.0,1.0,#base_x
    1.0,1.0,#base_y
    1.0,1.0,#base_z
    1.0,1.0,#base_roll
    1.0,1.0,#base_pitch
    1.0,1.0,#base_yaw
    1.0,1.0,#left hip
    1.0,1.0,#right hip
    1.0,1.0,#left_hip 2
    1.0,1.0,#right hip 2
    1.0,1.0,#left hip 3
    1.0,1.0,#right hip 3
    1.0,1.0,#left knee
    1.0,1.0,#right knee
    1.0,1.0,#left ankle
    1.0,1.0,#right ankle
    1.0,1.0,#left ankle 2
    1.0,1.0,#right ankle 2
]


HZD_R_weights = [
      1.0,#base x
      1.0,#base y
      1.0,#base z
      1.0,#base roll
      1.0,#base pitch
      1.0,#base yaw
      1.0,#left hip
      1.0,#right hip
      1.0,#left_hip 2
      1.0,#right hip 2
      1.0,#left hip 3
      1.0,#right hip 3
      1.0,#left knee
      1.0,#right knee
      1.0,#left ankle
      1.0,#right ankle
      1.0,#left ankle 2
      1.0,#right ankle 2
    ]


@configclass
class HZDCommandCfg(CommandTermCfg):
    """
    Configuration for the HZDCommandTerm.
    """
    class_type: type = HZDCommandTerm
    asset_name: str = "robot"
    foot_body_name: str = ".*_ankle_roll_link"
    num_joints: int = 12
    num_coeffs: int = 8
    joint_patterns: list = [".*HipJoint", ".*KneeJoint", ".*AnkleJoint"]  # Regex patterns to match joint names
    resampling_time_range: tuple[float, float] = (5.0, 15.0)
    traj_coeff: dict[str, torch.Tensor] = {}
    traj_coeff_remap: dict[str, torch.Tensor] = {}
    debug_vis: bool = False
    trajectory_tracking_visualizer_cfg: dict = {}
    Q_weights = HZD_Q_weights
    R_weights = HZD_R_weights
