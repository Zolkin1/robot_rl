from .hlip_cmd import HLIPCommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils


Q_weights = [
    25.0,   0.0,    # com_x pos, vel
    25.0,   2.22,   # com_y pos, vel
    100.0,  300.0,  # com_z pos, vel
    4.0,    4.0,    # pelvis_roll pos, vel
    8.0,    4.0,    # pelvis_pitch pos, vel
    4.0,    4.0,    # pelvis_yaw pos, vel
    1500.0, 125.0,  # swing_x pos, vel
    500.0,  125.0,  # swing_y pos, vel
    2000.0, 25.0,   # swing_z pos, vel
    4.0,    0.0,    # swing_ori_roll pos, vel
    0.0,    0.0,    # swing_ori_pitch pos, vel
    0.0,    0.0,    # swing_ori_yaw pos, vel
]


R_weights = [
        0.1, 0.1, 0.1,    # CoM inputs: allow moderate effort
        0.05,0.05,0.05,   # pelvis inputs: lower torque priority
        0.05,0.05,0.05,   # swing foot linear inputs
        0.02,0.02,0.02    # swing foot orientation inputs: small adjustments
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
    y_nom: float = 0.25        # nominal lateral foot offset (m)
    gait_period: float = 0.8   # gait cycle period (s)
    debug_vis: bool = False    # enable debug visualization
    z_sw_max: float = 0.14    # max swing foot z height (m); this is ankle height so different from actual foot position
    z_sw_min: float = 0.0
    pelv_pitch_ref: float = 0.2
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
