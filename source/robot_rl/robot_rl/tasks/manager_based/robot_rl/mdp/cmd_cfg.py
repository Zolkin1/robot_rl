from .hlip_cmd import HLIPCommandTerm
from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
@configclass
class HLIPCommandCfg(CommandTermCfg):
    """
    Configuration for the HLIPCommandTerm.
    """
    class_type: type = HLIPCommandTerm
    asset_name: str = "robot"
    T_ds: float = 0.0          # double support duration (s)
    z0: float = 0.78           # CoM height (m)
    y_nom: float = 0.3        # nominal lateral foot offset (m)
    gait_period: float = 0.6   # gait cycle period (s)
    debug_vis: bool = True    # enable debug visualization

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
