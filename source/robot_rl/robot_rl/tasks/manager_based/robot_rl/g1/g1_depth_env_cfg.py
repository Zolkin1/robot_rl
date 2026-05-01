"""G1 walking-CLF env variants with a forward-facing depth camera.

Two flavours:

- :class:`G1WalkingCLFDepthScanEnvCfg` adds a flattened ``depth_scan`` term to
  the policy observation group.
- :class:`G1WalkingCLFDepthImageEnvCfg` removes the flattened term and instead
  exposes the depth as an image-shape observation in its own ``depth_image``
  group, suitable for CNN backbones.

Migration notes (vs. legged_locomotion_rl):

- The custom ``SelfIntersectionRayCasterCfg`` is replaced by IsaacLab v3.0.0-beta's
  :class:`MultiMeshRayCasterCameraCfg`. ``RaycastTargetCfg(track_mesh_transforms=True)``
  on the per-leg-link entries gives self-occlusion against the dynamic robot meshes.
- ``mdp.depth_image`` no longer projects ray hits onto the forward axis manually;
  the camera reports ``distance_to_image_plane`` natively. Our wrapper just clips
  and applies the ``(d - offset) * scale`` normalisation.
- ``DepthNoiseCfg`` (plane tilt + per-pixel noise + Gaussian blur + dropout) is
  ported verbatim under ``robot_rl.tasks.manager_based.robot_rl.mdp.sensors``.
- ``randomize_camera_intrinsics`` is ported under
  ``robot_rl.tasks.manager_based.robot_rl.mdp.events.depth``, adapted to the
  new :meth:`RayCasterCamera.set_intrinsic_matrices` API.
"""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCameraCfg, patterns
from isaaclab.utils import configclass

from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.mdp.sensors import DepthNoiseCfg

from .g1_clf_tracking_base import (
    G1ClfTrackingEventsCfg,
    G1ClfTrackingObservationsCfg,
    G1ClfTrackingSceneCfg,
)
from .g1_walking_clf_env_cfg import G1WalkingCLFEnvCfg


# ---------------------------------------------------------------------------
# Depth camera sensor
# ---------------------------------------------------------------------------

# Pinhole intrinsics (3x3 K, flattened) — matches the legged_locomotion_rl reference.
_DEPTH_K = [25.9928, 0.0, 14.9936, 0.0, 27.8703, 12.9966, 0.0, 0.0, 1.0]
_DEPTH_W, _DEPTH_H = 30, 26
# 65° pitch-down. Reference quat in (w, x, y, z) was (0.843391, 0, 0.537300, 0);
# IsaacLab beta's OffsetCfg.rot expects (x, y, z, w).
_DEPTH_ROT_XYZW = (0.0, 0.537300, 0.0, 0.843391)
_DEPTH_OFFSET = (0.153246, 0.0, 0.106799)


def _make_depth_camera() -> MultiMeshRayCasterCameraCfg:
    # G1's USD nests links under /Geometry/pelvis_link/<chain>. find_matching_prims
    # matches each `/`-separated token as a standalone regex against the prim name,
    # so `.*` does not cross hierarchy boundaries — we have to spell out the chain.
    _LEG_PARENT = (
        "{ENV_REGEX_NS}/Robot/Geometry/pelvis_link"
        "/.*_hip_pitch_link/.*_hip_roll_link/.*_hip_yaw_link"
    )
    return MultiMeshRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/waist_yaw_link",
        update_period=1.0 / 30.0,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(
            pos=_DEPTH_OFFSET,
            rot=_DEPTH_ROT_XYZW,
            convention="world",
        ),
        data_types=["distance_to_image_plane"],
        max_distance=10.0,
        depth_clipping_behavior="max",
        pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
            intrinsic_matrix=_DEPTH_K,
            width=_DEPTH_W,
            height=_DEPTH_H,
        ),
        mesh_prim_paths=[
            "/World/ground",
            # hip_roll
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Robot/Geometry/pelvis_link/.*_hip_pitch_link/.*_hip_roll_link",
                track_mesh_transforms=True,
            ),
            # hip_yaw
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr=f"{_LEG_PARENT}",
                track_mesh_transforms=True,
            ),
            # knee
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr=f"{_LEG_PARENT}/.*_knee_link",
                track_mesh_transforms=True,
            ),
            # ankle_roll
            MultiMeshRayCasterCameraCfg.RaycastTargetCfg(
                prim_expr=f"{_LEG_PARENT}/.*_knee_link/.*_ankle_pitch_link/.*_ankle_roll_link",
                track_mesh_transforms=True,
            ),
        ],
        debug_vis=True,
    )


def _make_depth_noise() -> DepthNoiseCfg:
    """Factory for the ported `DepthNoiseCfg` (plane + uniform + blur + dropout).

    Currently unused — see the explanatory note in
    `G1WalkingCLFDepthScanObservationsCfg.PolicyCfg.depth_scan` about why
    ``noise=None`` is set for the depth obs terms. This factory is kept ready
    so noise can be re-enabled after the IsaacLab beta `obs_manager.py:626`
    `ResolvableString` issue is resolved (either upstream or via a local
    Modifier-based wrapper).
    """
    return DepthNoiseCfg(
        max_plane_deviation=0.05,
        dropout_prob=0.02,
        dropout_val=0.0,
        height=_DEPTH_H,
        width=_DEPTH_W,
    )


# ---------------------------------------------------------------------------
# Scene / observations / events
# ---------------------------------------------------------------------------

@configclass
class G1WalkingCLFDepthScanScene(G1ClfTrackingSceneCfg):
    """Walking scene + depth camera attached to the waist link."""

    depth_camera: MultiMeshRayCasterCameraCfg = _make_depth_camera()


@configclass
class G1WalkingCLFDepthScanObservationsCfg(G1ClfTrackingObservationsCfg):
    """Adds a flattened ``depth_scan`` term to the policy group."""

    @configclass
    class PolicyCfg(G1ClfTrackingObservationsCfg.PolicyCfg):
        # NOTE: `noise=DepthNoiseCfg(...)` would normally go here. Temporarily
        # left as `None` because IsaacLab v3.0.0-beta's `class_to_dict` /
        # `update_class_from_dict` round-trip converts a custom `class_type`
        # field into a `ResolvableString` (str subclass), and
        # `observation_manager.py:626` then does
        # ``issubclass(class_type, NoiseModel)`` without resolving — which
        # raises ``TypeError: issubclass() arg 1 must be a class``.
        # See `mdp/sensors/noise.py` for the ported `DepthNoiseCfg` (kept
        # ready for re-enable) and `_make_depth_noise()` below.
        depth_scan = ObsTerm(
            func=mdp.depth_image,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "offset": 0.75,
                "scale": 0.5,
            },
            noise=None,
            clip=(-2.0, 2.0),
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1WalkingCLFDepthEventCfg(G1ClfTrackingEventsCfg):
    """Walking events + camera-intrinsics randomisation on reset."""

    randomize_camera_intrinsics = EventTerm(
        func=mdp.randomize_camera_intrinsics,
        mode="reset",
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera"),
            "focal_length_distribution_params": (0.98, 1.02),
            "principal_point_distribution_params": (0.98, 1.02),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


# ---------------------------------------------------------------------------
# Env classes
# ---------------------------------------------------------------------------

@configclass
class G1WalkingCLFDepthScanEnvCfg(G1WalkingCLFEnvCfg):
    """Walking + flattened depth scan in the policy observation group."""

    scene: G1WalkingCLFDepthScanScene = G1WalkingCLFDepthScanScene(num_envs=4096, env_spacing=2.5)
    observations: G1WalkingCLFDepthScanObservationsCfg = G1WalkingCLFDepthScanObservationsCfg()
    events: G1WalkingCLFDepthEventCfg = G1WalkingCLFDepthEventCfg()


@configclass
class G1WalkingCLFDepthImageObservationsCfg(G1WalkingCLFDepthScanObservationsCfg):
    """Replaces the flattened scan with an image-shape obs in its own group."""

    @configclass
    class PolicyCfg(G1WalkingCLFDepthScanObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            # Flat scan moves out of the policy group; image is its own group.
            self.depth_scan = None

    @configclass
    class DepthImageCfg(ObsGroup):
        # See note on `depth_scan` in the parent cfg about `noise=None`.
        depth_image = ObsTerm(
            func=mdp.depth_image_4d,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "offset": 0.75,
                "scale": 0.5,
                "height": _DEPTH_H,
                "width": _DEPTH_W,
            },
            noise=None,
            clip=(-2.0, 2.0),
        )

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = True
            # Single-term group: concat=True is a no-op (cat of one tensor),
            # so the (B, 1, H, W) shape is preserved. Setting concat=False
            # would yield a TensorDict, which rsl_rl's `check_nan` and our
            # train.py debug print don't dispatch on.

    policy: PolicyCfg = PolicyCfg()
    depth_image: DepthImageCfg = DepthImageCfg()


@configclass
class G1WalkingCLFDepthImageEnvCfg(G1WalkingCLFDepthScanEnvCfg):
    """Walking + image-shape depth in its own ``depth_image`` group."""

    observations: G1WalkingCLFDepthImageObservationsCfg = G1WalkingCLFDepthImageObservationsCfg()


# ---------------------------------------------------------------------------
# PLAY variants — mirror the trims in G1WalkingCLFEnvCfg_PLAY.
# ---------------------------------------------------------------------------

def _apply_play_trims(cfg) -> None:
    cfg.scene.num_envs = 2
    cfg.scene.env_spacing = 2.5
    cfg.observations.policy.enable_corruption = False
    cfg.scene.terrain.size = (3, 3)
    cfg.scene.terrain.border_width = 0.0
    cfg.scene.terrain.num_rows = 3
    cfg.scene.terrain.num_cols = 2

    cfg.episode_length_s = 8.0

    cfg.events.randomize_ground_contact_friction = None
    cfg.events.add_base_mass = None
    cfg.events.base_com = None
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.events.gain_randomization = None
    cfg.events.randomize_camera_intrinsics = None

    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.4)
    cfg.commands.base_velocity.debug_vis = False


@configclass
class G1WalkingCLFDepthScanEnvCfg_PLAY(G1WalkingCLFDepthScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_trims(self)


@configclass
class G1WalkingCLFDepthImageEnvCfg_PLAY(G1WalkingCLFDepthImageEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_play_trims(self)
