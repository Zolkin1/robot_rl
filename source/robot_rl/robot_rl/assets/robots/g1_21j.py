import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# From beyond mimic
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

import pathlib


def _find_project_root() -> pathlib.Path:
    """Walk up from this file until we find a directory containing 'robot_assets'."""
    path = pathlib.Path(__file__).resolve().parent
    while path != path.parent:
        if (path / "robot_assets").is_dir():
            return path
        path = path.parent
    raise FileNotFoundError("Could not find project root containing 'robot_assets'")


ROBOT_ASSETS = str(_find_project_root() / "robot_assets" / "g1")
# TODO: Fix warnings about waist_roll_link and yaw_link inertia and mass
G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_ASSETS}/g1_21j/g1_21j_merged/g1_21j_merged.usda",
        # usd_path=f"{ROBOT_ASSETS}/g1_21j/g1_21j_merged/g1_21j_merged_flat.usda",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Bend up
        # pos=(0.0, 0.0, 0.62), #0.63),
        # rot=(0.0, 0.67, 0.0, 0.73),  # XYZW format

        # Standing
        pos=(0.0, 0.0, 0.785),

        joint_pos={
            # Bend up
            # ".*_hip_pitch_joint": -2.0,
            # ".*_hip_roll_joint": 0.0,
            # ".*_hip_yaw_joint": 0.0,
            # ".*_knee_joint": 1.0,
            # ".*_ankle_pitch_joint": -0.48,
            # ".*_ankle_roll_joint": 0.0,
            # "waist_yaw_joint": 0.0,
            # "left_shoulder_yaw_joint": 0.0,
            # "left_shoulder_pitch_joint": 0.1,
            # "left_shoulder_roll_joint": 0.5,
            # "right_shoulder_yaw_joint": 0.0,
            # "right_shoulder_pitch_joint": 0.1,
            # "right_shoulder_roll_joint": -0.5,
            # ".*_elbow_joint": 0.7,  # 1.39,

            # Standing
            ".*_hip_pitch_joint": -0.25,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.46,
            ".*_ankle_pitch_joint": -0.25,
            ".*_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.31,
            "left_shoulder_roll_joint": 0.24,
            "right_shoulder_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.31,
            "right_shoulder_roll_joint": -0.24,
            ".*_elbow_joint": 0.8, #1.39,

            # ".*_wrist_roll_joint": 0.,
            # ".*_wrist_pitch_joint": 0.,
            # ".*_wrist_yaw_joint": 0.,
            # "left_one_joint": 1.0,
            # "right_one_joint": -1.0,
            # "left_two_joint": 0.52,
            # "right_two_joint": -0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_yaw_joint": STIFFNESS_7520_14, #100.0,
                ".*_hip_roll_joint": STIFFNESS_7520_22, #100.0,
                ".*_hip_pitch_joint": STIFFNESS_7520_14, #100.0,
                ".*_knee_joint": STIFFNESS_7520_22, #150.0,
            },
            damping={
                ".*_hip_yaw_joint": DAMPING_7520_14, #2.0,
                ".*_hip_roll_joint": DAMPING_7520_22, #2.0,
                ".*_hip_pitch_joint":DAMPING_7520_14, # 2.0,
                ".*_knee_joint": DAMPING_7520_22, #4.0,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
                # ".*_hip_.*": 0.01,
                # ".*_knee_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
            # stiffness=40.0,
            # damping=2.0,
            # armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
            # stiffness=100.0,
            # damping=2.0,
            # armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                # ".*_shoulder_pitch_joint": 100.0,
                # ".*_shoulder_roll_joint": 100.0,
                # ".*_shoulder_yaw_joint": 50.0,
                # ".*_elbow_joint": 50.0,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                # ".*_shoulder_pitch_joint": 2.0,
                # ".*_shoulder_roll_joint": 2.0,
                # ".*_shoulder_yaw_joint": 2.0,
                # ".*_elbow_joint": 2.0,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                # ".*_shoulder_.*": 0.01,
                # ".*_elbow_.*": 0.01,
            },
        ),
    },
)
"""Configuration for the Unitree G1 Humanoid robot."""

G1_ACTION_SCALE = {}
for a in G1_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            G1_ACTION_SCALE[n] = 0.125 * e[n] / s[n]

G1_MINIMAL_CFG = G1_CFG.copy()


