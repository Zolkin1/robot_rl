import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ROBOT_ASSETS = "robot_assets/g1"
# TODO: Fix warnings about waist_roll_link and yaw_link inertia and mass
G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_ASSETS}/g1_21j_urdf_v3_min_contacts.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.01, rest_offset=0.0),

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),   # TODO: Consider setting this higher if I init issues
        joint_pos={
            ".*_hip_pitch_joint": -0.42,
            ".*_hip_roll_joint": 0.,
            ".*_hip_yaw_joint": 0.,
            ".*_knee_joint": 0.81,
            ".*_ankle_pitch_joint": -0.4,
            ".*_ankle_roll_joint": 0.,
            "waist_yaw_joint": 0.,
            "left_shoulder_pitch_joint": 0.,
            "left_shoulder_roll_joint": 0.27,
            "right_shoulder_pitch_joint": 0.,
            "right_shoulder_roll_joint": -0.27,
            ".*_elbow_joint": 0.5,
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
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=100.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness= {
                ".*_shoulder_pitch_joint": 100.0,
                ".*_shoulder_roll_joint": 100.0,
                ".*_shoulder_yaw_joint": 50.0,
                ".*_elbow_joint": 50.0,
            },
            damping= {
                ".*_shoulder_pitch_joint": 2.0,
                ".*_shoulder_roll_joint": 2.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
            },
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
            },
        ),
    },
)
"""Configuration for the Unitree G1 Humanoid robot."""


G1_MINIMAL_CFG = G1_CFG.copy()
# G1_MINIMAL_CFG.spawn.usd_path = f"{G1_CUSTOM_DIR}/Robots/Unitree/G1/g1_23dof_minimal.usda"
"""Configuration for the Unitree G1 Humanoid robot with fewer collision meshes.

This configuration removes most collision meshes to speed up simulation.
"""