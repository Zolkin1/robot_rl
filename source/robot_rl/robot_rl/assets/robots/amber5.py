import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# Path to Amber USD asset directory
ROBOT_ASSETS = "robot_assets/amber5/amber"

# Stiffness and damping constants for Amber joints
STIFFNESS = 1000.0
DAMPING = 50.0

# --- AMBER5 ROBOT CONFIGURATION ---
AMBER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_ASSETS}/amber2.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.01,
            rest_offset=0.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.3),
        joint_pos={
            "q1_left": 0.0,
            "q2_left": 0.0,
            "q1_right": 0.0,
            "q2_right": 0.0,
        },
        joint_vel={
            "q1_left":   0.0,
            "q2_left":   0.0,
            "q1_right":  0.0,
            "q2_right":  0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "left_thigh_act": ImplicitActuatorCfg(
            joint_names_expr=["q1_left"],
            effort_limit_sim=400.0,
            velocity_limit_sim=4.0,
            stiffness=STIFFNESS,
            damping=DAMPING,
        ),
        "left_shin_act": ImplicitActuatorCfg(
            joint_names_expr=["q2_left"],
            effort_limit_sim=400.0,
            velocity_limit_sim=4.0,
            stiffness=STIFFNESS,
            damping=DAMPING,
        ),
        "right_thigh_act": ImplicitActuatorCfg(
            joint_names_expr=["q1_right"],
            effort_limit_sim=400.0,
            velocity_limit_sim=4.0,
            stiffness=STIFFNESS,
            damping=DAMPING,
        ),
        "right_shin_act": ImplicitActuatorCfg(
            joint_names_expr=["q2_right"],
            effort_limit_sim=400.0,
            velocity_limit_sim=5.0,
            stiffness=STIFFNESS,
            damping=DAMPING,
        ),
    },
)

# Minimal configuration copy (in case fewer collision meshes or variants are needed)
AMBER_MINIMAL_CFG = AMBER_CFG.copy()
"""
Configuration for the Amber5 robot with base articulation and actuator settings.

This module provides `AMBER_CFG` for full simulation and a
copy `AMBER_MINIMAL_CFG` that can be used for faster instantiation or
experimentation with reduced contacts.
"""
