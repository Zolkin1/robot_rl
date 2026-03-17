import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

ROBOT_ASSETS = "robot_assets/double_integrator"

DOUBLE_INTEGRATOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_ASSETS}/double_integrator.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        joint_pos={"slide_joint": 0.0},
        joint_vel={"slide_joint": 0.0},
    ),
    actuators={
        "slide": ImplicitActuatorCfg(
            joint_names_expr=["slide_joint"],
            effort_limit=100.0,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
"""Configuration for the double integrator (1D ball on prismatic joint)."""
