import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from dataclasses import dataclass
import torch,yaml
import numpy as np
from copy import deepcopy
@dataclass
class JointTrajectoryConfig:
    """Configuration for storing joint trajectory data."""
    joints_yaml_order = [
        "LeftFrontalHipJoint", "LeftTransverseHipJoint", "LeftSagittalHipJoint", 
        "LeftSagittalKneeJoint", "LeftSagittalAnkleJoint", "LeftHenkeAnkleJoint",
        "RightFrontalHipJoint", "RightTransverseHipJoint", "RightSagittalHipJoint", 
        "RightSagittalKneeJoint", "RightSagittalAnkleJoint", "RightHenkeAnkleJoint"
    ]

    base_trajectories = {
        "x":[-0.12266671, -0.10308919, -0.08193415, -0.08567717, -0.04978699, -0.04058201, -0.02899778, -0.00220716],
        "y":[ 0.15255061,  0.10236909,  0.06714495,  0.05970906,  0.0498983 ,0.07733541,  0.10594412,  0.14625296],
        "z":[ 0.93524048,  0.93925662,  0.93629271,  0.96158756,  0.92655642,0.95167741,  0.94928602,  0.93578933],
        "roll":[ 0.03365673,  0.0840478 ,  0.05092976, -0.00791901, -0.0036338 ,0.01881114,  0.03720801, -0.03371127],
        "pitch": [ 0.26083729,  0.3131924 ,  0.31582513,  0.29973821,  0.18419699,0.19015824,  0.21317358,  0.26097091],
        "yaw": [-0.00204214, -0.04575562,  0.01270317,  0.15864688,  0.13853448,0.09162312,  0.04365285,  0.00206194]
    }

    joint_trajectories = {
        "LeftFrontalHipJoint": [0.0584, 0.0389, 0.1169, 0.1863, 0.2513, 0.1205, 0.1200, 0.1320],
        "RightFrontalHipJoint": [-0.1346, -0.1457, -0.0767, -0.0197, 0.0412, -0.0329, -0.0819, -0.0564],
        "LeftTransverseHipJoint": [-0.0446, 0.0532, 0.0369, -0.1133, -0.0761, -0.0337, -0.0313, -0.0349],
        "RightTransverseHipJoint": [0.0344, 0.0985, 0.1124, 0.0109, 0.0258, 0.0424, 0.0526, 0.0458],
        "LeftSagittalHipJoint": [-0.3070, -0.3056, -0.4172, -0.7477, -0.5681, -0.6444, -0.5866, -0.4932],
        "RightSagittalHipJoint": [-0.4929, -0.5712, -0.5351, -0.3030, -0.4423, -0.2313, -0.2596, -0.3062],
        "LeftSagittalKneeJoint": [0.2147, 0.2096, 0.3116, 0.6107, 0.4086, 0.5875, 0.4639, 0.2843],
        "RightSagittalKneeJoint": [0.2830, 0.3422, 0.3160, 0.0257, 0.4833, 0.1644, 0.1561, 0.2124],
        "LeftSagittalAnkleJoint": [-0.1642, -0.2070, -0.2000, -0.1416, -0.0845, -0.1001, -0.0902, -0.0460],
        "RightSagittalAnkleJoint": [-0.0461, -0.0866, -0.0964, -0.0025, -0.2497, -0.1070, -0.1093, -0.1637],
        "LeftHenkeAnkleJoint": [-0.1277, -0.1793, -0.3425, -0.0403, -0.3654, -0.2275, -0.2114, -0.1167],
        "RightHenkeAnkleJoint": [0.1192, 0.0613, 0.0067, -0.0504, -0.0242, -0.0022, 0.0456, 0.1241],
    }

    def load_joint_trajectories_from_yaml(self,file_path, joint_names, matrix_shape):
        """
        Load joint coefficients from a YAML file, reshape them, and assign them to joints.

        Args:
            file_path (str): Path to the YAML file.
            joint_names (list of str): List of joint names.
            matrix_shape (tuple of int): Shape to reshape each joint's coefficients.

        Returns:
            dict: A dictionary with joint names as keys and reshaped coefficients as values.
        """
        # Load the YAML file
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Extract and process coefficients
        coeff_jt = data.get('coeff_jt', [])
        
        if not coeff_jt:
            raise ValueError("Cannot find coeff_jt.")

        joint_trajectories = {}
        reshaped_coeffs = np.reshape(coeff_jt, matrix_shape, order='F').tolist() 

        for joint_name, coeffs in zip(joint_names, reshaped_coeffs):
            joint_trajectories[joint_name] = coeffs

        return joint_trajectories

    def get_joint_names(self) -> list:
        """Return the list of joint names."""
        return list(self.joint_trajectories.keys())
    
    def get_trajectory(self, joint_names) -> torch.Tensor:
        """Returns the trajectory for multiple joint names as a tensor."""
        if isinstance(joint_names, str):
            joint_names = [joint_names]

        trajectories = []
        for joint_name in joint_names:
            if joint_name in self.joint_trajectories:
                trajectories.append(torch.tensor(self.joint_trajectories[joint_name]))
            else:
                raise ValueError(f"Joint {joint_name} not found in the configuration.")
        return torch.stack(trajectories)

    def remap_base_symmetric(self)->dict:
        remapped_trajectories = deepcopy(self.base_trajectories)
        # Negate specific joint trajectories
        joints_to_negate = ["y", "roll", "yaw"]
        for joint in joints_to_negate:
            remapped_trajectories[joint] = [-value for value in remapped_trajectories[joint]]
        return remapped_trajectories

    def remap_jt_symmetric(self) -> dict:
        """Remap the joint trajectory coefficients symmetrically for left and right legs."""
        # Initialize remapped coefficients
        remapped_trajectories = {}

        # Flip left and right legs
        left_joints = ["LeftFrontalHipJoint", "LeftTransverseHipJoint", "LeftSagittalHipJoint", "LeftSagittalKneeJoint", "LeftSagittalAnkleJoint", "LeftHenkeAnkleJoint"]
        right_joints = ["RightFrontalHipJoint", "RightTransverseHipJoint", "RightSagittalHipJoint", "RightSagittalKneeJoint", "RightSagittalAnkleJoint", "RightHenkeAnkleJoint"]

        for i in range(6):
            remapped_trajectories[left_joints[i]] = torch.tensor(self.joint_trajectories[right_joints[i]])
            remapped_trajectories[right_joints[i]] = torch.tensor(self.joint_trajectories[left_joints[i]])

        # Negate specific joint trajectories
        joints_to_negate = ["LeftFrontalHipJoint", "LeftTransverseHipJoint", "LeftHenkeAnkleJoint", "RightFrontalHipJoint", "RightTransverseHipJoint", "RightHenkeAnkleJoint"]
        for joint in joints_to_negate:
            if isinstance(remapped_trajectories[joint], torch.Tensor):
                remapped_trajectories[joint] = -1 * remapped_trajectories[joint]
            else:
                remapped_trajectories[joint] = [-value for value in remapped_trajectories[joint]]
                
        return remapped_trajectories


    def get_joint_tensor(self, joint_trajectory_dict: dict, joint_names: list) -> torch.Tensor:
        """Convert the joint trajectory dictionary into a tensor with shape (num_joints, num_control_points)."""
        # Ensure a consistent ordering of the joints
        
        # Extract trajectories in the given order and stack them
        joint_tensors = [torch.tensor(joint_trajectory_dict[joint]).clone().detach() for joint in joint_names]
        joint_tensor = torch.stack(joint_tensors)  # Shape: (num_joints, num_control_points)

        return joint_tensor
    
    def get_base_tensor(self,base_trajectory_dict: dict) ->torch.Tensor:
        base_names = ["x","y","z","roll","pitch","yaw"]

        # Extract trajectories in the given order and stack them
        base_tensors = [torch.tensor(base_trajectory_dict[joint]).clone().detach() for joint in base_names]
        base_tensors = torch.stack(base_tensors)  # Shape: (num_joints, num_control_points)

        return base_tensors


    def get_trajectory_remap(self, joint_names) -> torch.Tensor:
        """Returns the remapped trajectory for multiple joint names as a tensor."""
        if isinstance(joint_names, str):
            joint_names = [joint_names]

        remapped_trajectories = self.remap_jt_symmetric()
        trajectories = []
        for joint_name in joint_names:
            if joint_name in remapped_trajectories:
                trajectories.append(remapped_trajectories[joint_name])
            else:
                raise ValueError(f"Joint {joint_name} not found in the remapped configuration.")
        return torch.stack(trajectories)


#TODO: add armature to the robot

ROBOT_ASSETS = "robot_assets/loadedExo"
EXO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ROBOT_ASSETS}/loadedExo.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    
    init_state=ArticulationCfg.InitialStateCfg(
        # Floating base (x, y, z) position and quaternion (w, x, y, z) for orientation
        pos=(-0.014008593589966438, 0.0796781833923272, 0.9354441930494577),  # Position in world coordinates
        rot=(0.9987690404068765, -0.006123545141481294, 0.04554959911180042, 0.01865851388459936),  # Quaternion for orientation

        # Joint positions
        joint_pos={
            "LeftFrontalHipJoint": 0.18028152949275983,
            "LeftTransverseHipJoint": -0.021132895502285864,
            "LeftSagittalHipJoint": -0.09113022902079292,
            "LeftSagittalKneeJoint": 0.08726666711960272,
            "LeftSagittalAnkleJoint": -0.08070105776323688,
            "LeftHenkeAnkleJoint": -0.22689276924459834,
            "RightFrontalHipJoint": 0.01216398608384502,
            "RightTransverseHipJoint": 0.08726530893357537,
            "RightSagittalHipJoint": -0.20524332222769187,
            "RightSagittalKneeJoint": 0.306769990413718,
            "RightSagittalAnkleJoint": -0.19197628136257563,
            "RightHenkeAnkleJoint": 0.021800551650373128,
        },

        # Joint velocities (assuming starting at rest)
        joint_vel={".*": 0.0},  # All joints start with zero velocity
    ),



    # init_state=ArticulationCfg.InitialStateCfg(
    #     # Floating base (x, y, z) position and quaternion (w, x, y, z) for orientation
    #     pos=(0.023188588723930748, 0.15, 0.9387507676324566),  # Position in world coordinates
    #     rot=(0.9972708203428684, 9.257248108811524E-19, 0.07383028438697906, 1.2906817772851895E-17),  # Quaternion for orientation

    #     # Joint positions
    #     joint_pos={
    #         "LeftFrontalHipJoint": 0.08349606121006133,
    #         "LeftTransverseHipJoint": -0.05015773774547203,
    #         "LeftSagittalHipJoint": -0.11985110552995228,
    #         "LeftSagittalKneeJoint": 0.13291080512603076,
    #         "LeftSagittalAnkleJoint": -0.1570693091810952,
    #         "LeftHenkeAnkleJoint": -0.1235453418940121,
    #         "RightFrontalHipJoint": -0.08349606121006133,
    #         "RightTransverseHipJoint": 0.05015773774547201,
    #         "RightSagittalHipJoint": -0.11985110552995229,
    #         "RightSagittalKneeJoint": 0.13291080512603076,
    #         "RightSagittalAnkleJoint": -0.1570693091810952,
    #         "RightHenkeAnkleJoint": 0.1235453418940121,
    #     },

    #     # Joint velocities (assuming starting at rest)
    #     joint_vel={".*": 0.0},  # All joints start with zero velocity
    # ),

    soft_joint_pos_limit_factor=0.95,
    actuators = {
    # Left Side Actuators
    "left_hips": ImplicitActuatorCfg(
        joint_names_expr=["LeftFrontalHipJoint", "LeftTransverseHipJoint", "LeftSagittalHipJoint"],
        effort_limit={
            "LeftFrontalHipJoint": 350.0,
            "LeftTransverseHipJoint": 180.0,
            "LeftSagittalHipJoint": 219.0,
        },
        velocity_limit={
            "LeftFrontalHipJoint": 2.61799,
            "LeftTransverseHipJoint": 4.18879,
            "LeftSagittalHipJoint": 4.18879,
        },
        stiffness={
            "LeftFrontalHipJoint": 21000.0,
            "LeftTransverseHipJoint": 16000.0,
            "LeftSagittalHipJoint": 16000.0,
        },
        damping={
            "LeftFrontalHipJoint": 500.0,
            "LeftTransverseHipJoint": 140.0,
            "LeftSagittalHipJoint": 140.0,
        },
    ),
    "left_knees": ImplicitActuatorCfg(
        joint_names_expr=["LeftSagittalKneeJoint"],
        effort_limit=219.0,
        velocity_limit=4.18879,
        stiffness={"LeftSagittalKneeJoint": 16000.0},
        damping={"LeftSagittalKneeJoint": 140.0},
    ),
    "left_ankles": ImplicitActuatorCfg(
        joint_names_expr=["LeftSagittalAnkleJoint", "LeftHenkeAnkleJoint"],
        effort_limit={
            "LeftSagittalAnkleJoint": 184.0,
            "LeftHenkeAnkleJoint": 82.0,
        },
        velocity_limit={
            "LeftSagittalAnkleJoint": 3.14,
            "LeftHenkeAnkleJoint": 5.24,
        },
        stiffness={
            "LeftSagittalAnkleJoint": 2500.0,
            "LeftHenkeAnkleJoint": 1100.0,
        },
        damping={
            "LeftSagittalAnkleJoint": 20.0,
            "LeftHenkeAnkleJoint": 15.0,
        },
    ),

    # Right Side Actuators
    "right_hips": ImplicitActuatorCfg(
        joint_names_expr=["RightFrontalHipJoint", "RightTransverseHipJoint", "RightSagittalHipJoint"],
        effort_limit={
            "RightFrontalHipJoint": 350.0,
            "RightTransverseHipJoint": 180.0,
            "RightSagittalHipJoint": 219.0,
        },
        velocity_limit={
            "RightFrontalHipJoint": 2.61799,
            "RightTransverseHipJoint": 4.18879,
            "RightSagittalHipJoint": 4.18879,
        },
        stiffness={
            "RightFrontalHipJoint": 21000.0,
            "RightTransverseHipJoint": 16000.0,
            "RightSagittalHipJoint": 16000.0,
        },
        damping={
            "RightFrontalHipJoint": 500.0,
            "RightTransverseHipJoint": 140.0,
            "RightSagittalHipJoint": 140.0,
        },
    ),
    "right_knees": ImplicitActuatorCfg(
        joint_names_expr=["RightSagittalKneeJoint"],
        effort_limit=219.0,
        velocity_limit=4.18879,
        stiffness={"RightSagittalKneeJoint": 16000.0},
        damping={"RightSagittalKneeJoint": 140.0},
    ),
    "right_ankles": ImplicitActuatorCfg(
        joint_names_expr=["RightSagittalAnkleJoint", "RightHenkeAnkleJoint"],
        effort_limit={
            "RightSagittalAnkleJoint": 184.0,
            "RightHenkeAnkleJoint": 82.0,
        },
        velocity_limit={
            "RightSagittalAnkleJoint": 3.14,
            "RightHenkeAnkleJoint": 5.24,
        },
        stiffness={
            "RightSagittalAnkleJoint": 2500.0,
            "RightHenkeAnkleJoint": 1100.0,
        },
        damping={
            "RightSagittalAnkleJoint": 20.0,
            "RightHenkeAnkleJoint": 15.0,
        },
    ),
}


)
"""Configuration for Atalante Exoskeleton."""


