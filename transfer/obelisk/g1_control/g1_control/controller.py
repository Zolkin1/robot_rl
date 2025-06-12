import os
import torch
import numpy as np
from typing import List, Optional
from abc import ABC, abstractmethod
from ament_index_python.packages import get_package_share_directory

from obelisk_control_msgs.msg import PDFeedForward
from obelisk_estimator_msgs.msg import EstimatedState
from obelisk_control_msgs.msg import VelocityCommand
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn
from obelisk_py.core.utils.ros import spin_obelisk
from rclpy.executors import SingleThreadedExecutor

from obelisk_py.core.control import ObeliskController
from obelisk_py.core.obelisk_typing import ObeliskControlMsg, is_in_bound


class VelocityTrackingController(ObeliskController, ABC):
    """Example position setpoint controller."""

    def __init__(self, node_name: str = "velocity_tracking_controller") -> None:
        """Initialize the example position setpoint controller."""
        super().__init__(node_name, PDFeedForward, EstimatedState)
        # Velocity limits        
        self.declare_parameter("v_x_max", 1.)
        self.declare_parameter("v_x_min", -1.)
        self.declare_parameter("v_y_max", 0.5)
        self.declare_parameter("w_z_max", 0.5)

        # Load policy
        self.declare_parameter("policy_name", "")
        policy_name = self.get_parameter("policy_name").get_parameter_value().string_value
        pkg_path = get_package_share_directory('g1_control')
        policy_path = os.path.join(pkg_path, f'resource/policies/{policy_name}')
        self.policy = torch.jit.load(policy_path)
        self.device = next(self.policy.parameters()).device

        # POlicy information
        self.num_obs = 74
        self.isaac_to_mujoco = {
            0: 0,       # left_hip_pitch
            1: 6,       # right_hip_pitch
            2: 12,       # waist_yaw
            3: 1,       # left_hip_roll
            4: 7,       # right_hip_roll
            5: 13,       # left_shoulder_pitch
            6: 17,       # right_shoulder_pitch
            7: 2,       # left_hip_yaw
            8: 8,       # right_hip_yaw
            9: 14,       # left_shoulder_roll
            10: 18,     # right_shoulder_roll
            11: 3,     # left_knee
            12: 9,     # right_knee
            13: 15,     # left_shoulder_yaw
            14: 19,     # right_shoulder_yaw
            15: 4,     # left_ankle_pitch
            16: 10,     # right_ankle_pitch
            17: 16,     # left_elbow
            18: 20,     # right_elbow
            19: 5,     # left_ankle_roll
            20: 11,     # right_ankle_roll
        }
        self.isaac_joints = {
            0: "left_hip_pitch_joint",
            1: "right_hip_pitch_joint",
            2: "waist_yaw_joint",
            3: "left_hip_roll_joint",
            4: "right_hip_roll_joint",
            5: "left_shoulder_pitch_joint",
            6: "right_shoulder_pitch_joint",
            7: "left_hip_yaw_joint",
            8: "right_hip_yaw_joint",
            9: "left_shoulder_roll_joint",
            10: "right_shoulder_roll_joint",
            11: "left_knee_joint",
            12: "right_knee_joint",
            13: "left_shoulder_yaw_joint",
            14: "right_shoulder_yaw_joint",
            15: "left_ankle_pitch_joint",
            16: "right_ankle_pitch_joint",
            17: "left_elbow_joint",
            18: "right_elbow_joint",
            19: "left_ankle_roll_joint",
            20: "right_ankle_roll_joint"
        }
        self.joint_names_mujoco = [
          "left_hip_pitch_joint",
          "left_hip_roll_joint",
          "left_hip_yaw_joint",
          "left_knee_joint",
          "left_ankle_pitch_joint",
          "left_ankle_roll_joint",
          "right_hip_pitch_joint",
          "right_hip_roll_joint",
          "right_hip_yaw_joint",
          "right_knee_joint",
          "right_ankle_pitch_joint",
          "right_ankle_roll_joint",
          "waist_yaw_joint",
          "left_shoulder_pitch_joint",
          "left_shoulder_roll_joint",
          "left_shoulder_yaw_joint",
          "left_elbow_joint",
          "right_shoulder_pitch_joint",
          "right_shoulder_roll_joint",
          "right_shoulder_yaw_joint",
          "right_elbow_joint",
        ]
        self.full_joint_names = [
          "left_hip_pitch_joint",
          "left_hip_roll_joint",
          "left_hip_yaw_joint",
          "left_knee_joint",
          "left_ankle_pitch_joint",
          "left_ankle_roll_joint",
          "right_hip_pitch_joint",
          "right_hip_roll_joint",
          "right_hip_yaw_joint",
          "right_knee_joint",
          "right_ankle_pitch_joint",
          "right_ankle_roll_joint",
          "waist_yaw_joint",
          "left_shoulder_pitch_joint",
          "left_shoulder_roll_joint",
          "left_shoulder_yaw_joint",
          "left_elbow_joint",
          "left_wrist_roll_joint",
          "left_wrist_pitch_joint",
          "left_wrist_yaw_joint",
          "right_shoulder_pitch_joint",
          "right_shoulder_roll_joint",
          "right_shoulder_yaw_joint",
          "right_elbow_joint",
          "right_wrist_roll_joint",
          "right_wrist_pitch_joint",
          "right_wrist_yaw_joint",
        ]

        self.num_motors = 21

        # Set scales and defaults
        self.declare_parameter("action_scale", 0.25)
        self.action_scale = self.get_parameter("action_scale").get_parameter_value().double_value

        self.declare_parameter("cmd_scale", [2., 2., 0.25])
        self.cmd_scale = self.get_parameter("cmd_scale").get_parameter_value().double_array_value

        self.declare_parameter("period", 0.8)
        self.period = self.get_parameter("period").get_parameter_value().double_value

        self.declare_parameter("num_actions", 0)
        self.num_actions = self.get_parameter("num_actions").get_parameter_value().integer_value

        self.declare_parameter("default_angles", [0.])
        self.declare_parameter("default_angles_names", [""])
        self.default_angles = self.get_parameter("default_angles").get_parameter_value().double_array_value
        self.default_angles_names = self.get_parameter("default_angles_names").get_parameter_value().string_array_value

        self.default_angles_isaac = self._convert_to_isaac(self.default_angles, self.default_angles_names)

        self.declare_parameter("qvel_scale", 1.)
        self.qvel_scale = self.get_parameter("qvel_scale").get_parameter_value().double_value

        self.declare_parameter("ang_vel_scale", 1.)
        self.ang_vel_scale = self.get_parameter("ang_vel_scale").get_parameter_value().double_value


        # Set PD gains
        self.declare_parameter("kps", [25.] * self.num_motors)
        self.declare_parameter("kds", [0.5] * self.num_motors)
        self.kps = self.get_parameter("kps").get_parameter_value().double_array_value
        self.kds = self.get_parameter("kds").get_parameter_value().double_array_value
        
        # Declare subscriber to velocity commands
        self.register_obk_subscription(
            "sub_vel_cmd_setting",
            self.vel_cmd_callback,  # type: ignore
            key="sub_vel_cmd_key",  # key can be specified here or in the config file
            msg_type=VelocityCommand
        )

        self.received_xhat = False

        self.get_logger().info(f"Policy: {policy_path} loaded on {self.device}.")
        self.get_logger().info("RL Velocity Tracking node constructor complete.")

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the controller."""
        super().on_configure(state)

        self.cmd_vel = np.zeros((3,))
        self.proj_g = np.zeros((3,))
        self.proj_g[2] = -1
        self.omega = np.zeros((3,))
        self.phase = np.zeros((2,))
        self.zero_action = np.zeros((self.num_motors,))
        self.action = self.zero_action.tolist()
        self.t_start = None

        self.get_logger().info("RL Velocity Tracking node configuration complete.")
        
        return TransitionCallbackReturn.SUCCESS

    def update_x_hat(self, x_hat_msg: EstimatedState) -> None:
        """Update the state estimate.

        Parameters:
            x_hat_msg: The Obelisk message containing the state estimate.
        """
        self.joint_pos = np.array(x_hat_msg.q_joints)
        self.joint_names = x_hat_msg.joint_names

        self.joint_vel = np.array(x_hat_msg.v_joints)

        self.proj_g = self.project_gravity(x_hat_msg.q_base[3:7])

        self.omega = np.array(x_hat_msg.v_base[3:6])
       
        self.time = x_hat_msg.header.stamp.sec + x_hat_msg.header.stamp.nanosec * 1e-9

        self.received_xhat = True

    def vel_cmd_callback(self, cmd_msg: VelocityCommand):
        self.cmd_vel[0] = min(
            max(cmd_msg.v_x, self.get_parameter("v_x_min").get_parameter_value().double_value),
            self.get_parameter("v_x_max").get_parameter_value().double_value
        )
        v_y_max = self.get_parameter("v_y_max").get_parameter_value().double_value
        self.cmd_vel[1] = min(max(cmd_msg.v_y, -v_y_max), v_y_max)
        w_z_max = self.get_parameter("w_z_max").get_parameter_value().double_value
        self.cmd_vel[2] = min(max(cmd_msg.w_z, -w_z_max), w_z_max)

    @staticmethod
    def project_gravity(quat):
        qx = quat[0]
        qy = quat[1]
        qz = quat[2]
        qw = quat[3]

        pg = np.zeros(3)

        pg[0] = 2 * (-qz * qx + qw * qy)
        pg[1] = -2 * (qz * qy + qw * qx)
        pg[2] = 1 - 2 * (qw * qw + qz * qz)

        return pg

    def get_obs(self) -> np.ndarray:
        """Create the observation from the estimated state."""
        obs = np.zeros(self.num_obs, dtype=np.float32)

        obs[:3] = self.omega*self.ang_vel_scale                                                 # Angular velocity
        obs[3:6] = self.proj_g                                        # Projected gravity
        obs[6] = self.cmd_vel[0]*self.cmd_scale[0]                                   # Command velocity
        obs[7] = self.cmd_vel[1]*self.cmd_scale[1]                                   # Command velocity
        obs[8] = self.cmd_vel[2]*self.cmd_scale[2]                                   # Command velocity

        joint_pos_isaac = self._convert_to_isaac(self.joint_pos, self.joint_names) - self.default_angles_isaac
        nj = len(joint_pos_isaac)

        joint_vel_isaac = self._convert_to_isaac(self.joint_vel, self.joint_names)

        obs[9 : 9 + nj] = joint_pos_isaac                                          # Joint pos
        obs[9 + nj : 9 + 2*nj] = joint_vel_isaac * self.qvel_scale                            # Joint vel
       

        obs[9 + 2*nj : 9 + 3*nj] = self.action                              # Past action

        sin_phase = np.sin(2 * np.pi * self.time/self.period)
        cos_phase = np.cos(2 * np.pi * self.time/self.period)
        obs[9 + 3*nj : 9 + 3*nj + 2] = np.array([sin_phase, cos_phase])     # Phases        

        # if height_map is not None:
        #     height_obs = self.convert_height_map_to_obs(height_map, sensor_pos)
        #     obs[9 + 3 * nj:9 + 3 * nj + height_obs.shape[0]] = height_obs
        #     obs[9 + 3 * nj + height_obs.shape[0] : 9 + 3 * nj + height_obs.shape[0] + 2] = np.array([sin_phase, cos_phase])     # Phases

        return obs
    
    def compute_control(self) -> PDFeedForward:
        """Compute the control signal for the dummy 2-link robot.

        Returns:
            obelisk_control_msg: The control message.
        """
        # Generate input to RL model
        if self.received_xhat:
            obs = self.get_obs()

            # Call RL model
            self.action = self.policy(torch.tensor(obs).to(self.device).float()).detach().cpu().numpy()

            # setting the message
            pd_ff_msg = PDFeedForward()
            pd_ff_msg.header.stamp = self.get_clock().now().to_msg()
            pos_targ = self._convert_to_mujoco(self.action * self.action_scale + self.default_angles_isaac)
            pd_ff_msg.pos_target = self._add_wrist_mujoco(pos_targ.tolist())
            pd_ff_msg.vel_target = self._add_wrist_mujoco(self.zero_action.tolist())
            pd_ff_msg.feed_forward = self._add_wrist_mujoco(self.zero_action.tolist())

            # Add in the wrist joints

            pd_ff_msg.u_mujoco = np.concatenate([
                self._add_wrist_mujoco(pos_targ.tolist()),
                self._add_wrist_mujoco(self.zero_action.tolist()),
                self._add_wrist_mujoco(self.zero_action.tolist())
            ]).tolist()
            pd_ff_msg.joint_names = self.full_joint_names
            pd_ff_msg.kp = self.kps
            pd_ff_msg.kd = self.kds
            self.obk_publishers["pub_ctrl"].publish(pd_ff_msg)

            # log here
            assert is_in_bound(type(pd_ff_msg), ObeliskControlMsg)
            return pd_ff_msg
    
    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        return TransitionCallbackReturn.SUCCESS

    def _convert_to_mujoco(self, vec):
        mj_vec = np.zeros(21)
        for isaac_index, mujoco_index in self.isaac_to_mujoco.items():
            mj_vec[mujoco_index] = vec[isaac_index]

        return mj_vec

    def _convert_to_isaac(self, joint_values, joint_names):
        ordered_values = np.zeros(len(self.isaac_joints), dtype=np.float64)
        name_to_index = {name: idx for idx, name in self.isaac_joints.items()}

        for name, value in zip(joint_names, joint_values):
            if name in name_to_index:
                index = name_to_index[name]
                ordered_values[index] = value
            else:
                self.get_logger().warning(f"Unknown joint name: {name}", once=True)
                # raise ValueError(f"Unknown joint name: {name}")

        return ordered_values
    
    def _add_wrist_mujoco(self, joint_values):
        """Add 0's to the wrist in mujoco ordering."""
        joint_values.insert(17, 0.)
        joint_values.insert(18, 0.)
        joint_values.insert(19, 0.)

        joint_values.insert(24, 0.)
        joint_values.insert(25, 0.)
        joint_values.insert(26, 0.)


        return joint_values


def main(args: Optional[List] = None) -> None:
    """Main entrypoint."""
    spin_obelisk(args, VelocityTrackingController, SingleThreadedExecutor)


if __name__ == "__main__":
    main()