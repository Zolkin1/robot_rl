import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from transfer.sim.rl_policy import RLPolicy

import mujoco
import numpy as np
import pygame
from scipy.spatial.transform import Rotation


class Robot:
    def __init__(self, robot_name: str, scene_name: str, joystick_scaling: np.ndarray,
                 input_function: Callable[[float], np.array] = None, use_pd: bool =False, gains: Dict[str, float]=None, rng=None):
        """Initialize the robot with its model and data."""
        if robot_name != "g1_21j" and robot_name != "g1_21j_M4" and robot_name != "g1_21j_compute":
            raise ValueError("Invalid robot name! Only support g1_21j for now.")

        self.robot_name = robot_name
        self.scene_name = scene_name
        self.mj_model, self.mj_data = self._get_model_data()
        self.commanded_vel = np.zeros(3)  # Store commanded velocity
        self.input_function = input_function
        self.rng = rng
        self.joystick_scaling = joystick_scaling
        self.joint_names = self.get_joint_names()
        self.gains = gains
        print(f"Mujoco joint names: {self.joint_names}")

        body_name = "torso_link"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.torso_ipos = self.mj_model.body_ipos[body_id]

        self.use_pd = use_pd

        if self.input_function is None:
            # Initialize joystick
            pygame.init()
            pygame.joystick.init()
            joystick_count = pygame.joystick.get_count()
            if joystick_count < 1:
                print("No joystick detected, using initial command from config instead.")
                self.joystick = None
            else:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                print(f"Using controller: {self.joystick.get_name()}")

    def _get_model_data(self):
        """Create the mj model and data from the given robot."""
        file_name = f"{self.robot_name}_{self.scene_name}.xml"
        relative_path = f"robots/g1/{file_name}"
        path = os.path.join(os.getcwd(), relative_path)
        print(f"Trying to load the xml at {path}")
        mj_model = mujoco.MjModel.from_xml_path(path)
        mj_data = mujoco.MjData(mj_model)

        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)

        return mj_model, mj_data

    def reset_robot(self):
        """Resets the robot."""
        self.mj_model, self.mj_data = self._get_model_data()

    def add_base_mass(self, added_mass):
        """Add mass to the robot base."""
        body_name = "torso_link"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        self.mj_model.body_mass[body_id] += added_mass

        print(f"Adjusting the mass of the {body_name} by adding {added_mass}.")

    def apply_force_disturbance(self, force_disturbance):
        """Apply the force_disturbance to the robot base."""
        body_name = "torso_link"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        self.mj_data.xfrc_applied[body_id] = force_disturbance

    def randomize_torso_mass_pos(self, max_movement: np.array):
        """Shift where the torso mass is randomly."""
        body_name = "torso_link"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        if self.rng is not None:
            rand_vec = self.rng.uniform(low=-max_movement, high=max_movement)
        else:
            rand_vec = np.random.uniform(low=-max_movement, high=max_movement)

        self.mj_model.body_ipos[body_id] += rand_vec

        print(f"Adjusting ipos of torso by: {rand_vec}.")

        return rand_vec

    def reset_torso_mass_pos(self):
        body_name = "torso_link"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)

        self.mj_model.body_ipos[body_id] += self.torso_ipos

    def scale_torso_mass(self, scale: float) -> None:
        """Scale the torso mass by the given factor.

        Args:
            scale: Multiplicative factor for the torso mass.
        """
        body_name = "torso_link"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.mj_model.body_mass[body_id] *= scale
        print(f"Scaling torso mass by {scale}. New mass: {self.mj_model.body_mass[body_id]}")

    def set_torso_mass_pos_offset(self, offset: np.ndarray) -> None:
        """Set the torso COM position offset to an exact value.

        Args:
            offset: 3D offset vector [x, y, z] in meters.
        """
        body_name = "torso_link"
        body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.mj_model.body_ipos[body_id] = self.torso_ipos + offset
        print(f"Setting torso COM offset to: {offset}")

    def scale_kd_gains(self, scale: float) -> None:
        """Scale all actuator kd gains by the given factor.

        Args:
            scale: Multiplicative factor for kd gains.
        """
        for i in range(self.mj_model.nu):
            self.mj_model.actuator_biasprm[i, 2] *= scale
        print(f"Scaling all kd gains by {scale}")

    def get_joystick_command(self, policy):
        """Get velocity commands from joystick."""
        des_vel = np.zeros(3)
        if self.joystick is not None:
            for event in pygame.event.get():
                pass
            # Left stick: control vx, vy (2D plane), right stick X-axis: vyaw
            vy = -(self.joystick_scaling[0]*self.joystick.get_axis(0)) * policy.get_max_vy()
            vx = -(self.joystick_scaling[0]*self.joystick.get_axis(1)) * policy.get_max_vx()
            vyaw = -(self.joystick_scaling[0]*self.joystick.get_axis(3)) * policy.get_max_vyaw()

            des_vel[0] = vx
            des_vel[1] = vy
            des_vel[2] = vyaw
        else:
            des_vel = np.array([0.5, 0.0, 0.0])
        self.commanded_vel = des_vel  # Store the commanded velocity
        # print(f"Commanded velocity: {des_vel}")
        return des_vel

    def create_observation(self, policy, height_map=None, sensor_pos=None):
        """Create observation for the policy."""
        qpos = self.mj_data.qpos
        qvel = self.mj_data.qvel
        sim_time = self.mj_data.time
        # pg = self.get_projected_gravity(qpos[3:7])
        if self.input_function is None:
            self.commanded_vel = self.get_joystick_command(policy)
        else:
            self.commanded_vel = self.input_function(sim_time)

        if self.use_pd:
            self.pd_control(qpos, qvel)

        return policy.create_obs(
            qpos[:7],
            qvel[3:6],
            qpos[7:],
            qvel[6:],
            sim_time,
            self.commanded_vel,
            self.joint_names
        )

    def get_log_data(self, policy, obs, action):
        """Get data to be logged."""
        torques = []
        for j in range(self.mj_model.nu):
            torques.append(self.mj_data.actuator_force[j])

        left_ankle_pos = self.mj_data.sensor("left_ankle_pos").data
        right_ankle_pos = self.mj_data.sensor("right_ankle_pos").data

        log = [
            self.mj_data.time,
            *self.mj_data.qpos.tolist(),
            *self.get_local_vel().tolist(),
            *self.mj_data.qvel[3:].tolist(),
            *obs[0, :].numpy().tolist(),
            *action.tolist(),
            *torques,
            *left_ankle_pos.tolist(),
            *right_ankle_pos.tolist(),
            *self.commanded_vel.tolist(),  # Add commanded velocity to log data
        ]
        return log

    def get_local_vel(self):
        """Convert the global floating base velocity to the local frame."""
        # Convert to scipy format [x, y, z, w]
        q_scipy = np.array([self.mj_data.qpos[4], self.mj_data.qpos[5], self.mj_data.qpos[6], self.mj_data.qpos[3]])

        # Create rotation object
        r = Rotation.from_quat(q_scipy)

        # Inverse rotate to get local velocity
        v_local = r.inv().apply(self.mj_data.qvel[:3])

        return v_local

    def pd_control(self, qpos: np.ndarray, qvel: np.ndarray):
        """Compute the PD control actions to keep the robot moving straight."""
        kp_y = self.gains["kp_y"]
        kd_y = self.gains["kd_y"]
        y_vel = np.sign(self.commanded_vel[0]) * np.clip(-kp_y * qpos[1] + -kd_y * qvel[1], -0.5, 0.5)
        self.commanded_vel[1] = y_vel

        kp_yaw = self.gains["kp_yaw"]
        kd_yaw = self.gains["kd_yaw"]
        siny_cosp = 2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5])
        cosy_cosp = 1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6])
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        angular_vel = np.sign(self.commanded_vel[0]) * np.clip(-kp_yaw * yaw + -kd_yaw * qvel[5], -0.5, 0.5)
        self.commanded_vel[2] = angular_vel

        # print(f"Commanded velocity: {self.commanded_vel}, y pos: {qpos[1]}, y vel: {qvel[1]}, yaw: {yaw}")

    def apply_action(self, action):
        """Apply control action to the robot."""
        self.mj_data.ctrl[: len(action)] = action

    def step(self):
        """Step the robot simulation."""
        mujoco.mj_step(self.mj_model, self.mj_data)

    def failed(self):
        return self.mj_data.qpos[2] < 0.2

    def get_joint_names(self) -> list[str]:
        """Extract joint names from qpos in MuJoCo order.

        Returns:
            List of joint names corresponding to qpos[7:] (excluding floating base)
        """
        joint_names = []
        for i in range(7, self.mj_model.nq):
            joint_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i - 6)
            if joint_name:
                joint_names.append(joint_name)
        return joint_names

    def set_initial_condition(self, policy: "RLPolicy") -> None:
        """Set the robot state from the valid IC in the policy YAML.

        The IC vectors are [base, joints_in_isaac_order]. Joint values are
        reordered to MuJoCo order before writing to qpos/qvel.

        Args:
            policy: The RLPolicy object containing the valid IC data.
        """
        ic_pos = policy.get_valid_ic_pos()
        ic_vel = policy.get_valid_ic_vel()
        if ic_pos is None or ic_vel is None:
            print("[WARNING] No valid IC found in policy params, using XML default.")
            return

        isaac_joint_names = policy.get_joint_names()

        # qpos: [base_pos(3), base_quat(4), joint_pos(N)]
        # Isaac stores quat as (x, y, z, w); MuJoCo expects (w, x, y, z).
        self.mj_data.qpos[:3] = ic_pos[:3]
        self.mj_data.qpos[3] = ic_pos[6]
        self.mj_data.qpos[4:7] = ic_pos[3:6]
        joint_pos_isaac = ic_pos[7:]
        self.mj_data.qpos[7:] = policy.convert_joint_order(
            joint_pos_isaac, isaac_joint_names, self.joint_names
        )

        # qvel: [base_lin_vel(3), base_ang_vel(3), joint_vel(N)]
        self.mj_data.qvel[:6] = ic_vel[:6]
        joint_vel_isaac = ic_vel[6:]
        self.mj_data.qvel[6:] = policy.convert_joint_order(
            joint_vel_isaac, isaac_joint_names, self.joint_names
        )

        # Forward kinematics to update derived quantities
        mujoco.mj_forward(self.mj_model, self.mj_data)
        print("[INFO] Set initial condition from policy YAML.")

    def set_pd_gains_from_policy(self, policy) -> None:
        """Set PD gains on MuJoCo actuators from the RLPolicy gains.

        Args:
            policy: The RLPolicy object containing kp and kd gains in Isaac order.

        Raises:
            ValueError: If an actuator for a joint is not found in the MuJoCo model.
        """
        kp_isaac = policy.get_kp()
        kd_isaac = policy.get_kd()
        isaac_joint_names = policy.get_joint_names()

        for i, joint_name in enumerate(isaac_joint_names):
            actuator_name = f"{joint_name}_pos"
            actuator_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
            )

            if actuator_id == -1:
                raise ValueError(
                    f"Actuator '{actuator_name}' for joint '{joint_name}' not found in MuJoCo model"
                )

            kp = kp_isaac[i]
            kd = kd_isaac[i]

            # For position actuators: force = kp*(ctrl-pos) - kd*vel
            self.mj_model.actuator_gainprm[actuator_id, 0] = kp
            self.mj_model.actuator_biasprm[actuator_id, 1] = -kp
            self.mj_model.actuator_biasprm[actuator_id, 2] = -kd

            print(f"Set gains for '{joint_name}': kp={kp}, kd={kd}")
