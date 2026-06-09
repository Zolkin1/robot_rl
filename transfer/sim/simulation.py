# Copyright 2026 Zachary Olkin. All rights reserved.

import csv
import math
import os
from collections.abc import Callable
from datetime import datetime
import time

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from sim.robot import Robot


def log_row_to_csv(filename, data):
    """
    Appends a single row of data to an existing CSV file.

    Args:
      filename (str): The path to the CSV file.
      data_row (list): A list of data points for the row.
    """
    try:
        # Create the file if it doesn't exist
        if not os.path.exists(filename):
            print(f"Creating new log file: {filename}")
            with open(filename, "w", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(data)
        else:
            # Append to existing file
            with open(filename, "a", newline="") as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(data)
                # Force write to disk
                csvfile.flush()
                os.fsync(csvfile.fileno())
    except Exception as e:
        print(f"Error appending row to {filename}: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"File exists: {os.path.exists(filename)}")
        print(f"File permissions: {oct(os.stat(filename).st_mode)[-3:] if os.path.exists(filename) else 'N/A'}")


class Simulation:
    def __init__(
        self,
        policy,
        robot: Robot,
        log: bool = False,
        log_dir: str = None,
        use_height_sensor: bool = False,
        tracking_body_name: str = "",
    ):
        """Initialize the simulation.

        Args:
            policy: The policy to use for control
            robot: Robot instance
            log: Whether to log data
            log_dir: Directory to save logs
            use_height_sensor: Whether to use height sensor (default: False)
        """
        self.policy = policy
        self.robot = robot
        self.log = log
        self.log_dir = log_dir
        self.log_file = None
        self.new_log_folder = ""
        self.use_height_sensor = use_height_sensor

        # Setup simulation parameters
        self.sim_steps_per_policy_update = int(policy.get_dt() / robot.mj_model.opt.timestep)
        self.sim_loop_rate = self.sim_steps_per_policy_update * robot.mj_model.opt.timestep
        self.viewer_rate = math.ceil((1 / 50) / robot.mj_model.opt.timestep)

        # Tracking body
        self.tracking_body_name = tracking_body_name

        # Setup logging if enabled
        if self.log:
            self._setup_logging()

    def _setup_logging(self):
        """Setup logging directory and files."""
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        self.new_log_folder = os.path.join(self.log_dir, timestamp_str)
        try:
            os.makedirs(self.new_log_folder, exist_ok=True)
            print(f"Successfully created folder: {self.new_log_folder}")
        except OSError as e:
            print(f"Error creating folder {self.new_log_folder}: {e}")

        print(f"Saving rerun logs to {self.new_log_folder}.")
        self.log_file = os.path.join(self.new_log_folder, "sim_log.csv")

        # Extract joint names and torque limits from actuators in the order they appear
        joint_names = []
        torque_limits = []
        for i in range(self.robot.mj_model.nu):
            actuator_id = i
            joint_id = self.robot.mj_model.actuator_trnid[actuator_id, 0]
            joint_name = mujoco.mj_id2name(self.robot.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            joint_names.append(joint_name)
            
            # Extract torque limits (actuatorfrcrange gives [min, max])
            torque_min = float(self.robot.mj_model.jnt_actfrcrange[actuator_id+1, 0])
            torque_max = float(self.robot.mj_model.jnt_actfrcrange[actuator_id+1, 1])
            torque_limits.append([torque_min, torque_max])

        # Save simulation configuration
        data_structure = [
            {"name": "time", "length": 1},
            {"name": "qpos", "length": self.robot.mj_data.qpos.shape[0]},
            {"name": "qvel", "length": self.robot.mj_data.qvel.shape[0]},
            {"name": "obs", "length": self.policy.get_num_obs()},
            {"name": "action", "length": self.policy.get_num_actions()},
            {"name": "torque", "length": self.robot.mj_model.nu},
            {"name": "left_ankle_pos", "length": 3},
            {"name": "right_ankle_pos", "length": 3},
            {"name": "commanded_vel", "length": 3},
        ]

        sim_config = {
            "simulator": "mujoco",
            "robot": self.robot.robot_name,
            "policy": self.policy.policy_path,
            "policy_dt": self.policy.get_dt(),
            "use_height_sensor": self.use_height_sensor,
            "data_structure": data_structure,
            "joint_names": joint_names,
            "torque_limits": torque_limits,
        }

        config_path = os.path.join(self.new_log_folder, "sim_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(sim_config, f)

    def get_logging_folder(self):
        return self.new_log_folder

    def run_headless(
        self,
        total_time: float,
        force_disturbance: Callable[[float], np.array] = None,
        skip_gain_setup: bool = False,
    ):
        """Run the simulation without a viewer.

        Args:
            total_time: Total simulation time in seconds.
            force_disturbance: Optional callable returning force vector at each timestep.
            skip_gain_setup: If True, skip calling set_pd_gains_from_policy. Use this
                when gains have been manually set and scaled before running.
        """
        print(
            f"Starting mujoco simulation with robot {self.robot.robot_name}.\n"
            f"Policy dt set to {self.policy.get_dt()} s ({self.sim_steps_per_policy_update} steps per policy update.)\n"
            f"Simulation dt set to {self.robot.mj_model.opt.timestep} s. Sim loop rate set to {self.sim_loop_rate} s.\n"
            f"Height sensor enabled: {self.use_height_sensor}\n"
        )

        # Setup height sensor visualization if enabled
        if self.use_height_sensor:
            grid_size = (1.5, 1.5)
            x_y_num_rays = (25, 25)
            height_map = self._ray_cast_sensor(
                self.robot.mj_model, self.robot.mj_data, "height_sensor_site", grid_size, x_y_num_rays
            )

        if total_time < 0:
            raise ValueError("Headless simulation must have a positive total time specified!")

        success = True

        if not skip_gain_setup:
            self.robot.set_pd_gains_from_policy(self.policy)

        while self.robot.mj_data.time < total_time:
            # Get observation and compute action
            if self.use_height_sensor:
                height_map = self._ray_cast_sensor(
                    self.robot.mj_model, self.robot.mj_data, "height_sensor_site", grid_size, x_y_num_rays
                )
                site_id = mujoco.mj_name2id(self.robot.mj_model, mujoco.mjtObj.mjOBJ_SITE, "height_sensor_site")
                sensor_pos = self.robot.mj_data.site_xpos[site_id]

                obs = self.robot.create_observation(self.policy, height_map=height_map, sensor_pos=sensor_pos)
            else:
                obs = self.robot.create_observation(self.policy)
            action = self.policy.get_action(obs, self.robot.joint_names)
            self.robot.apply_action(action)

            if self.robot.failed():
                success = False
                break

            # Step the simulator
            for i in range(self.sim_steps_per_policy_update):
                # Update height sensor visualization if enabled
                if self.use_height_sensor:
                    height_map = self._ray_cast_sensor(
                        self.robot.mj_model, self.robot.mj_data, "height_sensor_site", grid_size, x_y_num_rays
                    )

                if force_disturbance is not None:
                    self.robot.apply_force_disturbance(force_disturbance(self.robot.mj_data.time))

                # Step the sim
                self.robot.step()

                # Only log and sync viewer at viewer_rate intervals
                if i % self.viewer_rate == 0:
                    if self.log:
                        log_data = self.robot.get_log_data(self.policy, obs, action)
                        # if any(abs(v) > 1e-6 for v in log_data[-3:]):  # Only print if commanded velocity is non-zero
                        # print(f"Commanded velocity: {log_data[-3:]}")
                        log_row_to_csv(self.log_file, log_data)

        return success

    def run(
        self,
        total_time: float,
        force_disturbance: Callable[[float], np.array] = None,
    ):
        """Run the simulation."""
        print(
            f"Starting mujoco simulation with robot {self.robot.robot_name}.\n"
            f"Policy dt set to {self.policy.get_dt()} s ({self.sim_steps_per_policy_update} steps per policy update.)\n"
            f"Simulation dt set to {self.robot.mj_model.opt.timestep} s. Sim loop rate set to {self.sim_loop_rate} s.\n"
            f"Height sensor enabled: {self.use_height_sensor}\n"
        )

        success = True

        self.robot.set_pd_gains_from_policy(self.policy)

        with mujoco.viewer.launch_passive(self.robot.mj_model, self.robot.mj_data) as viewer:
            if self.tracking_body_name != "":
                viewer.cam.trackbodyid = mujoco.mj_name2id(
                    self.robot.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.tracking_body_name
                )
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                print(f"Camera tracking body: {self.tracking_body_name}")

            # Setup height sensor visualization if enabled
            if self.use_height_sensor:
                grid_size = (1.5, 1.5)
                x_y_num_rays = (25, 25)
                height_map = self._ray_cast_sensor(
                    self.robot.mj_model, self.robot.mj_data, "height_sensor_site", grid_size, x_y_num_rays
                )
                # Add custom debug spheres
                for ii, pos in height_map.reshape(-1, 3):
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[ii],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.01, 0, 0]),
                        pos=pos,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0, 0, 1]),
                    )
                    viewer.user_scn.ngeom += 1

            while viewer.is_running():
                if total_time > 0 and self.robot.mj_data.time > total_time:
                    break
                # Get observation and compute action
                if self.use_height_sensor:
                    height_map = self._ray_cast_sensor(
                        self.robot.mj_model, self.robot.mj_data, "height_sensor_site", grid_size, x_y_num_rays
                    )
                    site_id = mujoco.mj_name2id(self.robot.mj_model, mujoco.mjtObj.mjOBJ_SITE, "height_sensor_site")
                    sensor_pos = self.robot.mj_data.site_xpos[site_id]

                    obs = self.robot.create_observation(self.policy, height_map=height_map, sensor_pos=sensor_pos)
                else:
                    obs = self.robot.create_observation(self.policy)
                action = self.policy.get_action(obs, self.robot.joint_names)
                self.robot.apply_action(action)

                if self.robot.failed():
                    success = False
                    break

                # Step the simulator
                for i in range(self.sim_steps_per_policy_update):
                    # Update scene
                    scene = mujoco.MjvScene(self.robot.mj_model, maxgeom=1000)
                    # cam = mujoco.MjvCamera()
                    opt = mujoco.MjvOption()
                    mujoco.mjv_updateScene(
                        self.robot.mj_model,
                        self.robot.mj_data,
                        opt,
                        None,
                        viewer.cam,
                        mujoco.mjtCatBit.mjCAT_ALL,
                        scene,
                    )

                    # Update height sensor visualization if enabled
                    if self.use_height_sensor:
                        height_map = self._ray_cast_sensor(
                            self.robot.mj_model, self.robot.mj_data, "height_sensor_site", grid_size, x_y_num_rays
                        )
                        # print(height_map)
                        for ii, pos in height_map.reshape(-1, 3):
                            viewer.user_scn.geoms[ii].pos = pos

                    if force_disturbance is not None:
                        self.robot.apply_force_disturbance(force_disturbance(self.robot.mj_data.time))

                    # Step the sim
                    self.robot.step()

                    # time.sleep(0.01)

                    # Only log and sync viewer at viewer_rate intervals
                    if i % self.viewer_rate == 0:
                        if self.log:
                            log_data = self.robot.get_log_data(self.policy, obs, action)
                            # if any(abs(v) > 1e-6 for v in log_data[-3:]):  # Only print if commanded velocity is non-zero
                            # print(f"Commanded velocity: {log_data[-3:]}")
                            log_row_to_csv(self.log_file, log_data)
                        viewer.sync()

        return success

    def _ray_cast_sensor(self, model, data, site_name, size, x_y_num_rays, sen_offset=0):
        """Using a grid pattern, create a height map using ray casting."""
        ray_pos_shape = x_y_num_rays
        ray_pos_shape = ray_pos_shape + (3,)
        ray_pos = np.zeros(ray_pos_shape)

        # Get the site location
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        site_pos = data.site_xpos[site_id].copy()

        # Add to the global z
        site_pos[2] = site_pos[2] + 10

        site_pos[0] = site_pos[0] - size[0] / 2.0
        site_pos[1] = site_pos[1] - size[1] / 2.0

        # Ray information
        direction = np.zeros(3)
        direction[2] = -1
        geom_group = np.zeros(6, dtype=np.int32)
        geom_group[2] = 1  # Only include group 2

        # Ray spacing
        spacing = np.zeros(3)
        spacing[0] = size[0] / (x_y_num_rays[0] - 1)
        spacing[1] = size[1] / (x_y_num_rays[1] - 1)

        # Loop through the rays
        for xray in range(x_y_num_rays[0]):
            for yray in range(x_y_num_rays[1]):
                geom_id = np.zeros(1, dtype=np.int32)
                offset = spacing.copy()
                offset[0] = spacing[0] * xray
                offset[1] = spacing[1] * yray

                ray_origin = offset + site_pos
                ray_pos[xray, yray, 2] = -mujoco.mj_ray(
                    model, data, ray_origin.astype(np.float64), direction.astype(np.float64), geom_group, 1, -1, geom_id
                )

                ray_pos[xray, yray, :] = ray_origin + ray_pos[xray, yray, :]

        return ray_pos
