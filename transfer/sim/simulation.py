import os
import time
import math
import csv
import yaml
import mujoco
import mujoco.viewer
from datetime import datetime
import numpy as np

from transfer.sim.robot import Robot


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
            with open(filename, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(data)
        else:
            # Append to existing file
            with open(filename, 'a', newline='') as csvfile:
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
    def __init__(self, policy, robot: Robot, log: bool = False, log_dir: str = None, use_height_sensor: bool = False):
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
        self.use_height_sensor = use_height_sensor
        
        # Setup simulation parameters
        self.sim_steps_per_policy_update = int(policy.dt / robot.mj_model.opt.timestep)
        self.sim_loop_rate = self.sim_steps_per_policy_update * robot.mj_model.opt.timestep
        self.viewer_rate = math.ceil((1 / 50) / robot.mj_model.opt.timestep)
        
        # Setup logging if enabled
        if self.log:
            self._setup_logging()

    def _setup_logging(self):
        """Setup logging directory and files."""
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        new_folder_path = os.path.join(self.log_dir, timestamp_str)
        try:
            os.makedirs(new_folder_path, exist_ok=True)
            print(f"Successfully created folder: {new_folder_path}")
        except OSError as e:
            print(f"Error creating folder {new_folder_path}: {e}")
        
        print(f"Saving rerun logs to {new_folder_path}.")
        self.log_file = os.path.join(new_folder_path, "sim_log.csv")
        
        # Save simulation configuration
        data_structure = [
            {'name': 'time', 'length': 1},
            {'name': 'qpos', 'length': self.robot.mj_data.qpos.shape[0]},
            {'name': 'qvel', 'length': self.robot.mj_data.qvel.shape[0]},
            {'name': 'obs', 'length': self.policy.get_num_obs()},
            {'name': 'action', 'length': self.policy.get_num_actions()},
            {'name': 'torque', 'length': self.robot.mj_model.nu},
            {'name': 'left_ankle_pos', 'length': 3},
            {'name': 'right_ankle_pos', 'length': 3},
            {'name': 'commanded_vel', 'length': 3},
        ]
        
        sim_config = {
            'simulator': "mujoco",
            'robot': self.robot.robot_name,
            'policy': self.policy.get_chkpt_path(),
            'policy_dt': self.policy.dt,
            'use_height_sensor': self.use_height_sensor,
            'data_structure': data_structure
        }
        
        config_path = os.path.join(new_folder_path, "sim_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(sim_config, f)

    def run(self):
        """Run the simulation."""
        print(f"Starting mujoco simulation with robot {self.robot.robot_name}.\n"
              f"Policy dt set to {self.policy.dt} s ({self.sim_steps_per_policy_update} steps per policy update.)\n"
              f"Simulation dt set to {self.robot.mj_model.opt.timestep} s. Sim loop rate set to {self.sim_loop_rate} s.\n"
              f"Height sensor enabled: {self.use_height_sensor}\n")

        with mujoco.viewer.launch_passive(self.robot.mj_model, self.robot.mj_data) as viewer:
            # Setup height sensor visualization if enabled
            if self.use_height_sensor:
                grid_size = (1.5,1.5)
                x_y_num_rays = (16 , 16)
                height_map = self._ray_cast_sensor(self.robot.mj_model, self.robot.mj_data, "height_sensor_site", grid_size, x_y_num_rays)
                # Add custom debug spheres
                ii = 0
                for pos in height_map.reshape(-1, 3):
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[ii],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.01, 0, 0]),
                        pos=pos,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0, 0, 1]),
                    )
                    viewer.user_scn.ngeom += 1
                    ii += 1

            while viewer.is_running():
                # Get observation and compute action
                if self.use_height_sensor:
                    height_map = self._ray_cast_sensor(self.robot.mj_model, self.robot.mj_data, "height_sensor_site", grid_size, x_y_num_rays)
                    site_id = mujoco.mj_name2id(self.robot.mj_model, mujoco.mjtObj.mjOBJ_SITE, "height_sensor_site")
                    sensor_pos = self.robot.mj_data.site_xpos[site_id]

                obs = self.robot.create_observation(self.policy, height_map=height_map, sensor_pos=sensor_pos)
                action = self.policy.get_action(obs)
                self.robot.apply_action(action)

                # Step the simulator
                for i in range(self.sim_steps_per_policy_update):
                    # Update scene
                    scene = mujoco.MjvScene(self.robot.mj_model, maxgeom=1000)
                    cam = mujoco.MjvCamera()
                    opt = mujoco.MjvOption()
                    mujoco.mjv_updateScene(
                        self.robot.mj_model, self.robot.mj_data, opt, None, cam,
                        mujoco.mjtCatBit.mjCAT_ALL, scene
                    )
                    
                    # Update height sensor visualization if enabled
                    if self.use_height_sensor:
                        height_map = self._ray_cast_sensor(self.robot.mj_model, self.robot.mj_data, "height_sensor_site", grid_size, x_y_num_rays)
                        # print(height_map)
                        ii = 0
                        for pos in height_map.reshape(-1, 3):
                            viewer.user_scn.geoms[ii].pos = pos
                            ii += 1
                    
                    # Get latest joystick command before stepping
                    self.robot.get_joystick_command()
                    self.robot.step()
                    
                    # Only log and sync viewer at viewer_rate intervals
                    if i % self.viewer_rate == 0:
                        if self.log:
                            log_data = self.robot.get_log_data(self.policy, obs, action)
                            if any(abs(v) > 1e-6 for v in log_data[-3:]):  # Only print if commanded velocity is non-zero
                                print(f"Commanded velocity: {log_data[-3:]}")
                            log_row_to_csv(self.log_file, log_data)
                        viewer.sync()

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

        site_pos[0] = site_pos[0] - size[0] / 2.
        site_pos[1] = site_pos[1] - size[1] / 2.


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
                    model, data,
                    ray_origin.astype(np.float64),
                    direction.astype(np.float64),
                    geom_group, 1, -1, geom_id
                )

                ray_pos[xray, yray, :] = ray_origin + ray_pos[xray, yray, :]

        return ray_pos 