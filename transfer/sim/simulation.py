import os
import time
import math
import csv
import yaml
import mujoco
import mujoco.viewer
from datetime import datetime

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
    def __init__(self, policy, robot: Robot, log: bool = False, log_dir: str = None):
        """Initialize the simulation."""
        self.policy = policy
        self.robot = robot
        self.log = log
        self.log_dir = log_dir
        self.log_file = None
        
        # Setup simulation parameters
        self.sim_steps_per_policy_update = int(policy.dt / robot.mj_model.opt.timestep)
        self.sim_loop_rate = self.sim_steps_per_policy_update * robot.mj_model.opt.timestep
        self.viewer_rate = math.ceil((1 / 100) / robot.mj_model.opt.timestep)
        
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
            'data_structure': data_structure
        }
        
        config_path = os.path.join(new_folder_path, "sim_config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(sim_config, f)

    def run(self):
        """Run the simulation."""
        print(f"Starting mujoco simulation with robot {self.robot.robot_name}.\n"
              f"Policy dt set to {self.policy.dt} s ({self.sim_steps_per_policy_update} steps per policy update.)\n"
              f"Simulation dt set to {self.robot.mj_model.opt.timestep} s. Sim loop rate set to {self.sim_loop_rate} s.\n")

        with mujoco.viewer.launch_passive(self.robot.mj_model, self.robot.mj_data) as viewer:
            while viewer.is_running():
                start_time = time.time()

                # Get observation and compute action
                obs = self.robot.create_observation(self.policy)
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
                    
                    # Get latest joystick command before stepping
                    self.robot.get_joystick_command()
                    self.robot.step()
                    
                    if self.log:
                        log_data = self.robot.get_log_data(self.policy, obs, action)
                        if i == 0 and any(abs(v) > 1e-6 for v in log_data[-3:]):  # Only print if commanded velocity is non-zero
                            print(f"Commanded velocity: {log_data[-3:]}")
                        log_row_to_csv(self.log_file, log_data)
                    
                    if i % self.viewer_rate == 0:
                        viewer.sync()

                # Try to run in roughly realtime
                elapsed = time.time() - start_time
                if elapsed < 1 * self.sim_loop_rate:
                    time.sleep(1 * self.sim_loop_rate - elapsed) 