import torch
import numpy as np

class RLPolicy():
    """RL Policy Wrapper"""
    def __init__(self, dt: float, checkpoint_path: str, num_obs: int, num_action: int,
                 cmd_scale: list, period: float, action_scale: float, default_angles: np.array, qvel_scale: float,
                 ang_vel_scale: float, height_map_scale=None):
        """Initialize RL Policy Wrapper.
            freq: time between actions (s)
        """
        self.dt = dt
        self.checkpoint_path = checkpoint_path
        self.num_obs = num_obs
        self.cmd_scale = cmd_scale
        self.period = period
        self.num_actions = num_action
        self.action_scale = action_scale
        self.default_angles = default_angles
        self.qvel_scale = qvel_scale
        self.ang_vel_scale = ang_vel_scale
        self.height_map_scale = height_map_scale

        self.action_isaac = np.zeros(num_action)

        if self.checkpoint_path == "newest":
            # TODO: Find the newest policy in the normal location
            pass

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

        # Load in the policy
        self.load()

    def load(self):
        """Load RL Policy"""
        self.policy = torch.jit.load(self.checkpoint_path)
        # load to cuda
        if torch.cuda.is_available():
            self.policy = self.policy.cuda()

    def create_obs(self, qjoints, body_ang_vel, qvel, time, projected_gravity, des_vel,
                   height_map=None, sensor_pos=None, convention="mj"):
        """Create the observation vector from the sensor data"""
        obs = np.zeros(self.num_obs, dtype=np.float32)

        obs[:3] = body_ang_vel*self.ang_vel_scale                                                 # Angular velocity
        obs[3:6] = projected_gravity                                        # Projected gravity
        obs[6] = des_vel[0]*self.cmd_scale[0]                                   # Command velocity
        obs[7] = des_vel[1]*self.cmd_scale[1]                                   # Command velocity
        obs[8] = des_vel[2]*self.cmd_scale[2]     
                                     # Command velocity

        nj = len(qjoints)
        if convention == "mj":
            qj = qjoints - self.default_angles
            obs[9 : 9 + nj] = self.convert_to_isaac(qj)                                          # Joint pos
            obs[9 + nj : 9 + 2*nj] = self.convert_to_isaac(qvel) * self.qvel_scale                            # Joint vel
        else:
            qj = qjoints - self.convert_to_isaac(self.default_angles)
            obs[9: 9 + nj] = qj  # Joint pos
            obs[9 + nj: 9 + 2 * nj] = qvel * self.qvel_scale  # Joint vel

        obs[9 + 2*nj : 9 + 3*nj] = self.action_isaac                              # Past action

        sin_phase = np.sin(2 * np.pi * time/self.period)
        cos_phase = np.cos(2 * np.pi * time/self.period)

        if height_map is not None:
            height_obs = self.convert_height_map_to_obs(height_map, sensor_pos)
            # print(height_obs)
            # height_obs = np.ones(256)*0.25
            obs[9 + 3 * nj:9 + 3 * nj + height_obs.shape[0]] = height_obs
            obs[9 + 3 * nj + height_obs.shape[0] : 9 + 3 * nj + height_obs.shape[0] + 2] = np.array([sin_phase, cos_phase])     # Phases
        else:
            obs[9 + 3*nj : 9 + 3*nj + 2] = np.array([sin_phase, cos_phase])     # Phases

        obs_tensor = torch.from_numpy(obs).unsqueeze(0)

        # print(obs_tensor)

        return obs_tensor

    def get_action(self, obs: torch.Tensor) -> np.array:
        """Get action from RL Policy"""
        if torch.cuda.is_available():
            obs_cuda = obs.cuda()
            self.action_isaac = self.policy(obs_cuda).detach().cpu().numpy().squeeze()
        else:
            self.action_isaac = self.policy(obs).detach().numpy().squeeze()

        return self.convert_to_mujoco(self.action_isaac) * self.action_scale + self.default_angles

    def get_num_actions(self) -> int:
        return self.num_actions

    def get_num_obs(self) -> int:
        return self.num_obs

    def convert_to_mujoco(self, vec):
        mj_vec = np.zeros(21)
        for isaac_index, mujoco_index in self.isaac_to_mujoco.items():
            mj_vec[mujoco_index] = vec[isaac_index]

        return mj_vec

    def convert_to_isaac(self, vec):
        isaac_vec = np.zeros(21)
        for isaac_index, mujoco_index in self.isaac_to_mujoco.items():
            isaac_vec[isaac_index] = vec[mujoco_index]

        return isaac_vec

    def get_chkpt_path(self):
        return self.checkpoint_path

    def get_action_isaac(self):
        default_isaac = self.convert_to_isaac(self.default_angles)
        return self.action_isaac * self.action_scale + default_isaac

    def convert_height_map_to_obs(self, height_map, sensor_pos, offset=0.5):
        """Converts the (N, M, 3) height map to an observation vector.
            sensor_pos is the position of the position of the sensor
            offset is the same as the height_scan issac lab function.
        """
        obs = np.zeros(height_map.shape[0] * height_map.shape[1])
        if self.height_map_scale is not None:
            # IsaacLab default is "xy" for the grid ordering
            for x in range(height_map.shape[0]):
                for y in range(height_map.shape[1]):
                    # TODO: Verify that it is clipped
                    obs[x * height_map.shape[1] + y] = np.clip(sensor_pos[2] - height_map[x, y, 2] - offset, -1, 1)
        else:
            raise ValueError("Height map scale is none but a height map was passed in!")
        return obs