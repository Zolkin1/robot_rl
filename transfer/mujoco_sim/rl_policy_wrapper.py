import torch
import numpy as np

class RLPolicy():
    """RL Policy Wrapper"""
    def __init__(self, dt: float, checkpoint_path: str, num_obs: int, num_action: int,
                 cmd_scale: float, period: float, action_scale: float, default_angles: np.array, qvel_scale: float):
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

    def create_obs(self, qpos, qvel, time, projected_gravity, des_vel):
        """Create the observation vector from the sensor data"""
        obs = np.zeros(self.num_obs, dtype=np.float32)

        qj = qpos[7:] - self.default_angles

        obs[:3] = qvel[3:6]                                                 # Angular velocity
        obs[3:6] = projected_gravity                                        # Projected gravity
        obs[6:9] = des_vel*self.cmd_scale                                   # Command velocity

        nj = len(qj)
        obs[9 : 9 + nj] = self.convert_to_isaac(qj)                                          # Joint pos
        obs[9 + nj : 9 + 2*nj] = self.convert_to_isaac(qvel[6:]) * self.qvel_scale                            # Joint vel
        obs[9 + 2*nj : 9 + 3*nj] = self.action_isaac                              # Past action

        sin_phase = np.sin(2 * np.pi * time/self.period)
        cos_phase = np.cos(2 * np.pi * time/self.period)
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