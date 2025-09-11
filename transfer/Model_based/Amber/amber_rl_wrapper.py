import numpy as np
import torch


class RLPolicy:
    """RL Policy Wrapper"""

    def __init__(
        self,
        dt: float,
        checkpoint_path: str,
        num_obs: int,
        num_action: int,
        cmd_scale: list,
        period: float,
        action_scale: float,
        default_angles: np.array,
        qvel_scale: float,
        ang_vel_scale: float,
        height_map_scale=None,
    ):
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
            0: 0,  # left_hip_pitch
            1: 6,  # right_hip_pitch
            2: 12,  # waist_yaw
            3: 1,  # left_hip_roll
            4: 7,  # right_hip_roll
            5: 13,  # left_shoulder_pitch
            6: 17,  # right_shoulder_pitch
            7: 2,  # left_hip_yaw
            8: 8,  # right_hip_yaw
            9: 14,  # left_shoulder_roll
            10: 18,  # right_shoulder_roll
            11: 3,  # left_knee
            12: 9,  # right_knee
            13: 15,  # left_shoulder_yaw
            14: 19,  # right_shoulder_yaw
            15: 4,  # left_ankle_pitch
            16: 10,  # right_ankle_pitch
            17: 16,  # left_elbow
            18: 20,  # right_elbow
            19: 5,  # left_ankle_roll
            20: 11,  # right_ankle_roll
        }

        # Load in the policy
        self.load()

    def load(self):
        """Load RL Policy"""
        self.policy = torch.jit.load(self.checkpoint_path)
        # load to cuda
        if torch.cuda.is_available():
            self.policy = self.policy.cuda()

    def create_obs(
        self,
        qjoints: np.ndarray,         # shape (7,)
        body_ang_vel: np.ndarray,    # shape (3,)
        qvel: np.ndarray,            # shape (7,)
        time: float,
        projected_gravity: np.ndarray,  # shape (3,)
        des_vel: np.ndarray           # shape (3,)
    ) -> torch.Tensor:
        """
        Build policy obs of shape (1,29) from:
        0:3   base_ang_vel        (3,)
        3:6   projected_gravity   (3,)
        6:9   velocity_commands   (3,)
        9:16  joint_pos           (7,)
        16:23 joint_vel           (7,)
        23:27 past actions        (4,)
        27    sin_phase           (1,)
        28    cos_phase           (1,)
        """
        obs = np.zeros(29, dtype=np.float32)

        # 0:3 — base angular velocity
        obs[0:3] = body_ang_vel * self.ang_vel_scale
        # print("scale----------------",self.ang_vel_scale)
        # 3:6 — projected gravity
        obs[3:6] = projected_gravity

        # 6:9 — commanded base velocity [vx, vy, vyaw]
        # self.cmd_scale should be a length-3 list or array
        obs[6:9] = des_vel * np.array(self.cmd_scale, dtype=np.float32)

        # 9:16 — joint positions relative to default stance
        # self.default_angles must now be length-7
        q_rel = qjoints #- np.array(self.default_angles, dtype=np.float32)
        obs[9:16] = q_rel

        # 16:23 — joint velocities (scaled)
        obs[16:23] = qvel * self.qvel_scale
        # print(f"qvel shape:{qvel}")
        # print(f"qscale:{self.qvel_scale}")
        # 23:27 — previous action (isaac-ordered) shape (4,)
        obs[23:27] = self.action_isaac

        # 27 & 28 — phase clock
        sin_p = np.sin(2 * np.pi * time / self.period)
        cos_p = np.cos(2 * np.pi * time / self.period)
        obs[27] = sin_p
        obs[28] = cos_p

        # turn into a (1,29) torch tensor
        return torch.from_numpy(obs).unsqueeze(0)

    def get_action(self, obs: torch.Tensor) -> np.array:
        """Get action from RL Policy"""
        if torch.cuda.is_available():
            obs_cuda = obs.cuda()
            self.action_isaac = self.policy(obs_cuda).detach().cpu().numpy().squeeze()
        else:
            self.action_isaac = self.policy(obs).detach().numpy().squeeze()

        return (self.action_isaac) * self.action_scale + self.default_angles[:self.num_actions]

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
        default_isaac = self.default_angles
        # print(default_isaac)
        # print(self.action_isaac)
        default_isaac_only_actuated = default_isaac[: self.num_actions]
        return self.action_isaac * self.action_scale + default_isaac_only_actuated

    def convert_height_map_to_obs(self, height_map, sensor_pos, offset=0.5):
        """Converts the (N, M, 3) height map to an observation vector.
        sensor_pos is the position of the position of the sensor
        offset is the same as the height_scan issac lab function.
        """
        obs = np.zeros(height_map.shape[0] * height_map[1])
        if self.height_map_scale is not None:
            # IsaacLab default is "xy" for the grid ordering
            for x in range(height_map.shape[0]):
                for y in range(height_map[1]):
                    # TODO: Verify that it is clipped
                    obs[x * height_map[1] + y] = np.clip(sensor_pos[2] - height_map[x, y, 2] - offset, -1, 1)
        else:
            raise ValueError("Height map scale is none but a height map was passed in!")
