from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import numpy as np
from hid import device

from robot_rl.assets.robots.cartpole import CARTPOLE_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from .cartpole_mpc import SpcMpcCartpoleConfig, SpcMpcCartpole
from judo.controller import Controller, ControllerConfig
from judo.optimizers import MPPI, MPPIConfig

@configclass
class SpcCartpoleEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0
    action_scale = 100.0  # [N]
    action_space = 8 # quadratic and linear term of the MPC
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # reset
    max_cart_pos = 3.0  # the cart is reset if it exceeds that position [m]
    initial_pole_angle_range = [-0.25, 0.25]  # the range in which the pole angle is sampled from on reset [rad]

    # reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005

    # SPC Controller
    horizon = 1.0
    opt_iters = 1
    max_num_traces = 1

    sigma = 0.1
    temp = 0.05
    num_samples = 32
    use_noise_ramp = False
    num_nodes = 4
    nu = 1


class SpcCartpoleEnv(DirectRLEnv):
    cfg: SpcCartpoleEnvCfg

    def __init__(self, cfg: SpcCartpoleEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._cart_dof_idx, _ = self.cartpole.find_joints(self.cfg.cart_dof_name)
        self._pole_dof_idx, _ = self.cartpole.find_joints(self.cfg.pole_dof_name)
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        # SPC Controllers - array of judo controllers
        mp = "/home/zolkin/AmberLab/Project-Isaac-RL/robot-rl/robot_rl/source/robot_rl/robot_rl/assets/robots/mujoco/cartpole.xml"
        # Note: Numpy on CPU
        # TODO: Make a controller in addition to the task
        ctrl_config = ControllerConfig(horizon=cfg.horizon, spline_order="linear", max_opt_iters=cfg.opt_iters,
                                       max_num_traces=cfg.max_num_traces)
        optimizer_config = MPPIConfig(sigma=cfg.sigma, temperature=cfg.temp, num_rollouts=cfg.num_samples,
                                      use_noise_ramp=cfg.use_noise_ramp,
                                      num_nodes=cfg.num_nodes)
        optimizer = MPPI(optimizer_config, cfg.nu)
        self.spc_control_configs = [SpcMpcCartpoleConfig(Q=np.zeros(4), q=np.zeros(4)) for i in range(self.num_envs)]
        self.spc_tasks = [SpcMpcCartpole(model_path=mp) for i in range(self.num_envs)]
        self.spc_controllers = []
        for i in range(self.num_envs):
            self.spc_controllers.append(Controller(ctrl_config, SpcMpcCartpole(model_path=mp),
                                               self.spc_control_configs[i], optimizer, optimizer_config))

    def _setup_scene(self):
        self.cartpole = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation to scene
        self.scene.articulations["cartpole"] = self.cartpole
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Called once per RL step."""
        # Update the config with the actions
        for i in range(self.num_envs):
            self.spc_control_configs[i].Q = actions[i, :4].cpu().numpy()
            self.spc_control_configs[i].q = actions[i, 4:].cpu().numpy()

            time = self.sim.current_time
            state = np.concatenate((self.joint_pos[i, (self._cart_dof_idx[0], self._pole_dof_idx[0])].cpu().numpy(),
                             self.joint_vel[i, (self._cart_dof_idx[0], self._pole_dof_idx[0])].cpu().numpy()))
            # Compute the MPC
            self.spc_controllers[i].task_config = self.spc_control_configs[i]
            self.spc_controllers[i].update_action(state, time)


    def _apply_action(self) -> None:
        """Called once for every simulation step."""
        # Interpolate into the trajectory
        # self.cartpole.set_joint_effort_target(self.actions, joint_ids=self._cart_dof_idx)
        for i in range(self.num_envs):
            self.cartpole.set_joint_position_target(torch.from_numpy(
                self.spc_controllers[i].action(self.sim.current_time)))

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
                self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
                self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
            ),
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.cfg.rew_scale_alive,
            self.cfg.rew_scale_terminated,
            self.cfg.rew_scale_pole_pos,
            self.cfg.rew_scale_cart_vel,
            self.cfg.rew_scale_pole_vel,
            self.joint_pos[:, self._pole_dof_idx[0]],
            self.joint_vel[:, self._pole_dof_idx[0]],
            self.joint_pos[:, self._cart_dof_idx[0]],
            self.joint_vel[:, self._cart_dof_idx[0]],
            self.reset_terminated,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.cartpole.data.joint_pos
        self.joint_vel = self.cartpole.data.joint_vel

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_bounds = torch.any(torch.abs(self.joint_pos[:, self._cart_dof_idx]) > self.cfg.max_cart_pos, dim=1)
        out_of_bounds = out_of_bounds | torch.any(torch.abs(self.joint_pos[:, self._pole_dof_idx]) > math.pi / 2, dim=1)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.cartpole._ALL_INDICES
        super()._reset_idx(env_ids)

        joint_pos = self.cartpole.data.default_joint_pos[env_ids]
        joint_pos[:, self._pole_dof_idx] += sample_uniform(
            self.cfg.initial_pole_angle_range[0] * math.pi,
            self.cfg.initial_pole_angle_range[1] * math.pi,
            joint_pos[:, self._pole_dof_idx].shape,
            joint_pos.device,
        )
        joint_vel = self.cartpole.data.default_joint_vel[env_ids]

        default_root_state = self.cartpole.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.joint_pos[env_ids] = joint_pos
        self.joint_vel[env_ids] = joint_vel

        self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # TODO: Reset the SPC controllers
        for i in range(self.num_envs):
            self.spc_controllers[i].reset()


@torch.jit.script
def compute_rewards(
    rew_scale_alive: float,
    rew_scale_terminated: float,
    rew_scale_pole_pos: float,
    rew_scale_cart_vel: float,
    rew_scale_pole_vel: float,
    pole_pos: torch.Tensor,
    pole_vel: torch.Tensor,
    cart_pos: torch.Tensor,
    cart_vel: torch.Tensor,
    reset_terminated: torch.Tensor,
):
    rew_alive = rew_scale_alive * (1.0 - reset_terminated.float())
    rew_termination = rew_scale_terminated * reset_terminated.float()
    rew_pole_pos = rew_scale_pole_pos * torch.sum(torch.square(pole_pos).unsqueeze(dim=1), dim=-1)
    rew_cart_vel = rew_scale_cart_vel * torch.sum(torch.abs(cart_vel).unsqueeze(dim=1), dim=-1)
    rew_pole_vel = rew_scale_pole_vel * torch.sum(torch.abs(pole_vel).unsqueeze(dim=1), dim=-1)
    total_reward = rew_alive + rew_termination + rew_pole_pos + rew_cart_vel + rew_pole_vel
    return total_reward