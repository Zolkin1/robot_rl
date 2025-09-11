# amber/tasks/manager_based/robot_rl/amber/mdp/termination.py
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.sensors import ContactSensor
from isaaclab.envs import ManagerBasedRLEnv

def torso_contact_termination(
    env: ManagerBasedRLEnv,
    sensor_cfg,
    asset_cfg,
) -> torch.Tensor:
    """
    Terminate an env whenever the torso’s contact sensor fires.
    Returns a (num_envs,) boolean mask.
    """
    # grab the sensor by name
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w_history: [n_envs, hist, n_bodies, 3]
    forces = contact_sensor.data.net_forces_w_history
    # compute max force magnitude over history → [n_envs, n_bodies]
    contact_detected = forces.norm(dim=-1).max(dim=1)[0] > 0.0
    # if *any* torso body got hit → terminate that env
    return contact_detected.any(dim=-1)