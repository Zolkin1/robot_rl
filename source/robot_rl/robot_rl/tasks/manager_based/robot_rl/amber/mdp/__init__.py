# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the environment."""


from .termination import torso_contact_termination


from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *    #Inherit from the base envs
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import generated_commands as _base_gen

from .rewards import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403


# def generated_commands(env, command_name: str):
#     """
#     Wrap the default velocity‐command MDP so we zero out lateral (y) and yaw (z).
#     """
#     # get the 3‐vector [vx, vy, wz] from the base MDP
#     cmds = _base_gen(env, command_name)

#     # enforce planar: only X velocity allowed
#     # cmds is shape [num_envs, 3]
#     #   [:,0] = forward, [:,1] = lateral, [:,2] = yaw-rate
#     cmds = cmds.clone()            # avoid mutating any shared tensor
#     cmds[:, 1] = 0.0               # zero out lateral
#     cmds[:, 2] = 0.0               # zero out yaw-rate
#     return cmds