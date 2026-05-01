# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the environment."""

# Lazy-load isaaclab.envs.mdp to avoid triggering pxr imports before SimulationApp.
# Wildcard import from this module resolves all lazy attributes, which pulls in pxr.
# Instead, we import the module and fall back to it via __getattr__.
import isaaclab.envs.mdp as _isaaclab_envs_mdp

from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # Inherit from the base envs

from .rewards.rewards import *  # noqa: F401, F403
from .observations.observations import *  # noqa: F401, F403
from .observations.depth import depth_image, depth_image_4d  # noqa: F401
from .curriculums.curriculums import *  # noqa: F401, F403
from .terminations.terminations import *  # noqa: F401, F403
from .commands import *  # noqa: F401, F403
from .events.init_config import *  # noqa: F401, F403
from .events.physical_randomization import *  # noqa: F401, F403
from .events.resets import *  # noqa: F401, F403
from .events.depth import randomize_camera_intrinsics  # noqa: F401


def __getattr__(name):
    """Lazily resolve attributes from isaaclab.envs.mdp."""
    return getattr(_isaaclab_envs_mdp, name)