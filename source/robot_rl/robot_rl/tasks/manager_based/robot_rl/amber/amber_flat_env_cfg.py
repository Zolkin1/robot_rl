import math
import torch

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm

# from .g1_rough_env_cfg import G1RoughEnvCfg

##
# Pre-defined configs
##
from .amber_env_cfg import AmberEnvCfg
# from .amber_env import AmberFlatEnv
##
# Environment configuration
##

@configclass
class AmberFlatEnvCfg(AmberEnvCfg):
    # env: type = AmberFlatEnv    
    def __post_init__(self):
        super().__post_init__()
        # override terrain to flat plane...
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None
        # (any other flat‐specific tweaks)
