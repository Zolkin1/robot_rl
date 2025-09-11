import math
import torch

from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
    RewardsCfg,
    EventCfg,
)
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg as ObsTerm

import robot_rl.tasks.manager_based.robot_rl.amber.mdp as mdp

# from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidEnvCfg

##
# Pre-defined configs
##
from .amber_env_cfg import AmberEnvCfg
from .amber5 import AMBER_CFG

def sampled_forward_only(env, params):
    """
    Always outputs a constant vx, with vy=0, ωz=0.
    """
    print("custom")
    n = env.num_envs
    vx = torch.full((n,), params.get("vx", 0.5), device=env.device)
    return torch.stack([vx, torch.zeros_like(vx), torch.zeros_like(vx)], dim=1)
##
# Environment configuration

@configclass
class AmberRoughEnvCfg(AmberEnvCfg):
    """Configuration for the G1 Flat environment."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # from isaaclab.sim import schemas, schemas_cfg
        # from pxr import UsdPhysics             # need this to apply the API
        # import omni.usd

        # def set_torso_mass(mass_kg: float, env_id: int = 0):
        #     stage      = omni.usd.get_context().get_stage()
        #     torso_path = f"/World/envs/env_{env_id}/Amber/amber3_PF/torso"

        #     # Ensure MassAPI exists; modify_mass_properties() returns False otherwise
        #     prim = stage.GetPrimAtPath(torso_path)
        #     if not UsdPhysics.MassAPI(prim):
        #         UsdPhysics.MassAPI.Apply(prim)                          # :contentReference[oaicite:2]{index=2}

        #     cfg = schemas_cfg.MassPropertiesCfg(mass=mass_kg)           # only this attr is set :contentReference[oaicite:3]{index=3}
        #     schemas.modify_mass_properties(torso_path, cfg, stage)      # returns True on success :contentReference[oaicite:4]{index=4}
        # set_torso_mass(10)

        base_reset = self.events.reset_base
        base_reset.params["pose_range"]["yaw"] = (0.0, 0.0)

        self.scene.robot = AMBER_CFG.replace(prim_path="{ENV_REGEX_NS}/Amber")
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Amber/amber3_PF/torso",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
        )
        self.scene.contact_forces_right = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Amber/amber3_PF/right_shin",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
        )
        self.scene.contact_forces_left = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Amber/amber3_PF/left_shin",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
        )
        
        
        #-------------- Change the observations from base to a link based
        self.observations.base_velocity = None


        # add your custom XY-vel observation
        # self.observations.policy.body_link_vel_xy = ObsTerm(
        #     func=body_link_vel_xy,
        #     params={
        #         "command_name": "base_velocity",       # whatever your command alias is
        #         "asset_cfg": SceneEntityCfg(name="robot")
        #     },
        #     # you may need to set history_length, scale, etc.
        #     history_length=1,
        #     scale=1.0,
        # )
        ##
        # Scene
        ##
        # self.scene.robot = AMBER_CFG.replace(prim_path="{ENV_REGEX_NS}/Amber")
        # self.scene.robot = AMBER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis_link"

        # No height scanner for now
        self.scene.height_scanner = None

        self.terminations.base_contact = TerminationTermCfg(
            func=mdp.torso_contact_termination,
            params={
                "sensor_cfg": SceneEntityCfg(name="contact_forces"),
                "asset_cfg":  SceneEntityCfg(name="robot"),
            },
        )
        ##
        # Randomization
        ##
        # self.events.push_robot = None
        # self.events.push_robot.params["velocity_range"] = {"x": (-1, 1), "y": (-1, 1), "roll": (-0.4, 0.4),
        #                                                    "pitch": (-0.4, 0.4), "yaw": (-0.4, 0.4)}
        # self.events.add_base_mass.params["asset_cfg"].body_names = ["torso"]
        # self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        # self.events.add_base_mass.params["operation"] = "scale"
        # self.events.randomize_ground_contact_friction.params["static_friction_range"] = (0.1, 1.25)
        # self.events.randomize_ground_contact_friction.params["dynamic_friction_range"] = (0.1, 1.25)
        # self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["pelvis_link"]
        # self.events.reset_base.params = {
        #     "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        #     "velocity_range": {
        #         "x": (0.0, 0.0),
        #         "y": (0.0, 0.0),
        #         "z": (0.0, 0.0),
        #         "roll": (0.0, 0.0),
        #         "pitch": (0.0, 0.0),
        #         "yaw": (0.0, 0.0),
        #     },
        # }
        # preserve whatever you did with external forces
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso"]

        ##
        # Commands
        ##
        # self.commands.base_velocity.sampler = sampled_forward_only
        # self.commands.base_velocity.params = {"vx": 0.5}
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1) # 0 - 1
        # self.commands.base_velocity.ranges.lin_vel_y = (0,0) #(-1.0, 1.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (0, 0)
        self.events.add_base_mass = None
        ##
        # Terminations
        ##
        # self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["pelvis_link"]

        ##
        # Rewards
        ##
        
        # self.rewards.track_lin_vel_xy_exp.weight = 1.2
        # self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight =  -.2 # TODO reduce this maybe?
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        # self.rewards.dof_vel_l2.weight = -1.0e-3
        self.rewards.action_rate_l2.weight = -0.01
        # self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_pos_limits.weight = -5.0
        self.rewards.alive.weight = 0.15
        self.rewards.undesired_contacts = None

        # self.rewards.contact_no_vel.weight = -0.2
        # self.rewards.joint_deviation_hip.weight = -1.0
        # self.rewards.height_torso.weight = -10.0
        # self.rewards.feet_clearance.weight = -20.0
        # self.rewards.phase_contact.weight = 0.25

       # big penalty on fall (torso contact)
        self.rewards.termination_penalty.weight           = -200.0  
        # reward forward x‐velocity tracking
        self.rewards.track_lin_vel_xy.weight              =  30#80.0  
        # penalize asymmetric joints per cycle
        self.rewards.joint_symmetry_reward.weight        =  0#2#
        # self.rewards.joint_symmetry_reward.params["diff_threshold"] = 5
        self.rewards.joint_symmetry_reward.params["debug"] = False
        # penalize asymmetric joints per cycle
        self.rewards.stride_consistency.weight        =  10#10#
        # self.rewards.joint_symmetry_reward.params["diff_threshold"] = 5
        self.rewards.stride_consistency.params["debug"] = False
        # Reward phase based contacts: stance and wing
        self.rewards.phase_contact.weight                 =  0
        # Reward phase based contacts: stance and wing
        self.rewards.phase_contact_per_cycle.weight       =  0
        # punish large arm joint deviations
        self.rewards.joint_angles.weight                  =   -.2#-4.0  
        # reward maintaining torso upright within window, penalize beyond threshold
        self.rewards.torso_orientation.weight             =    4#2.0  
        # reward alternating foot contacts vs repeats
        self.rewards.alternation_contact.weight           =   10
        # reward progressive foot placement per cycle
        self.rewards.progressive_step.weight              =    3#15.0  
        # per‐cycle foot‐contact correctness (+5 for exactly one each, else penalty)
        self.rewards.foot_cycle_sym.weight                =   0#4
        # penalize asymmetric foot airtime
        self.rewards.symmetric_foot_airtime.weight        =  0#
        self.rewards.symmetric_foot_airtime.params["diff_threshold"] = 5
        self.rewards.symmetric_foot_airtime.params["reward_good"] = 8
        # penalize foot sliding (squared speed during contact)
        self.rewards.feet_no_slip_condition.weight        =  -1.5#-10.0  
        # cycle based reward
        self.rewards.paper_cycle_reward.weight            = 4#5
        self.rewards.paper_cycle_reward.params["debug"]   = False
        # Reward for contact step ahead of com
        self.rewards.com_step_reward.weight               = 0
        self.rewards.com_step_reward.params["debug"]      = False

        # forward feet placement:
        # self.rewards.foot_forward_placement = 0.8
        # forward and progressive foot placement:
        # self.rewards.alternating_forward_step = 0.8
        # # -- Regularization
        # self.rewards.dof_torques_l2.weight = -1e-4                  # Joint torques
        # # self.rewards.torque_lim.weight = -1e-2                      # Torque limits
        # self.rewards.joint_vel.weight = -1e-3                       # Joint velocity
        # self.rewards.dof_pos_limits.weight = -1 #-10                    # Joint limits
        # self.rewards.joint_reg.weight = 0. #0.25                    # Regularize positions of leg joints (relative to a nominal)
        # self.rewards.ang_vel_xy_l2.weight = -0.05                   # Base x-y angular velocity
        # self.rewards.lin_vel_z_l2.weight = -2.0                     # Base z linear velocity
        # self.rewards.flat_orientation_l2.weight = -1.5              # Tilting
        # self.rewards.action_rate_l2.weight = -0.005                 # Action smoothing
        # self.rewards.joint_deviation_hip.weight = -1.0              # Hip yaw and roll regularization
        # self.rewards.joint_deviation_arms.weight = -0.5             # Arms regularization
        # self.rewards.joint_deviation_torso.weight = -1.0
        # self.rewards.phase_feet_contacts.weight = 0.25 #1.               # Contact location
        # self.rewards.height_torso.weight = -2.                     # Base height
        # self.rewards.height_torso.params["target_height"] = 0.76
        # self.rewards.feet_clearance.weight = -20.
        # self.rewards.feet_clearance.params["target_height"] = 0.1
        # self.rewards.feet_slide.weight = -0.3
        # self.rewards.dof_acc_l2.weight = -1.25e-7
        # self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        # )
        # self.rewards.dof_torques_l2.weight = -1.5e-7
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        # )
        #
        # # -- Task
        # self.rewards.track_lin_vel_xy_exp.weight = 2 #2.5
        # self.rewards.track_ang_vel_z_exp.weight = 0.5 #0.75  # 0
        #
        # # -- Unused
        # self.rewards.track_heading.weight = 0.                     # Base heading
        # self.rewards.feet_air_time.weight = 0.
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso"]
        
        