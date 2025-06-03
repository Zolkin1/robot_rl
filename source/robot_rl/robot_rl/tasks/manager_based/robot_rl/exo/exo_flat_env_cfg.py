from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidEnvCfg


from robot_rl.tasks.manager_based.robot_rl.g1.g1_rough_env_lip_cfg import G1RoughLipCommandsCfg, G1RoughLipRewards
from robot_rl.tasks.manager_based.robot_rl.g1_isaac.rough_env_cfg import G1RoughLipObservationsCfg, G1_Observations
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg  #Inherit from the base envs


from robot_rl.assets.robots.exo_cfg import EXO_CFG  # isort: skip

from robot_rl.tasks.manager_based.robot_rl.mdp.cmd_cfg import HLIPCommandCfg, HZDCommandCfg
EXO_PERIOD = 2.0

@configclass
class ExoHLIPCommandsCfg(CommandsCfg):

     hlip_ref = HLIPCommandCfg()
     hlip_ref.foot_body_name = ".*HenkeAnkleLink"
     hlip_ref.gait_period = EXO_PERIOD
     hlip_ref.z0 = 0.85
     hlip_ref.y_nom =0.275
     hlip_ref.z_sw_max = 0.08

     def __post_init__(self):
        
        super().__post_init__()
        
class ExoHZDCommandsCfg(CommandsCfg):

     hzd_ref = HZDCommandCfg()
     hzd_ref.foot_body_name = ".*HenkeAnkleLink"

@configclass
class ExoRewardsCfg(G1RoughLipRewards):

     
     def __post_init__(self):

          super().__post_init__()
          self.feet_air_time.params["sensor_cfg"].body_names = ".*HenkeAnkleLink"
          self.feet_air_time.params["threshold"] = EXO_PERIOD/2
          self.phase_contact.params["sensor_cfg"].body_names = ".*HenkeAnkleLink"


          self.joint_deviation_hip.params["asset_cfg"].joint_names = [".*FrontalHipJoint", ".*TransverseHipJoint"]
          self.joint_deviation_arms = None
          self.joint_deviation_torso = None
          self.height_torso.params["target_height"] = 0.92
          self.feet_clearance.params["sensor_cfg"].body_names = ".*HenkeAnkleLink"
          self.feet_clearance.params["asset_cfg"].body_names = ".*HenkeAnkleLink"
          self.feet_clearance.params["target_height"] = 0.0
          self.contact_no_vel.params["sensor_cfg"].body_names = ".*HenkeAnkleLink"
          self.contact_no_vel.params["asset_cfg"].body_names = ".*HenkeAnkleLink"

          

                  

@configclass
class ExoObservationsCfg(G1RoughLipObservationsCfg):

     def __post_init__(self):
          super().__post_init__()
          self.policy.sin_phase.params["period"] = EXO_PERIOD
          self.policy.cos_phase.params["period"] = EXO_PERIOD

\
@configclass
class ExoFlatEnvCfg(HumanoidEnvCfg):
   """Config for the ExoRoughEnv."""
   commands: ExoHLIPCommandsCfg = ExoHLIPCommandsCfg()
   observations: ExoObservationsCfg = ExoObservationsCfg()
   rewards: ExoRewardsCfg = ExoRewardsCfg()


   def __post_init__(self):
     super().__post_init__()


     self.scene.robot = EXO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
     self.commands.base_velocity.ranges.lin_vel_x = (0.13,0.13)
     self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
     self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)


     self.scene.terrain.terrain_type = "plane"
     self.scene.terrain.terrain_generator = None
     # no height scan
     self.scene.height_scanner = None
     self.observations.policy.height_scan = None
     # no terrain curriculum
     self.curriculum.terrain_levels = None

     # Set base contact sensor to use PelvisLink
     self.terminations.base_contact.params["sensor_cfg"].body_names = "PelvisLink"   

     self.events.push_robot.params["velocity_range"] = {"x": (-1, 1), "y": (-1, 1), "roll": (-0.4, 0.4),
                                                       "pitch": (-0.4, 0.4), "yaw": (-0.4, 0.4)}
     self.events.add_base_mass.params["asset_cfg"].body_names = ["PelvisLink"]
     self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
     self.events.add_base_mass.params["operation"] = "scale"
     self.events.randomize_ground_contact_friction.params["asset_cfg"].body_names = ".*HenkeAnkleLink"
     # self.events.randomize_ground_contact_friction.params["static_friction_range"] = (0.1, 1.25)
     # self.events.randomize_ground_contact_friction.params["dynamic_friction_range"] = (0.1, 1.25)
     self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
     self.events.base_external_force_torque.params["asset_cfg"].body_names = ["PelvisLink"]
     self.events.reset_base.params = {
     "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
     "velocity_range": {
          "x": (0.0, 0.0),
          "y": (0.0, 0.0),
          "z": (0.0, 0.0),
          "roll": (0.0, 0.0),
          "pitch": (0.0, 0.0),
          "yaw": (0.0, 0.0),
     },
     }

     self.events.base_external_force_torque = None



@configclass
class ExoHZDEnvCfg(ExoFlatEnvCfg):
   """Config for the ExoRoughEnv."""
   commands: ExoHZDCommandsCfg = ExoHZDCommandsCfg()
   observations: G1_Observations = G1_Observations()
 


   def __post_init__(self):
     super().__post_init__()

     self.rewards.holonomic_constraint.params["command_name"] = "hzd_ref"
     self.rewards.holonomic_constraint_vel.params["command_name"] = "hzd_ref"
     self.rewards.reference_tracking.params["command_name"] = "hzd_ref"
     self.rewards.reference_vel_tracking.params["command_name"] = "hzd_ref"
     self.rewards.clf_reward.params["command_name"] = "hzd_ref"
     self.rewards.clf_decreasing_condition.params["command_name"] = "hzd_ref"
     



@configclass
class ExoPlayEnvCfg(ExoFlatEnvCfg):
    
    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.13,0.13)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5