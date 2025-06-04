from robot_rl.tasks.manager_based.robot_rl.humanoid_env_cfg import HumanoidEnvCfg


from robot_rl.tasks.manager_based.robot_rl.g1.g1_rough_env_lip_cfg import G1RoughLipCommandsCfg, G1RoughLipRewards
from robot_rl.tasks.manager_based.robot_rl.g1_isaac.rough_env_cfg import G1RoughLipObservationsCfg, G1_Observations
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg  #Inherit from the base envs
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from robot_rl.assets.robots.exo_cfg import EXO_CFG  # isort: skip
from robot_rl.tasks.manager_based.robot_rl import mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from robot_rl.tasks.manager_based.robot_rl.mdp.ResidualActionCfg import ResidualActionCfg
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

# class ExoAction
        
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
          self.feet_clearance.params["target_height"] = 0.05
          self.contact_no_vel.params["sensor_cfg"].body_names = ".*HenkeAnkleLink"
          self.contact_no_vel.params["asset_cfg"].body_names = ".*HenkeAnkleLink"
          self.holonomic_constraint.params["z_offset"] = 0.163

          
@configclass
class ExoResidualActionCfg():
     joint_pos = ResidualActionCfg(scale=0.1, joint_names=[".*"], asset_name="robot")

@configclass
class ExoObservationsCfg(G1RoughLipObservationsCfg):


     def __post_init__(self):
          super().__post_init__()
          self.policy.sin_phase.params["period"] = EXO_PERIOD
          self.policy.cos_phase.params["period"] = EXO_PERIOD
          self.policy.joint_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
          # self.policy.des_jt_pos = ObsTerm(func=mdp.joint_pos_des,params={"command_name":"hzd_ref"})


@configclass
class ExoFlatEnvCfg(HumanoidEnvCfg):
   """Config for the ExoRoughEnv."""
   commands: ExoHLIPCommandsCfg = ExoHLIPCommandsCfg()
   observations: ExoObservationsCfg = ExoObservationsCfg()
   rewards: ExoRewardsCfg = ExoRewardsCfg()

   
   def __post_init__(self):
     super().__post_init__()

     # self.observations.policy.des_jt_pos = None
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
          "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0.0, 0.0)},
          "velocity_range": {
               "x": (0.0, 0.4),
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
   actions: ExoResidualActionCfg = ExoResidualActionCfg()


   def __post_init__(self):
     super().__post_init__()

     self.rewards.holonomic_constraint.params["command_name"] = "hzd_ref"
     self.rewards.holonomic_constraint_vel.params["command_name"] = "hzd_ref"
     self.rewards.reference_tracking.params["command_name"] = "hzd_ref"
     self.rewards.reference_vel_tracking.params["command_name"] = "hzd_ref"
     self.rewards.clf_reward.params["command_name"] = "hzd_ref"
     self.rewards.clf_decreasing_condition.params["command_name"] = "hzd_ref"


     self.rewards.reference_tracking.params["term_std"] = [0.1,0.1,0.1,
                                                           0.1,0.1,0.1,
                                                           0.1,0.1,0.1,
                                                           0.1,0.1,0.1,
                                                           0.1,0.1,0.1,
                                                           0.1,0.1,0.1]
     self.rewards.reference_tracking.params["term_weight"] = [2.0,2.0,2.0,
                                                              2.0,5.0,1.0,
                                                               1.0,1.0,1.0,
                                                               1.0,1.0,1.0,
                                                               1.0,1.0,1.0,
                                                               1.0,1.0,1.0]

     self.rewards.reference_vel_tracking.params["term_std"] = [0.1,0.1,0.1,
                                                               0.1,0.1,0.1,
                                                               0.1,0.1,0.1,
                                                               0.1,0.1,0.1,
                                                               0.1,0.1,0.1,
                                                               0.1,0.1,0.1]
     self.rewards.reference_vel_tracking.params["term_weight"] = [1.0,1.0,1.0,
                                                                  1.0,1.0,1.0,
                                                                  1.0,1.0,1.0,
                                                                  1.0,1.0,1.0,
                                                                  1.0,1.0,1.0,
                                                                  1.0,1.0,1.0]

     self.rewards.feet_air_time = None
     self.rewards.phase_contact = None
     self.rewards.lin_vel_z_l2 = None
     # self.rewards.height_torso = None
     self.rewards.feet_clearance = None
     self.rewards.ang_vel_xy_l2 = None
     self.rewards.termination_penalty = None
     self.rewards.flat_orientation_l2 = None
     self.rewards.joint_deviation_hip = None
     self.rewards.contact_no_vel = None
     self.rewards.alive = None
     self.rewards.track_lin_vel_xy_exp = None
     self.rewards.track_ang_vel_z_exp = None
     self.rewards.action_rate_l2.weight = -1e-5
     self.rewards.dof_torque_l2.weight = -1e-7
 



@configclass
class ExoPlayEnvCfg(ExoFlatEnvCfg):
    
    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.13,0.13)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5

@configclass
class ExoHZDPlayEnvCfg(ExoHZDEnvCfg):
    
    def __post_init__(self):
        super().__post_init__()

        self.commands.base_velocity.ranges.lin_vel_x = (0.13,0.13)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5