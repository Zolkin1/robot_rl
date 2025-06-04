
import torch, math
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers.action_manager import ActionTerm
import isaaclab.utils.math as math_utils
from robot_rl.assets.robots.exo_cfg import JointTrajectoryConfig
from robot_rl.tasks.manager_based.robot_rl.mdp import ResidualActionCfg
class ResidualAction(ActionTerm):
     def __init__(self, cfg: ResidualActionCfg,env: ManagerBasedRLEnv):
          super().__init__(cfg, env)

          self._num_joints = 12
          
          self.jt_config = JointTrajectoryConfig()
          right_jt_coeffs = self.jt_config.joint_trajectories
          right_base_coeffs = self.jt_config.base_trajectories
          left_jt_coeffs = self.jt_config.remap_jt_symmetric()
          left_base_coeffs = self.jt_config.remap_base_symmetric()

          
          left_coeffs = []
          right_coeffs = []
          for key in self.jt_config.base_trajectories.keys():
               right_coeffs.append(right_base_coeffs[key])
               left_coeffs.append(left_base_coeffs[key])

          for key in self.jt_config.joint_trajectories.keys():
               
               right_coeffs.append(right_jt_coeffs[key])
               left_coeffs.append(left_jt_coeffs[key])

          self.right_coeffs = torch.tensor(right_coeffs, device=self.device)
          self.left_coeffs = torch.tensor(left_coeffs, device=self.device)
          self.env = env

          self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=self.cfg.preserve_order
          )


     @property
     def action_dim(self):
          return self._num_joints

     @property
     def raw_actions(self):
          return self._raw_actions
     
     @property
     def processed_actions(self):
          return self._jt_actions

     def process_actions(self,actions: torch.Tensor):
          
          #get nominal joint trajectory from hzd_cmd
          ref_cmf = self.env.command_manager.get_term("hzd_ref")
          des_jt_pos = ref_cmf.y_out[:,6:]
          des_jt_vel = ref_cmf.dy_out[:,6:]

          self._raw_actions = actions
          self._jt_actions = des_jt_pos + self.cfg.scale*actions
          self._jt_vel_actions = des_jt_vel 


     def apply_actions(self):
          processed_jt_actions = math_utils.saturate(
               self._jt_actions,
               self._asset.data.soft_joint_pos_limits[:, :, 0],
               self._asset.data.soft_joint_pos_limits[:, :, 1],
          )

          self._asset.set_joint_position_target(processed_jt_actions)
          self._asset.set_joint_velocity_target(self._jt_vel_actions)
     #    self._asset.set_joint_position_target(processed_jt_actions,
     #                                         joint_ids=self._joint_ids)
       

    

        
     


     
        
        