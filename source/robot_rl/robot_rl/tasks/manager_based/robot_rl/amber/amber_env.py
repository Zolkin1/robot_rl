# amber_env.py
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils

# from isaaclab.managers import ResetCallback
# from .amber_flat_env_cfg import AmberFlatEnvCfg

def settle_physics_after_reset(env, env_ids):
    # runs 5 substeps on the sim stage
    print("_______________--reset________________________")
    for _ in range(5):
        sim_utils.gym.simulate(env.sim)
        sim_utils.gym.fetch_results(env.sim, True)


class AmberFlatEnv(ManagerBasedRLEnv):
    """Override the default reset so we zero velocities and settle."""
    def __init__(self, cfg ,**kwargs):
        super().__init__(cfg,**kwargs)

        # # register our post‐reset hook
        # self.reset_manager.register_callback(
            
        #         name="amber_post_reset",
        #         func=self._immediate_reset,
        #         order=100,   # run after all built‐ins
            
        # )


    def reset(self, seed=None, options=None):
        # 1) call the base reset
        obs = super().reset(seed=seed, options=options)

        # 2) now perform your "immediate reset" logic
        # ids = env_ids if env_ids is not None else slice(None)
        ids = slice(None)
        print("----- post-reset hook! env_ids:", ids)

        # zero any leftover base & joint velocities
        self.tensors.root_vel[ids, :] = 0.0
        self.tensors.dof_vel[ids, :]  = 0.0

        # bump the robot up by 1 cm to avoid penetration
        self.tensors.root_pos[ids, 2] += 0.01

        # run a few substeps so PhysX settles
        for _ in range(5):
            self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # 3) return the (possibly updated) initial observations
        return obs


    def _immediate_reset(self, env_ids):
        print("____________________post-reset hook!____________________________________", env_ids)

        # 1) zero any leftover base & joint velocities
        self.tensors.root_vel[env_ids, :] = 0.0
        self.tensors.dof_vel[env_ids, :]  = 0.0

        # 2) bump the robot up by 1 cm (avoid any penetration)
        self.tensors.root_pos[env_ids, 2] += 0.01

        # 3) run 5 small sub‐steps to let PhysX settle contacts
        for _ in range(5):
            self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
