import os
import time
import math
import numpy as np
import mujoco
import mujoco.viewer

def get_model_data(robot: str):
    """Create the mj model and data from the given robot."""
    if robot != "g1_21j":
        raise ValueError("Invalid robot name! Only support g1_21j for now.")

    relative_path = "robots/g1/g1_21j_basic_scene.xml"
    path = os.path.join(os.getcwd(), relative_path)
    print(f"Trying to load the xml at {path}")
    mj_model = mujoco.MjModel.from_xml_path(path)
    mj_data = mujoco.MjData(mj_model)

    return mj_model, mj_data

def get_projected_gravity(quat):
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    pg = np.zeros(3)

    pg[0] = 2 * (-qz * qx + qw * qy)
    pg[1] = -2 * (qz * qy + qw * qx)
    pg[2] = 1 - 2 * (qw * qw + qz * qz)

    return pg

def run_simulation(policy, robot: str):
    """Run the simulation."""

    # Setup mj model and data
    mj_model, mj_data = get_model_data(robot)

    keyframe_index = mj_model.keyframe("standing").id
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, keyframe_index)

    # Compute sim steps per policy update
    sim_steps_per_policy_update = int(policy.dt / mj_model.opt.timestep)
    sim_loop_rate = sim_steps_per_policy_update * mj_model.opt.timestep
    viewer_rate = math.ceil((1/100) / mj_model.opt.timestep)
    print(viewer_rate)

    print(f"Starting mujoco simulation with robot {robot}.\n"
          f"Policy dt set to {policy.dt} s ({sim_steps_per_policy_update} steps per policy update.)\n"
          f"Simulation dt set to {mj_model.opt.timestep} s. Sim loop rate set to {sim_loop_rate} s.\n")

    nu = policy.get_num_actions()

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

        des_vel = np.zeros(3)
        des_vel[0] = 0.2

        while viewer.is_running():
            start_time = time.time()

            # Extract relevant info
            sim_time = mj_data.time
            qpos = mj_data.qpos
            qvel = mj_data.qvel

            # Compute the control action
            pg = get_projected_gravity(qpos[3:7])
            obs = policy.create_obs(qpos, qvel, sim_time, pg, des_vel)
            u = policy.get_action(obs)
            mj_data.ctrl[:nu] = u

            # Step the simulator
            for i in range(sim_steps_per_policy_update):
                mujoco.mj_step(mj_model, mj_data)
                if i % viewer_rate == 0:
                    viewer.sync()

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < 10*sim_loop_rate:
                time.sleep(10*sim_loop_rate - elapsed)

