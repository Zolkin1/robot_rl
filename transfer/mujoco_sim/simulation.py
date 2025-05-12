import os
import mujoco
import mujoco.viewer

def get_model_data(robot: str):
    """Create the mj model and data from the given robot."""
    if robot != "g1_21j":
        raise ValueError("Invalid robot name! Only support g1_21j for now.")

    relative_path = "robots/g1/g1_21j.xml"
    path = os.path.join(relative_path)
    print(f"Trying to load the xml at {path}")
    mj_model = mujoco.MjModel.from_xml_path(path)
    mj_data = mujoco.MjData(mj_model)

    return mj_model, mj_data

def run_simulation(policy, robot: str):
    """Run the simulation."""

    # Setup mj model and data
    mj_model, mj_data = get_model_data(robot)

    # Compute sim steps per policy update
    sim_steps_per_policy_update = (1./policy.freq) / mj_model.opt.timestep

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:

        while viewer.is_running():
            # Extract relevant info
            sim_time = mj_data.time
            qpos = mj_data.qpos
            qvel = mj_data.qvel

            # Compute the control action
            # u = policy.get_action(qpos, qvel, sim_time, imu)
            # mj_data.ctrl[:] = u

            # Step the simulator
            for i in range(sim_steps_per_policy_update):
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

