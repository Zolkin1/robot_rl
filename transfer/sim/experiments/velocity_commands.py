# Copyright 2026 Zachary Olkin. All rights reserved.

import numpy as np

def step_to_max(sim_time):
    """Pass a given input as a function of time."""
    if sim_time > 1.:
        return np.array([0.75,0.0,0.0])
    else:
        return np.array([0.0,0.0,0.0])

def smooth_ramp_running(sim_time):
    """Compute a ramp input over a few seconds up to max speed."""
    RAMP_TIME = 3.0 #1.0

    MAX_SPEED = 1.4 #3.8 #3.2 #2.6 #2.0 #1.4

    slope = MAX_SPEED / RAMP_TIME

    y_vel = 0.0
    if sim_time > 6.0 and sim_time < 8.0:
        y_vel = -0.75
    elif sim_time > 8.0 and sim_time < 10.0:
        y_vel = 0.75

    yaw_vel = 0.0
    if sim_time > 10.0 and sim_time < 12.0:
        yaw_vel = -0.75
    elif sim_time > 12.0 and sim_time < 14.0:
        yaw_vel = 0.75

    return np.array([min(slope * sim_time, MAX_SPEED),y_vel,yaw_vel])

def smooth_ramp(sim_time):
    """Compute a ramp input over a few seconds up to max speed."""
    RAMP_TIME = 2.0

    MAX_SPEED = 0.75

    slope = MAX_SPEED / RAMP_TIME

    return np.array([min(slope * sim_time, MAX_SPEED),0.0,0.0])

def speed_steps_running(sim_time):
    # time_steps = np.array([3, 3, 3, 3, 6, 6])
    # speeds = np.array([0, 0.25, -0.25, 0.5, 0.75, -0.5])
    time_steps = np.array([3, 6, 6, 6])
    speeds = np.array([0, 2.0, 2.5, 1.6])

    # Compute start times of each interval
    start_times = np.cumsum(np.insert(time_steps[:-1], 0, 0))

    def _get_velocity_at_time(t):
        # Ensure t is within bounds
        if t < 0 or t >= np.sum(time_steps):
            raise ValueError("Time t is out of bounds.")

        # Find the index of the bin t belongs to
        idx = np.searchsorted(start_times, t, side='right') - 1
        return speeds[idx]

    return np.array([_get_velocity_at_time(sim_time), 0, 0])

def speed_steps(sim_time):
    # time_steps = np.array([3, 3, 3, 3, 6, 6])
    # speeds = np.array([0, 0.25, -0.25, 0.5, 0.75, -0.5])
    time_steps = np.array([3, 3, 6, 6, 6])
    speeds = np.array([0, 0.5, 0.75, -0.75, 0.75])

    # Compute start times of each interval
    start_times = np.cumsum(np.insert(time_steps[:-1], 0, 0))

    def _get_velocity_at_time(t):
        # Ensure t is within bounds
        if t < 0 or t >= np.sum(time_steps):
            raise ValueError("Time t is out of bounds.")

        # Find the index of the bin t belongs to
        idx = np.searchsorted(start_times, t, side='right') - 1
        return speeds[idx]

    return np.array([_get_velocity_at_time(sim_time), 0, 0])




def ramped_speed_steps(sim_time):
    time_steps = np.array([3, 3, 3, 3])
    speeds = np.array([0.0, 0.5, -0.5, 0.75])


    RAMP_TIME = 1.0  # Duration over which to ramp between speeds

    # Compute start times of each interval
    start_times = np.cumsum(np.insert(time_steps[:-1], 0, 0))
    end_times = np.cumsum(time_steps)

    total_time = end_times[-1]
    if sim_time < 0:
        return np.array([speeds[0], 0.0, 0.0])
    elif sim_time >= total_time:
        return np.array([speeds[-1], 0.0, 0.0])

    # Find current interval index
    idx = np.searchsorted(start_times, sim_time, side='right') - 1

    t0 = start_times[idx]
    t1 = end_times[idx]
    v0 = speeds[idx]
    v1 = speeds[idx + 1] if idx + 1 < len(speeds) else speeds[idx]

    # Ramp forward only within the last RAMP_TIME seconds of the interval
    ramp_start = t1 - RAMP_TIME
    if sim_time < ramp_start or RAMP_TIME <= 0:
        vx = v0
    else:
        alpha = (sim_time - ramp_start) / RAMP_TIME
        vx = (1 - alpha) * v0 + alpha * v1

    return np.array([vx, 0.0, 0.0])
