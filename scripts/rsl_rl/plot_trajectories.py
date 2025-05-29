import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load the saved data
with open("y_out_list.pkl", "rb") as f:
    y_out_list = pickle.load(f)
with open("dy_out_list.pkl", "rb") as f:
    dy_out_list = pickle.load(f)
with open("base_velocity_list.pkl", "rb") as f:
    base_velocity_list = pickle.load(f)
with open("cur_swing_time_list.pkl", "rb") as f:
    cur_swing_time_list = pickle.load(f)
with open("y_act_list.pkl", "rb") as f:
    y_act_list = pickle.load(f)
with open("dy_act_list.pkl", "rb") as f:
    dy_act_list = pickle.load(f)

# Convert lists to numpy arrays
# Shape: (time_steps, num_envs, dim)
y_out_array = np.array([y.cpu().numpy() for y in y_out_list])
dy_out_array = np.array([dy.cpu().numpy() for dy in dy_out_list])
y_act_array = np.array([y.cpu().numpy() for y in y_act_list])
dy_act_array = np.array([dy.cpu().numpy() for dy in dy_act_list])
base_velocity_array = np.array([v.cpu().numpy() for v in base_velocity_list])
cur_swing_time_array = np.array([t for t in cur_swing_time_list])

time_steps = np.arange(len(y_out_list))
env_idx = 0

# Only plot the first 6 dimensions (COM position and pelvis orientation)
state_labels = [
    'COM x', 'COM y', 'COM z',
    'Pelvis Roll', 'Pelvis Pitch', 'Pelvis Yaw'
]
units = ['m', 'm', 'm', 'rad', 'rad', 'rad']

fig, axs = plt.subplots(2, 3, figsize=(18, 8))
fig.suptitle('Reference vs Actual Trajectories (First Environment)', fontsize=16)

for i in range(6):
    ax = axs[i // 3, i % 3]
    ax.plot(time_steps, y_out_array[:, env_idx, i], label='y_des', color='b')
    ax.plot(time_steps, y_act_array[:, env_idx, i], label='y_act', color='r', linestyle='--')
    ax.set_title(state_labels[i])
    ax.set_xlabel('Time Steps')
    ax.set_ylabel(units[i])
    ax.legend()
    ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('reference_vs_actual_y.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot velocities (dy_out vs dy_act)
fig, axs = plt.subplots(2, 3, figsize=(18, 8))
fig.suptitle('Reference vs Actual Velocities (First Environment)', fontsize=16)

for i in range(6):
    ax = axs[i // 3, i % 3]
    ax.plot(time_steps, dy_out_array[:, env_idx, i], label='dy_des', color='b')
    ax.plot(time_steps, dy_act_array[:, env_idx, i], label='dy_act', color='r', linestyle='--')
    ax.set_title(state_labels[i] + ' Velocity')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel(units[i] + '/s')
    ax.legend()
    ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('reference_vs_actual_dy.png', dpi=300, bbox_inches='tight')
plt.show() 