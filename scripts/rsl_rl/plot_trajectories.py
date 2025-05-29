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

# Convert lists to numpy arrays
y_out_array = np.array([y.cpu().numpy() for y in y_out_list])  # Shape: (time_steps, num_envs, 12)
dy_out_array = np.array([dy.cpu().numpy() for dy in dy_out_list])  # Shape: (time_steps, num_envs, 12)
base_velocity_array = np.array([v.cpu().numpy() for v in base_velocity_list])  # Shape: (time_steps, num_envs, 3)

# Print shapes for debugging
print("y_out_array shape:", y_out_array.shape)
print("dy_out_array shape:", dy_out_array.shape)
print("base_velocity_array shape:", base_velocity_array.shape)

# Create time array
time_steps = np.arange(len(y_out_list))

# Plot for the first environment (index 0)
env_idx = 0

# Create subplots - now 7 rows to include velocity commands
fig, axs = plt.subplots(7, 4, figsize=(20, 17))
fig.suptitle('Reference Trajectories - Position and Velocity', fontsize=16)

# State names and their indices
states = [
    ('COM Position', 0, 'm'),
    ('COM Position', 1, 'm'),
    ('COM Position', 2, 'm'),
    ('Pelvis Orientation', 3, 'rad'),
    ('Pelvis Orientation', 4, 'rad'),
    ('Pelvis Orientation', 5, 'rad'),
    ('Swing Foot Position', 6, 'm'),
    ('Swing Foot Position', 7, 'm'),
    ('Swing Foot Position', 8, 'm'),
    ('Swing Foot Orientation', 9, 'rad'),
    ('Swing Foot Orientation', 10, 'rad'),
    ('Swing Foot Orientation', 11, 'rad')
]

# Plot each state variable
for i, (state_name, idx, unit) in enumerate(states):
    row = i // 2
    col = (i % 2) * 2
    
    # Position plot
    axs[row, col].plot(time_steps, y_out_array[:, env_idx, idx], 'b-', label='Position')
    axs[row, col].set_title(f'{state_name} {["x", "y", "z"][idx % 3]}')
    axs[row, col].set_xlabel('Time Steps')
    axs[row, col].set_ylabel(f'Position ({unit})')
    axs[row, col].grid(True)
    
    # Velocity plot - only plot if the index exists in dy_out_array
    if idx < dy_out_array.shape[2]:
        axs[row, col + 1].plot(time_steps, dy_out_array[:, env_idx, idx], 'r-', label='Velocity')
        axs[row, col + 1].set_title(f'{state_name} {["x", "y", "z"][idx % 3]} Velocity')
        axs[row, col + 1].set_xlabel('Time Steps')
        axs[row, col + 1].set_ylabel(f'Velocity ({unit}/s)')
        axs[row, col + 1].grid(True)
    else:
        axs[row, col + 1].text(0.5, 0.5, 'No velocity data', 
                              horizontalalignment='center',
                              verticalalignment='center',
                              transform=axs[row, col + 1].transAxes)
        axs[row, col + 1].set_title(f'{state_name} {["x", "y", "z"][idx % 3]} Velocity')
        axs[row, col + 1].grid(True)

# Plot base velocity commands in the last row
row = 6
vel_names = ['Linear X', 'Linear Y', 'Angular Z']
for i in range(3):
    axs[row, i].plot(time_steps, base_velocity_array[:, env_idx, i], 'g-', label='Command')
    axs[row, i].set_title(f'Base Velocity {vel_names[i]}')
    axs[row, i].set_xlabel('Time Steps')
    axs[row, i].set_ylabel('Velocity (m/s or rad/s)')
    axs[row, i].grid(True)

plt.tight_layout()
plt.savefig('reference_trajectories_detailed.png', dpi=300, bbox_inches='tight')
plt.show() 