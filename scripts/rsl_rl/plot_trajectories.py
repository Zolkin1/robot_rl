import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch

def find_most_recent_log_dir(base_path="logs/play"):
    """Find the most recent log directory"""
    if not os.path.exists(base_path):
        print(f"Error: Log directory {base_path} does not exist")
        return None
    # Get all timestamped directories
    dirs = glob.glob(os.path.join(base_path, "*"))
    if not dirs:
        print(f"No log directories found in {base_path}")
        return None
    # Sort by modification time (newest first)
    latest_dir = max(dirs, key=os.path.getmtime)
    return latest_dir

def load_data(log_dir):
    """Load all pickle files from the log directory"""
    data = {}
    for pkl_file in glob.glob(os.path.join(log_dir, "*.pkl")):
        var_name = os.path.basename(pkl_file).replace(".pkl", "")
        with open(pkl_file, "rb") as f:
            data[var_name] = pickle.load(f)
    return data

def plot_trajectories(data, save_dir=None):
    """Plot all trajectories with proper labels and units"""
    # Convert lists to numpy arrays and handle torch tensors
    processed_data = {}
    for key, values in data.items():
        if isinstance(values[0], torch.Tensor):
            processed_data[key] = np.array([v.cpu().numpy() for v in values])
        else:
            processed_data[key] = np.array(values)
    
    # Create time array
    time_steps = np.arange(len(processed_data[list(processed_data.keys())[0]]))
    
    # Define state labels and units
    state_labels = {
        'y_out': [
            'COM x', 'COM y', 'COM z', 'Pelvis Roll', 'Pelvis Pitch', 'Pelvis Yaw',
            'swing foot x', 'swing foot y', 'swing foot z',
            'swing foot roll', 'swing foot pitch', 'swing foot yaw',
            "waist_yaw",
            'left sholder pitch', 'right sholder pitch',
            'left sholder roll', 'right sholder roll',
            'left sholder yaw', 'right sholder yaw',
            'left elbow ', 'right elbow '
        ],
        'dy_out': [
            'COM x', 'COM y', 'COM z', 'Pelvis Roll', 'Pelvis Pitch', 'Pelvis Yaw',
            'swing foot x', 'swing foot y', 'swing foot z',
            'swing foot roll', 'swing foot pitch', 'swing foot yaw',
            "waist_yaw",
            'left sholder pitch', 'right sholder pitch',
            'left sholder roll', 'right sholder roll',
            'left sholder yaw', 'right sholder yaw',
            'left elbow ', 'right elbow '
        ],
        'base_velocity': ['Linear x', 'Linear y', 'Angular z'],
        "stance_foot_pos": ['x', 'y', 'z'],
        "stance_foot_ori": ['roll', 'pitch', 'yaw'],
        'cur_swing_time': ['Time'],
        'y_act': [
            'COM x', 'COM y', 'COM z', 'Pelvis Roll', 'Pelvis Pitch', 'Pelvis Yaw',
            'swing foot x', 'swing foot y', 'swing foot z',
            'swing foot roll', 'swing foot pitch', 'swing foot yaw',
            "waist_yaw",
            'left sholder pitch', 'right sholder pitch',
            'left sholder roll', 'right sholder roll',
            'left sholder yaw', 'right sholder yaw',
            'left elbow ', 'right elbow '
        ],
        'dy_act': [
            'COM x', 'COM y', 'COM z', 'Pelvis Roll', 'Pelvis Pitch', 'Pelvis Yaw',
            'swing foot x', 'swing foot y', 'swing foot z',
            'swing foot roll', 'swing foot pitch', 'swing foot yaw',
            "waist_yaw",
            'left sholder pitch', 'right sholder pitch',
            'left sholder roll', 'right sholder roll',
            'left sholder yaw', 'right sholder yaw',
            'left elbow ', 'right elbow '
        ],
        'v': ['Velocity'],
        'vdot': ['Acceleration'],
        'reward': ['Reward']
    }
    
    units = {
        'y_out': ['m', 'm', 'm', 'rad', 'rad', 'rad', 'm', 'm', 'm', 'rad', 'rad', 'rad'],
        'dy_out': ['m/s', 'm/s', 'm/s', 'rad/s', 'rad/s', 'rad/s', 'm/s', 'm/s', 'm/s', 'rad/s', 'rad/s', 'rad/s'],
        'base_velocity': ['m/s', 'm/s', 'rad/s'],
        'stance_foot_pos': ['m', 'm', 'm'],
        'stance_foot_ori': ['rad', 'rad', 'rad'],
        'cur_swing_time': ['s'],
        'y_act': ['m', 'm', 'm', 'rad', 'rad', 'rad', 'm', 'm', 'm', 'rad', 'rad', 'rad'],
        'dy_act': ['m/s', 'm/s', 'm/s', 'rad/s', 'rad/s', 'rad/s', 'm/s', 'm/s', 'm/s', 'rad/s', 'rad/s', 'rad/s'],
        'v': ['m/s'],
        'vdot': ['m/s²'],
        'reward': ['']
    }

    # Add error metrics labels and units
    error_labels = {
        'error_sw_x': 'Swing Foot X Error',
        'error_sw_y': 'Swing Foot Y Error',
        'error_sw_z': 'Swing Foot Z Error',
        'error_sw_roll': 'Swing Foot Roll Error',
        'error_sw_pitch': 'Swing Foot Pitch Error',
        'error_sw_yaw': 'Swing Foot Yaw Error',
        'error_com_x': 'COM X Error',
        'error_com_y': 'COM Y Error',
        'error_com_z': 'COM Z Error',
        'error_pelvis_roll': 'Pelvis Roll Error',
        'error_pelvis_pitch': 'Pelvis Pitch Error',
        'error_pelvis_yaw': 'Pelvis Yaw Error'
    }

    error_units = {
        'error_sw_x': 'm',
        'error_sw_y': 'm',
        'error_sw_z': 'm',
        'error_sw_roll': 'rad',
        'error_sw_pitch': 'rad',
        'error_sw_yaw': 'rad',
        'error_com_x': 'm',
        'error_com_y': 'm',
        'error_com_z': 'm',
        'error_pelvis_roll': 'rad',
        'error_pelvis_pitch': 'rad',
        'error_pelvis_yaw': 'rad'
    }
    
    # Helper for subplot indexing
    def get_ax(axs, idx, n_cols):
        if axs.ndim == 1:
            return axs[idx]
        return axs[idx // n_cols, idx % n_cols]

    env_ids = 0

    if "stance_foot_pos" and "stance_foot_ori" in processed_data:
        pos_data = processed_data["stance_foot_pos"]
        ori_data = processed_data["stance_foot_ori"]
        pos_data_0 = processed_data["stance_foot_pos_0"]
        ori_data_0 = processed_data["stance_foot_ori_0"]
        # Assume shape: (timesteps, envs, 3) for both
        env_ids = 0  # or loop over envs if you want

        fig, axs = plt.subplots(2, 3, figsize=(15, 6))
        fig.suptitle("Stance Foot Position and Orientation", fontsize=16)

        # Position
        pos_labels = ["x", "y", "z"]
        for i in range(3):
            ax = axs[0, i]
            ax.plot(time_steps, pos_data[:, env_ids, i], label=f"pos {pos_labels[i]}")
            ax.plot(time_steps, pos_data_0[:, env_ids, i], label=f"pos_0 {pos_labels[i]}", linestyle='--')
            ax.set_title(f"Position {pos_labels[i]}")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("m")
            ax.grid(True)
            ax.legend()

        # Orientation
        ori_labels = ["roll", "pitch", "yaw"]
        for i in range(3):
            ax = axs[1, i]
            ax.plot(time_steps, ori_data[:, env_ids, i], label=f"ori {ori_labels[i]}")
            ax.plot(time_steps, ori_data_0[:, env_ids, i], label=f"ori_0 {ori_labels[i]}", linestyle='--')
            ax.set_title(f"Orientation {ori_labels[i]}")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("rad")
            ax.grid(True)
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(os.path.join(save_dir, "stance_foot_pos_ori.png"), dpi=300, bbox_inches="tight")


    # Plot positions (y_out vs y_act)
    if 'y_out' in processed_data and 'y_act' in processed_data:
        n_dims = processed_data['y_out'].shape[2]
        n_cols = 4
        n_rows = (n_dims + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
        fig.suptitle('Reference vs Actual Positions', fontsize=16)
        axs = np.array(axs)
        for i in range(n_dims):
            ax = get_ax(axs, i, n_cols)
            ax.plot(time_steps, processed_data['y_out'][:, env_ids, i], label='Reference', color='b')
            ax.plot(time_steps, processed_data['y_act'][:, env_ids, i], label='Actual', color='r', linestyle='--')
            # Use label if available, else fallback
            label = state_labels['y_out'][i] if i < len(state_labels['y_out']) else f'Var {i}'
            unit = units['y_out'][i] if i < len(units['y_out']) else ''
            ax.set_title(label)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(unit)
            ax.grid(True)
            if i == 0:
                ax.legend()
        # Hide unused subplots
        for i in range(n_dims, n_rows * n_cols):
            ax = get_ax(axs, i, n_cols)
            ax.set_visible(False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'positions.png'), dpi=300, bbox_inches='tight')

    
    # Plot velocities (dy_out vs dy_act)
    if 'dy_out' in processed_data and 'dy_act' in processed_data:
        n_dims = processed_data['dy_out'].shape[2]
        n_cols = 4
        n_rows = (n_dims + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
        fig.suptitle('Reference vs Actual Velocities', fontsize=16)
        axs = np.array(axs)
        for i in range(n_dims):
            ax = get_ax(axs, i, n_cols)
            ax.plot(time_steps, processed_data['dy_out'][:, env_ids, i], label='Reference', color='b')
            ax.plot(time_steps, processed_data['dy_act'][:, env_ids, i], label='Actual', color='r', linestyle='--')
            label = state_labels['dy_out'][i] if i < len(state_labels['dy_out']) else f'Var {i}'
            unit = units['dy_out'][i] if i < len(units['dy_out']) else ''
            ax.set_title(label)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(unit)
            ax.grid(True)
            if i == 0:
                ax.legend()
        for i in range(n_dims, n_rows * n_cols):
            ax = get_ax(axs, i, n_cols)
            ax.set_visible(False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'velocities.png'), dpi=300, bbox_inches='tight')

    
    # Plot base velocity
    if 'base_velocity' in processed_data:
        n_dims = processed_data['base_velocity'].shape[2]
        fig, axs = plt.subplots(1, n_dims, figsize=(5 * n_dims, 3))
        fig.suptitle('Base Velocity', fontsize=16)
        for i in range(n_dims):
            ax = axs[i] if n_dims > 1 else axs
            ax.plot(time_steps, processed_data['base_velocity'][:, env_ids, i])
            label = state_labels['base_velocity'][i] if i < len(state_labels['base_velocity']) else f'Var {i}'
            unit = units['base_velocity'][i] if i < len(units['base_velocity']) else ''
            ax.set_title(label)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(unit)
            ax.grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'base_velocity.png'), dpi=300, bbox_inches='tight')

    
    # Plot swing time
    # if 'cur_swing_time' in processed_data:
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(time_steps, processed_data['cur_swing_time'])
    #     plt.title('Swing Time')
    #     plt.xlabel('Time Steps')
    #     plt.ylabel('Time (s)')
    #     plt.grid(True)
    #     if save_dir:
    #         plt.savefig(os.path.join(save_dir, 'swing_time.png'), dpi=300, bbox_inches='tight')
    #     plt.show()

    # Plot v and vdot as two subplots in one figure
    if 'v' in processed_data and 'vdot' in processed_data:
        v_data = processed_data['v']
        vdot_data = processed_data['vdot']
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        
        axs[0].plot(time_steps, v_data[:, env_ids], label='v', color='g')
        axs[0].set_title('clf v')
        axs[0].set_ylabel(units['v'][0] if 'v' in units else '')
        axs[0].grid(True)
        axs[0].legend()
        
        axs[1].plot(time_steps, vdot_data[:, env_ids], label='vdot', color='m')
        axs[1].set_title('clf vdot')
        axs[1].set_xlabel('Time Steps')
        axs[1].set_ylabel(units['vdot'][0] if 'vdot' in units else '')
        axs[1].grid(True)
        axs[1].legend()

        alpha=1.0
        decay = alpha*v_data[:, env_ids] + vdot_data[:, env_ids]
        axs[2].plot(time_steps, decay, label='decay', color='y')
        axs[2].set_title('clf decay')
        axs[2].set_xlabel('Time Steps')
        axs[2].set_ylabel(units['decay'][0] if 'decay' in units else '')
        axs[2].grid(True)
        axs[2].legend()
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'v_and_vdot.png'), dpi=300, bbox_inches='tight')


    # Plot error metrics
    error_metrics = [key for key in processed_data.keys() if key.startswith('error_')]
    if error_metrics:
        n_metrics = len(error_metrics)
        n_cols = 4
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
        fig.suptitle('Error Metrics', fontsize=16)
        axs = np.array(axs)
        
        for i, metric in enumerate(error_metrics):
            ax = get_ax(axs, i, n_cols)
            data = processed_data[metric]
            # Handle both 1D and 2D arrays
            if data.ndim > 1:
                plot_data = data[:, env_ids]
            else:
                plot_data = data
            ax.plot(time_steps, plot_data, label=error_labels.get(metric, metric))
            ax.set_title(error_labels.get(metric, metric))
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(error_units.get(metric, ''))
            ax.grid(True)
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_metrics, n_rows * n_cols):
            ax = get_ax(axs, i, n_cols)
            ax.set_visible(False)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'error_metrics.png'), dpi=300, bbox_inches='tight')
        # plt.show()

    # Plot log_terms.pkl if it exists
    # if save_dir:
    #     log_terms_path = os.path.join(save_dir, '..', 'log_terms.pkl')
    #     log_terms_path = os.path.abspath(log_terms_path)
    #     if os.path.exists(log_terms_path):
    #         with open(log_terms_path, 'rb') as f:
    #             log_terms_list = pickle.load(f)
    #         if len(log_terms_list) > 0:
    #             # Get all keys
    #             keys = list(log_terms_list[0].keys())
    #             n_keys = len(keys)
    #             n_cols = 3
    #             n_rows = (n_keys + n_cols - 1) // n_cols
    #             fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
    #             fig.suptitle('Log Terms', fontsize=16)
    #             axs = np.array(axs)
    #             for i, key in enumerate(keys):
    #                 ax = axs[i // n_cols, i % n_cols] if n_rows > 1 or n_cols > 1 else axs
    #                 # Gather the series for this key
    #                 series = [entry.get(key, None) for entry in log_terms_list]
    #                 # Convert to numpy array, handle tensors
    #                 series = np.array([v.item() if hasattr(v, 'item') else v for v in series])
    #                 ax.plot(np.arange(len(series)), series, label=key)
    #                 ax.set_title(key)
    #                 ax.set_xlabel('Time Steps')
    #                 ax.grid(True)
    #                 ax.legend()
    #             # Hide unused subplots
    #             for i in range(n_keys, n_rows * n_cols):
    #                 ax = axs[i // n_cols, i % n_cols]
    #                 ax.set_visible(False)
    #             plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    #             plt.savefig(os.path.join(save_dir, 'log_terms.png'), dpi=300, bbox_inches='tight')
    #             plt.show()

def plot_hzd_trajectories(data, save_dir=None):
    """Plot HZD trajectories with proper labels and units"""
    # Convert lists to numpy arrays and handle torch tensors
    processed_data = {}
    for key, values in data.items():
        if isinstance(values[0], torch.Tensor):
            processed_data[key] = np.array([v.cpu().numpy() for v in values])
        else:
            processed_data[key] = np.array(values)
    
    # Create time array
    time_steps = np.arange(len(processed_data[list(processed_data.keys())[0]]))
    
    # Define state labels and units for HZD
    state_labels = {
        'y_out': [
            'Pelvis x', 'Pelvis y', 'Pelvis z',
            'Pelvis roll', 'Pelvis pitch', 'Pelvis yaw',
            'LeftFrontalHipJoint', 'RightFrontalHipJoint',
            'LeftTransverseHipJoint', 'RightTransverseHipJoint',
            'LeftSagittalHipJoint', 'RightSagittalHipJoint',
            'LeftSagittalKneeJoint', 'RightSagittalKneeJoint',
            'LeftSagittalAnkleJoint', 'RightSagittalAnkleJoint',
            'LeftHenkeAnkleJoint', 'RightHenkeAnkleJoint'
        ],
        'dy_out': [
            'Pelvis x', 'Pelvis y', 'Pelvis z',
            'Pelvis roll', 'Pelvis pitch', 'Pelvis yaw',
            'LeftFrontalHipJoint', 'RightFrontalHipJoint',
            'LeftTransverseHipJoint', 'RightTransverseHipJoint',
            'LeftSagittalHipJoint', 'RightSagittalHipJoint',
            'LeftSagittalKneeJoint', 'RightSagittalKneeJoint',
            'LeftSagittalAnkleJoint', 'RightSagittalAnkleJoint',
            'LeftHenkeAnkleJoint', 'RightHenkeAnkleJoint'
        ],
        'base_velocity': ['Linear x', 'Linear y', 'Angular z'],
        "stance_foot_pos": ['x', 'y', 'z'],
        "stance_foot_ori": ['roll', 'pitch', 'yaw'],
        'cur_swing_time': ['Time'],
        'y_act': [
            'Pelvis x', 'Pelvis y', 'Pelvis z',
            'Pelvis roll', 'Pelvis pitch', 'Pelvis yaw',
            'LeftFrontalHipJoint', 'RightFrontalHipJoint',
            'LeftTransverseHipJoint', 'RightTransverseHipJoint',
            'LeftSagittalHipJoint', 'RightSagittalHipJoint',
            'LeftSagittalKneeJoint', 'RightSagittalKneeJoint',
            'LeftSagittalAnkleJoint', 'RightSagittalAnkleJoint',
            'LeftHenkeAnkleJoint', 'RightHenkeAnkleJoint'
        ],
        'dy_act': [
            'Pelvis x', 'Pelvis y', 'Pelvis z',
            'Pelvis roll', 'Pelvis pitch', 'Pelvis yaw',
            'LeftFrontalHipJoint', 'RightFrontalHipJoint',
            'LeftTransverseHipJoint', 'RightTransverseHipJoint',
            'LeftSagittalHipJoint', 'RightSagittalHipJoint',
            'LeftSagittalKneeJoint', 'RightSagittalKneeJoint',
            'LeftSagittalAnkleJoint', 'RightSagittalAnkleJoint',
            'LeftHenkeAnkleJoint', 'RightHenkeAnkleJoint'
        ],
        'v': ['Velocity'],
        'vdot': ['Acceleration'],
        'reward': ['Reward']
    }
    
    units = {
        'y_out': ['rad'] * 12,  # All joint angles are in radians
        'dy_out': ['rad/s'] * 12,  # All joint velocities are in rad/s
        'base_velocity': ['m/s', 'm/s', 'rad/s'],
        'stance_foot_pos': ['m', 'm', 'm'],
        'stance_foot_ori': ['rad', 'rad', 'rad'],
        'cur_swing_time': ['s'],
        'y_act': ['rad'] * 12,
        'dy_act': ['rad/s'] * 12,
        'v': ['m/s'],
        'vdot': ['m/s²'],
        'reward': ['']
    }

    # Add error metrics labels and units
    error_labels = {
        'error_LeftFrontalHipJoint': 'Left Frontal Hip Error',
        'error_RightFrontalHipJoint': 'Right Frontal Hip Error',
        'error_LeftTransverseHipJoint': 'Left Transverse Hip Error',
        'error_RightTransverseHipJoint': 'Right Transverse Hip Error',
        'error_LeftSagittalHipJoint': 'Left Sagittal Hip Error',
        'error_RightSagittalHipJoint': 'Right Sagittal Hip Error',
        'error_LeftSagittalKneeJoint': 'Left Knee Error',
        'error_RightSagittalKneeJoint': 'Right Knee Error',
        'error_LeftSagittalAnkleJoint': 'Left Ankle Error',
        'error_RightSagittalAnkleJoint': 'Right Ankle Error',
        'error_LeftHenkeAnkleJoint': 'Left Henke Ankle Error',
        'error_RightHenkeAnkleJoint': 'Right Henke Ankle Error'
    }

    error_units = {
        'error_LeftFrontalHipJoint': 'rad',
        'error_RightFrontalHipJoint': 'rad',
        'error_LeftTransverseHipJoint': 'rad',
        'error_RightTransverseHipJoint': 'rad',
        'error_LeftSagittalHipJoint': 'rad',
        'error_RightSagittalHipJoint': 'rad',
        'error_LeftSagittalKneeJoint': 'rad',
        'error_RightSagittalKneeJoint': 'rad',
        'error_LeftSagittalAnkleJoint': 'rad',
        'error_RightSagittalAnkleJoint': 'rad',
        'error_LeftHenkeAnkleJoint': 'rad',
        'error_RightHenkeAnkleJoint': 'rad'
    }
    
    # Helper for subplot indexing
    def get_ax(axs, idx, n_cols):
        if axs.ndim == 1:
            return axs[idx]
        return axs[idx // n_cols, idx % n_cols]

    env_ids = 0

    if "stance_foot_pos" and "stance_foot_ori" in processed_data:
        pos_data = processed_data["stance_foot_pos"]
        ori_data = processed_data["stance_foot_ori"]
        pos_data_0 = processed_data["stance_foot_pos_0"]
        ori_data_0 = processed_data["stance_foot_ori_0"]

        fig, axs = plt.subplots(2, 3, figsize=(15, 6))
        fig.suptitle("Stance Foot Position and Orientation", fontsize=16)

        # Position
        pos_labels = ["x", "y", "z"]
        for i in range(3):
            ax = axs[0, i]
            ax.plot(time_steps, pos_data[:, env_ids, i], label=f"pos {pos_labels[i]}")
            ax.plot(time_steps, pos_data_0[:, env_ids, i], label=f"pos_0 {pos_labels[i]}", linestyle='--')
            ax.set_title(f"Position {pos_labels[i]}")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("m")
            ax.grid(True)
            ax.legend()

        # Orientation
        ori_labels = ["roll", "pitch", "yaw"]
        for i in range(3):
            ax = axs[1, i]
            ax.plot(time_steps, ori_data[:, env_ids, i], label=f"ori {ori_labels[i]}")
            ax.plot(time_steps, ori_data_0[:, env_ids, i], label=f"ori_0 {ori_labels[i]}", linestyle='--')
            ax.set_title(f"Orientation {ori_labels[i]}")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("rad")
            ax.grid(True)
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(os.path.join(save_dir, "stance_foot_pos_ori.png"), dpi=300, bbox_inches="tight")
        plt.show()

    # Plot positions (y_out vs y_act)
    if 'y_out' in processed_data and 'y_act' in processed_data:
        n_dims = processed_data['y_out'].shape[2]
        n_cols = 4
        n_rows = (n_dims + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
        fig.suptitle('Reference vs Actual Positions', fontsize=16)
        axs = np.array(axs)
        for i in range(n_dims):
            ax = get_ax(axs, i, n_cols)
            ax.plot(time_steps, processed_data['y_out'][:, env_ids, i], label='Reference', color='b')
            ax.plot(time_steps, processed_data['y_act'][:, env_ids, i], label='Actual', color='r', linestyle='--')
            label = state_labels['y_out'][i] if i < len(state_labels['y_out']) else f'Var {i}'
            unit = units['y_out'][i] if i < len(units['y_out']) else ''
            ax.set_title(label)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(unit)
            ax.grid(True)
            if i == 0:
                ax.legend()
        for i in range(n_dims, n_rows * n_cols):
            ax = get_ax(axs, i, n_cols)
            ax.set_visible(False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'positions.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Plot velocities (dy_out vs dy_act)
    if 'dy_out' in processed_data and 'dy_act' in processed_data:
        n_dims = processed_data['dy_out'].shape[2]
        n_cols = 4
        n_rows = (n_dims + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
        fig.suptitle('Reference vs Actual Velocities', fontsize=16)
        axs = np.array(axs)
        for i in range(n_dims):
            ax = get_ax(axs, i, n_cols)
            ax.plot(time_steps, processed_data['dy_out'][:, env_ids, i], label='Reference', color='b')
            ax.plot(time_steps, processed_data['dy_act'][:, env_ids, i], label='Actual', color='r', linestyle='--')
            label = state_labels['dy_out'][i] if i < len(state_labels['dy_out']) else f'Var {i}'
            unit = units['dy_out'][i] if i < len(units['dy_out']) else ''
            ax.set_title(label)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(unit)
            ax.grid(True)
            if i == 0:
                ax.legend()
        for i in range(n_dims, n_rows * n_cols):
            ax = get_ax(axs, i, n_cols)
            ax.set_visible(False)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'velocities.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    # Plot base velocity
    if 'base_velocity' in processed_data:
        n_dims = processed_data['base_velocity'].shape[2]
        fig, axs = plt.subplots(1, n_dims, figsize=(5 * n_dims, 3))
        fig.suptitle('Base Velocity', fontsize=16)
        for i in range(n_dims):
            ax = axs[i] if n_dims > 1 else axs
            ax.plot(time_steps, processed_data['base_velocity'][:, env_ids, i])
            label = state_labels['base_velocity'][i] if i < len(state_labels['base_velocity']) else f'Var {i}'
            unit = units['base_velocity'][i] if i < len(units['base_velocity']) else ''
            ax.set_title(label)
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(unit)
            ax.grid(True)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'base_velocity.png'), dpi=300, bbox_inches='tight')
        plt.show()

    # Plot v and vdot as two subplots in one figure
    if 'v' in processed_data and 'vdot' in processed_data:
        v_data = processed_data['v']
        vdot_data = processed_data['vdot']
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        axs[0].plot(time_steps, v_data[:, env_ids], label='v', color='g')
        axs[0].set_title('clf v')
        axs[0].set_ylabel(units['v'][0] if 'v' in units else '')
        axs[0].grid(True)
        axs[0].legend()
        
        axs[1].plot(time_steps, vdot_data[:, env_ids], label='vdot', color='m')
        axs[1].set_title('clf vdot')
        axs[1].set_xlabel('Time Steps')
        axs[1].set_ylabel(units['vdot'][0] if 'vdot' in units else '')
        axs[1].grid(True)
        axs[1].legend()
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'v_and_vdot.png'), dpi=300, bbox_inches='tight')
        plt.show()

    # Plot error metrics
    error_metrics = [key for key in processed_data.keys() if key.startswith('error_')]
    if error_metrics:
        n_metrics = len(error_metrics)
        n_cols = 4
        n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
        fig.suptitle('Error Metrics', fontsize=16)
        axs = np.array(axs)
        
        for i, metric in enumerate(error_metrics):
            ax = get_ax(axs, i, n_cols)
            data = processed_data[metric]
            # Handle both 1D and 2D arrays
            if data.ndim > 1:
                plot_data = data[:, env_ids]
            else:
                plot_data = data
            ax.plot(time_steps, plot_data, label=error_labels.get(metric, metric))
            ax.set_title(error_labels.get(metric, metric))
            ax.set_xlabel('Time Steps')
            ax.set_ylabel(error_units.get(metric, ''))
            ax.grid(True)
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_metrics, n_rows * n_cols):
            ax = get_ax(axs, i, n_cols)
            ax.set_visible(False)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'error_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Find the most recent log directory
    log_dir = find_most_recent_log_dir()
    if not log_dir:
        return
    print(f"Loading data from {log_dir}")
    # Load the data
    data = load_data(log_dir)
    # Create a directory for plots
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    # Plot the data
    plot_trajectories(data, save_dir=plot_dir)


if __name__ == "__main__":
    main() 