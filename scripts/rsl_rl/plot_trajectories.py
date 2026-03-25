import os
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
import glob
import torch
import argparse


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


def format_joint_name(joint_name):
    """Format joint name for better readability in plots"""
    # Remove '_joint' suffix and replace underscores with spaces
    formatted = joint_name.replace('_joint', '').replace('_', ' ')
    
    # Capitalize first letter of each word
    formatted = ' '.join(word.capitalize() for word in formatted.split())
    
    # Special formatting for common terms
    formatted = formatted.replace('Hip', 'Hip')
    formatted = formatted.replace('Knee', 'Knee')
    formatted = formatted.replace('Ankle', 'Ankle')
    formatted = formatted.replace('Shoulder', 'Shoulder')
    formatted = formatted.replace('Elbow', 'Elbow')
    formatted = formatted.replace('Waist', 'Waist')
    
    return formatted


def plot_trajectories(data, save_dir=None, trajectory_type=None, paper=False):
    """Plot all trajectories with proper labels and units."""
    # Convert lists to numpy arrays and handle torch tensors
    processed_data = {}
    for key, values in data.items():
        if isinstance(values[0], torch.Tensor):
            processed_data[key] = np.array([v.cpu().numpy() for v in values])
        else:
            processed_data[key] = np.array(values)
    
    # Create time array
    time_steps = np.arange(len(processed_data[list(processed_data.keys())[0]]))
    
    # Detect trajectory type if not provided
    if trajectory_type is None:
        # Check if we have end effector style metrics (contains frame names)
        ee_metrics = [key for key in processed_data.keys()
                     if '_ee_pos_' in key or '_ee_ori_' in key]
        if ee_metrics:
            trajectory_type = 'end_effector'
        else:
            # Check if we have joint style metrics
            joint_metrics = [key for key in processed_data.keys()
                           if key.startswith('error_') and '_joint' in key]
            if joint_metrics:
                trajectory_type = 'joint'
            else:
                # Default to joint if we can't determine
                trajectory_type = 'joint'
    
    print(f"Detected trajectory type: {trajectory_type}")
    
    # Generate dynamic labels and units based on trajectory type
   
    if trajectory_type == 'end_effector':
        # End effector trajectory - generate labels from metric names
        # Try to get axis names from the data if available
        axis_names = []
        if 'axis_names' in data and data['axis_names']:
            # Extract names from axis_names data (take first entry since it's the same for all timesteps)
            axis_names_data = data['axis_names'][0] if isinstance(data['axis_names'], list) else data['axis_names']
            if isinstance(axis_names_data, list):
                axis_names = [axis_info['name'] for axis_info in axis_names_data]
            else:
                axis_names = []
        else:
            # Fallback: try to infer from error metrics
            error_keys = [key for key in processed_data.keys() if key.startswith('error_')]
            if error_keys:
                # Extract dimension names from error keys
                axis_names = [key.replace('error_', '') for key in error_keys]
            else:
                # Final fallback: generic names
                n_dims = processed_data.get('y_des', [[]]).shape[2] if 'y_des' in processed_data else 0
                axis_names = [f'Dimension {i}' for i in range(n_dims)]
        
        state_labels = {
            'y_des': axis_names,
            'dy_des': [f"{name} Rate" for name in axis_names],
            'base_velocity': ['Linear X', 'Linear Y', 'Angular Z'],
            "stance_foot_pos": ['X', 'Y', 'Z'],
            "stance_foot_ori": ['Roll', 'Pitch', 'Yaw'],
            'cur_swing_time': ['Time'],
            'y_act': axis_names,
            'dy_act': [f"{name} Rate" for name in axis_names],
            'v': ['v'],
            'vdot': ['vdot'],
            'reward': ['Reward']
        }
        
        # Generate units based on axis names
        def get_unit_for_axis(axis_name):
            if 'pos' in axis_name or 'com_pos' in axis_name:
                return 'm'
            elif 'ori' in axis_name:
                return 'rad'
            elif 'joint' in axis_name:
                return 'rad'
            else:
                return 'mixed'
        
        axis_units = [get_unit_for_axis(name) for name in axis_names]
        rate_units = [f"{unit}/s" for unit in axis_units]
        
        units = {
            'y_des': axis_units,
            'dy_des': rate_units,
            'base_velocity': ['m/s', 'm/s', 'rad/s'],
            'stance_foot_pos': ['m', 'm', 'm'],
            'stance_foot_ori': ['rad', 'rad', 'rad'],
            'cur_swing_time': ['s'],
            'y_act': axis_units,
            'dy_act': rate_units,
            'v': ['m/s'],
            'vdot': ['m/s²'],
            'reward': ['']
        }
        
        # Generate error labels dynamically from end effector metrics
        error_labels = {}
        error_units = {}
        for key in processed_data.keys():
            if key.startswith('error_'):
                # Parse end effector error metrics
                parts = key.split('_')
                if len(parts) >= 3:
                    if '_ee_pos_' in key:
                        # Position constraint: error_frame_ee_pos_axis
                        frame_name = parts[1]  # e.g., 'left_palm'
                        axis = parts[-1].upper()
                        error_labels[key] = f"{frame_name.replace('_', ' ').title()} Position {axis}"
                        error_units[key] = 'm'
                    elif '_ee_ori_' in key:
                        # Orientation constraint: error_frame_ee_ori_axis
                        frame_name = parts[1]  # e.g., 'left_palm'
                        axis = parts[-1].title()
                        error_labels[key] = f"{frame_name.replace('_', ' ').title()} Orientation {axis}"
                        error_units[key] = 'rad'
                    elif '_com_pos_' in key:
                        # COM position constraint: error_com_pos_axis
                        axis = parts[-1].upper()
                        error_labels[key] = f"COM Position {axis}"
                        error_units[key] = 'm'
                    elif '_pelvis_ori_' in key:
                        # Pelvis orientation constraint: error_pelvis_ori_axis
                        axis = parts[-1].title()
                        error_labels[key] = f"Pelvis Orientation {axis}"
                        error_units[key] = 'rad'
                    else:
                        # Generic end effector error
                        error_labels[key] = key.replace('error_', '').replace('_', ' ').title()
                        error_units[key] = 'mixed'
                else:
                    # Fallback for simple error names
                    error_labels[key] = key.replace('error_', '').replace('_', ' ').title()
                    error_units[key] = 'mixed'
    
    else:
        # Fallback for unknown trajectory types
        state_labels = {}
        units = {}
        error_labels = {}
        error_units = {}

    # Helper for subplot indexing
    def get_ax(axs, idx, n_cols):
        if axs.ndim == 1:
            return axs[idx]
        return axs[idx // n_cols, idx % n_cols]

    # Hard code number of envs to plot
    N_ENVS_TO_PLOT = 2
    env_ids = list(range(N_ENVS_TO_PLOT))

    # TODO: Don't plot None data
    for env_id in env_ids:
        # --- Stance Foot Position and Orientation ---
        if "stance_foot_pos" and "stance_foot_ori" in processed_data:
            pos_data = processed_data["stance_foot_pos"]
            ori_data = processed_data["stance_foot_ori"]
            pos_data_0 = processed_data["stance_foot_pos_0"]
            ori_data_0 = processed_data["stance_foot_ori_0"]
            fig, axs = plt.subplots(2, 3, figsize=(15, 6))
            fig.suptitle(f"Stance Foot Position and Orientation (Env {env_id})", fontsize=16)
            pos_labels = ["X", "Y", "Z"]
            for i in range(3):
                ax = axs[0, i]
                ax.plot(time_steps, pos_data[:, env_id, i], label=f"Position {pos_labels[i]}", linewidth=2)
                ax.plot(time_steps, pos_data_0[:, env_id, i], label=f"Initial {pos_labels[i]}", linestyle='--', linewidth=2)
                ax.set_title(f"Position {pos_labels[i]}")
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("m")
                ax.grid(True, alpha=0.3)
                ax.legend()
            ori_labels = ["Roll", "Pitch", "Yaw"]
            for i in range(3):
                ax = axs[1, i]
                ax.plot(time_steps, ori_data[:, env_id, i], label=f"Orientation {ori_labels[i]}", linewidth=2)
                ax.plot(time_steps, ori_data_0[:, env_id, i], label=f"Initial {ori_labels[i]}", linestyle='--', linewidth=2)
                ax.set_title(f"Orientation {ori_labels[i]}")
                ax.set_xlabel("Time Steps")
                ax.set_ylabel("rad")
                ax.grid(True, alpha=0.3)
                ax.legend()
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"stance_foot_pos_ori_env{env_id}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

        # --- Positions (y_out vs y_act) ---
        if 'y_des' in processed_data and 'y_act' in processed_data:
            n_dims = processed_data['y_des'].shape[2]
            if trajectory_type == 'end_effector':
                title = f'Reference vs Actual End Effector Positions (Env {env_id})'
            else:
                title = f'Reference vs Actual Joint Positions (Env {env_id})'
            n_cols = 4
            n_rows = (n_dims + n_cols - 1) // n_cols
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
            fig.suptitle(title, fontsize=16)
            axs = np.array(axs)
            for i in range(n_dims):
                ax = get_ax(axs, i, n_cols)
                ax.plot(time_steps, processed_data['y_des'][:, env_id, i], label='Reference', linewidth=2)
                ax.plot(time_steps, processed_data['y_act'][:, env_id, i], label='Actual', linestyle='--', linewidth=2)
                # label = state_labels['y_des'][i] if i < len(state_labels['y_des']) else f'Dimension {i}'
                label = processed_data['ordered_pos_output_names'][0, i]
                print(f"using label {label}, i: {i}, n_dims: {n_dims}")
                unit = units['y_des'][i] if i < len(units['y_des']) else ''
                ax.set_title(label, fontsize=10)
                ax.set_xlabel('Time Steps')
                ax.set_ylabel(f'Position ({unit})')
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend()
            for i in range(n_dims, n_rows * n_cols):
                ax = get_ax(axs, i, n_cols)
                ax.set_visible(False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'positions_env{env_id}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

        # --- Velocities (dy_out vs dy_act) ---
        if 'dy_des' in processed_data and 'dy_act' in processed_data:
            n_dims = processed_data['dy_des'].shape[2]
            if trajectory_type == 'end_effector':
                title = f'Reference vs Actual End Effector Velocities (Env {env_id})'
            else:
                title = f'Reference vs Actual Joint Velocities (Env {env_id})'
            n_cols = 4
            n_rows = (n_dims + n_cols - 1) // n_cols
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
            fig.suptitle(title, fontsize=16)
            axs = np.array(axs)
            for i in range(n_dims):
                ax = get_ax(axs, i, n_cols)
                ax.plot(time_steps, processed_data['dy_des'][:, env_id, i], label='Reference', linewidth=2)
                ax.plot(time_steps, processed_data['dy_act'][:, env_id, i], label='Actual', linestyle='--', linewidth=2)
                label = processed_data['ordered_vel_output_names'][0, i]
                unit = units['dy_des'][i] if i < len(units['dy_des']) else ''
                ax.set_title(label, fontsize=10)
                ax.set_xlabel('Time Steps')
                ax.set_ylabel(f'Velocity ({unit})')
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend()
            for i in range(n_dims, n_rows * n_cols):
                ax = get_ax(axs, i, n_cols)
                ax.set_visible(False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'velocities_env{env_id}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

        # --- Joint Angle Targets vs Actual ---
        if 'action_targets' in processed_data and 'joint_pos' in processed_data:
            n_dims = processed_data['action_targets'].shape[2]
            title = f'Joint Angle Targets vs Actual (Env {env_id})'
            n_cols = 4
            n_rows = (n_dims + n_cols - 1) // n_cols
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
            fig.suptitle(title, fontsize=16)
            axs = np.array(axs)

            # Get joint names for labels
            joint_names = None
            if 'joint_names' in processed_data:
                joint_names = processed_data['joint_names'][0]  # Same for all timesteps

            for i in range(n_dims):
                ax = get_ax(axs, i, n_cols)
                ax.plot(time_steps, processed_data['action_targets'][:, env_id, i], label='Target', linewidth=2)
                ax.plot(time_steps, processed_data['joint_pos'][:, env_id, i], label='Actual', linestyle='--', linewidth=2)
                if joint_names is not None and i < len(joint_names):
                    label = format_joint_name(str(joint_names[i]))
                else:
                    label = f'Dim {i}'
                ax.set_title(label, fontsize=10)
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Angle (rad)')
                ax.grid(True, alpha=0.3)
                if i == 0:
                    ax.legend()
            for i in range(n_dims, n_rows * n_cols):
                ax = get_ax(axs, i, n_cols)
                ax.set_visible(False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'joint_targets_vs_actual_env{env_id}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

        # --- Joint Torques ---
        if 'applied_torque' in processed_data:
            n_dims = processed_data['applied_torque'].shape[2]
            title = f'Joint Torques (Env {env_id})'
            n_cols = 4
            n_rows = (n_dims + n_cols - 1) // n_cols
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
            fig.suptitle(title, fontsize=16)
            axs = np.array(axs)

            joint_names = None
            if 'joint_names' in processed_data:
                joint_names = processed_data['joint_names'][0]

            for i in range(n_dims):
                ax = get_ax(axs, i, n_cols)
                ax.plot(time_steps, processed_data['applied_torque'][:, env_id, i], linewidth=2)
                if joint_names is not None and i < len(joint_names):
                    label = format_joint_name(str(joint_names[i]))
                else:
                    label = f'Dim {i}'
                ax.set_title(label, fontsize=10)
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Torque (Nm)')
                ax.grid(True, alpha=0.3)
            for i in range(n_dims, n_rows * n_cols):
                ax = get_ax(axs, i, n_cols)
                ax.set_visible(False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'joint_torques_env{env_id}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

        # --- Base Velocity ---
        if 'base_velocity' in processed_data:
            n_dims = processed_data['base_velocity'].shape[2]
            fig, axs = plt.subplots(1, n_dims, figsize=(5 * n_dims, 3))
            fig.suptitle(f'Base Velocity (Env {env_id})', fontsize=16)
            for i in range(n_dims):
                ax = axs[i] if n_dims > 1 else axs
                ax.plot(time_steps, processed_data['base_velocity'][:, env_id, i], linewidth=2)
                label = state_labels['base_velocity'][i] if i < len(state_labels['base_velocity']) else f'Component {i}'
                unit = units['base_velocity'][i] if i < len(units['base_velocity']) else ''
                ax.set_title(label)
                ax.set_xlabel('Time Steps')
                ax.set_ylabel(f'Velocity ({unit})')
                ax.grid(True, alpha=0.3)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'base_velocity_env{env_id}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

        if 'phasing_var' in processed_data:
            phase_var = processed_data['phasing_var']
            current_domains = processed_data['current_domain']

            # Check if v_log data is available
            has_v_log = 'v_log' in processed_data and 'phi_keys' in processed_data
            n_dims = 3 if has_v_log else 2

            vars = [phase_var, current_domains]
            labels = ['Phasing Var', 'Current Domain']
            fig, axs = plt.subplots(1, n_dims, figsize=(5 * n_dims, 3))
            fig.suptitle(f'Domain Info (Env {env_id})', fontsize=16)

            # Plot phasing var and current domain
            for i in range(2):
                ax = axs[i] if n_dims > 1 else axs
                ax.plot(time_steps, vars[i][:, env_id], linewidth=2)
                ax.set_title(labels[i])
                ax.set_xlabel('Time Steps')
                ax.grid(True, alpha=0.3)

            # Plot V moving average vs phasing variable
            if has_v_log:
                ax = axs[2]
                phi_keys = processed_data['phi_keys'][-1]  # Take final snapshot
                v_log = processed_data['v_log'][-1]        # Take final snapshot
                if isinstance(phi_keys, torch.Tensor):
                    phi_keys = phi_keys.cpu().numpy()
                if isinstance(v_log, torch.Tensor):
                    v_log = v_log.cpu().numpy()
                # Handle LibraryManager case where v_log is [num_envs, num_bins]
                if v_log.ndim > 1:
                    v_log = v_log[env_id]
                ax.bar(phi_keys, v_log, width=0.08, alpha=0.7)
                ax.set_title('V Moving Avg vs Phase')
                ax.set_xlabel('Phasing Variable')
                ax.set_ylabel('V (EMA)')
                ax.grid(True, alpha=0.3)

            if save_dir:
                plt.savefig(os.path.join(save_dir, f'domain_info_env{env_id}.png'), dpi=300, bbox_inches='tight')

            plt.close(fig)

        # --- v and vdot ---
        if 'v' in processed_data and 'vdot' in processed_data:
            v_data = processed_data['v']
            vdot_data = processed_data['vdot']
            fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            axs[0].plot(time_steps, v_data[:, env_id], label='CLF v', linewidth=2)
            axs[0].set_title('CLF (v)')
            axs[0].set_ylabel(units['v'][0] if 'v' in units else '')
            axs[0].grid(True, alpha=0.3)
            axs[0].legend()
            # axs[0].set_ylim(0, 20.0)
            axs[1].plot(time_steps, vdot_data[:, env_id], label='CLF vdot', linewidth=2)
            axs[1].set_title('CLF (v̇)')
            axs[1].set_xlabel('Time Steps')
            axs[1].set_ylabel(units['vdot'][0] if 'vdot' in units else '')
            axs[1].grid(True, alpha=0.3)
            axs[1].legend()
            # axs[1].set_ylim(-100.0, 100.0)
            alpha = 1.0
            decay = alpha * v_data[:, env_id] + vdot_data[:, env_id]
            axs[2].plot(time_steps, decay, label='CLF Decay', linewidth=2)
            axs[2].set_title('CLF Decay (v + αv̇)')
            axs[2].set_xlabel('Time Steps')
            axs[2].set_ylabel('Decay Rate')
            axs[2].grid(True, alpha=0.3)
            axs[2].legend()
            # axs[2].set_ylim(-100.0, 100.0)
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'v_and_vdot_env{env_id}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)


        # --- Error Metrics ---
        error_metrics = [key for key in processed_data.keys() if key.startswith('error_')]
        if error_metrics:
            n_metrics = len(error_metrics)
            n_cols = 4
            n_rows = (n_metrics + n_cols - 1) // n_cols
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows))
            fig.suptitle(f'Error Metrics (Env {env_id})', fontsize=16)
            axs = np.array(axs)
            for i, metric in enumerate(error_metrics):
                ax = get_ax(axs, i, n_cols)
                data = processed_data[metric]
                if data.ndim > 1:
                    plot_data = data[:, env_id]
                    ax.plot(time_steps, plot_data, label=error_labels.get(metric, metric), linewidth=2)
                else:
                    ax.plot(time_steps, data, label=error_labels.get(metric, metric), linewidth=2)
                ax.set_title(error_labels.get(metric, metric), fontsize=10)
                ax.set_xlabel('Time Steps')
                ax.set_ylabel(error_units.get(metric, ''))
                ax.grid(True, alpha=0.3)
                ax.legend()
            for i in range(n_metrics, n_rows * n_cols):
                ax = get_ax(axs, i, n_cols)
                ax.set_visible(False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'error_metrics_env{env_id}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

    # --- CLF EMA Plot (not per-env) ---
    NUM_EMA = 11
    clf_ema_keys = [f'CLF_EMA_{i}' for i in range(NUM_EMA)]
    if all(key in processed_data for key in clf_ema_keys):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        fig.suptitle('CLF EMA vs Phase Index', fontsize=16)

        # Get the last logged value for each CLF_EMA_X
        clf_values = [float(processed_data[f'CLF_EMA_{i}'][-1][-1]) for i in range(NUM_EMA)]

        indices = list(range(NUM_EMA))
        ax.bar(indices, clf_values, width=0.7, alpha=0.7)
        ax.set_title('CLF EMA Values')
        ax.set_xlabel('Phase Index')
        ax.set_ylabel('CLF Value (EMA)')
        ax.set_xticks(indices)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'clf_ema.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)

    # # Generate focused COM and ankle plot
    # plot_focused_com_ankle(data, save_dir=save_dir, trajectory_type=trajectory_type)

    if paper:
        plot_paper_figure(data, save_dir=save_dir)


def _find_name_index(names_array: np.ndarray, target: str) -> int | None:
    """Find the index of a target name in an ordered output names array.

    Args:
        names_array: 2-D array of shape [1, num_dims] from ordered_pos/vel_output_names.
        target: The name string to search for (e.g. ``"pelvis_link:pos_x"``).

    Returns:
        The column index, or ``None`` if not found.
    """
    for i in range(names_array.shape[1]):
        if str(names_array[0, i]) == target:
            return i
    return None


def plot_paper_figure(data: dict, save_dir: str | None = None) -> None:
    """Generate a 2x2 paper-ready figure with LaTeX fonts.

    Plots pelvis x position, pelvis x velocity, left hip pitch angle,
    and left hip pitch angular velocity (reference vs actual).

    Args:
        data: Raw logged data dict (values are lists of torch tensors).
        save_dir: Directory to save the figure. If ``None``, only displays.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
    })

    # Convert torch tensors to numpy
    processed: dict[str, np.ndarray] = {}
    for key, values in data.items():
        if isinstance(values[0], torch.Tensor):
            processed[key] = np.array([v.cpu().numpy() for v in values])
        else:
            processed[key] = np.array(values)

    for required in ('y_des', 'y_act', 'dy_des', 'dy_act',
                     'ordered_pos_output_names', 'ordered_vel_output_names'):
        if required not in processed:
            print(f"[paper] Missing required data key '{required}'. Skipping paper figure.")
            return

    # Time axis (assume 50 Hz)
    n_steps = len(processed['y_des'])
    time = np.arange(n_steps) * 0.02

    env_id = 0

    # Look up indices
    pos_names = processed['ordered_pos_output_names']
    vel_names = processed['ordered_vel_output_names']

    pelvis_pos_idx = _find_name_index(pos_names, "pelvis_link:pos_x")
    pelvis_vel_idx = _find_name_index(vel_names, "pelvis_link:pos_x")
    hip_pos_idx = _find_name_index(pos_names, "joint:left_hip_pitch_joint")
    hip_vel_idx = _find_name_index(vel_names, "joint:left_hip_pitch_joint")

    if any(idx is None for idx in (pelvis_pos_idx, pelvis_vel_idx, hip_pos_idx, hip_vel_idx)):
        print(f"[paper] Could not find all required indices:")
        print(f"  pelvis pos: {pelvis_pos_idx}, pelvis vel: {pelvis_vel_idx}")
        print(f"  hip pos: {hip_pos_idx}, hip vel: {hip_vel_idx}")
        print(f"  Available pos names: {[str(pos_names[0, i]) for i in range(pos_names.shape[1])]}")
        print(f"  Available vel names: {[str(vel_names[0, i]) for i in range(vel_names.shape[1])]}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))

    # Shared plot helper
    def _plot_panel(ax, idx_pos, y_key, dy_key, title, ylabel):
        ax.plot(time, processed[y_key][:, env_id, idx_pos], '--', linewidth=2, label='Reference')
        ax.plot(time, processed[dy_key][:, env_id, idx_pos], linewidth=2, label='Actual')
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(r'Time (s)', fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14)

    # (0,0) Pelvis x position
    _plot_panel(axes[0, 0], pelvis_pos_idx, 'y_des', 'y_act',
                r'Pelvis $x$ Position', r'Position (m)')

    # (0,1) Pelvis x velocity
    _plot_panel(axes[0, 1], pelvis_vel_idx, 'dy_des', 'dy_act',
                r'Pelvis $x$ Velocity', r'Velocity (m/s)')

    # (1,0) Left hip pitch angle
    _plot_panel(axes[1, 0], hip_pos_idx, 'y_des', 'y_act',
                r'Left Hip Pitch Angle', r'Angle (rad)')

    # (1,1) Left hip pitch angular velocity
    _plot_panel(axes[1, 1], hip_vel_idx, 'dy_des', 'dy_act',
                r'Left Hip Pitch Angular Velocity', r'Angular Velocity (rad/s)')

    plt.tight_layout()

    if save_dir:
        svg_path = os.path.join(save_dir, 'paper_figure.svg')
        plt.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"[paper] Saved SVG to {svg_path}")

        png_path = os.path.join(save_dir, 'paper_figure.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"[paper] Saved PNG to {png_path}")

        pdf_path = os.path.join(save_dir, 'paper_figure.pdf')
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
        print(f"[paper] Saved PDF to {pdf_path}")

    plt.close(fig)


def plot_focused_com_ankle(data, save_dir=None, trajectory_type=None):
    """Plot focused view of COM positions and left ankle position/orientation (desired vs actual)"""
    # Set nice font for plots (LaTeX disabled due to missing packages)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
    })
    
    # Convert lists to numpy arrays and handle torch tensors
    processed_data = {}
    for key, values in data.items():
        if isinstance(values[0], torch.Tensor):
            processed_data[key] = np.array([v.cpu().numpy() for v in values])
        else:
            processed_data[key] = np.array(values)
    
    # Create time array
    time_steps = np.arange(len(processed_data[list(processed_data.keys())[0]]))
    time = time_steps * 0.02    # Assume 50 Hz
    
    # Hard code number of envs to plot
    N_ENVS_TO_PLOT = 2
    env_ids = list(range(N_ENVS_TO_PLOT))
    
    # Check if we have required data
    if 'y_des' not in processed_data or 'y_act' not in processed_data:
        print("Warning: y_out or y_act data not found. Cannot create focused COM/ankle plots.")
        return
    
    # Check if we have base_velocity data
    has_base_velocity = 'base_velocity' in processed_data
    if not has_base_velocity:
        print("Warning: base_velocity data not found. Will skip velocity plots.")
    
    # Get axis names if available to identify which dimensions correspond to what
    axis_names = []
    if 'axis_names' in data and data['axis_names']:
        axis_names_data = data['axis_names'][0] if isinstance(data['axis_names'], list) else data['axis_names']
        if isinstance(axis_names_data, list):
            axis_names = [axis_info['name'] for axis_info in axis_names_data]
    
    print(f"Debug - Available axis names: {axis_names}")
    print(f"Debug - y_out shape: {processed_data['y_des'].shape}")
    print(f"Debug - y_act shape: {processed_data['y_act'].shape}")
    
    for env_id in env_ids:
        # Find indices for the metrics we want
        com_x_idx = com_y_idx = com_z_idx = None
        left_ankle_x_idx = left_ankle_z_idx = left_ankle_pitch_idx = None
        
        for i, name in enumerate(axis_names):
            name_lower = name.lower()
            if 'com' in name_lower and 'pos' in name_lower and 'x' in name_lower:
                com_x_idx = i
            elif 'com' in name_lower and 'pos' in name_lower and 'y' in name_lower:
                com_y_idx = i
            elif 'com' in name_lower and 'pos' in name_lower and 'z' in name_lower:
                com_z_idx = i
            elif 'left' in name_lower and 'ankle' in name_lower and 'pos' in name_lower and 'x' in name_lower:
                left_ankle_x_idx = i
            elif 'left' in name_lower and 'ankle' in name_lower and 'pos' in name_lower and 'z' in name_lower:
                left_ankle_z_idx = i
            elif 'left' in name_lower and 'ankle' in name_lower and 'pitch' in name_lower:
                left_ankle_pitch_idx = i
        
        print(f"Debug - Found indices:")
        print(f"  COM X: {com_x_idx}")
        print(f"  COM Y: {com_y_idx}")
        print(f"  COM Z: {com_z_idx}")
        print(f"  Left Ankle X: {left_ankle_x_idx}")
        print(f"  Left Ankle Z: {left_ankle_z_idx}")
        print(f"  Left Ankle Pitch: {left_ankle_pitch_idx}")
        
        # Create the focused plot - new layout: Col 0=Commanded Velocities, Col 1=COM Positions, Col 2=Ankle Values
        fig, axes = plt.subplots(3, 3, figsize=(18, 6))
        
        # Column 0: Base Velocities (commanded velocities)
        if has_base_velocity and processed_data['base_velocity'].shape[2] >= 2:
            # Plot X velocity
            axes[0, 0].plot(time, processed_data['base_velocity'][:, env_id, 0], linewidth=2, color='tab:blue')
            axes[0, 0].set_title('Commanded $x$ Velocity', fontsize=18)
            axes[0, 0].set_xlabel('Time (s)', fontsize=14)
            axes[0, 0].set_ylabel('Velocity (m/s)', fontsize=14)
            axes[0, 0].tick_params(axis='both', which='major', labelsize=12)
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot Y velocity
            axes[1, 0].plot(time, processed_data['base_velocity'][:, env_id, 1], linewidth=2, color='tab:blue')
            axes[1, 0].set_title('Commanded $y$ Velocity', fontsize=18)
            axes[1, 0].set_xlabel('Time (s)', fontsize=14)
            axes[1, 0].set_ylabel('Velocity (m/s)', fontsize=14)
            axes[1, 0].tick_params(axis='both', which='major', labelsize=12)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot Yaw velocity
            axes[2, 0].plot(time, processed_data['base_velocity'][:, env_id, 2], linewidth=2, color='tab:blue')
            axes[2, 0].set_title('Commanded Yaw Velocity', fontsize=18)
            axes[2, 0].set_xlabel('Time (s)', fontsize=14)
            axes[2, 0].set_ylabel('Angular Velocity (rad/s)', fontsize=14)
            axes[2, 0].tick_params(axis='both', which='major', labelsize=12)
            axes[2, 0].grid(True, alpha=0.3)
        
        # Column 1: COM Positions
        if com_x_idx is not None:
            axes[0, 1].plot(time, processed_data['y_des'][:, env_id, com_x_idx], '--', linewidth=2, label='Reference')
            axes[0, 1].plot(time, processed_data['y_act'][:, env_id, com_x_idx], linewidth=2, label='Actual')
            axes[0, 1].set_title('COM Position $x$', fontsize=18)
            axes[0, 1].set_xlabel('Time (s)', fontsize=14)
            axes[0, 1].set_ylabel('Position (m)', fontsize=14)
            axes[0, 1].tick_params(axis='both', which='major', labelsize=12)
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend(fontsize=12, loc='lower right')
        
        if com_y_idx is not None:
            axes[1, 1].plot(time, processed_data['y_des'][:, env_id, com_y_idx], '--', linewidth=2, label='Reference')
            axes[1, 1].plot(time, processed_data['y_act'][:, env_id, com_y_idx], linewidth=2, label='Actual')
            axes[1, 1].set_title('COM Position $y$', fontsize=18)
            axes[1, 1].set_xlabel('Time (s)', fontsize=14)
            axes[1, 1].set_ylabel('Position (m)', fontsize=14)
            axes[1, 1].tick_params(axis='both', which='major', labelsize=12)
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend(fontsize=12, loc='lower right')
        
        if com_z_idx is not None:
            axes[2, 1].plot(time, processed_data['y_des'][:, env_id, com_z_idx], '--', linewidth=2, label='Reference')
            axes[2, 1].plot(time, processed_data['y_act'][:, env_id, com_z_idx], linewidth=2, label='Actual')
            axes[2, 1].set_title('COM Position $z$', fontsize=18)
            axes[2, 1].set_xlabel('Time (s)', fontsize=14)
            axes[2, 1].set_ylabel('Position (m)', fontsize=14)
            axes[2, 1].tick_params(axis='both', which='major', labelsize=12)
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].legend(fontsize=12, loc='lower right')
        
        # Column 2: Ankle Values
        if left_ankle_x_idx is not None:
            axes[0, 2].plot(time, processed_data['y_des'][:, env_id, left_ankle_x_idx], '--', linewidth=2, label='Reference')
            axes[0, 2].plot(time, processed_data['y_act'][:, env_id, left_ankle_x_idx], linewidth=2, label='Actual')
            axes[0, 2].set_title('Swing Ankle Position $x$', fontsize=18)
            axes[0, 2].set_xlabel('Time (s)', fontsize=14)
            axes[0, 2].set_ylabel('Position (m)', fontsize=14)
            axes[0, 2].tick_params(axis='both', which='major', labelsize=12)
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].legend(fontsize=12, loc='lower right')
        
        if left_ankle_z_idx is not None:
            axes[1, 2].plot(time, processed_data['y_des'][:, env_id, left_ankle_z_idx], '--', linewidth=2, label='Reference')
            axes[1, 2].plot(time, processed_data['y_act'][:, env_id, left_ankle_z_idx], linewidth=2, label='Actual')
            axes[1, 2].set_title('Swing Ankle Position $z$', fontsize=18)
            axes[1, 2].set_xlabel('Time (s)', fontsize=14)
            axes[1, 2].set_ylabel('Position (m)', fontsize=14)
            axes[1, 2].tick_params(axis='both', which='major', labelsize=12)
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].legend(fontsize=12, loc='lower right')
        
        if left_ankle_pitch_idx is not None:
            axes[2, 2].plot(time, processed_data['y_des'][:, env_id, left_ankle_pitch_idx], '--', linewidth=2, label='Reference')
            axes[2, 2].plot(time, processed_data['y_act'][:, env_id, left_ankle_pitch_idx], linewidth=2, label='Actual')
            axes[2, 2].set_title('Swing Ankle Pitch Angle', fontsize=18)
            axes[2, 2].set_xlabel('Time (s)', fontsize=14)
            axes[2, 2].set_ylabel('Angle (rad)', fontsize=14)
            axes[2, 2].tick_params(axis='both', which='major', labelsize=12)
            axes[2, 2].grid(True, alpha=0.3)
            axes[2, 2].legend(fontsize=12, loc='lower right')
        
        # Hide any empty subplots
        for i in range(3):
            for j in range(3):
                if not axes[i, j].lines:  # If no data was plotted
                    axes[i, j].text(0.5, 0.5, 'No Data\nAvailable', ha='center', va='center', 
                                   transform=axes[i, j].transAxes, fontsize=12)
                    axes[i, j].set_title(f'Plot {i},{j} - No Data', fontsize=18)
        
        plt.tight_layout()
        
        # Save as SVG automatically
        if save_dir:
            svg_path = os.path.join(save_dir, f'focused_com_ankle_env{env_id}.svg')
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"Saved focused COM and ankle plot to: {svg_path}")
            
            # Also save as PNG for backup
            png_path = os.path.join(save_dir, f'focused_com_ankle_env{env_id}.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)

def compute_and_save_stats(data: dict, save_dir: str | None = None,
                           start_idx: int | None = None, end_idx: int | None = None) -> None:
    """Compute summary statistics (mean V, position/velocity errors) and save to file.

    Computes per-env means first, then reports the mean and std across envs.

    Args:
        data: Raw data dict from load_data (values may be torch tensors or lists).
        save_dir: Directory to save play_stats.txt. If None, only prints.
        start_idx: Start timestep index (inclusive). Defaults to 0.
        end_idx: End timestep index (exclusive). Defaults to last timestep.
    """
    # Convert to numpy
    processed: dict[str, np.ndarray] = {}
    for key, values in data.items():
        if isinstance(values[0], torch.Tensor):
            processed[key] = np.array([v.cpu().numpy() for v in values])
        else:
            processed[key] = np.array(values)

    # Determine timestep range
    sample_key = next(iter(processed))
    n_timesteps = processed[sample_key].shape[0]
    si = start_idx if start_idx is not None else 0
    ei = end_idx if end_idx is not None else n_timesteps

    lines: list[str] = []
    csv_rows: list[dict[str, str]] = []

    def _log(text: str = "") -> None:
        print(text)
        lines.append(text)

    def _csv(metric: str, value: float) -> None:
        csv_rows.append({"metric": metric, "value": f"{value:.6f}"})

    _log("=" * 60)
    _log("  Play Statistics")
    _log("=" * 60)
    _log(f"Timestep range: [{si}, {ei}) of {n_timesteps} total")

    # --- Number of envs ---
    # Infer from a 2D+ array
    n_envs = None
    for arr in processed.values():
        if arr.ndim >= 2:
            n_envs = arr.shape[1]
            break
    if n_envs is not None:
        _log(f"Number of envs: {n_envs}")

    # --- Commanded velocity ranges ---
    if 'base_velocity' in processed:
        bv = processed['base_velocity'][si:ei]  # [T, envs, 3]
        vel_labels = ['Linear X', 'Linear Y', 'Angular Z']
        vel_units = ['m/s', 'm/s', 'rad/s']
        _log("")
        _log("Commanded Velocity Ranges:")
        csv_vel_keys = ['cmd_vel_lin_x', 'cmd_vel_lin_y', 'cmd_vel_ang_z']
        for i in range(min(bv.shape[2], 3)):
            v_min = bv[:, :, i].min()
            v_max = bv[:, :, i].max()
            _log(f"  {vel_labels[i]:>12s}: [{v_min:+.4f}, {v_max:+.4f}] {vel_units[i]}")
            _csv(f"{csv_vel_keys[i]}_min", float(v_min))
            _csv(f"{csv_vel_keys[i]}_max", float(v_max))

    # --- Mean V (Lyapunov) ---
    if 'v' in processed:
        v_data = processed['v'][si:ei]  # [T, envs] or [T, envs, 1]
        if v_data.ndim == 3:
            v_data = v_data.squeeze(-1)
        per_env_mean = v_data.mean(axis=0)  # [envs]
        _log("")
        _log("Lyapunov Function (V):")
        _log(f"  Mean across envs: {per_env_mean.mean():.6f}")
        _log(f"  Std  across envs: {per_env_mean.std():.6f}")
        _csv("lyapunov_v_mean", float(per_env_mean.mean()))
        _csv("lyapunov_v_std", float(per_env_mean.std()))

    # --- Norm squared error ---
    if 'y_des' in processed and 'y_act' in processed and 'dy_des' in processed and 'dy_act' in processed:
        e_pos = processed['y_des'][si:ei] - processed['y_act'][si:ei]  # [T, envs, pos_dims]
        e_vel = processed['dy_des'][si:ei] - processed['dy_act'][si:ei]  # [T, envs, vel_dims]
        e = np.concatenate([e_pos, e_vel], axis=2)  # [T, envs, pos_dims + vel_dims]
        norm_sq = (e ** 2).sum(axis=2)  # [T, envs]
        per_env_mean = norm_sq.mean(axis=0)  # [envs]
        _log("")
        _log("Norm Squared Error (dot(e,e)):")
        _log(f"  Mean across envs: {per_env_mean.mean():.6f}")
        _log(f"  Std  across envs: {per_env_mean.std():.6f}")
        _csv("norm_sq_error_mean", float(per_env_mean.mean()))
        _csv("norm_sq_error_std", float(per_env_mean.std()))

    # --- Position errors ---
    if 'y_des' in processed and 'y_act' in processed:
        y_des = processed['y_des'][si:ei]  # [T, envs, dims]
        y_act = processed['y_act'][si:ei]
        pos_err = (y_des - y_act)**2  # [T, envs, dims]
        n_dims = pos_err.shape[2]

        # Get labels
        pos_names: list[str] = []
        if 'ordered_pos_output_names' in processed:
            for i in range(n_dims):
                pos_names.append(str(processed['ordered_pos_output_names'][0, i]))
        else:
            pos_names = [f"Dim {i}" for i in range(n_dims)]

        per_env_mean_err = pos_err.mean(axis=0)  # [envs, dims]
        mean_over_envs = per_env_mean_err.mean(axis=0)  # [dims]
        std_over_envs = per_env_mean_err.std(axis=0)    # [dims]

        _log("")
        _log("Position Errors (|y_des - y_act|):")
        _log(f"  {'Name':<40s} {'Mean':>12s} {'Std':>12s}")
        _log(f"  {'-'*40} {'-'*12} {'-'*12}")
        for i in range(n_dims):
            _log(f"  {pos_names[i]:<40s} {mean_over_envs[i]:12.6f} {std_over_envs[i]:12.6f}")

        # Group summaries for position errors
        groups = {"Positions": ":pos_", "Orientations": ":ori_", "Joints": "joint:"}
        _log("")
        _log("  Position Error Group Summaries:")
        _log(f"  {'Group':<40s} {'Mean':>12s} {'Std':>12s}")
        _log(f"  {'-'*40} {'-'*12} {'-'*12}")
        for group_name, pattern in groups.items():
            idxs = [i for i, name in enumerate(pos_names) if pattern in name]
            if idxs:
                group_means = mean_over_envs[idxs]
                group_stds = std_over_envs[idxs]
                _log(f"  {group_name:<40s} {group_means.mean():12.6f} {group_stds.mean():12.6f}")
                csv_key = f"pos_error_{group_name.lower()}"
                _csv(f"{csv_key}_mean", float(group_means.mean()))
                _csv(f"{csv_key}_std", float(group_stds.mean()))

    # --- Velocity errors ---
    if 'dy_des' in processed and 'dy_act' in processed:
        dy_des = processed['dy_des'][si:ei]
        dy_act = processed['dy_act'][si:ei]
        vel_err = (dy_des - dy_act)**2
        n_dims = vel_err.shape[2]

        vel_names: list[str] = []
        if 'ordered_vel_output_names' in processed:
            for i in range(n_dims):
                vel_names.append(str(processed['ordered_vel_output_names'][0, i]))
        else:
            vel_names = [f"Dim {i}" for i in range(n_dims)]

        per_env_mean_err = vel_err.mean(axis=0)
        mean_over_envs = per_env_mean_err.mean(axis=0)
        std_over_envs = per_env_mean_err.std(axis=0)

        _log("")
        _log("Velocity Errors (|dy_des - dy_act|):")
        _log(f"  {'Name':<40s} {'Mean':>12s} {'Std':>12s}")
        _log(f"  {'-'*40} {'-'*12} {'-'*12}")
        for i in range(n_dims):
            _log(f"  {vel_names[i]:<40s} {mean_over_envs[i]:12.6f} {std_over_envs[i]:12.6f}")

        # Group summaries for velocity errors
        groups = {"Positions": ":pos_", "Orientations": ":ori_", "Joints": "joint:"}
        _log("")
        _log("  Velocity Error Group Summaries:")
        _log(f"  {'Group':<40s} {'Mean':>12s} {'Std':>12s}")
        _log(f"  {'-'*40} {'-'*12} {'-'*12}")
        for group_name, pattern in groups.items():
            idxs = [i for i, name in enumerate(vel_names) if pattern in name]
            if idxs:
                group_means = mean_over_envs[idxs]
                group_stds = std_over_envs[idxs]
                _log(f"  {group_name:<40s} {group_means.mean():12.6f} {group_stds.mean():12.6f}")
                csv_key = f"vel_error_{group_name.lower()}"
                _csv(f"{csv_key}_mean", float(group_means.mean()))
                _csv(f"{csv_key}_std", float(group_stds.mean()))

    _log("")
    _log("=" * 60)

    # Save to file
    if save_dir:
        stats_path = os.path.join(save_dir, "play_stats.txt")
        with open(stats_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nStats saved to: {stats_path}")

        # Save CSV
        if csv_rows:
            csv_path = os.path.join(save_dir, "tracking_stats.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["metric", "value"])
                writer.writeheader()
                writer.writerows(csv_rows)
            print(f"Stats saved to: {csv_path}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot trajectory data from log files')
    parser.add_argument('--log_dir', type=str, help='Specific log directory to plot (optional)')
    parser.add_argument('--trajectory_type', type=str,
                       choices=['joint', 'end_effector', 'auto'],
                       default='auto', help='Type of trajectory to plot (default: auto-detect)')
    parser.add_argument('--base_path', type=str, default='logs/play',
                       help='Base path to search for log directories (default: logs/play)')
    parser.add_argument('--start_idx', type=int, default=None,
                       help='Start timestep index for stats computation (default: 0)')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='End timestep index for stats computation (default: last)')

    args = parser.parse_args()
    
    # Find the log directory
    if args.log_dir:
        log_dir = args.log_dir
        if not os.path.exists(log_dir):
            print(f"Error: Specified log directory {log_dir} does not exist")
            return
    else:
        log_dir = find_most_recent_log_dir(args.base_path)
        if not log_dir:
            return
    
    print(f"Loading data from {log_dir}")
    # Load the data
    data = load_data(log_dir)
    # Create a directory for plots
    plot_dir = os.path.join(log_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # Determine trajectory type
    trajectory_type = None if args.trajectory_type == 'auto' else args.trajectory_type
    
    # Plot the data with specified or auto-detected trajectory type
    plot_trajectories(data, save_dir=plot_dir, trajectory_type=trajectory_type)

    # Compute and save stats
    compute_and_save_stats(data, save_dir=plot_dir,
                           start_idx=args.start_idx, end_idx=args.end_idx)


if __name__ == "__main__":
    main() 