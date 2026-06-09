# Copyright 2026 Zachary Olkin. All rights reserved.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from velocity_commands import smooth_ramp
import os
def parse_g1_control_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = [
        "time", 
        "pos_x", "pos_y", "pos_z", 
        "ori_x", "ori_y", "ori_z", "ori_w"
    ]
    return df

def smooth_signal(signal, window_size):
    return np.convolve(signal, np.ones(window_size)/window_size, mode='same')

def estimate_velocity(df, smooth_window=10):
    df["vel_x"] = np.gradient(df["pos_x"], df["time"])
    df["vel_y"] = np.gradient(df["pos_y"], df["time"])
    df["vel_z"] = np.gradient(df["pos_z"], df["time"])

    # if vel_x is larger than 2, set it to 2
    df.loc[df["vel_x"] > 2, "vel_x"] = 2
    df.loc[df["vel_x"] < -2, "vel_x"] = -2

    df["vel_x"] = smooth_signal(df["vel_x"], smooth_window)
    df["vel_y"] = smooth_signal(df["vel_y"], smooth_window)
    df["vel_z"] = smooth_signal(df["vel_z"], smooth_window)

    return df

def compute_body_frame_vx(df, t_start=None, t_end=None):
    if t_start is not None:
        df = df[df["time"] >= t_start]
    if t_end is not None:
        df = df[df["time"] <= t_end]
    df = df.reset_index(drop=True)

    v_world = np.stack([df["vel_x"], df["vel_y"], df["vel_z"]], axis=1)
    quats = df[["ori_x", "ori_y", "ori_z", "ori_w"]].values
    rot = R.from_quat(quats)
    yaw = rot.as_euler('xyz', degrees=False)[:, 2]

    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    rot_mats = np.stack([
        np.stack([cos_yaw, -sin_yaw], axis=1),
        np.stack([sin_yaw,  cos_yaw], axis=1)
    ], axis=1)

    v_xy_world = v_world[:, :2][:, :, None]
    v_body = np.matmul(rot_mats, v_xy_world).squeeze(-1)
    vx_body = v_body[:, 0]

    return df["time"], vx_body

def plot_multiple_policies(file_paths, labels=None, time_ranges=None, smooth_window=10,title=None,save_path=None):
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({
        "text.usetex": True,  # Use LaTeX for all text
        "font.family": "serif",  # Use serif font (default LaTeX style)
        "font.serif": ["Computer Modern Roman"],  # LaTeX default font
    })
    fig, axes = plt.subplots(1, 1, figsize=(10, 5), sharex=True)

    if labels is None:
        labels = [f"Policy {i+1}" for i in range(len(file_paths))]
    if time_ranges is None:
        time_ranges = [(None, None)] * len(file_paths)

    for file_path, label, (t_start, t_end) in zip(file_paths, labels, time_ranges):
        df = parse_g1_control_csv(file_path)
        df = estimate_velocity(df, smooth_window)
        time, vx_body = compute_body_frame_vx(df, t_start=t_start, t_end=t_end)
        time = time - time.iloc[0]  # reset to start from 0
        plt.plot(time, vx_body, label=label, linewidth=1.5)


    # Add reference velocity ramp
    # Segments
    
#     t_pad = np.arange(0, 2, 0.01)
#     vx_pad = np.zeros_like(t_pad)

    vx_max = 0.75
    t_ref = np.linspace(0, 8, 500)
    ramp_time = 2.0
    slope = vx_max / ramp_time
    vx_ref = np.minimum(slope * t_ref, vx_max)
    #add a few more entries so that the first two seconds is 0
    #at the end just set to 0
    vx_ref[-1] = 0.0
    t_ref = np.concatenate([np.linspace(0,3.5,100),t_ref+3.5,np.linspace(t_ref[-1]+3.5,t_ref[-1]+7,100)])
    vx_ref = np.concatenate([np.zeros(100),vx_ref,np.zeros(100)])
    axes.plot(t_ref, vx_ref, "k--", label=r"$v_x^d$", linewidth=2)



    axes.set_xlabel("Time (s)", fontsize=20)
    axes.set_ylabel(r"$v_x$ (m/s)", fontsize=20)
    axes.grid(True)
    axes.legend(fontsize=20,loc="upper left")
     # bbox_to_anchor=(0.5, 0.96),
     # ncol=len(labels)+1,  # Puts all legend entries in one row
     # framealpha=0.0,
     # columnspacing=0.8,)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if title is not None:
        plt.title(title,fontsize=20)
    plt.tight_layout()

    if save_path is not None:
     os.makedirs(os.path.dirname(save_path), exist_ok=True)
     fig.savefig(save_path + ".png", bbox_inches='tight', transparent=True)
     fig.savefig(save_path + ".pdf", bbox_inches='tight', transparent=True)
     fig.savefig(save_path + ".svg", bbox_inches='tight', transparent=True)


    plt.show()


def plot_policy_subplot_comparison(hzd_paths, baseline_paths, labels=None,
                                   hzd_time_ranges=None, baseline_time_ranges=None,
                                   smooth_window=10, title=None, save_path=None):
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })
    fig, axes = plt.subplots(2, 1, figsize=(6.7, 8), sharex=True)

    if labels is None:
        labels = [f"Policy {i+1}" for i in range(len(hzd_paths))]
    if hzd_time_ranges is None:
        hzd_time_ranges = [(None, None)] * len(hzd_paths)
    if baseline_time_ranges is None:
        baseline_time_ranges = [(None, None)] * len(baseline_paths)

    # Plot HZD
    for file_path, label, (t_start, t_end) in zip(hzd_paths, labels, hzd_time_ranges):
        df = parse_g1_control_csv(file_path)
        df = estimate_velocity(df, smooth_window)
        time, vx_body = compute_body_frame_vx(df, t_start=t_start, t_end=t_end)
        time = time - time.iloc[0]
        axes[0].plot(time, vx_body, label=label, linewidth=1.5)

    # Plot Baseline
    for file_path, label, (t_start, t_end) in zip(baseline_paths, labels, baseline_time_ranges):
        df = parse_g1_control_csv(file_path)
        df = estimate_velocity(df, smooth_window)
        time, vx_body = compute_body_frame_vx(df, t_start=t_start, t_end=t_end)
        time = time - time.iloc[0]
        axes[1].plot(time, vx_body, label=label, linewidth=1.5)

    # Reference velocity
    vx_max = 0.75
    t_ref = np.linspace(0, 8, 500)
    ramp_time = 2.0
    slope = vx_max / ramp_time
    vx_ref = np.minimum(slope * t_ref, vx_max)
    vx_ref[-1] = 0.0
    t_ref = np.concatenate([np.linspace(0, 3.5, 100), t_ref + 3.5, np.linspace(t_ref[-1]+3.5, t_ref[-1]+7, 100)])
    vx_ref = np.concatenate([np.zeros(100), vx_ref, np.zeros(100)])
    for ax in axes:
        ax.plot(t_ref, vx_ref, "k--", label=r"$v_x^d$", linewidth=2)
        ax.set_ylabel(r"$v_x$ (m/s)", fontsize=20)
        ax.grid(True)
        ax.legend(fontsize=14, loc="upper left")
        ax.tick_params(axis='both', which='major', labelsize=16)

    axes[0].set_title("HZD-CLF", fontsize=20)
    axes[1].set_title("Baseline", fontsize=20)
    axes[1].set_xlabel("Time (s)", fontsize=20)
#     axes[1].text(7.7, 0.38, "Collision", fontsize=20, fontweight='bold',color='red')

    if title is not None:
        fig.suptitle(title, fontsize=22)

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # leave space for suptitle

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path + ".png", bbox_inches='tight')
        fig.savefig(save_path + ".pdf", bbox_inches='tight', transparent=True)
        fig.savefig(save_path + ".svg", bbox_inches='tight', transparent=True)

    plt.show()


def plot_global_position(file_paths, labels=None, time_ranges=None, smooth_window=10, title=None, save_path=None):
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
    })

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if labels is None:
        labels = [f"Policy {i+1}" for i in range(len(file_paths))]
    if time_ranges is None:
        time_ranges = [(None, None)] * len(file_paths)

    for file_path, label, (t_start, t_end) in zip(file_paths, labels, time_ranges):
        df = parse_g1_control_csv(file_path)

        if t_start is not None:
            df = df[df["time"] >= t_start]
        if t_end is not None:
            df = df[df["time"] <= t_end]
        df = df.reset_index(drop=True)

        pos_x = df["pos_x"]
        pos_y = df["pos_y"]
     #    pos_x = smooth_signal(df["pos_x"], smooth_window)
     #    pos_y = smooth_signal(df["pos_y"], smooth_window)

        ax.plot(pos_x, pos_y, label=label, linewidth=1.5)

    ax.set_xlabel(r"$x$ (m)", fontsize=20)
    ax.set_ylabel(r"$y$ (m)", fontsize=20)
    ax.grid(True)
    ax.legend(fontsize=18, loc="best")
    ax.set_aspect("equal", adjustable="box")

    if title is not None:
        ax.set_title(title, fontsize=22)

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path + ".png", bbox_inches='tight', transparent=True)
        fig.savefig(save_path + ".pdf", bbox_inches='tight', transparent=True)
        fig.savefig(save_path + ".svg", bbox_inches='tight', transparent=True)

    plt.show()



# === Updated Example Usage ===
if __name__ == "__main__":
    hzd_file_paths = [
        "experiments/hardware_logs/g1_control_hzd_v2.csv",
        "experiments/hardware_logs/g1_control_hzd_1.765_v2.csv",
        "experiments/hardware_logs/g1_control_hzd_3.55.csv",
    ]

    baseline_file_paths = [
        "experiments/hardware_logs/g1_control_baseline.csv",
        "experiments/hardware_logs/g1_control_baseline_1.765.csv",
        "experiments/hardware_logs/g1_control_baseline_3.55.csv",
    ]

    labels = [
        "0 kg",
        "1.765 kg",
        "3.55 kg",
    ]

    hzd_time_ranges = [
        (7.5, 22.5),
        (11, 26),
        (48, 63)
    ]

    baseline_time_ranges = [
        (124, 139),
        (88, 103),
        (108.5, 116),
    ]

    plot_policy_subplot_comparison(
        hzd_paths=hzd_file_paths,
        baseline_paths=baseline_file_paths,
        labels=labels,
        hzd_time_ranges=hzd_time_ranges,
        baseline_time_ranges=baseline_time_ranges,
        smooth_window=10,
        title="",
        save_path="experiments/plots/hardware_hzd_vs_baseline"
    )

#     file_paths = [
#         "experiments/hardware_logs/g1_control_lip_v2.csv",
#         "experiments/hardware_logs/g1_control_baseline.csv",
#     ]

#     labels = [
#         "LIP",
#         "Baseline",
#     ]
    
#     time_ranges = [
#         (110, 125),
#         (124, 139),
#     ]

#     plot_multiple_policies(
#           file_paths=file_paths,
#           labels=labels,
#           time_ranges=time_ranges,
#           smooth_window=10,
#           title=""
#      )


#     plot_global_position(
#         file_paths=baseline_file_paths,
#         labels=labels,
#         time_ranges=baseline_time_ranges,
#         smooth_window=10,
#         title="Global Position Trajectory",
#         save_path="experiments/plots/hardware_global_traj_baseline"
#     )


# # === Example Usage ===
# if __name__ == "__main__":
#     hzd_file_paths = [
#         "experiments/hardware_logs/g1_control_hzd_v2.csv",
#         "experiments/hardware_logs/g1_control_hzd_1.765_v2.csv",
#         "experiments/hardware_logs/g1_control_hzd_3.55.csv",
#     ]

#     baseline_file_paths = [
#         "experiments/hardware_logs/g1_control_baseline.csv",
#         "experiments/hardware_logs/g1_control_baseline_1.765.csv",
#         "experiments/hardware_logs/g1_control_baseline_3.55.csv",
#     ]
#     labels = [
#         "0 kg",
#         "1.765 kg",
#         "3.55 kg",
#     ]
#     hzd_time_ranges = [
#      (7.5, 22.5),   # for first policy
#      (11, 26),
#         (48, 63)
#     ]

#     baseline_time_ranges = [
#           (124,148),
#         (88, 100),
#         (108.5, 130),
#     ]
#     smooth_window = 10

#     plot_multiple_policies(
#         file_paths=file_paths,
#         labels=labels,
#         time_ranges=time_ranges,
#         smooth_window=smooth_window,
#         title="HZD-CLF",
#         save_path="experiments/plots/hardware_hzd_clf"
#     )
