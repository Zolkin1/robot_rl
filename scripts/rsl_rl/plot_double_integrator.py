"""Plotting utilities for the double integrator environment."""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch


def _to_numpy(tensor_list: list) -> np.ndarray:
    """Convert a list of tensors (one per timestep) to a numpy array of shape [T, N].

    Each tensor has shape [num_envs, 1]. Returns array of shape [T, num_envs].
    """
    arrays = []
    for t in tensor_list:
        if isinstance(t, torch.Tensor):
            arrays.append(t.detach().cpu().numpy().squeeze())
        else:
            arrays.append(np.asarray(t).squeeze(-1))
    return np.stack(arrays, axis=0)


def plot_double_integrator(data: dict, save_dir: str, dt: float,
                           gamma: float = 0.99, horizon: int = 24) -> None:
    """Plot position, velocity, force, and discounted return for the double integrator.

    Args:
        data: Dictionary with keys 'joint_pos', 'joint_vel', 'applied_torque', 'reward'.
              Each value is a list of tensors (one per timestep).
        save_dir: Directory to save plots.
        dt: Environment step dt (seconds per logged step).
        gamma: Discount factor for computing discounted return.
        horizon: Number of future steps for finite-horizon return. Negative for infinite horizon.
    """
    os.makedirs(save_dir, exist_ok=True)

    pos = _to_numpy(data["joint_pos"])       # [T, N]
    vel = _to_numpy(data["joint_vel"])       # [T, N]
    force = _to_numpy(data["applied_torque"])  # [T, N]

    num_steps = pos.shape[0]
    time = np.arange(num_steps) * dt

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # --- Position ---
    ax = axes[0]
    ax.plot(time, pos, alpha=0.15, color="C0", linewidth=0.5)
    ax.plot(time, pos.mean(axis=1), color="C0", linewidth=2, label="mean")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8, label="target")
    ax.set_ylabel("Position (m)")
    ax.legend(loc="upper right")
    ax.set_title("Double Integrator")

    # --- Velocity ---
    ax = axes[1]
    ax.plot(time, vel, alpha=0.15, color="C1", linewidth=0.5)
    ax.plot(time, vel.mean(axis=1), color="C1", linewidth=2, label="mean")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Velocity (m/s)")
    ax.legend(loc="upper right")

    # --- Force ---
    ax = axes[2]
    ax.plot(time, force, alpha=0.15, color="C2", linewidth=0.5)
    ax.plot(time, force.mean(axis=1), color="C2", linewidth=2, label="mean")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Force (N)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    path = os.path.join(save_dir, "double_integrator.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved double integrator plot to {path}")

    # --- Reward ---
    if "reward" in data:
        reward = _to_numpy(data["reward"])  # [T, N]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(time, reward, alpha=0.15, color="C3", linewidth=0.5)
        ax.plot(time, reward.mean(axis=1), color="C3", linewidth=2, label="mean")
        ax.set_ylabel("Reward")
        ax.set_xlabel("Time (s)")
        ax.set_title("Reward")
        ax.legend(loc="upper right")
        plt.tight_layout()
        path = os.path.join(save_dir, "reward.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved reward plot to {path}")

        # --- Discounted Return ---
        n_steps, n_envs = reward.shape
        use_finite = horizon > 0

        # Backward pass: G[t] = r[t] + gamma * G[t+1]
        discounted_return = np.zeros_like(reward)
        discounted_return[-1] = reward[-1]
        for t in range(n_steps - 2, -1, -1):
            discounted_return[t] = reward[t] + gamma * discounted_return[t + 1]

        if use_finite:
            # Subtract contributions beyond the horizon window:
            # G_H[t] = G_full[t] - gamma^H * G_full[t+H]
            gamma_h = gamma ** horizon
            g_full = discounted_return.copy()
            for t in range(n_steps - horizon - 1, -1, -1):
                discounted_return[t] = g_full[t] - gamma_h * g_full[t + horizon]

        fig, ax = plt.subplots(figsize=(10, 4))

        if use_finite:
            split_idx = max(n_steps - horizon, 0)
            # Full horizon portion (solid)
            if split_idx > 0:
                ax.plot(time[:split_idx + 1], discounted_return[:split_idx + 1],
                        alpha=0.15, color="C4", linewidth=0.5)
                ax.plot(time[:split_idx + 1], discounted_return[:split_idx + 1].mean(axis=1),
                        color="C4", linewidth=2, label="mean (full horizon)")
            # Truncated portion (dashed)
            ax.plot(time[split_idx:], discounted_return[split_idx:],
                    alpha=0.15, color="C5", linewidth=0.5, linestyle="--")
            ax.plot(time[split_idx:], discounted_return[split_idx:].mean(axis=1),
                    color="C5", linewidth=2, linestyle="--", label="mean (truncated)")
            ax.set_title(f"Discounted Return (\u03b3={gamma}, H={horizon})")
            ax.legend(loc="upper right")
        else:
            ax.plot(time, discounted_return, alpha=0.15, color="C4", linewidth=0.5)
            ax.plot(time, discounted_return.mean(axis=1), color="C4", linewidth=2, label="mean")
            ax.set_title(f"Discounted Return (\u03b3={gamma}, infinite horizon)")
            ax.legend(loc="upper right")

        ax.set_ylabel("Discounted Return")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, "discounted_return.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved discounted return plot to {path}")
