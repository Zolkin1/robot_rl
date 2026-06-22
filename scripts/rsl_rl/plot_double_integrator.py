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
                           gamma: float = 0.99, horizon: int = 100,
                           alpha: float = 0.02, a: float = 0.99,
                           b: float = 1.0) -> None:
    """Plot position, velocity, force, discounted return, and CLF metrics for the double integrator.

    Args:
        data: Dictionary with keys 'joint_pos', 'joint_vel', 'applied_torque', 'reward',
              and optionally 'v', 'vdot', 'clf_reward'. Each value is a list of tensors
              (one per timestep).
        save_dir: Directory to save plots.
        dt: Environment step dt (seconds per logged step).
        gamma: Discount factor for computing discounted return.
        horizon: Number of future steps for finite-horizon return. Negative for infinite horizon.
        alpha: CLF decay rate for computing the decay condition (vdot + alpha * v).
        a: Base for the exponential decay reference line b * a^k * ||x_0|| on the state norm plot.
        b: Constant multiplier for the exponential decay reference line.
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

    # --- State Norm ---
    state_norm = np.sqrt(pos ** 2 + vel ** 2)  # [T, N]
    x0_norm = state_norm[0].max()
    steps = np.arange(num_steps)
    decay_line = b * (a ** (steps / 2)) * x0_norm

    # Use LaTeX rendering for this figure
    rc = {
        "text.usetex": True,
        "font.family": "serif",
        "axes.titlesize": 28,
        "axes.labelsize": 32,
        "xtick.labelsize": 26,
        "ytick.labelsize": 26,
        "legend.fontsize": 24,
        "lines.linewidth": 6,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(13, 4))
        ax.plot(time, state_norm, alpha=0.15, color="C0", linewidth=0.5)
        ax.plot(time, decay_line, color="k", linestyle="--",
                label=(r"$\sqrt{\frac{(\zeta_+ + c_{\mathrm{reg}})c_2}"
                       r"{\zeta_{-}(\bar{c})\, c_1 (1 - \delta (1 - \lambda))}}"
                       r"\; q_{\bar{c}}^{\,k/2} \|x_0\|$"))
        ax.set_ylabel(r"$\|x\|$")
        ax.set_xlabel(r"Time (s)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path_png = os.path.join(save_dir, "state_norm.png")
        path_svg = os.path.join(save_dir, "state_norm.svg")
        fig.savefig(path_png, dpi=150)
        fig.savefig(path_svg)
        plt.close(fig)
    print(f"[INFO] Saved state norm plot to {path_png} and {path_svg}")

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

    # --- V and Vdot ---
    if "v" in data and "vdot" in data:
        v = _to_numpy(data["v"])        # [T, N]
        vdot = _to_numpy(data["vdot"])  # [T, N]

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # V
        ax = axes[0]
        ax.plot(time, v, alpha=0.15, color="C0", linewidth=0.5)
        ax.plot(time, v.mean(axis=1), color="C0", linewidth=2, label="mean")
        ax.set_ylabel("V")
        ax.set_title("CLF (V)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Vdot
        ax = axes[1]
        ax.plot(time, vdot, alpha=0.15, color="C1", linewidth=0.5)
        ax.plot(time, vdot.mean(axis=1), color="C1", linewidth=2, label="mean")
        ax.set_ylabel(r"$\Delta V$")
        ax.set_title(r"CLF ($\Delta V$)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # CLF Decay: vdot + alpha * v
        ax = axes[2]
        decay = vdot + alpha * v
        ax.plot(time, decay, alpha=0.15, color="C2", linewidth=0.5)
        ax.plot(time, decay.mean(axis=1), color="C2", linewidth=2, label="mean")
        ax.axhline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Decay Rate")
        ax.set_title(rf"CLF Decay ($\Delta V + \alpha V$, $\alpha$={alpha})")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(save_dir, "v_and_vdot.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved V and Vdot plot to {path}")

    # --- Discounted CLF Reward ---
    if "clf_reward" in data:
        clf_reward = _to_numpy(data["clf_reward"])  # [T, N]
        n_steps, n_envs = clf_reward.shape
        use_finite = horizon > 0

        # Forward loop: for each starting time t, sum discounted rewards
        # over the horizon. If the horizon extends past the trajectory,
        # use the last available reward value.
        H = horizon if use_finite else n_steps
        disc_clf = np.zeros_like(clf_reward)
        for t in range(n_steps):
            for k in range(H):
                idx = min(t + k, n_steps - 1)
                disc_clf[t] += (gamma ** k) * clf_reward[idx]

        fig, ax = plt.subplots(figsize=(10, 4))

        title_suffix = f"H={horizon}" if use_finite else "infinite"
        ax.plot(time, disc_clf, alpha=0.15, color="C6", linewidth=0.5)
        ax.plot(time, disc_clf.mean(axis=1), color="C6", linewidth=2, label="mean")
        ax.set_title(f"Discounted CLF Reward (\u03b3={gamma}, {title_suffix})")

        ax.set_ylabel("Discounted CLF Reward")
        ax.set_xlabel("Time (s)")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(save_dir, "discounted_clf_reward.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved discounted CLF reward plot to {path}")

    # --- Save trajectory data as npz ---
    npz_data = {"time": time, "pos": pos, "vel": vel, "force": force}
    if "v" in data:
        npz_data["v"] = _to_numpy(data["v"])
    if "vdot" in data:
        npz_data["vdot"] = _to_numpy(data["vdot"])
    if "clf_reward" in data:
        npz_data["clf_reward"] = _to_numpy(data["clf_reward"])
    path = os.path.join(save_dir, "trajectory_data.npz")
    np.savez(path, **npz_data)
    print(f"[INFO] Saved trajectory data to {path}")
