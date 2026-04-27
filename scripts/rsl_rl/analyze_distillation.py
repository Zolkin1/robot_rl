"""Distillation analysis: study how teacher and student behave during distillation.

Two stages:
  1. Collect rollouts driven by the student and/or the teacher. At every step,
     query *both* policies and log their observations + actions. When the student
     drives, this is on-policy DAgger style; when the teacher drives, this is
     off-policy distillation style.
  2. Run k-NN with Mahalanobis-whitened observations on the transformer student's
     history space, then plot k-NN obs distance vs action distance, with and
     without the velocity command included in the obs. Episode-aware tagging
     separates "neighbor in same episode within W steps" from "cross-episode
     neighbor" so the cross-episode scatter is the primary diagnostic.

Hard requirements: distillation cfg only, transformer student only.
"""

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
import robot_rl  # noqa: F401 - register gym envs
from isaaclab_tasks.utils import add_launcher_args, get_checkpoint_path, launch_simulation
from isaaclab_tasks.utils.hydra import hydra_task_config

import cli_args  # isort: skip

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Analyze a distilled policy by comparing teacher and student "
                "rollouts and the local smoothness of the action map in obs space."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--rollout", choices=["student", "teacher", "both"], default="both",
    help="Which policy drives the env. 'both' runs two sequential rollouts.",
)
parser.add_argument("--num_steps", type=int, default=1000, help="Steps per rollout.")
parser.add_argument("--k", type=int, default=10, help="k for k-nearest-neighbors.")
parser.add_argument(
    "--num_queries", type=int, default=1000,
    help="Sampled query points for k-NN (full N x N is too expensive).",
)
parser.add_argument(
    "--episode_window", type=int, default=10,
    help="Steps within same episode considered 'near' (W).",
)
parser.add_argument(
    "--vel_buckets", type=str, default="0,0.1,1.5,3.7",
    help="Comma list of velocity-command bucket edges (on |vx|).",
)
parser.add_argument(
    "--from_data", nargs="?", const="<auto>", default=None,
    help="Skip Stage 1 and analyze existing .npz files. Pass a directory to point "
         "at a specific dataset, or pass the flag with no value to use the "
         "<log_dir>/distillation_analysis folder of the just-loaded checkpoint.",
)
parser.add_argument(
    "--output_dir", type=str, default=None,
    help="Where to put data + plots. Default: <log_dir>/distillation_analysis.",
)
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

installed_version = metadata.version("rsl-rl-lib")


# ---------------------------------------------------------------------------
# Stage-1 helpers (need the simulator)
# ---------------------------------------------------------------------------
@dataclass
class CollectionMeta:
    """Metadata captured once at the start of collection.

    Used by the analysis stage to reshape histories into [N, H, D] tokens and
    to find the velocity-command slice within a single token.
    """
    history_length: int
    single_obs_dim: int
    student_term_names: list[str]
    student_term_dims_per_step: list[int]   # per-step dim for each student term
    teacher_obs_dim: int
    n_actions: int
    vel_token_start: int
    vel_token_end: int


def _build_collection_meta(env, runner) -> CollectionMeta:
    """Read history length, term layout, and velocity-command slice from the env.

    The velocity slice is computed from the env's observation manager so the
    script survives obs-config edits without code changes.
    """
    mgr = env.unwrapped.observation_manager
    student_term_names: list[str] = list(mgr.active_terms["student"])
    student_term_dims_flat = [int(td[0]) for td in mgr.group_obs_term_dim["student"]]

    transformer = runner.alg.student.mlp
    history_length = int(transformer.history_length)
    single_obs_dim = int(transformer.single_obs_dim)

    per_step = [d // history_length for d in student_term_dims_flat]
    if sum(per_step) != single_obs_dim:
        raise RuntimeError(
            f"Per-step dims {per_step} (sum={sum(per_step)}) inconsistent with "
            f"transformer single_obs_dim={single_obs_dim}."
        )

    if "velocity_commands" not in student_term_names:
        raise RuntimeError(
            f"Could not find 'velocity_commands' in student obs terms: {student_term_names}. "
            "Update the script if the term has been renamed."
        )
    vel_idx = student_term_names.index("velocity_commands")
    vel_token_start = sum(per_step[:vel_idx])
    vel_token_end = vel_token_start + per_step[vel_idx]

    teacher_groups = list(runner.alg.teacher.obs_groups)
    teacher_obs_dim = int(sum(env.get_observations()[g].shape[-1] for g in teacher_groups))

    return CollectionMeta(
        history_length=history_length,
        single_obs_dim=single_obs_dim,
        student_term_names=student_term_names,
        student_term_dims_per_step=per_step,
        teacher_obs_dim=teacher_obs_dim,
        n_actions=int(env.num_actions),
        vel_token_start=vel_token_start,
        vel_token_end=vel_token_end,
    )


def _gather_obs(obs, groups: list[str]) -> torch.Tensor:
    """Concat raw (pre-normalization) obs slices the given model would consume."""
    return torch.cat([obs[g] for g in groups], dim=-1)


def collect_rollout(
    env,
    runner,
    driver: str,
    num_steps: int,
    meta: CollectionMeta,
) -> dict[str, np.ndarray]:
    """Run one rollout driven by `driver` ('student' or 'teacher'); query both policies.

    Returns a flat dict of np arrays of shape [num_steps * num_envs, ...].
    """
    student = runner.alg.student
    teacher = runner.alg.teacher
    student_groups = list(student.obs_groups)
    teacher_groups = list(teacher.obs_groups)

    n_envs = env.num_envs
    device = env.unwrapped.device

    obs = env.get_observations()
    # Distinct ids per env so step-0 observations from different envs are NOT
    # treated as same-episode neighbors during k-NN tagging.
    episode_id = torch.arange(n_envs, dtype=torch.int64, device=device)
    step_in_ep = torch.zeros(n_envs, dtype=torch.int64, device=device)
    next_episode_id = int(n_envs)  # fresh ids handed out as envs reset

    buf: dict[str, list[torch.Tensor]] = {
        "student_obs": [], "teacher_obs": [],
        "student_action": [], "teacher_action": [],
        "velocity_command": [],
        "episode_id": [], "step_in_episode": [],
    }

    print(f"[INFO] Collecting {num_steps} steps with driver='{driver}' on {n_envs} envs...")
    for step in range(num_steps):
        with torch.inference_mode():
            student_act = student(obs)
            teacher_act = teacher(obs)
            student_obs = _gather_obs(obs, student_groups).detach().clone()
            teacher_obs = _gather_obs(obs, teacher_groups).detach().clone()
            vel_cmd = env.unwrapped.command_manager.get_command("base_velocity").detach().clone()

        buf["student_obs"].append(student_obs.cpu())
        buf["teacher_obs"].append(teacher_obs.cpu())
        buf["student_action"].append(student_act.detach().cpu())
        buf["teacher_action"].append(teacher_act.detach().cpu())
        buf["velocity_command"].append(vel_cmd.cpu())
        buf["episode_id"].append(episode_id.detach().cpu().clone())
        buf["step_in_episode"].append(step_in_ep.detach().cpu().clone())

        drive_action = student_act if driver == "student" else teacher_act
        with torch.inference_mode():
            obs, _, dones, _ = env.step(drive_action)

        # episode bookkeeping: envs that finished step at done get a fresh id and
        # a reset step counter for the *next* observation, which corresponds to
        # the post-reset state.
        done_mask = dones.to(torch.bool)
        n_done = int(done_mask.sum().item())
        if n_done > 0:
            new_ids = torch.arange(next_episode_id, next_episode_id + n_done,
                                   device=device, dtype=torch.int64)
            episode_id = episode_id.clone()
            episode_id[done_mask] = new_ids
            next_episode_id += n_done
            step_in_ep = step_in_ep.clone()
            step_in_ep[done_mask] = 0
            step_in_ep[~done_mask] += 1
        else:
            step_in_ep = step_in_ep + 1

        if (step + 1) % 100 == 0:
            print(f"  step {step + 1}/{num_steps}")

    out = {k: torch.stack(v, dim=0).numpy() for k, v in buf.items()}  # [T, N, ...]
    # Flatten time and env into one sample axis: [T*N, ...]
    flat = {}
    for k, v in out.items():
        if v.ndim == 1:  # shouldn't happen, but be defensive
            flat[k] = v
        else:
            flat[k] = v.reshape(-1, *v.shape[2:]) if v.ndim >= 3 else v.reshape(-1)
    # Reshape student_obs to [N, H, D] for clarity; analysis flattens as needed.
    flat["student_obs"] = flat["student_obs"].reshape(
        -1, meta.history_length, meta.single_obs_dim
    )
    return flat


# ---------------------------------------------------------------------------
# Stage-2: analysis
# ---------------------------------------------------------------------------
def _mahalanobis_whiten(H: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Cholesky-based whitening: returns H_white where L2(H_white) = Mahalanobis(H).

    H is [N, F]. Σ + eps*I is Cholesky-factored to make the whitening O(F^2).
    """
    from scipy.linalg import solve_triangular

    mu = H.mean(axis=0)
    Sigma = np.cov(H, rowvar=False)
    Sigma_reg = Sigma + eps * np.eye(Sigma.shape[0], dtype=Sigma.dtype)
    L = np.linalg.cholesky(Sigma_reg)
    H_white = solve_triangular(L, (H - mu).T, lower=True).T
    return H_white


def _check_whitening(H_white: np.ndarray) -> tuple[float, float]:
    """Sanity stats: mean of per-dim std (≈1 ideal) and max abs off-diagonal corr (≈0 ideal)."""
    std_mean = float(H_white.std(axis=0).mean())
    sub = H_white if H_white.shape[1] <= 256 else H_white[:, :256]
    corr = np.corrcoef(sub.T)
    np.fill_diagonal(corr, 0.0)
    return std_mean, float(np.max(np.abs(corr)))


@dataclass
class KNNResult:
    """All per-pair quantities needed for plotting, flat across (query, neighbor)."""
    obs_dist: np.ndarray
    student_act_dist: np.ndarray
    teacher_act_dist: np.ndarray
    category: np.ndarray  # 0=same-ep-near, 1=same-ep-far, 2=cross-ep
    query_vx: np.ndarray  # |vx| of the query point


CAT_LABELS = ("same-ep-near", "same-ep-far", "cross-ep")
CAT_COLORS = ("#4c72b0", "#dd8452", "#55a868")  # blue, orange, green


def run_knn(
    student_obs_flat: np.ndarray,        # [N, F]
    student_actions: np.ndarray,         # [N, A]
    teacher_actions: np.ndarray,         # [N, A]
    velocity_command: np.ndarray,        # [N, 3]
    episode_id: np.ndarray,              # [N]
    step_in_episode: np.ndarray,         # [N]
    k: int,
    num_queries: int,
    episode_window: int,
    rng: np.random.Generator,
) -> KNNResult:
    """Whiten, k-NN-search, episode-tag, and accumulate per-pair distances."""
    from sklearn.neighbors import NearestNeighbors

    print(f"  whitening {student_obs_flat.shape}...")
    H_white = _mahalanobis_whiten(student_obs_flat)
    std_mean, max_offdiag = _check_whitening(H_white)
    print(f"  whitening sanity: per-dim std mean={std_mean:.3f} (≈1.0), "
          f"max off-diag corr={max_offdiag:.3f} (≈0.0)")

    print(f"  building k-NN index (n_neighbors={k + 1})...")
    nn_index = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(H_white)

    n_total = H_white.shape[0]
    n_q = min(num_queries, n_total)
    query_idx = rng.choice(n_total, size=n_q, replace=False)
    print(f"  querying {n_q} points...")
    dists, neigh_idx = nn_index.kneighbors(H_white[query_idx])
    # drop self (first column)
    dists = dists[:, 1:]
    neigh_idx = neigh_idx[:, 1:]

    q_rep = np.repeat(query_idx, k)              # [n_q * k]
    n_flat = neigh_idx.reshape(-1)               # [n_q * k]

    obs_dist = dists.reshape(-1)
    student_act_dist = np.linalg.norm(
        student_actions[q_rep] - student_actions[n_flat], axis=-1
    )
    teacher_act_dist = np.linalg.norm(
        teacher_actions[q_rep] - teacher_actions[n_flat], axis=-1
    )

    same_ep = episode_id[q_rep] == episode_id[n_flat]
    step_diff = np.abs(step_in_episode[q_rep].astype(np.int64) - step_in_episode[n_flat].astype(np.int64))
    near = same_ep & (step_diff <= episode_window)
    far = same_ep & (step_diff > episode_window)
    category = np.full(obs_dist.shape, 2, dtype=np.int8)  # default cross-ep
    category[far] = 1
    category[near] = 0

    query_vx = np.abs(velocity_command[q_rep, 0])

    return KNNResult(
        obs_dist=obs_dist,
        student_act_dist=student_act_dist,
        teacher_act_dist=teacher_act_dist,
        category=category,
        query_vx=query_vx,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _scatter_by_category(ax, x, y, category, title, xlabel, ylabel, max_pts=20000):
    """Scatter points colored by pair category; downsample for legibility."""
    n = len(x)
    if n > max_pts:
        idx = np.random.default_rng(0).choice(n, size=max_pts, replace=False)
        x, y, category = x[idx], y[idx], category[idx]
    for c, label, color in zip((2, 1, 0), ("cross-ep", "same-ep-far", "same-ep-near"),
                               (CAT_COLORS[2], CAT_COLORS[1], CAT_COLORS[0])):
        m = category == c
        if not np.any(m):
            continue
        ax.scatter(x[m], y[m], s=4, alpha=0.35, c=color, label=label, edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3)


def make_plots(
    res: KNNResult,
    label: str,                # e.g. "student_vel_kept"
    out_dir: str,
    vel_buckets: list[float],
):
    """Write the four diagnostic figures for one (dataset, vel-mode) combo."""
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # Plot 1: scatter, student-action distance
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_by_category(
        ax, res.obs_dist, res.student_act_dist, res.category,
        title=f"{label}: obs vs student-action distance",
        xlabel="Mahalanobis student-obs distance",
        ylabel="L2 student action distance",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label}_scatter_student.png"), dpi=130)
    plt.close(fig)

    # Plot 2: scatter, teacher-action distance
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_by_category(
        ax, res.obs_dist, res.teacher_act_dist, res.category,
        title=f"{label}: obs vs teacher-action distance",
        xlabel="Mahalanobis student-obs distance",
        ylabel="L2 teacher action distance",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label}_scatter_teacher.png"), dpi=130)
    plt.close(fig)

    # Plot 3: histogram of obs distances (per category)
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, lbl, color in zip((0, 1, 2), CAT_LABELS, CAT_COLORS):
        m = res.category == c
        if np.any(m):
            ax.hist(res.obs_dist[m], bins=60, alpha=0.55, color=color, label=lbl)
    ax.set_xlabel("Mahalanobis student-obs distance")
    ax.set_ylabel("count")
    ax.set_title(f"{label}: k-NN obs distance distribution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label}_obsdist_hist.png"), dpi=130)
    plt.close(fig)

    # Plot 4: per-velocity-bucket grid
    n_buckets = len(vel_buckets) - 1
    fig, axes = plt.subplots(n_buckets, 2, figsize=(11, 3.5 * n_buckets), squeeze=False)
    for i in range(n_buckets):
        lo, hi = vel_buckets[i], vel_buckets[i + 1]
        m_bucket = (res.query_vx >= lo) & (res.query_vx < hi)
        bucket_label = f"|vx|∈[{lo:.2f},{hi:.2f})  n={int(m_bucket.sum())}"
        _scatter_by_category(
            axes[i, 0], res.obs_dist[m_bucket], res.student_act_dist[m_bucket],
            res.category[m_bucket],
            title=f"{bucket_label}  student",
            xlabel="Mahalanobis student-obs distance",
            ylabel="L2 student action distance",
        )
        _scatter_by_category(
            axes[i, 1], res.obs_dist[m_bucket], res.teacher_act_dist[m_bucket],
            res.category[m_bucket],
            title=f"{bucket_label}  teacher",
            xlabel="Mahalanobis student-obs distance",
            ylabel="L2 teacher action distance",
        )
    fig.suptitle(f"{label}: per velocity bucket", y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label}_per_bucket.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def _save_dataset(path: str, data: dict[str, np.ndarray], meta: CollectionMeta):
    """Persist a collected rollout + metadata to a single .npz file."""
    np.savez_compressed(
        path,
        student_obs=data["student_obs"],
        teacher_obs=data["teacher_obs"],
        student_action=data["student_action"],
        teacher_action=data["teacher_action"],
        velocity_command=data["velocity_command"],
        episode_id=data["episode_id"],
        step_in_episode=data["step_in_episode"],
        history_length=np.int64(meta.history_length),
        single_obs_dim=np.int64(meta.single_obs_dim),
        teacher_obs_dim=np.int64(meta.teacher_obs_dim),
        n_actions=np.int64(meta.n_actions),
        vel_token_start=np.int64(meta.vel_token_start),
        vel_token_end=np.int64(meta.vel_token_end),
        student_term_names=np.array(meta.student_term_names),
        student_term_dims_per_step=np.array(meta.student_term_dims_per_step),
    )


def _load_dataset(path: str) -> tuple[dict[str, np.ndarray], CollectionMeta]:
    """Load a .npz produced by `_save_dataset`."""
    z = np.load(path, allow_pickle=False)
    data = {
        "student_obs": z["student_obs"],
        "teacher_obs": z["teacher_obs"],
        "student_action": z["student_action"],
        "teacher_action": z["teacher_action"],
        "velocity_command": z["velocity_command"],
        "episode_id": z["episode_id"],
        "step_in_episode": z["step_in_episode"],
    }
    meta = CollectionMeta(
        history_length=int(z["history_length"]),
        single_obs_dim=int(z["single_obs_dim"]),
        student_term_names=[str(x) for x in z["student_term_names"]],
        student_term_dims_per_step=[int(x) for x in z["student_term_dims_per_step"]],
        teacher_obs_dim=int(z["teacher_obs_dim"]),
        n_actions=int(z["n_actions"]),
        vel_token_start=int(z["vel_token_start"]),
        vel_token_end=int(z["vel_token_end"]),
    )
    return data, meta


def _drop_velocity(student_obs: np.ndarray, meta: CollectionMeta) -> np.ndarray:
    """Zero out the velocity-command slice on every token, then flatten to [N, H*D]."""
    out = student_obs.copy()  # [N, H, D]
    out[:, :, meta.vel_token_start:meta.vel_token_end] = 0.0
    return out.reshape(out.shape[0], -1)


def analyze_dataset(
    rollout_label: str,
    data: dict[str, np.ndarray],
    meta: CollectionMeta,
    out_root: str,
    args,
):
    """Run vel-kept + vel-removed analyses for one rollout dataset."""
    rng = np.random.default_rng(0)
    vel_buckets = [float(x) for x in args.vel_buckets.split(",")]
    plots_dir = os.path.join(out_root, "plots")

    student_obs = data["student_obs"]  # [N, H, D]

    for vel_mode, flat_obs in (
        ("vel_kept", student_obs.reshape(student_obs.shape[0], -1)),
        ("vel_removed", _drop_velocity(student_obs, meta)),
    ):
        label = f"{rollout_label}_{vel_mode}"
        print(f"[INFO] Analyzing {label} (N={flat_obs.shape[0]}, F={flat_obs.shape[1]})...")
        res = run_knn(
            student_obs_flat=flat_obs.astype(np.float64),
            student_actions=data["student_action"].astype(np.float64),
            teacher_actions=data["teacher_action"].astype(np.float64),
            velocity_command=data["velocity_command"].astype(np.float64),
            episode_id=data["episode_id"],
            step_in_episode=data["step_in_episode"],
            k=args.k,
            num_queries=args.num_queries,
            episode_window=args.episode_window,
            rng=rng,
        )
        make_plots(res, label, plots_dir, vel_buckets)


def _resolve_resume_path(agent_cfg: RslRlBaseRunnerCfg) -> str:
    """Resolve the checkpoint path the same way play.py does, without the simulator."""
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    if getattr(args_cli, "teacher_experiment", None):
        play_log_path = os.path.abspath(os.path.join("logs", "rsl_rl", args_cli.teacher_experiment))
    else:
        play_log_path = log_root_path
    return get_checkpoint_path(play_log_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _analyze_only(out_root: str):
    """Load any data_<rollout>.npz files in out_root and run Stage 2 on each."""
    rollout_files = []
    for r in ("student", "teacher"):
        p = os.path.join(out_root, f"data_{r}.npz")
        if os.path.exists(p):
            rollout_files.append((r, p))
    if not rollout_files:
        raise FileNotFoundError(
            f"No data_student.npz or data_teacher.npz found in {out_root}."
        )
    for r, p in rollout_files:
        print(f"[INFO] Loading {p}...")
        data, meta = _load_dataset(p)
        analyze_dataset(r, data, meta, out_root, args_cli)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Entry point. Either collect rollouts then analyze, or analyze existing data."""
    # Resolve the data source. With --from_data and an explicit path, jump in.
    # With --from_data and no value, derive the default analysis dir from the
    # checkpoint that the loaded cfg points at -- no simulator needed for that.
    if args_cli.from_data is not None and args_cli.from_data != "<auto>":
        _analyze_only(args_cli.from_data)
        return

    if agent_cfg.class_name != "DistillationRunner":
        raise ValueError(
            f"analyze_distillation requires a DistillationRunner config; got "
            f"{agent_cfg.class_name}."
        )

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    resume_path = _resolve_resume_path(agent_cfg)
    log_dir = os.path.dirname(resume_path)
    out_root = args_cli.output_dir or os.path.join(log_dir, "distillation_analysis")

    if args_cli.from_data == "<auto>":
        print(f"[INFO] Resolved analysis dir from checkpoint: {out_root}")
        _analyze_only(out_root)
        return

    with launch_simulation(env_cfg, args_cli):
        # Recursive contact sensor patch (matches play.py)
        import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: F401

        from rsl_rl.runners import DistillationRunner

        from robot_rl.network.transformer_network import CausalTransformer

        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.seed = agent_cfg.seed
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        env_cfg.log_dir = log_dir

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path)

        if not isinstance(runner.alg.student.mlp, CausalTransformer):
            raise TypeError(
                "analyze_distillation requires a transformer student "
                f"(robot_rl.network.transformer_network.CausalTransformer); "
                f"got {type(runner.alg.student.mlp).__name__}. "
                "Use a transformer distillation cfg."
            )

        runner.alg.eval_mode()
        meta = _build_collection_meta(env, runner)
        print(f"[INFO] Student token layout: H={meta.history_length}, D={meta.single_obs_dim}, "
              f"velocity slice in token = [{meta.vel_token_start}:{meta.vel_token_end})")

        os.makedirs(out_root, exist_ok=True)

        rollouts = (["student", "teacher"] if args_cli.rollout == "both" else [args_cli.rollout])
        collected: list[tuple[str, dict[str, np.ndarray]]] = []
        for r in rollouts:
            data = collect_rollout(env, runner, r, args_cli.num_steps, meta)
            path = os.path.join(out_root, f"data_{r}.npz")
            _save_dataset(path, data, meta)
            print(f"[INFO] Saved {path}: student_obs shape {data['student_obs'].shape}")
            collected.append((r, data))

        env.close()

    for r, data in collected:
        analyze_dataset(r, data, meta, out_root, args_cli)


if __name__ == "__main__":
    main()
