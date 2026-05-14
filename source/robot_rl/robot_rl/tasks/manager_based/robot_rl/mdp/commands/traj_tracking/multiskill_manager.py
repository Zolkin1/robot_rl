"""MultiSkillManager: batched trajectory management across multiple skills.

Replaces the loop-per-trajectory pattern in LibraryManager with fully batched
tensor operations. Supports multi-dimensional conditioning (velocity, terrain).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor

from .manager_base import ManagerBase
from .sagittal_reflector import swap_left_right
from .trajectory_manager import TrajectoryManager, TrajectoryType


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConditionerData:
    """Structured conditioning information parsed from a trajectory YAML."""

    vel_x: float = 0.0
    vel_y: float = 0.0
    vel_yaw: float = 0.0
    terrain: str = "flat"
    terrain_width: float = 0.0
    terrain_height: float = 0.0
    terrain_length: float = 0.0

    def to_continuous_tensor(self) -> list[float]:
        """Return the continuous conditioning dimensions as a flat list."""
        return [
            self.vel_x, self.vel_y, self.vel_yaw,
            self.terrain_width, self.terrain_height, self.terrain_length,
        ]


@dataclass
class SkillData:
    """Metadata for a single skill (a group of related trajectories)."""

    name: str
    traj_indices: list[int] = field(default_factory=list)
    num_trajectories: int = 0
    has_terrain: bool = False
    conditioning_tensor: Tensor | None = None  # [num_traj, C] continuous dims


# ---------------------------------------------------------------------------
# Trajectory type int encoding (matches TrajectoryType enum order)
# ---------------------------------------------------------------------------

_TRAJ_TYPE_TO_INT = {
    TrajectoryType.HALF_PERIODIC: 0,
    TrajectoryType.FULL_PERIODIC: 1,
    TrajectoryType.EPISODIC: 2,
    TrajectoryType.PERPETUAL: 3,
}

_HALF_PERIODIC_INT = _TRAJ_TYPE_TO_INT[TrajectoryType.HALF_PERIODIC]
_FULL_PERIODIC_INT = _TRAJ_TYPE_TO_INT[TrajectoryType.FULL_PERIODIC]
_EPISODIC_INT = _TRAJ_TYPE_TO_INT[TrajectoryType.EPISODIC]
_PERPETUAL_INT = _TRAJ_TYPE_TO_INT[TrajectoryType.PERPETUAL]


# ---------------------------------------------------------------------------
# MultiSkillManager
# ---------------------------------------------------------------------------
class MultiSkillManager(ManagerBase):
    """Manages trajectories across multiple skills with fully batched evaluation.

    All trajectory data is flattened into global tensors so that every method
    (get_output, get_contact_state, …) runs as a single batched operation
    over all environments — no per-trajectory Python loops at runtime.

    Skills are groups of trajectories that share a semantic label (e.g.
    "walking", "running", "stairs").  Within each skill, trajectory selection
    uses nearest-neighbour lookup on continuous conditioning dimensions
    (vel_x, vel_y, vel_yaw, terrain dims).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        path: str,
        device: str | torch.device,
        env: Any = None,
        conditioner_generator_name: str | None = None,
        hf_repo: str | None = None,
    ):
        """Load trajectory YAMLs from a top-level folder containing per-skill subfolders.

        The expected layout is::

            path/
            ├── walking/
            │   ├── walk_20.yaml
            │   └── walk_40.yaml
            ├── running/
            │   ├── run_160.yaml
            │   └── run_180.yaml
            └── ...

        Each immediate subdirectory of ``path`` becomes a skill, named after
        the subdirectory.  The velocity command's ``_skill_list`` may be a
        strict subset of the loaded skills — extras are loaded into the
        pool but never sampled (see :meth:`_lazy_init_skill_filter`).

        Args:
            path: Top-level folder containing one subfolder per skill.
                If ``hf_repo`` is set, this is relative to the HF repo root.
            device: Torch device.
            env: IsaacLab environment (needed for conditioner queries at
                runtime).  Can be ``None`` for offline / test usage.
            conditioner_generator_name: Name of the command term that provides
                the conditioning signal at runtime.
            hf_repo: Optional HuggingFace repo ID (e.g. ``"zolkin/robot_rl"``).
                When set, ``path`` is treated as a path within the repo and
                downloaded to a local ``hf/`` cache directory.
        """
        self.device = torch.device(device)
        self.env = env
        self.conditioner_generator_name = conditioner_generator_name

        # --- Discover skill folders from subdirectories --------------------
        skill_folders = self._discover_skill_folders(path, hf_repo)

        # --- Load all trajectories via TrajectoryManager (then discard) ----
        managers: list[TrajectoryManager] = []
        skill_labels: list[str] = []
        conditioner_datas: list[ConditionerData] = []

        for skill_name, folder_path in skill_folders.items():
            yaml_paths = self._find_yaml_files(str(folder_path))
            for yp in yaml_paths:
                mgr = TrajectoryManager(str(yp), None, device)
                managers.append(mgr)
                skill_labels.append(skill_name)
                conditioner_datas.append(
                    self._parse_conditioner(mgr.traj_data.conditioner, skill_name)
                )

        if len(managers) == 0:
            raise ValueError("No trajectory YAML files found in any skill folder.")

        # --- Sort trajectories by conditioner within each skill ------------
        # This matches LibraryManager's sorting convention and ensures
        # consistent indexing.
        combined = list(zip(managers, skill_labels, conditioner_datas))
        combined.sort(key=lambda x: (x[1], x[2].vel_x))
        managers = [c[0] for c in combined]
        skill_labels = [c[1] for c in combined]
        conditioner_datas = [c[2] for c in combined]

        # --- Capture per-trajectory names (for stats reporting) ------------
        # Stored in the same global order as ``self._global_conditioning``.
        self.trajectory_names: list[str] = [m.traj_data.name for m in managers]
        self.skill_labels: list[str] = list(skill_labels)

        # --- Verify compatibility across all trajectories ------------------
        ref = managers[0]
        self.pos_output_names = list(ref.traj_data.pos_output_names)
        self.vel_output_names = list(ref.traj_data.vel_output_names)
        self.num_pos_outputs = ref.traj_data.num_pos_outputs
        self.num_vel_outputs = ref.traj_data.num_vel_outputs
        self.spline_order = ref.traj_data.spline_order
        self.ref_frames = list(ref.traj_data.reference_frames)

        for i, mgr in enumerate(managers):
            td = mgr.traj_data
            if td.num_pos_outputs != self.num_pos_outputs:
                raise ValueError(
                    f"Trajectory {td.name} has {td.num_pos_outputs} pos outputs, "
                    f"expected {self.num_pos_outputs}."
                )
            if td.num_vel_outputs != self.num_vel_outputs:
                raise ValueError(
                    f"Trajectory {td.name} has {td.num_vel_outputs} vel outputs, "
                    f"expected {self.num_vel_outputs}."
                )
            if td.pos_output_names != self.pos_output_names:
                raise ValueError(
                    f"Trajectory {td.name} pos output names mismatch.\n"
                    f"Expected: {self.pos_output_names}\nGot: {td.pos_output_names}"
                )
            if td.vel_output_names != self.vel_output_names:
                raise ValueError(
                    f"Trajectory {td.name} vel output names mismatch.\n"
                    f"Expected: {self.vel_output_names}\nGot: {td.vel_output_names}"
                )

        # --- Build global flat tensors from managers -----------------------
        self.num_trajectories = len(managers)
        self._build_global_tensors(managers, skill_labels, conditioner_datas)

        # --- Build per-skill metadata --------------------------------------
        self._build_skill_data(skill_labels, conditioner_datas)

        # --- Skill filter state (lazy-init at first selection) -------------
        # ``_traj_skill_idx`` is the per-trajectory skill index (built by
        # ``_build_global_tensors`` at ``data["skill_idx"]``). Cached here as a
        # plain Tensor for the per-env skill mask in ``_select_trajectories``.
        # The vel→traj skill-id gather and the alignment check are deferred to
        # ``_lazy_init_skill_filter`` because the conditioner command term may
        # not exist on ``env.command_manager`` at manager-construction time.
        self._traj_skill_idx: Tensor = self.data["skill_idx"].to(self.device)
        self._vel_skill_to_traj_skill: Tensor | None = None
        self._skill_filter_ready: bool = False

        # Owner that exposes the per-env active ``skill_id`` tensor and the
        # ordered ``_skill_list`` for the skill-filter validation.  Set by
        # :meth:`set_skill_owner`, typically called from the owning
        # trajectory cmd's ``_post_init``.  Lazy-init runs against this
        # owner instead of the velocity (conditioner) term — the velocity
        # term is still used as the ``command`` source via
        # ``conditioner_generator_name``.
        self._skill_owner: Any = None

        # --- Pre-compute binomial coefficients for Bezier evaluation -------
        self._binomial_coeffs: dict[int, Tensor] = {}
        for d in range(self.spline_order + 1):
            self._binomial_coeffs[d] = torch.tensor(
                [math.comb(d, i) for i in range(d + 1)],
                dtype=torch.float32, device=self.device,
            )

        # --- Cache for per-step trajectory assignment ----------------------
        self._cache_valid = False
        self._cached_global_indices: Tensor | None = None
        self._cached_env_ids: Tensor | None = None

        # --- Trajectory transition tracking --------------------------------
        # ``_traj_changed`` is populated by :meth:`_ensure_cache` on each
        # cache rebuild and read by the owning command term to re-arm the
        # contact gate.  Snapshot is committed by :meth:`commit_traj_state`,
        # not eagerly inside ``_ensure_cache`` — see that method's docstring.
        self._prev_global_indices: Tensor | None = None
        self._traj_changed: Tensor | None = None

        # --- Contact-gate metadata ----------------------------------------
        self._build_gate_tables()

        # --- Per-env phase + gate state -----------------------------------
        # Allocated lazily (we may not know num_envs at construction time
        # for offline / test usage).  ``phase`` lives in [0, 1]; advances
        # by ``step_dt / total_time`` each step (wrap for periodic, clamp
        # for episodic, perpetual stays at 0).  ``next_gate_idx`` is the
        # contact-gate currently armed per env, or -1 if none.
        # ``gate_rel_phi`` is signed phi-distance from the armed gate
        # (positive = past, negative = before), accumulated monotonically
        # — does NOT wrap with ``phase``.  Re-anchored on every gate
        # (re)arm by :meth:`_refresh_gate_rel_phi`.
        # ``gate_rel_phi_pre_snap`` captures ``gate_rel_phi`` immediately
        # after :meth:`update_phase` runs and *before* any contact-gate
        # snap mutates it.  It's the value the gate logic actually
        # decides on, so it's what the data-logger / play plots should
        # display to show "how early/late was the contact when the fire
        # happened."
        self.phase: Tensor | None = None
        self.next_gate_idx: Tensor | None = None
        self.gate_rel_phi: Tensor | None = None
        self.gate_rel_phi_pre_snap: Tensor | None = None

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _discover_skill_folders(
        path: str, hf_repo: str | None
    ) -> dict[str, Path]:
        """Discover per-skill subfolders under a top-level path.

        If ``hf_repo`` is set, downloads the folder from HuggingFace first.
        Each immediate subdirectory that contains YAML files becomes a skill.

        Args:
            path: Top-level folder (local or HF-relative).
            hf_repo: Optional HuggingFace repo ID.

        Returns:
            Dict mapping skill name to resolved local ``Path``.
        """
        if hf_repo is not None:
            resolved = MultiSkillManager._download_from_hf(hf_repo, path)
        else:
            resolved = Path(path)

        if not resolved.exists():
            raise FileNotFoundError(f"Skill root folder not found: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {resolved}")

        skill_folders: dict[str, Path] = {}
        for child in sorted(resolved.iterdir()):
            if not child.is_dir():
                continue
            yamls = list(child.glob("*.yaml")) + list(child.glob("*.yml"))
            if len(yamls) > 0:
                skill_folders[child.name] = child

        if len(skill_folders) == 0:
            # Fall back: maybe the path itself contains YAMLs (single-skill)
            yamls = list(resolved.glob("*.yaml")) + list(resolved.glob("*.yml"))
            if len(yamls) > 0:
                skill_folders["default"] = resolved
            else:
                raise ValueError(
                    f"No skill subfolders with YAML files found under: {resolved}"
                )

        return skill_folders

    @staticmethod
    def _download_from_hf(hf_repo: str, hf_path: str) -> Path:
        """Download a folder from HuggingFace to a local cache.

        Args:
            hf_repo: HuggingFace repo ID (e.g. ``"zolkin/robot_rl"``).
            hf_path: Path within the repo to download.

        Returns:
            Local ``Path`` to the downloaded folder.
        """
        root = os.path.abspath(os.getcwd())
        cache_dir = os.path.join(root, "hf")
        os.makedirs(cache_dir, exist_ok=True)

        local_folder = os.path.join(cache_dir, hf_path)

        # Check if already cached
        if os.path.isdir(local_folder):
            # Check for YAMLs in subdirectories or directly
            has_yamls = (
                list(Path(local_folder).glob("*.yaml"))
                + list(Path(local_folder).glob("*.yml"))
                + list(Path(local_folder).glob("**/*.yaml"))
            )
            if len(has_yamls) > 0:
                print(f"Using cached trajectories from {local_folder}")
                return Path(local_folder)

        try:
            from huggingface_hub import snapshot_download

            print(f"Downloading {hf_path} from {hf_repo}...")
            # Use multiple patterns: fnmatch doesn't support **
            # Match flat YAMLs and one level of skill subdirectories
            snapshot_download(
                repo_id=hf_repo,
                allow_patterns=[
                    f"{hf_path}/*",      # flat files
                    f"{hf_path}/*/*",    # files in skill subdirectories
                ],
                local_dir=cache_dir,
            )
            print(f"Successfully downloaded to {local_folder}")
            return Path(local_folder)

        except ImportError:
            raise RuntimeError(
                "huggingface_hub is required for downloading trajectories. "
                "Install with: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download from HuggingFace: {e}"
            )

    @staticmethod
    def _find_yaml_files(folder_path: str) -> list[Path]:
        """Find all YAML files in a folder."""
        p = Path(folder_path)
        if not p.exists():
            raise FileNotFoundError(f"Skill folder not found: {folder_path}")
        if not p.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {folder_path}")
        yamls = sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml"))
        if len(yamls) == 0:
            raise ValueError(f"No YAML trajectory files in: {folder_path}")
        return yamls

    @staticmethod
    def _parse_conditioner(raw_conditioner: Any, skill_name: str) -> ConditionerData:
        """Parse a conditioner field from YAML into ConditionerData.

        Supports both old format (list of floats) and new format (dict).
        """
        if isinstance(raw_conditioner, list):
            # Old format: [min, max] — use midpoint as vel_x, zeros elsewhere
            mid = sum(raw_conditioner) / len(raw_conditioner)
            return ConditionerData(vel_x=mid)
        elif isinstance(raw_conditioner, dict):
            return ConditionerData(
                vel_x=raw_conditioner.get("vel_x", 0.0),
                vel_y=raw_conditioner.get("vel_y", 0.0),
                vel_yaw=raw_conditioner.get("vel_yaw", 0.0),
                terrain=raw_conditioner.get("terrain", "flat"),
                terrain_width=raw_conditioner.get("terrain_width", 0.0),
                terrain_height=raw_conditioner.get("terrain_height", 0.0),
                terrain_length=raw_conditioner.get("terrain_length", 0.0),
            )
        else:
            raise ValueError(
                f"Unrecognised conditioner format for skill '{skill_name}': "
                f"{type(raw_conditioner)}"
            )

    def _build_skill_data(
        self,
        skill_labels: list[str],
        conditioner_datas: list[ConditionerData],
    ) -> None:
        """Build per-skill metadata and conditioning tensors."""
        unique_skills = list(dict.fromkeys(skill_labels))
        self.skill_name_to_idx: dict[str, int] = {
            name: i for i, name in enumerate(unique_skills)
        }
        self.skills: list[SkillData] = []

        for skill_name in unique_skills:
            indices = [
                i for i, label in enumerate(skill_labels) if label == skill_name
            ]
            conds = [conditioner_datas[i] for i in indices]
            has_terrain = any(c.terrain != "flat" for c in conds)

            cond_tensor = torch.tensor(
                [c.to_continuous_tensor() for c in conds],
                dtype=torch.float32,
                device=self.device,
            )

            self.skills.append(
                SkillData(
                    name=skill_name,
                    traj_indices=indices,
                    num_trajectories=len(indices),
                    has_terrain=has_terrain,
                    conditioning_tensor=cond_tensor,
                )
            )

        # Build global conditioning tensor across all trajectories.
        # Ordered by global trajectory index (0 .. T-1).
        all_conds = [cd.to_continuous_tensor() for cd in conditioner_datas]
        self._global_conditioning = torch.tensor(
            all_conds, dtype=torch.float32, device=self.device,
        )  # [T, C]

        # Terrain mask: True if trajectory has real terrain (positive dims).
        # Flat trajectories use -1 as a sentinel for terrain dimensions.
        terrain_dims = self._global_conditioning[:, 3:]  # [T, 3]
        self._terrain_mask = (terrain_dims > 0).any(dim=1)  # [T]

    # ------------------------------------------------------------------
    # Output ordering (called by TrajectoryCommand after construction)
    # ------------------------------------------------------------------

    def order_outputs(
        self,
        pos_output_names: list[str],
        vel_output_names: list[str],
    ) -> None:
        """Reorder outputs in the global coefficient tensors.

        This must be called before evaluation if the downstream consumer
        expects a different output ordering than the YAML default.

        Args:
            pos_output_names: Desired position output name order.
            vel_output_names: Desired velocity output name order.
        """
        # Build permutation indices for position outputs
        pos_perm = [self.pos_output_names.index(n) for n in pos_output_names]
        vel_perm = [self.vel_output_names.index(n) for n in vel_output_names]

        pos_perm_t = torch.tensor(pos_perm, dtype=torch.long, device=self.device)
        vel_perm_t = torch.tensor(vel_perm, dtype=torch.long, device=self.device)

        # Reorder coefficient tensors: [T, D, P, K+1] -> index dim 2
        self.data["coeffs_pos"] = self.data["coeffs_pos"][:, :, pos_perm_t, :]
        self.data["coeffs_vel"] = self.data["coeffs_vel"][:, :, vel_perm_t, :]

        self.pos_output_names = list(pos_output_names)
        self.vel_output_names = list(vel_output_names)
        self.num_pos_outputs = len(pos_output_names)
        self.num_vel_outputs = len(vel_output_names)

    # ------------------------------------------------------------------
    # Properties mirroring LibraryManager / TrajectoryManager interface
    # ------------------------------------------------------------------

    @property
    def get_pos_output_names(self) -> list[str]:
        """Get position output names (includes ori_w)."""
        return self.pos_output_names

    @property
    def get_vel_output_names(self) -> list[str]:
        """Get velocity output names (excludes ori_w)."""
        return self.vel_output_names

    def get_reference_frames(self) -> list[str]:
        """Get the reference frame names."""
        return self.ref_frames

    def get_num_pos_outputs(self) -> int:
        """Get number of position outputs (includes ori_w)."""
        return self.num_pos_outputs

    def get_num_vel_outputs(self) -> int:
        """Get number of velocity outputs (excludes ori_w)."""
        return self.num_vel_outputs

    def get_trajectory_type(self) -> TrajectoryType:
        """Get the trajectory type of the first trajectory.

        Note: in a multi-skill setting trajectory types may vary per env.
        This returns the most common type for compatibility.
        """
        types = self.data["traj_type"]
        mode_val = types.mode().values.item()
        for tt, iv in _TRAJ_TYPE_TO_INT.items():
            if iv == mode_val:
                return tt
        return TrajectoryType.HALF_PERIODIC

    # ------------------------------------------------------------------
    # Cache: per-step trajectory assignment
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        """Invalidate the per-step cache.  Call once at the start of each step."""
        self._cache_valid = False

    def commit_traj_state(self, env_ids: Tensor | None = None) -> None:
        """Snapshot the current cached trajectory assignment as the
        "previous" state used by the next :meth:`_ensure_cache` call to
        detect transitions.

        Decoupled from :meth:`_ensure_cache` so external callers (reset
        events, observation queries, snap helpers, loggers) do not
        silently consume ``_traj_changed`` before the owning command's
        ``_pre_update_phase`` has read it.  The owning command term must
        call this exactly once per step, after gate re-arm has run.

        Args:
            env_ids: Optional subset; ``None`` commits the full batch.
        """
        if self._cached_global_indices is None:
            return
        if env_ids is None:
            self._prev_global_indices = self._cached_global_indices.clone()
        else:
            self._prev_global_indices[env_ids] = self._cached_global_indices[env_ids]

    def set_trajectory_indices(self, global_indices: Tensor) -> None:
        """Directly set the trajectory index for each environment.

        This bypasses the conditioner lookup and is useful for testing or
        when the caller already knows the assignment.

        Args:
            global_indices: Shape ``[N]`` tensor of global trajectory indices.
        """
        self._cached_global_indices = global_indices.to(self.device)
        self._cached_env_ids = None
        self._cache_valid = True

    def _ensure_cache(self, env_ids: Tensor | None = None) -> None:
        """Ensure the trajectory assignment cache is populated.

        Selects the best trajectory per environment across all skills based
        on the commanded velocity (and terrain, when applicable).  Also
        tracks per-env trajectory transitions between steps.

        Args:
            env_ids: Optional subset of environment indices.  ``None`` means
                the full batch.
        """
        if self._cache_valid and self._cached_env_ids is env_ids:
            return

        if self.env is not None and self.conditioner_generator_name:
            conditioner = self._get_conditioner_from_env(env_ids)
            new_indices = self._select_trajectories(conditioner, env_ids=env_ids)

            # --- Trajectory transition detection --------------------------
            # Compute ``_traj_changed`` against the previous snapshot, but
            # do NOT update ``_prev_global_indices`` here.  Many callers
            # (reset events, observation fns, snap helpers) invoke
            # ``_ensure_cache`` between two ``_pre_update_phase`` calls; if
            # they updated the prev array they would silently consume the
            # transition before the owning command saw it.  The owning
            # command term must call :meth:`commit_traj_state` exactly
            # once per step, after gate re-arm has read ``_traj_changed``.
            if self._prev_global_indices is not None:
                if env_ids is not None:
                    self._traj_changed = new_indices != self._prev_global_indices[env_ids]
                else:
                    self._traj_changed = new_indices != self._prev_global_indices
            else:
                # First-call init: prev array must be populated so
                # subsequent comparisons have a baseline.  This is the
                # only place ``_ensure_cache`` writes it.
                self._traj_changed = torch.zeros(
                    new_indices.shape[0], dtype=torch.bool, device=self.device,
                )
                if env_ids is not None:
                    N_total = self.env.num_envs
                    self._prev_global_indices = torch.zeros(
                        N_total, dtype=torch.long, device=self.device,
                    )
                    self._prev_global_indices[env_ids] = new_indices
                else:
                    self._prev_global_indices = new_indices.clone()

            self._cached_global_indices = new_indices
        else:
            raise RuntimeError(
                "Cannot auto-populate cache: no env or conditioner_generator_name. "
                "Call set_trajectory_indices() explicitly."
            )

        self._cached_env_ids = env_ids
        self._cache_valid = True

    def _get_conditioner_from_env(self, env_ids: Tensor | None) -> Tensor:
        """Query the environment command manager for conditioning values.

        Returns:
            Tensor of shape ``[N, C]`` where C is the number of continuous
            conditioning dimensions.
        """
        cond_term = self.env.command_manager.get_term(self.conditioner_generator_name)
        # cond_term.command is typically [num_envs, cmd_dim]
        raw = cond_term.command
        if env_ids is not None:
            raw = raw[env_ids]
        # Pad or truncate to 6 continuous dims expected by ConditionerData
        N = raw.shape[0]
        C = 6  # vel_x, vel_y, vel_yaw, terrain_w, terrain_h, terrain_l
        cond = torch.zeros(N, C, device=self.device)
        cond[:, :min(raw.shape[1], C)] = raw[:, :min(raw.shape[1], C)]
        return cond

    def _lazy_init_skill_filter(self) -> None:
        """One-time validation + skill-id mapping build.

        Resolves the conditioner term, requires it to be a
        :class:`MultiskillVelocityTrackingCommand`-shaped object exposing a
        per-env ``skill_id`` and an ordered ``_skill_list``. Validates that
        the velocity command's skills are a subset of the trajectory
        manager's skills, then caches a gather tensor that maps vel-cmd
        skill indices to trajectory-manager skill indices.  Extra skills
        in the manager (loaded subfolders that the env's terrain doesn't
        advertise) are allowed and remain unused — they live in the
        trajectory pool but never get sampled.

        Deferred to first selection because the conditioner term may not be
        registered on ``env.command_manager`` at manager construction time.
        """
        if self._skill_filter_ready:
            return

        if self._skill_owner is None:
            raise RuntimeError(
                "MultiSkillManager has no skill owner registered. Call "
                "set_skill_owner(...) from the trajectory command's "
                "_post_init before any trajectory selection runs."
            )
        owner = self._skill_owner
        for required in ("skill_id", "_skill_list"):
            if not hasattr(owner, required):
                raise TypeError(
                    f"MultiSkillManager's skill owner "
                    f"({type(owner).__name__}) must expose '{required}'."
                )

        owner_skills: list[str] = list(owner._skill_list)
        traj_skills = set(self.skill_name_to_idx.keys())
        missing_in_traj = sorted(set(owner_skills) - traj_skills)
        if missing_in_traj:
            raise KeyError(
                "Skill-owner declares skills the trajectory pool does not "
                f"contain: {missing_in_traj}. "
                f"Available trajectory skills: {sorted(traj_skills)}."
            )

        self._vel_skill_to_traj_skill = torch.tensor(
            [self.skill_name_to_idx[name] for name in owner_skills],
            dtype=torch.long, device=self.device,
        )
        self._skill_filter_ready = True

    def set_skill_owner(self, owner: Any) -> None:
        """Register the term that owns the per-env active ``skill_id``.

        The owner must expose:

        - ``skill_id`` — ``[num_envs]`` long tensor of the active skill
          per env, indexing into the owner's ``_skill_list``.
        - ``_skill_list`` — ordered list of skill names (= the active
          skill index space).

        Validation is deferred to :meth:`_lazy_init_skill_filter`; calling
        ``set_skill_owner`` only stashes the reference.
        """
        self._skill_owner = owner

    def _select_trajectories(
        self,
        conditioner: Tensor,
        env_ids: Tensor | None = None,
    ) -> Tensor:
        """Select the nearest trajectory per env, restricted to the env's skill.

        The active per-env ``skill_id`` comes from the registered skill
        owner (the trajectory cmd) — not the velocity cmd.  The velocity
        cmd's sampled bucket may be in a different skill while
        ``vel_target_b`` is mid-ramp; the owner's active skill follows
        wherever ``vel_target_b`` actually sits.

        Args:
            conditioner: ``(N, C)`` conditioning vectors. Only the first three
                dims (vel_x, vel_y, vel_yaw) participate in matching now —
                skill replaces the legacy terrain-dim filter.
            env_ids: Optional env subset matching ``conditioner``'s rows. When
                provided, the owner's ``skill_id`` is sliced the same way so
                row counts stay aligned (callers like ``reset_on_reference``
                pass an env subset).

        Returns:
            ``(N,)`` global trajectory indices.
        """
        self._lazy_init_skill_filter()

        owner_skill_ids: Tensor = self._skill_owner.skill_id  # (num_envs,)
        if env_ids is not None:
            owner_skill_ids = owner_skill_ids[env_ids]

        # Sanity guard: catches any future caller that slices the conditioner
        # without also passing the matching ``env_ids`` (or vice versa).
        if conditioner.shape[0] != owner_skill_ids.shape[0]:
            raise RuntimeError(
                f"conditioner ({conditioner.shape[0]}) and owner.skill_id "
                f"({owner_skill_ids.shape[0]}) must have matching env count."
            )

        env_traj_skill_ids = self._vel_skill_to_traj_skill[owner_skill_ids]  # (N,)

        vel_cmd = conditioner[:, :3]                           # (N, 3)
        vel_global = self._global_conditioning[:, :3]          # (T, 3)
        vel_dists = torch.cdist(vel_cmd, vel_global)           # (N, T)

        allowed = self._traj_skill_idx.unsqueeze(0) == env_traj_skill_ids.unsqueeze(1)
        vel_dists = vel_dists.masked_fill(~allowed, float("inf"))

        return vel_dists.argmin(dim=1)

    # ------------------------------------------------------------------
    # Core batched helpers
    # ------------------------------------------------------------------

    def _get_global_indices(self, env_ids: Tensor | None = None) -> Tensor:
        """Return the cached global trajectory indices, ensuring cache is valid."""
        self._ensure_cache(env_ids)
        return self._cached_global_indices

    def get_current_trajectory_indices(self, env_ids: Tensor | None = None) -> Tensor:
        """Public wrapper around the trajectory assignment cache.

        Resolves the current trajectory index per environment without
        computing any outputs.  Callers can use this to detect skill/
        trajectory changes between steps.

        Args:
            env_ids: Optional environment index subset.  ``None`` resolves
                the full batch.

        Returns:
            ``[N]`` global trajectory indices for the selected envs.
        """
        return self._get_global_indices(env_ids)

    def _get_domain_indices(self, phase: Tensor, traj_idx: Tensor) -> Tensor:
        """Batched domain lookup via searchsorted on per-trajectory boundaries.

        Args:
            phase: ``[N]`` phase per environment.
            traj_idx: ``[N]`` global trajectory index per environment.

        Returns:
            ``[N]`` domain indices (into the expanded domain dimension).
        """
        total = self.data["total_time"][traj_idx]  # [N]
        t = phase * total  # [N]

        # Gather per-trajectory boundaries: [N, D_max+1]
        boundaries = self.data["domain_boundaries"][traj_idx]  # [N, D_max+1]

        # searchsorted along dim=1
        domain_idx = torch.searchsorted(boundaries, t.unsqueeze(1), right=False).squeeze(1) - 1

        # Clamp to valid range per trajectory
        max_dom = self.data["expanded_domains"][traj_idx] - 1
        domain_idx = torch.clamp(domain_idx, min=torch.zeros_like(max_dom), max=max_dom)

        return domain_idx

    def _compute_normalized_tau(
        self, phase: Tensor, traj_idx: Tensor, domain_idx: Tensor
    ) -> Tensor:
        """Compute tau ∈ [0,1] within the current domain for each environment.

        Args:
            phase: ``[N]`` phase in [0, 1].
            traj_idx: ``[N]`` global trajectory indices.
            domain_idx: ``[N]`` expanded domain indices.

        Returns:
            ``[N]`` normalized tau values.
        """

        # For half-periodic: fold the second half onto the first
        # ([0.5, 1) → [0, 0.5)) and scale to [0, 1).
        is_half = self.data["traj_type"][traj_idx] == _HALF_PERIODIC_INT
        phase = torch.where(is_half & (phase >= 0.5), phase - 0.5, phase)
        phase = torch.where(is_half, phase * 2.0, phase)

        # Domain index within original (non-reflected) domains
        num_orig = self.data["num_original_domains"][traj_idx]  # [N]
        dom_in_half = domain_idx % num_orig

        # Relative start of domain within the half-period
        # cumsum_T[traj, dom] / total_original_T[traj]
        rel_prev = self._cumsum_T[traj_idx, dom_in_half] / self._total_original_T[traj_idx]

        # Duration of this domain
        T_dom = self.data["domain_times"][traj_idx, domain_idx]

        # Normalize
        tau_norm = (phase - rel_prev) * self._total_original_T[traj_idx] / T_dom
        return torch.clamp(tau_norm, 0.0, 1.0)

    def _compute_bezier_batched(
        self, phase: Tensor, ctrl_pts: Tensor, T_dom: Tensor, derivative: bool
    ) -> Tensor:
        """Batched Bernstein polynomial evaluation.

        Args:
            phase: ``[N]`` phase in [0, 1].
            ctrl_pts: ``[N, num_outputs, degree+1]`` control points.
            T_dom: ``[N]`` domain durations (for velocity scaling).
            derivative: If True, compute first derivative of the Bezier curve.

        Returns:
            ``[N, num_outputs]`` interpolated values.
        """
        degree = ctrl_pts.shape[-1] - 1

        if not derivative:
            coefs = self._binomial_coeffs[degree]  # [degree+1]
            i_vec = torch.arange(degree + 1, device=ctrl_pts.device)
            tau_pow = phase.unsqueeze(1) ** i_vec  # [N, degree+1]
            one_minus_pow = (1 - phase).unsqueeze(1) ** (degree - i_vec)
            weights = coefs * tau_pow * one_minus_pow  # [N, degree+1]
            return torch.einsum("nd,nod->no", weights, ctrl_pts)
        else:
            coefs = self._binomial_coeffs[degree - 1]  # [degree]
            i_vec = torch.arange(degree, device=ctrl_pts.device)
            tau_pow = phase.unsqueeze(1) ** i_vec
            one_minus_pow = (1 - phase).unsqueeze(1) ** (degree - 1 - i_vec)
            weights = degree * coefs * tau_pow * one_minus_pow
            cp_diff = ctrl_pts[:, :, 1:] - ctrl_pts[:, :, :-1]
            result = torch.einsum("nd,nod->no", weights, cp_diff)
            return result / T_dom.unsqueeze(1)

    # ------------------------------------------------------------------
    # ManagerBase interface — fully batched
    # ------------------------------------------------------------------

    def get_output(
        self, phase: Tensor, env_ids: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Compute position and velocity outputs for all environments.

        Args:
            phase: ``[N]`` phase in [0, 1] per environment.
            env_ids: Optional environment index subset.

        Returns:
            ``(pos_outputs, vel_outputs)`` each of shape ``[N, P]`` / ``[N, V]``.
        """
        traj_idx = self._get_global_indices(env_ids)
        domain_idx = self._get_domain_indices(phase, traj_idx)
        tau = self._compute_normalized_tau(phase, traj_idx, domain_idx)

        # Gather coefficients: [N, P/V, K+1]
        env_coeffs_pos = self.data["coeffs_pos"][traj_idx, domain_idx]
        env_coeffs_vel = self.data["coeffs_vel"][traj_idx, domain_idx]
        T_dom = self.data["domain_times"][traj_idx, domain_idx]

        pos_out = self._compute_bezier_batched(tau, env_coeffs_pos, T_dom, derivative=False)
        vel_out = self._compute_bezier_batched(tau, env_coeffs_vel, T_dom, derivative=False)

        return pos_out, vel_out

    def get_current_domains(
        self, phase: Tensor, env_ids: Tensor | None = None
    ) -> Tensor:
        """Return the domain index for each environment.

        Args:
            phase: ``[N]`` phase in [0, 1] per environment.
            env_ids: Optional environment index subset.

        Returns:
            ``[N]`` domain indices.
        """
        traj_idx = self._get_global_indices(env_ids)
        return self._get_domain_indices(phase, traj_idx)

    def get_domain_times(
        self, phase: Tensor, env_ids: Tensor | None = None
    ) -> Tensor:
        """Get the duration of each environment's current domain.

        Args:
            phase: ``[N]`` phase in [0, 1] per environment.
            env_ids: Optional environment index subset.

        Returns:
            ``[N]`` domain durations.
        """
        traj_idx = self._get_global_indices(env_ids)
        domain_idx = self._get_domain_indices(phase, traj_idx)
        return self.data["domain_times"][traj_idx, domain_idx]

    def get_num_domains(self, env_ids: Tensor | None = None) -> Tensor:
        """Get the expanded domain count for each environment.

        Args:
            env_ids: Optional environment index subset.

        Returns:
            ``[N]`` expanded domain counts.
        """
        traj_idx = self._get_global_indices(env_ids)
        return self.data["expanded_domains"][traj_idx]

    def get_total_time(self) -> Tensor:
        """Get the total time of the first trajectory (compatibility)."""
        return self.data["total_time"][0]

    def get_ref_frames_in_use(
        self, phase: Tensor, ref_frames: list[str], env_ids: Tensor | None = None
    ) -> Tensor:
        """Determine the active reference frame per environment.

        Args:
            phase: ``[N]`` phase in [0, 1] per environment.
            ref_frames: List of reference frame names.
            env_ids: Optional environment index subset.

        Returns:
            ``[N]`` indices into ``ref_frames``.
        """
        # Lazily build lookup table on first call
        if not hasattr(self, "_ref_frame_map") or self._ref_frame_key != tuple(ref_frames):
            self._precompute_ref_frame_map(ref_frames)

        traj_idx = self._get_global_indices(env_ids)
        domain_idx = self._get_domain_indices(phase, traj_idx)
        return self._ref_frame_map[traj_idx, domain_idx]

    def _precompute_ref_frame_map(self, ref_frames: list[str]) -> None:
        """Build a [T, D_max] lookup table mapping (traj, domain) → ref frame index."""
        T = self.num_trajectories
        D = self.max_expanded_domains
        table = torch.zeros(T, D, dtype=torch.long, device=self.device)

        # TODO: Double check this comment - is this right?
        # We need domain→frame mappings.  Since we discarded managers, we
        # rebuild from the stored data.  For now, use a simple approach:
        # all domains within a trajectory share the same set of reference frames
        # in alternating order matching the original domain data.
        #
        # This is populated during _build_global_tensors if we store it.
        # For robustness, store a ref_frame_per_domain tensor at load time.
        if hasattr(self, "_ref_frame_domain_map"):
            table = self._ref_frame_domain_map
        else:
            # Fallback: all zeros (first ref frame).  This will be overridden
            # once _build_ref_frame_map is called during loading.
            pass

        self._ref_frame_map = table
        self._ref_frame_key = tuple(ref_frames)

    def get_contact_state(
        self, phase: Tensor, contact_frames: list[str], env_ids: Tensor | None = None
    ) -> Tensor:
        """Return the contact state per environment and contact frame.

        Args:
            phase: ``[N]`` phase in  [0, 1] per environment.
            contact_frames: List of contact frame names.
            env_ids: Optional environment index subset.

        Returns:
            ``[N, num_contacts]`` binary contact states.
        """
        if not hasattr(self, "_contact_table") or self._contact_table_key != tuple(contact_frames):
            self._precompute_contact_table(contact_frames)

        traj_idx = self._get_global_indices(env_ids)
        domain_idx = self._get_domain_indices(phase, traj_idx)

        # For half-periodic trajectories, use reflected table in second half
        is_half = self.data["traj_type"][traj_idx] == _HALF_PERIODIC_INT
        in_second_half = is_half & (phase >= 0.5)

        result = self._contact_table_first[traj_idx, domain_idx]
        reflected = self._contact_table_second[traj_idx, domain_idx]
        return torch.where(in_second_half.unsqueeze(1), reflected, result)

    def _precompute_contact_table(self, contact_frames: list[str]) -> None:
        """Build contact lookup tables of shape [T, D_max, num_contacts]."""
        # This requires domain-level contact body information which we need
        # to store during loading.  Use the stored _contact_bodies_per_domain.
        T = self.num_trajectories
        D = self.max_expanded_domains
        C = len(contact_frames)

        table_first = torch.zeros(T, D, C, device=self.device)
        table_second = torch.zeros(T, D, C, device=self.device)

        if hasattr(self, "_contact_bodies_per_domain"):
            for ti in range(T):
                ed = self.data["expanded_domains"][ti].item()
                nd = self.data["num_original_domains"][ti].item()
                is_half = self.data["traj_type"][ti].item() == _HALF_PERIODIC_INT

                for di in range(ed):
                    bodies = self._contact_bodies_per_domain[ti][di % nd]
                    reflected_bodies = [swap_left_right(b) for b in bodies]
                    for ci, frame in enumerate(contact_frames):
                        if frame in bodies:
                            table_first[ti, di, ci] = 1.0
                        if is_half and frame in reflected_bodies:
                            table_second[ti, di, ci] = 1.0
                        elif not is_half and frame in bodies:
                            table_second[ti, di, ci] = 1.0

        self._contact_table_first = table_first
        self._contact_table_second = table_second
        self._contact_table_key = tuple(contact_frames)

    # ------------------------------------------------------------------
    # Tensor construction
    # ------------------------------------------------------------------

    def _build_global_tensors(
        self,
        managers: list[TrajectoryManager],
        skill_labels: list[str],
        conditioner_datas: list[ConditionerData],
    ) -> None:
        """Extract data from TrajectoryManagers into padded global tensors.

        This method also stores per-domain metadata (contact bodies, ref frames)
        needed for lazy table construction.
        """
        T = self.num_trajectories
        P = self.num_pos_outputs
        V = self.num_vel_outputs
        K = self.spline_order

        # Determine max expanded domain count
        expanded_counts = [mgr.expanded_num_domains for mgr in managers]
        D_max = max(expanded_counts)
        self.max_expanded_domains = D_max

        # Allocate tensors
        coeffs_pos = torch.zeros(T, D_max, P, K + 1, device=self.device)
        coeffs_vel = torch.zeros(T, D_max, V, K + 1, device=self.device)
        domain_times = torch.zeros(T, D_max, device=self.device)
        domain_boundaries = torch.zeros(T, D_max + 1, device=self.device)
        total_time = torch.zeros(T, device=self.device)
        expanded_domains = torch.zeros(T, dtype=torch.long, device=self.device)
        traj_type = torch.zeros(T, dtype=torch.long, device=self.device)
        skill_idx_tensor = torch.zeros(T, dtype=torch.long, device=self.device)
        num_original_domains = torch.zeros(T, dtype=torch.long, device=self.device)
        ref_frame_offset = torch.zeros(T, D_max, 3, device=self.device)
        origin_relative_to_stair_center = torch.zeros(T, 3, device=self.device)

        unique_skills = list(dict.fromkeys(skill_labels))
        skill_name_to_idx = {name: i for i, name in enumerate(unique_skills)}

        # Per-domain metadata (stored as Python lists for lazy table building)
        contact_bodies_per_domain: list[list[list[str]]] = []
        ref_frame_per_domain: list[list[str]] = []

        original_domain_times = torch.zeros(T, D_max, device=self.device)

        for i, mgr in enumerate(managers):
            ed = mgr.expanded_num_domains
            nd = mgr.num_domains
            expanded_domains[i] = ed
            num_original_domains[i] = nd
            traj_type[i] = _TRAJ_TYPE_TO_INT[mgr.traj_data.trajectory_type]
            skill_idx_tensor[i] = skill_name_to_idx[skill_labels[i]]

            # Coefficients
            coeffs_pos[i, :ed, :, :] = mgr._all_coeffs_pos
            coeffs_vel[i, :ed, :, :] = mgr._all_coeffs_vel

            # Domain times
            domain_times[i, :ed] = mgr._T_all
            original_domain_times[i, :nd] = mgr.T

            # Domain boundaries
            db = mgr.domain_boundaries
            domain_boundaries[i, : ed + 1] = db
            if ed + 1 < D_max + 1:
                domain_boundaries[i, ed + 1 :] = db[-1]

            # Total time
            if mgr.traj_data.trajectory_type == TrajectoryType.HALF_PERIODIC:
                total_time[i] = mgr.traj_data.total_time * 2
            else:
                total_time[i] = mgr.traj_data.total_time

            # Contact bodies per domain (original domains only)
            bodies_list: list[list[str]] = []
            frames_list: list[str] = []
            for di, domain_name in enumerate(mgr.traj_data.domain_order):
                dd = mgr.traj_data.domain_data[domain_name]
                bodies_list.append(list(dd.contact_bodies))
                frames_list.append(dd.bezier_frame)
                ref_frame_offset[i, di, :] = dd.ref_frame_offset
                if mgr.traj_data.trajectory_type == TrajectoryType.HALF_PERIODIC:
                    # Reflected half reuses the same offset.
                    ref_frame_offset[i, di + mgr.num_domains, :] = dd.ref_frame_offset
            # Fail loudly if no domain declares any contact bodies. The most
            # common cause is a YAML field-name typo (e.g. ``contact_frames``
            # instead of the expected ``contact_bodies``), or values that
            # don't match the scene's URDF link names — both of which
            # silently produce empty lists here, an all-zero contact table,
            # and a frozen ``ref_poses`` that's almost impossible to debug
            # from symptoms alone.
            if not any(bodies_list):
                traj_name = getattr(mgr.traj_data, "name", f"<index {i}>")
                raise ValueError(
                    f"Trajectory '{traj_name}' has no contact bodies declared "
                    f"in any of its {len(bodies_list)} domain(s). Check the "
                    f"trajectory YAML: per-domain field must be named "
                    f"'contact_bodies' and values must match scene URDF link "
                    f"names (got {bodies_list!r})."
                )

            contact_bodies_per_domain.append(bodies_list)
            ref_frame_per_domain.append(frames_list)

            # Per-trajectory stair-origin offset (top-level YAML field).
            origin_relative_to_stair_center[i] = mgr.traj_data.origin_relative_to_stair_center

        self.data = TensorDict(
            {
                "coeffs_pos": coeffs_pos,
                "coeffs_vel": coeffs_vel,
                "domain_times": domain_times,
                "domain_boundaries": domain_boundaries,
                "total_time": total_time,
                "expanded_domains": expanded_domains,
                "num_original_domains": num_original_domains,
                "traj_type": traj_type,
                "skill_idx": skill_idx_tensor,
                "ref_frame_offset": ref_frame_offset,
                "origin_relative_to_stair_center": origin_relative_to_stair_center,
            },
            batch_size=[T],
            device=self.device,
        )

        # Store metadata for lazy table building
        self._contact_bodies_per_domain = contact_bodies_per_domain
        self._ref_frame_names_per_domain = ref_frame_per_domain

        # Tau normalisation helpers
        self._original_domain_times = original_domain_times
        cumsum = torch.zeros(T, D_max, device=self.device)
        cumsum[:, 1:] = torch.cumsum(original_domain_times[:, :-1], dim=1)
        self._cumsum_T = cumsum
        self._total_original_T = original_domain_times.sum(dim=1)

    # ------------------------------------------------------------------
    # Placeholder stubs for future features
    # ------------------------------------------------------------------

    def transform_terrain_relative(
        self, positions: Tensor, env_ids: Tensor | None = None
    ) -> Tensor:
        """Transform positions to be relative to terrain geometry.

        Currently a no-op stub.  Future implementation will use
        ``env.scene.terrain`` to position terrain-dependent trajectories
        in the correct global frame.

        Args:
            positions: ``[N, ...]`` position tensor.
            env_ids: Optional environment indices.

        Returns:
            The input positions unchanged.
        """
        return positions

    # ------------------------------------------------------------------
    # Contact-gate tables
    # ------------------------------------------------------------------

    def _build_gate_tables(self) -> None:
        """Pre-compute contact-gate phi values and per-gate body name lists.

        Gate points per trajectory type:

        - Half-periodic: 2 gates (phi=0.5 entering reflected second half;
          phi=1.0 wrapping back to first half).
        - Full-periodic: 1 gate at phi=1.0 (cycle wrap).
        - Episodic: 1 gate at phi=1.0 (end of trajectory).
        - Perpetual: 0 gates.

        The expected-contact body names at each gate are taken from the
        first domain of the next period — for half-periodic phi=0.5 this is
        the sagittally reflected first domain.  The boolean mask in the
        command-term's contact-body ordering is materialised lazily via
        :meth:`set_gate_contact_layout`.
        """
        T = self.num_trajectories
        MAX_GATES = 2

        gate_phi = torch.zeros(T, MAX_GATES, device=self.device)
        gate_active = torch.zeros(T, MAX_GATES, dtype=torch.bool, device=self.device)
        # Phi-distance from the previous gate (wrapping for periodic types)
        # to this gate.  Used to scale the contact-gate early window in
        # the command term so a configured ``contact_gate_window_frac``
        # is interpreted as "fraction of the local domain", not raw phi.
        gate_pre_size = torch.ones(T, MAX_GATES, device=self.device)
        num_gates = torch.zeros(T, dtype=torch.long, device=self.device)
        gate_body_names: list[list[list[str]]] = [
            [[] for _ in range(MAX_GATES)] for _ in range(T)
        ]

        for ti in range(T):
            tt = self.data["traj_type"][ti].item()
            first_domain_bodies = list(self._contact_bodies_per_domain[ti][0])

            if tt == _HALF_PERIODIC_INT:
                gate_phi[ti, 0] = 0.5
                gate_active[ti, 0] = True
                gate_body_names[ti][0] = [swap_left_right(b) for b in first_domain_bodies]
                gate_pre_size[ti, 0] = 0.5

                gate_phi[ti, 1] = 1.0
                gate_active[ti, 1] = True
                gate_body_names[ti][1] = list(first_domain_bodies)
                gate_pre_size[ti, 1] = 0.5
                num_gates[ti] = 2
            elif tt == _FULL_PERIODIC_INT or tt == _EPISODIC_INT:
                gate_phi[ti, 0] = 1.0
                gate_active[ti, 0] = True
                gate_body_names[ti][0] = list(first_domain_bodies)
                gate_pre_size[ti, 0] = 1.0
                num_gates[ti] = 1
            # Perpetual: no gates.

        self._gate_phi_table = gate_phi
        self._gate_active_table = gate_active
        self._gate_pre_size_table = gate_pre_size
        self._num_gates_per_traj = num_gates
        self._gate_body_names_per_gate = gate_body_names
        self._max_gates = MAX_GATES
        self._gate_contact_mask: Tensor | None = None

    def set_gate_contact_layout(self, contact_bodies: list[str]) -> None:
        """Materialise the [T, MAX_GATES, B] expected-contact mask in the
        ordering of *contact_bodies*.

        Args:
            contact_bodies: Ordered list of contact body names matching the
                command term's runtime contact tensor layout.
        """
        T = self.num_trajectories
        MAX_GATES = self._max_gates
        B = len(contact_bodies)

        mask = torch.zeros(T, MAX_GATES, B, dtype=torch.bool, device=self.device)
        body_to_idx = {n: i for i, n in enumerate(contact_bodies)}

        for ti in range(T):
            for gi in range(MAX_GATES):
                if not self._gate_active_table[ti, gi].item():
                    continue
                for body in self._gate_body_names_per_gate[ti][gi]:
                    if body in body_to_idx:
                        mask[ti, gi, body_to_idx[body]] = True

        self._gate_contact_mask = mask

    # ------------------------------------------------------------------
    # Per-env phase state and snap operations
    # ------------------------------------------------------------------

    def _ensure_phase_state(self, num_envs: int) -> None:
        """Lazily allocate ``phase``, ``next_gate_idx``, ``gate_rel_phi``,
        and ``gate_rel_phi_pre_snap``."""
        if (
            self.phase is not None
            and self.phase.shape[0] == num_envs
        ):
            return
        self.phase = torch.zeros(num_envs, device=self.device)
        self.next_gate_idx = -torch.ones(num_envs, dtype=torch.long, device=self.device)
        self.gate_rel_phi = torch.zeros(num_envs, device=self.device)
        self.gate_rel_phi_pre_snap = torch.zeros(num_envs, device=self.device)
        # Per-env wrap-aware phi delta written by :meth:`update_phase` for
        # the envs that actually advanced this tick; consumers (cross-fade
        # state, future analytics) read this instead of diffing
        # ``phase`` themselves.  Envs that didn't advance get 0.
        self.last_phase_delta = torch.zeros(num_envs, device=self.device)

    def _refresh_gate_rel_phi(self, env_ids: Tensor) -> None:
        """Recompute ``gate_rel_phi`` from current phase + armed gate.

        Sets ``gate_rel_phi[i] = phase[i] - gate_phi[next_gate_idx[i]]`` for
        envs with an active gate, or ``0`` for envs whose gate is ``-1``
        (perpetual / post-episodic).  Called after any operation that
        changes which gate is armed (reseed, advance, snap-with-advance).

        Args:
            env_ids: Env indices to refresh.
        """
        if env_ids.numel() == 0:
            return
        traj_idx = self._get_global_indices()[env_ids]
        gate_idx = self.next_gate_idx[env_ids]
        active = gate_idx >= 0
        safe_idx = torch.clamp(gate_idx, min=0)
        gate_phi = self._gate_phi_table[traj_idx, safe_idx]
        rel = self.phase[env_ids] - gate_phi
        self.gate_rel_phi[env_ids] = torch.where(active, rel, torch.zeros_like(rel))

    _EPS_PHI = 0.001
    """Small phi-space nudge used by the snap operations.

    Snap operations land phase at ``gate_phi ± _EPS_PHI`` so that the
    post-snap value falls strictly inside the intended (new vs. previous)
    domain rather than on the boundary, where ``searchsorted(...,
    right=False)`` would assign it to the previous domain.  ``0.001`` is
    small enough not to perturb training and large enough to clear
    float-precision and the boundary tie.
    """

    def _reseed_gate_for_envs(self, env_ids: Tensor) -> None:
        """Re-arm ``next_gate_idx`` for the given envs based on their current phase.

        Picks the smallest active gate whose ``gate_phi >= phase``;  if no
        upcoming gate this period, arms gate 0 of the next period.  If the
        trajectory has no gates at all, sets ``-1``.
        """
        if env_ids.numel() == 0:
            return
        traj_idx = self._get_global_indices()[env_ids]
        phi = self.phase[env_ids]
        gates = self._gate_phi_table[traj_idx]                # [n, MAX_GATES]
        gate_active = self._gate_active_table[traj_idx]       # [n, MAX_GATES]
        upcoming = gate_active & (gates >= phi.unsqueeze(1))
        has_upcoming = upcoming.any(dim=1)
        first_upcoming_idx = upcoming.to(torch.long).argmax(dim=1)
        init_idx = torch.where(
            has_upcoming, first_upcoming_idx, torch.zeros_like(first_upcoming_idx)
        )
        num_gates = self._num_gates_per_traj[traj_idx]
        init_idx = torch.where(num_gates > 0, init_idx, -torch.ones_like(init_idx))
        self.next_gate_idx[env_ids] = init_idx
        self._refresh_gate_rel_phi(env_ids)

    def _advance_gate_for_envs(self, env_ids: Tensor) -> None:
        """Advance ``next_gate_idx`` after a fire / expiry.

        Wraps to gate 0 for periodic; sets to -1 once the last gate of an
        episodic trajectory has been processed.
        """
        if env_ids.numel() == 0:
            return
        traj_idx = self._get_global_indices()[env_ids]
        cur_idx = self.next_gate_idx[env_ids]
        new_idx = cur_idx + 1
        num_gates = self._num_gates_per_traj[traj_idx]
        traj_type = self.data["traj_type"][traj_idx]

        last = new_idx >= num_gates
        is_episodic = traj_type == _EPISODIC_INT

        new_idx = torch.where(
            last & ~is_episodic, torch.zeros_like(new_idx), new_idx
        )
        new_idx = torch.where(
            last & is_episodic, -torch.ones_like(new_idx), new_idx
        )
        self.next_gate_idx[env_ids] = new_idx
        self._refresh_gate_rel_phi(env_ids)

    def advance_phase_for_traj(
        self,
        phase: Tensor,
        traj_idx: Tensor,
        step_dt: float,
    ) -> tuple[Tensor, Tensor]:
        """Advance per-row phase by ``step_dt`` using each row's trajectory rule.

        Pure (side-effect-free) version of the per-trajectory-type
        advancement logic in :meth:`update_phase`.  Called by
        ``update_phase`` for ``self.phase`` and by the cross-fade
        machinery for ``CrossfadeState.old_phase`` so the old
        trajectory keeps its own clock running during a transition
        whose target is perpetual (otherwise the blend can't progress).

        Half/full periodic wrap mod 1.0.  Episodic clamps to ``[0, 1]``.
        Perpetual snaps to 0 (unchanged).

        Args:
            phase: ``(K,)`` current phase per row.
            traj_idx: ``(K,)`` trajectory index per row.
            step_dt: sim step duration (seconds).

        Returns:
            ``(new_phase, phi_movement)`` — both ``(K,)``.  ``phi_movement``
            is the wrap-aware signed advancement (positive for periodic
            including across a wrap, can be negative for the single
            perpetual-snap step where ``new_phase=0`` from a non-zero
            ``prev_phase``).  Caller decides how to interpret negatives.
        """
        total = self.data["total_time"][traj_idx]
        tt = self.data["traj_type"][traj_idx]

        delta = step_dt / total
        new_phase = phase + delta

        is_periodic = (tt == _HALF_PERIODIC_INT) | (tt == _FULL_PERIODIC_INT)
        is_episodic = tt == _EPISODIC_INT
        is_perpetual = tt == _PERPETUAL_INT

        new_phase = torch.where(is_periodic, new_phase % 1.0, new_phase)
        new_phase = torch.where(
            is_episodic, torch.clamp(new_phase, 0.0, 1.0), new_phase
        )
        new_phase = torch.where(
            is_perpetual, torch.zeros_like(new_phase), new_phase
        )

        phi_movement = new_phase - phase
        phi_movement = torch.where(
            is_periodic & (new_phase < phase),
            phi_movement + 1.0,
            phi_movement,
        )
        return new_phase, phi_movement

    def update_phase(
        self, step_dt: float, env_ids: Tensor | None = None
    ) -> None:
        """Advance ``self.phase`` by ``step_dt / total_time[traj_idx]``.

        Half/full periodic phases wrap mod 1.0.  Episodic phases clamp at
        1.0.  Perpetual phases stay at 0.

        ``gate_rel_phi`` advances by the same delta but does *not* wrap or
        clamp — it tracks signed phi-distance from the currently armed
        gate monotonically.  The contact-gate logic in the command term
        reads ``gate_rel_phi`` rather than recomputing ``phase - gate_phi``,
        which means no wrap-advance is needed for phi=1.0 gates: a missed
        wrap-gate contact just becomes a (very) late-side fire on the
        next contact event.

        Args:
            step_dt: Sim step duration (seconds).
            env_ids: Optional subset; if ``None``, advances all envs.
        """
        traj_idx_full = self._get_global_indices()
        n_total = traj_idx_full.shape[0]
        self._ensure_phase_state(n_total)

        if env_ids is None:
            sel = torch.arange(n_total, device=self.device)
        else:
            sel = env_ids
        if sel.numel() == 0:
            return

        traj_idx = traj_idx_full[sel]
        prev_phase = self.phase[sel].clone()
        new_phase, phi_movement = self.advance_phase_for_traj(
            prev_phase, traj_idx, step_dt
        )

        self.phase[sel] = new_phase

        # Surface the wrap-aware per-env delta for downstream consumers
        # (e.g. ``CrossfadeState.step``).  Envs not in ``sel`` keep their
        # previous value zeroed out below to avoid drift.
        self.last_phase_delta[:] = 0.0
        self.last_phase_delta[sel] = phi_movement
        gate_armed = self.next_gate_idx[sel] >= 0
        phi_movement = torch.where(gate_armed, phi_movement, torch.zeros_like(phi_movement))
        self.gate_rel_phi[sel] = self.gate_rel_phi[sel] + phi_movement
        # Snapshot the post-update, pre-snap value for diagnostic logging.
        self.gate_rel_phi_pre_snap[sel] = self.gate_rel_phi[sel]

    def reset_phase(self, env_ids: Tensor, randomize: bool = True) -> None:
        """Reset phase for the given envs.

        Args:
            env_ids: Env indices to reset.
            randomize: If True, sample uniform phase ∈ [0, 1).  Otherwise
                set to 0.
        """
        if env_ids.numel() == 0:
            return
        n_total = self._get_global_indices().shape[0]
        self._ensure_phase_state(n_total)

        if randomize:
            new_phase = torch.rand(env_ids.shape[0], device=self.device)
        else:
            new_phase = torch.zeros(env_ids.shape[0], device=self.device)
        self.phase[env_ids] = new_phase
        self._reseed_gate_for_envs(env_ids)

    def set_phase(self, phase: Tensor, env_ids: Tensor) -> None:
        """Explicitly set phase for the given envs.

        Used by ``reset_on_reference`` to align the phase with a sampled
        reference pose.

        Args:
            phase: ``[len(env_ids)]`` phase values in [0, 1].
            env_ids: Env indices to set.
        """
        if env_ids.numel() == 0:
            return
        n_total = self._get_global_indices().shape[0]
        self._ensure_phase_state(n_total)
        self.phase[env_ids] = phase
        self._reseed_gate_for_envs(env_ids)

    def snap_phase_to_new_domain(self, env_ids: Tensor) -> None:
        """Early-contact snap.  Phase currently in old domain (before gate);
        jump *forward* to ``(gate_phi + _EPS_PHI) % 1.0`` — start of the
        new domain — and advance ``next_gate_idx``.
        """
        if env_ids.numel() == 0:
            return
        traj_idx = self._get_global_indices()[env_ids]
        gate_idx = self.next_gate_idx[env_ids]
        gate_phi = self._gate_phi_table[traj_idx, gate_idx]
        self.phase[env_ids] = (gate_phi + self._EPS_PHI) % 1.0
        self._advance_gate_for_envs(env_ids)

    def snap_phase_to_start_of_current_domain(self, env_ids: Tensor) -> None:
        """Late-contact snap (hold-off).  Phase already past the gate (in
        the new domain); pull *backward* a small amount to
        ``(gate_phi + _EPS_PHI) % 1.0`` — start of the same new domain —
        and advance ``next_gate_idx``.

        Same numerical target as :meth:`snap_phase_to_new_domain` but the
        domain-membership intent (already inside vs. crossing into) is
        different at the call site.
        """
        if env_ids.numel() == 0:
            return
        traj_idx = self._get_global_indices()[env_ids]
        gate_idx = self.next_gate_idx[env_ids]
        gate_phi = self._gate_phi_table[traj_idx, gate_idx]
        self.phase[env_ids] = (gate_phi + self._EPS_PHI) % 1.0
        self._advance_gate_for_envs(env_ids)

    def snap_phase_to_end_of_previous_domain(self, env_ids: Tensor) -> None:
        """Hold-on-late-contact snap.  Phase has just crossed the gate
        boundary into the new domain; pull *backward* to
        ``(gate_phi - _EPS_PHI) % 1.0`` — end of the old (previous) domain.
        Does NOT advance ``next_gate_idx`` — we're still waiting for this
        gate's contact event.
        """
        if env_ids.numel() == 0:
            return
        traj_idx = self._get_global_indices()[env_ids]
        gate_idx = self.next_gate_idx[env_ids]
        gate_phi = self._gate_phi_table[traj_idx, gate_idx]
        self.phase[env_ids] = (gate_phi - self._EPS_PHI) % 1.0
        self._refresh_gate_rel_phi(env_ids)

    # ------------------------------------------------------------------
    # Ref frame map building (needs stored metadata)
    # ------------------------------------------------------------------

    def build_ref_frame_map(self, ref_frames: list[str]) -> None:
        """Pre-compute the reference frame lookup table.

        Call this after construction if you need ``get_ref_frames_in_use``
        to work correctly.  It maps each (trajectory, domain) to the
        appropriate index in ``ref_frames``.

        Args:
            ref_frames: List of reference frame names.
        """
        T = self.num_trajectories
        D = self.max_expanded_domains
        table = torch.zeros(T, D, dtype=torch.long, device=self.device)

        for ti in range(T):
            nd = self.data["num_original_domains"][ti].item()
            ed = self.data["expanded_domains"][ti].item()
            is_half = self.data["traj_type"][ti].item() == _HALF_PERIODIC_INT

            for di in range(nd):
                frame = self._ref_frame_names_per_domain[ti][di]
                if frame in ref_frames:
                    table[ti, di] = ref_frames.index(frame)
                else:
                    raise ValueError(
                        f"Bezier frame '{frame}' not found in ref_frames: {ref_frames}"
                    )

            if is_half:
                for di in range(nd):
                    frame = self._ref_frame_names_per_domain[ti][di]
                    reflected = frame.replace("right", "TEMP").replace("left", "right").replace("TEMP", "left")
                    if reflected in ref_frames:
                        table[ti, nd + di] = ref_frames.index(reflected)
                    else:
                        raise ValueError(
                            f"Reflected frame '{reflected}' not found in ref_frames: {ref_frames}"
                        )

        self._ref_frame_domain_map = table
        self._ref_frame_map = table
        self._ref_frame_key = tuple(ref_frames)
