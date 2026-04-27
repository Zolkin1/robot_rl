"""Distillation algorithm with sagittal-plane symmetry data augmentation.

Upstream rsl-rl rejects ``symmetry_cfg`` in ``Distillation`` because the frozen
teacher is not guaranteed to be symmetric: ``teacher(mirror(obs))`` may differ
from ``mirror(teacher(obs))``, which would create contradictory targets.

This subclass enables symmetry augmentation for distillation in two modes:

* ``"mirror"``: assume the teacher is approximately symmetric (true here, since
  it was trained with PPO + symmetry augmentation) and mirror the stored
  teacher action together with the observation. Cheap.
* ``"requery"``: re-run the frozen teacher on the mirrored observation each
  update to obtain a self-consistent target. One extra teacher forward per
  minibatch; no symmetry assumption on the teacher.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups


_TEACHER_TARGET_MODES = ("mirror", "requery")


class DistillationWithSymmetry(Distillation):
    """Distillation algorithm with optional symmetry-based data augmentation."""

    def __init__(
        self,
        student: MLPModel,
        teacher: MLPModel,
        storage: RolloutStorage,
        symmetry_cfg: dict | None = None,
        teacher_target_mode: Literal["mirror", "requery"] = "mirror",
        **kwargs,
    ) -> None:
        """Initialize the algorithm and resolve the symmetry configuration.

        Args:
            student: Student model to train.
            teacher: Frozen teacher model providing behavior-cloning targets.
            storage: Rollout storage shared with the base algorithm.
            symmetry_cfg: Resolved symmetry config dict (with ``_env`` key set by
                :func:`resolve_symmetry_config`). When ``None`` or with both
                ``use_data_augmentation`` and ``use_mirror_loss`` False, this
                subclass behaves identically to :class:`Distillation`.
            teacher_target_mode: How to obtain teacher targets for the mirrored
                observations. ``"mirror"`` reflects the stored teacher action
                (assumes symmetric teacher). ``"requery"`` re-runs the frozen
                teacher on the mirrored obs (no assumption, costs one extra
                forward per minibatch).
            **kwargs: Forwarded to :class:`Distillation`.
        """
        super().__init__(student, teacher, storage, **kwargs)

        if teacher_target_mode not in _TEACHER_TARGET_MODES:
            raise ValueError(
                f"teacher_target_mode must be one of {_TEACHER_TARGET_MODES}, got {teacher_target_mode!r}"
            )
        self.teacher_target_mode = teacher_target_mode

        if symmetry_cfg is not None:
            symmetry_cfg["data_augmentation_func"] = resolve_callable(symmetry_cfg["data_augmentation_func"])
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            if symmetry_cfg.get("use_mirror_loss", False):
                raise ValueError(
                    "use_mirror_loss is not supported for distillation; the behavior-cloning loss "
                    "across the augmented batch already enforces consistency."
                )
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        self.num_updates += 1
        mean_behavior_loss = 0.0
        loss = 0
        cnt = 0

        use_aug = self.symmetry is not None and self.symmetry["use_data_augmentation"]
        aug_func = self.symmetry["data_augmentation_func"] if use_aug else None
        aug_env = self.symmetry["_env"] if use_aug else None

        for _ in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()
            for batch in self.storage.generator():
                obs = batch.observations
                target = batch.privileged_actions

                if use_aug:
                    n_orig = target.shape[0]
                    obs, _ = aug_func(env=aug_env, obs=obs, actions=None)
                    if self.teacher_target_mode == "mirror":
                        _, target = aug_func(env=aug_env, obs=None, actions=target)
                    else:
                        with torch.no_grad():
                            target_mirror = self.teacher(obs[n_orig:]).detach()
                        target = torch.cat([target, target_mirror], dim=0)

                actions = self.student(obs)
                behavior_loss = self.loss_fn(actions, target)

                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.student.detach_hidden_state()
                    loss = 0

                self.student.reset(batch.dones.view(-1))
                self.teacher.reset(batch.dones.view(-1))
                self.student.detach_hidden_state(batch.dones.view(-1))

        mean_behavior_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
        self.student.detach_hidden_state()

        return {"behavior": mean_behavior_loss}

    @staticmethod
    def construct_algorithm(obs, env: VecEnv, cfg: dict, device: str) -> "DistillationWithSymmetry":
        """Construct the algorithm, allowing ``symmetry_cfg`` instead of rejecting it."""
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
        student_class: type[MLPModel] = resolve_callable(cfg["student"].pop("class_name"))  # type: ignore
        teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))  # type: ignore

        default_sets = ["student", "teacher"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        if cfg["algorithm"].get("rnd_cfg") is not None:
            raise ValueError("The RND extension is not compatible with Distillation.")
        cfg["algorithm"]["rnd_cfg"] = None

        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        student = student_class(obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"]).to(device)
        print(f"Student Model: {student}")
        teacher = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"]).to(device)
        print(f"Teacher Model: {teacher}")

        storage = RolloutStorage(
            "distillation", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device
        )

        alg = alg_class(
            student, teacher, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )
        return alg
