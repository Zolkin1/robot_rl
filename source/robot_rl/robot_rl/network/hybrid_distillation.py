# TODO: Need to test and verify this file. Also verify all the other touched files are ok
"""Hybrid DAgger + PPO distillation algorithm.

Implements the curriculum-based training approach from "Perceptive Humanoid Parkour"
(arXiv 2602.15827): starts with pure behavior cloning (DAgger) loss and gradually
shifts to PPO loss via a weighted sum, keeping a small DAgger regularization term
throughout to prevent jittery behavior.

Loss: L = lambda_PPO * L_PPO + lambda_D * dagger_coef * L_D
"""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


class HybridDistillation:
    """Hybrid DAgger + PPO distillation algorithm.

    Combines behavior cloning from a frozen teacher with PPO reinforcement learning,
    using a curriculum that linearly shifts from DAgger-dominated to PPO-dominated
    training over a configurable number of iterations.
    """

    student: MLPModel
    """The student model (acts as the actor for PPO)."""

    teacher: MLPModel
    """The frozen teacher model."""

    critic: MLPModel
    """The critic model for value estimation."""

    teacher_loaded: bool = False
    """Indicates whether the teacher model parameters have been loaded."""

    def __init__(
        self,
        student: MLPModel,
        teacher: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        # Distillation params
        behavior_loss_coef: float = 10.0,
        loss_type: str = "mse",
        # PPO params
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.001,
        num_learning_epochs: int = 2,
        num_mini_batches: int = 4,
        learning_rate: float = 3e-4,
        critic_learning_rate: float = 1e-3,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        # Curriculum params
        curriculum_end_iteration: int = 10000,
        min_dagger_weight: float = 0.1,
        lambda_d_freeze_threshold: float = 0.0,
        # Standard
        optimizer: str = "adam",
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        **kwargs: dict,
    ) -> None:
        """Initialize the hybrid distillation algorithm."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Models
        self.student = student.to(self.device)
        self.teacher = teacher.to(self.device)
        self.critic = critic.to(self.device)

        # Create the optimizer over student params (teacher is frozen)
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.student.parameters()), lr=learning_rate
        )

        # Separate optimizer for the critic
        self.critic_optimizer = resolve_optimizer(optimizer)(chain(self.critic.parameters()), lr=critic_learning_rate)

        # Storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # Teacher actions buffer (allocated externally via construct_algorithm)
        self.teacher_actions: torch.Tensor | None = None
        self.teacher_step = 0

        # Distillation parameters
        self.behavior_loss_coef = behavior_loss_coef
        loss_fn_dict = {
            "mse": nn.functional.mse_loss,
            "huber": nn.functional.huber_loss,
        }
        if loss_type in loss_fn_dict:
            self.loss_fn = loss_fn_dict[loss_type]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: {list(loss_fn_dict.keys())}")

        # PPO parameters
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.learning_rate = learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule

        # Curriculum parameters
        self.curriculum_end_iteration = curriculum_end_iteration
        self.min_dagger_weight = min_dagger_weight
        self.lambda_d_freeze_threshold = lambda_d_freeze_threshold

        # Locate the student's state-independent std parameter (Gaussian scalar/log).
        dist = self.student.distribution
        self._std_param: torch.nn.Parameter | None = getattr(dist, "std_param", None)
        if self._std_param is None:
            self._std_param = getattr(dist, "log_std_param", None)
        if self._std_param is None and self.lambda_d_freeze_threshold > 0.0:
            raise ValueError(
                "HybridDistillation: lambda_d_freeze_threshold > 0 requires a "
                "GaussianDistribution student with a state-independent std_param/log_std_param."
            )

        # Initial lambda_d is 1.0, so warmup is active if threshold > 0.
        self._std_frozen = False
        if self._std_param is not None and self.lambda_d_freeze_threshold > 0.0:
            self._std_param.requires_grad = False
            self._std_frozen = True

        self.num_updates = 0

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions from the student and record transition data."""
        # Store hidden states for recurrent policies
        self.transition.hidden_states = (self.student.get_hidden_state(), self.critic.get_hidden_state())

        # Student: stochastic actions + log probs + distribution params
        self.transition.actions = self.student(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.student.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.student.output_distribution_params)

        # Teacher: deterministic actions stored in separate buffer
        teacher_actions = self.teacher(obs).detach()
        self.teacher_actions[self.teacher_step] = teacher_actions

        self.teacher_step += 1

        # Record the observations
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update normalizers."""
        # Update the normalizers
        self.student.update_normalization(obs)
        self.critic.update_normalization(obs)

        # Record the rewards and dones (clone rewards for bootstrapping)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),
                1,
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.student.reset(dones)
        self.teacher.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute GAE return and advantage targets from stored transitions."""
        st = self.storage
        last_values = self.critic(obs).detach()

        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]

        st.advantages = st.returns - st.values
        st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Run optimization epochs with curriculum-weighted hybrid loss."""
        self.num_updates += 1

        # --- Curriculum schedule ---
        lambda_d = max(self.min_dagger_weight, 1.0 - self.num_updates / self.curriculum_end_iteration)
        lambda_ppo = 1.0 - lambda_d

        # --- Warmup: freeze std + LR while DAgger dominates ---
        warmup_active = lambda_d >= self.lambda_d_freeze_threshold > 0.0
        if self._std_param is not None and warmup_active != self._std_frozen:
            self._std_param.requires_grad = not warmup_active
            self._std_frozen = warmup_active

        # --- Flatten storage data for mini-batching ---
        batch_size = self.storage.num_envs * self.storage.num_transitions_per_env
        mini_batch_size = batch_size // self.num_mini_batches

        observations = self.storage.observations.flatten(0, 1)
        actions = self.storage.actions.flatten(0, 1)
        values = self.storage.values.flatten(0, 1)
        returns = self.storage.returns.flatten(0, 1)
        old_actions_log_prob = self.storage.actions_log_prob.flatten(0, 1)
        advantages = self.storage.advantages.flatten(0, 1)
        old_distribution_params = tuple(p.flatten(0, 1) for p in self.storage.distribution_params)
        teacher_actions_flat = self.teacher_actions.flatten(0, 1)

        # --- Accumulators ---
        mean_surrogate_loss = 0.0
        mean_value_loss = 0.0
        mean_entropy = 0.0
        mean_behavior_loss = 0.0
        num_batch_updates = 0

        for _epoch in range(self.num_learning_epochs):
            indices = torch.randperm(self.num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

            for i in range(self.num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                # Slice the mini-batch
                batch_obs = observations[batch_idx]
                batch_actions = actions[batch_idx]
                batch_values = values[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_log_prob = old_actions_log_prob[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_old_dist_params = tuple(p[batch_idx] for p in old_distribution_params)
                batch_teacher_actions = teacher_actions_flat[batch_idx]

                # --- Forward pass ---
                self.student(batch_obs, stochastic_output=True)
                student_mean_actions = self.student.output_distribution_params[0]   # TODO: Double check this line
                new_actions_log_prob = self.student.get_output_log_prob(batch_actions)
                new_values = self.critic(batch_obs)
                distribution_params = tuple(p for p in self.student.output_distribution_params)
                entropy = self.student.output_entropy

                # --- Adaptive LR based on KL divergence (skipped during warmup) ---
                if self.desired_kl is not None and self.schedule == "adaptive" and not warmup_active:
                    with torch.inference_mode():
                        kl = self.student.get_kl_divergence(batch_old_dist_params, distribution_params)
                        kl_mean = torch.mean(kl)

                        if self.is_multi_gpu:
                            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                            kl_mean /= self.gpu_world_size

                        if self.gpu_global_rank == 0:
                            if kl_mean > self.desired_kl * 2.0:
                                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                        if self.is_multi_gpu:
                            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                            torch.distributed.broadcast(lr_tensor, src=0)
                            self.learning_rate = lr_tensor.item()

                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate

                # --- PPO surrogate loss ---
                ratio = torch.exp(new_actions_log_prob - torch.squeeze(batch_old_log_prob))
                surrogate = -torch.squeeze(batch_advantages) * ratio
                surrogate_clipped = -torch.squeeze(batch_advantages) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # --- Value loss ---
                if self.use_clipped_value_loss:
                    value_clipped = batch_values + (new_values - batch_values).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (new_values - batch_returns).pow(2)
                    value_losses_clipped = (value_clipped - batch_returns).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (batch_returns - new_values).pow(2).mean()

                # --- PPO total loss ---
                # ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
                ppo_loss_annealing = surrogate_loss - self.entropy_coef * entropy.mean()

                # --- Behavior cloning loss (student mean vs teacher mean) ---
                # student_mean_actions = self.student(batch_obs)  # deterministic forward   # NOTE: Moved up to avoid recomputation
                behavior_loss = self.loss_fn(student_mean_actions, batch_teacher_actions)

                # --- Combined loss with curriculum weighting ---
                # Only anneal the non-value part of the PPO
                loss = lambda_ppo * ppo_loss_annealing + lambda_d * self.behavior_loss_coef * behavior_loss + self.value_loss_coef * value_loss

                # --- Backward + gradient step ---
                self.optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()

                if self.is_multi_gpu:
                    self.reduce_parameters()

                nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.critic_optimizer.step()

                # --- Accumulate for logging ---
                mean_surrogate_loss += surrogate_loss.item()
                mean_value_loss += value_loss.item()
                mean_entropy += entropy.mean().item()
                mean_behavior_loss += behavior_loss.item()
                num_batch_updates += 1

        # --- Finalize ---
        mean_surrogate_loss /= num_batch_updates
        mean_value_loss /= num_batch_updates
        mean_entropy /= num_batch_updates
        mean_behavior_loss /= num_batch_updates

        self.storage.clear()
        self.teacher_step = 0

        return {
            "surrogate": mean_surrogate_loss,
            "value": mean_value_loss,
            "entropy": mean_entropy,
            "behavior": mean_behavior_loss,
            "lambda_d": lambda_d,
            "lambda_ppo": lambda_ppo,
            "warmup_active": float(warmup_active),
            "lr": self.learning_rate,
        }

    def train_mode(self) -> None:
        """Set train mode for student and critic; teacher always eval."""
        self.student.train()
        self.critic.train()
        self.teacher.eval()

    def eval_mode(self) -> None:
        """Set evaluation mode for all models."""
        self.student.eval()
        self.critic.eval()
        self.teacher.eval()

    def save(self) -> dict:
        """Return a dict of all model states for saving."""
        return {
            "student_state_dict": self.student.state_dict(),
            "teacher_state_dict": self.teacher.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load model states from a saved dict.

        Handles loading from:
        - PPO checkpoints (teacher from actor_state_dict)
        - Hybrid distillation checkpoints (all three models)
        - Distillation checkpoints (student + teacher)
        """
        if load_cfg is None:
            # Auto-detect checkpoint type
            if "critic_state_dict" in loaded_dict and "student_state_dict" in loaded_dict:
                # Hybrid distillation checkpoint
                load_cfg = {
                    "student": True,
                    "teacher": True,
                    "critic": True,
                    "optimizer": True,
                    "iteration": True,
                }
            elif any("actor_state_dict" in key for key in loaded_dict):
                # PPO checkpoint — load teacher from actor weights
                load_cfg = {"teacher": True, "iteration": False}
            else:
                # Distillation checkpoint
                load_cfg = {
                    "student": True,
                    "teacher": True,
                    "optimizer": False,
                    "iteration": False,
                }

        if load_cfg.get("student"):
            self.student.load_state_dict(loaded_dict["student_state_dict"], strict=strict)
        if load_cfg.get("teacher"):
            self.teacher.load_state_dict(
                loaded_dict.get("teacher_state_dict") or loaded_dict["actor_state_dict"], strict=strict
            )
            self.teacher_loaded = True
        if load_cfg.get("critic") and "critic_state_dict" in loaded_dict:
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer") and "optimizer_state_dict" in loaded_dict:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("optimizer") and "critic_optimizer_state_dict" in loaded_dict:
            self.critic_optimizer.load_state_dict(loaded_dict["critic_optimizer_state_dict"])

        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the student policy model."""
        return self.student

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> HybridDistillation:
        """Construct the hybrid distillation algorithm from config."""
        # Resolve class callables
        alg_class: type[HybridDistillation] = resolve_callable(cfg["algorithm"].pop("class_name"))
        student_class: type[MLPModel] = resolve_callable(cfg["student"].pop("class_name"))
        teacher_class: type[MLPModel] = resolve_callable(cfg["teacher"].pop("class_name"))
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))

        # Resolve observation groups
        default_sets = ["student", "teacher", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Hybrid distillation is not compatible with RND and symmetry extensions
        if cfg["algorithm"].get("rnd_cfg") is not None:
            raise ValueError("The RND extension is not compatible with HybridDistillation.")
        cfg["algorithm"]["rnd_cfg"] = None
        if cfg["algorithm"].get("symmetry_cfg") is not None:
            raise ValueError("The symmetry extension is not compatible with HybridDistillation.")
        cfg["algorithm"]["symmetry_cfg"] = None

        # Initialize models
        student: MLPModel = student_class(obs, cfg["obs_groups"], "student", env.num_actions, **cfg["student"]).to(
            device
        )
        print(f"Student Model: {student}")
        teacher: MLPModel = teacher_class(obs, cfg["obs_groups"], "teacher", env.num_actions, **cfg["teacher"]).to(
            device
        )
        print(f"Teacher Model: {teacher}")
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Initialize "rl" type storage for PPO data
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm
        alg: HybridDistillation = alg_class(
            student, teacher, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )

        # Allocate teacher actions buffer
        alg.teacher_actions = torch.zeros(cfg["num_steps_per_env"], env.num_envs, env.num_actions, device=device)

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        model_params = [self.student.state_dict(), self.teacher.state_dict(), self.critic.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.student.load_state_dict(model_params[0])
        self.teacher.load_state_dict(model_params[1])
        self.critic.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them."""
        all_params = list(chain(self.student.parameters(), self.critic.parameters()))
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
