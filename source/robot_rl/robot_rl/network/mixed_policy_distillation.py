import torch
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation


class MixedPolicyDistillation(Distillation):
    """Distillation where each episode is driven entirely by either the teacher or the student.

    On every env reset, a Bernoulli(teacher_action_probability) is flipped per env.
    While the coin is True, that env's executed actions come from the (deterministic)
    teacher; otherwise from the stochastic student. Both policies are always evaluated
    every step so the teacher action is recorded as the behavior-cloning target for
    every transition. Only the rollout state distribution changes — the loss in
    update() is inherited unchanged.
    """

    def __init__(
        self,
        student,
        teacher,
        storage,
        teacher_action_probability: float = 0.2,
        **kwargs,
    ) -> None:
        """Initialize the algorithm and the per-env teacher/student mask.

        Args:
            student: Student model to train.
            teacher: Frozen teacher model providing BC targets.
            storage: Rollout storage shared with the base Distillation algorithm.
            teacher_action_probability: Probability that an env, on reset, becomes
                teacher-driven for the next episode. Must be in [0, 1].
            **kwargs: Forwarded to ``Distillation.__init__``.
        """
        super().__init__(student, teacher, storage, **kwargs)
        if not 0.0 <= teacher_action_probability <= 1.0:
            raise ValueError(
                f"teacher_action_probability must be in [0, 1], got {teacher_action_probability}"
            )
        self.teacher_action_probability = float(teacher_action_probability)
        self.teacher_mask: torch.Tensor | None = None

    def _sample_mask(self, num_envs: int) -> torch.Tensor:
        """Draw a fresh per-env Bernoulli mask of shape ``(num_envs,)``."""
        probs = torch.full((num_envs,), self.teacher_action_probability, device=self.device)
        return torch.bernoulli(probs).bool()

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample student and teacher actions, store the transition, and return the mixed action."""
        student_actions = self.student(obs, stochastic_output=True).detach()
        teacher_actions = self.teacher(obs).detach()

        if self.teacher_mask is None:
            self.teacher_mask = self._sample_mask(student_actions.shape[0])

        self.transition.actions = student_actions
        self.transition.privileged_actions = teacher_actions
        self.transition.observations = obs

        mask = self.teacher_mask.view(-1, 1)
        return torch.where(mask, teacher_actions, student_actions)

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        """Delegate to the base class, then resample the teacher mask for envs that reset."""
        super().process_env_step(obs, rewards, dones, extras)
        if self.teacher_mask is None:
            return
        dones_bool = dones.bool().view(-1)
        if dones_bool.any():
            fresh = self._sample_mask(self.teacher_mask.shape[0])
            self.teacher_mask = torch.where(dones_bool, fresh, self.teacher_mask)
