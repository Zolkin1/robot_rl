import torch
import torch.nn as nn

from rsl_rl.algorithms import Distillation

from .moe_network import load_balancing_loss


class MoEDistillation(Distillation):
    """Distillation with an auxiliary load-balancing loss for MoE students.

    Only the update() method changes. Everything else — rollout
    collection, transition recording, logging — is inherited as-is.
    """

    def __init__(
        self,
        student,
        teacher,
        storage,
        load_balance_coef: float = 0.01,
        **kwargs,
    ):
        super().__init__(student, teacher, storage, **kwargs)
        self.load_balance_coef = load_balance_coef

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_lb_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.student.reset(hidden_state=self.last_hidden_states[0])
            self.teacher.reset(hidden_state=self.last_hidden_states[1])
            self.student.detach_hidden_state()
            for batch in self.storage.generator():

                # inference the student for gradient computation
                actions = self.student(batch.observations)

                # behavior cloning loss
                behavior_loss = self.loss_fn(actions, batch.privileged_actions)

                # MoE load balancing loss on the student's gate weights
                latent = self.student.get_latent(batch.observations)
                gate_weights = self.student.mlp.gate(latent)
                lb_loss = load_balancing_loss(gate_weights)     # TODO: Need to make sure this makes sense with the sampled data. need to make sure in each batch we have a good distribution of each skill

                # total loss (accumulated over gradient_length steps)
                loss = loss + behavior_loss + self.load_balance_coef * lb_loss
                mean_behavior_loss += behavior_loss.item()
                mean_lb_loss += lb_loss.item()
                cnt += 1

                # gradient step
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

                # reset dones
                self.student.reset(batch.dones.view(-1))
                self.teacher.reset(batch.dones.view(-1))
                self.student.detach_hidden_state(batch.dones.view(-1))

        mean_behavior_loss /= cnt
        mean_lb_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = (self.student.get_hidden_state(), self.teacher.get_hidden_state())
        self.student.detach_hidden_state()

        return {"behavior": mean_behavior_loss, "lb_loss": mean_lb_loss}
