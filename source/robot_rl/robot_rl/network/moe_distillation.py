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
        policy,
        load_balance_coef: float = 0.01,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        self.load_balance_coef = load_balance_coef

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        self.num_updates += 1
        mean_behavior_loss = 0
        mean_lb_loss = 0
        loss = 0
        cnt = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, privileged_actions, dones in self.storage.generator():

                # inference the student for gradient computation
                actions = self.policy.act_inference(obs)

                # behavior cloning loss
                behavior_loss = self.loss_fn(actions, privileged_actions)

                # MoE load balancing loss on the student's gate weights
                obs_flat = self.policy.student.get_latent(obs)
                gate_weights = self.policy.student.mlp.gate(obs_flat)
                lb_loss = load_balancing_loss(gate_weights)

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
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                # reset dones
                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_behavior_loss /= cnt
        mean_lb_loss /= cnt
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        return {"behavior": mean_behavior_loss, "lb_loss": mean_lb_loss}
