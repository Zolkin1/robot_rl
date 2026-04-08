##
## A Mixture of Experts (MoE) network architecture
##

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.models import MLPModel
class ExpertMLP(nn.Module):
    """A single expert: a small feedforward MLP.

    Each expert independently maps input_dim -> output_dim through
    hidden layers. All experts share the same architecture but have
    independent (randomly initialized) weights.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: list,
                 activation: str = 'elu',
                 ):
        super().__init__()

        activation_fn = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "selu": nn.SELU,
            "leaky_relu": nn.LeakyReLU,
        }[activation]

        layers = []

        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation_fn())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class GatingNetwork(nn.Module):
    """Produces softmax weights over N experts.

    A lightweight MLP that takes the same observation input and
    outputs a probability distribution (via softmax) over which
    experts to trust for this particular input.
    """

    def __init__(self,
                 input_dim: int,
                 num_experts: int,
                 hidden_dims: list[int],
                 activation: str = 'elu',
                 ):
        super().__init__()
        self.num_experts = num_experts

        activation_fn = {
            "elu": nn.ELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
        }[activation]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation_fn())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_experts))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [batch, num_experts] gate weights summing to 1."""
        logits = self.net(x)
        return F.softmax(logits, dim=-1)

class MixtureOfExperts(nn.Module):
    """Mixture of Experts: N expert MLPs + a gating network.

    Forward pass:
        1. Gate produces weights  w_i = gate(x)[i]  for each expert
        2. Each expert produces    e_i = expert_i(x)
        3. Output = sum_i( w_i * e_i )

    All experts run and outputs are blended by the gate weights (soft MoE).

    Args:
        input_dim:      Observation size (e.g. num_obs from your env)
        output_dim:     Action size (e.g. num_actions) or value (1)
        num_experts:    How many expert MLPs to create
        expert_hidden_dims: Hidden layer sizes inside each expert
        gate_hidden_dims:   Hidden layer sizes inside the gate
        activation:     Activation function name
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_experts: int = 4,
        expert_hidden_dims: list[int] = [256, 128],
        gate_hidden_dims: list[int] = [64],
        activation: str = "elu",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.output_dim = output_dim

        self.experts = nn.ModuleList([
            ExpertMLP(input_dim, output_dim, expert_hidden_dims, activation,)
            for _ in range(num_experts)
        ])

        self.gate = GatingNetwork(input_dim, num_experts, gate_hidden_dims, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim] observation tensor

        Returns:
            [batch_size, output_dim] weighted combination of expert outputs
        """
        # Gate weighs: [batch, num_experts]
        gate_weights = self.gate(x)

        # Stack all expert outputs: [batch, num_experts, output_dim]
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )

        # Weighted sum: [batch, output_dim]
        # gate_weights[:, :, None] broadcasts to [batch, num_experts, 1]
        output = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)

        return output

    def get_expert_utilization(self, x: torch.Tensor) -> dict:
        """Diagnostic: see how the gate distributes weight across experts.

        Useful for monitoring during training — if one expert always
        gets weight ~1.0, your MoE has collapsed to a single expert
        and you might want load-balancing loss.
        """
        with torch.no_grad():
            weights = self.gate(x)  # [batch, num_experts]
        return {
            "mean_weights": weights.mean(dim=0),     # avg weight per expert
            "max_weights": weights.max(dim=0).values, # peak weight per expert
            "entropy": -(weights * (weights + 1e-8).log()).sum(dim=-1).mean(),
        }


# To go into an RSL-RL pipeline
class MoEModel(MLPModel):
    """Drop-in MLPModel replacement that uses MoE internally.

    PPO instantiates this exactly like MLPModel:
        actor = MoEModel(obs, obs_groups, "actor", num_actions, **cfg)

    Extra kwargs (num_experts, expert_hidden_dims, etc.) pass through
    because MLPModel accepts **kwargs and ignores unknowns.
    """

    def __init__(
            self,
            obs,
            obs_groups,
            role,
            num_outputs,
            hidden_dims=[256, 128],
            activation="elu",
            # MoE-specific args
            num_experts=4,
            expert_hidden_dims=None,
            gate_hidden_dims=None,
            **kwargs,
    ):
        # 1) Let MLPModel build everything: normalization, distribution,
        #    and a temporary self.mlp we're about to replace
        super().__init__(
            obs, obs_groups, role, num_outputs,
            hidden_dims=hidden_dims,
            activation=activation,
            **kwargs,
        )

        # 2) Grab the input dim MLPModel computed from obs_groups
        input_dim = self.mlp[0].in_features

        # 3) Replace the plain MLP with our MoE
        self.mlp = MixtureOfExperts(
            input_dim=input_dim,
            output_dim=num_outputs,
            num_experts=num_experts,
            expert_hidden_dims=expert_hidden_dims or hidden_dims,
            gate_hidden_dims=gate_hidden_dims or [64],
            activation=activation,
        )

# ─── Load-Balancing Loss ──────────────────────────────────────────────
# Without this, MoE often collapses: one expert dominates and the rest
# go unused. This auxiliary loss encourages the gate to spread load
# evenly across experts.

def load_balancing_loss(gate_weights: torch.Tensor) -> torch.Tensor:
    """Encourages uniform expert utilization.

    From the Switch Transformer paper (Fedus et al., 2021).
    Computes: N * sum_i( fraction_i * weight_i )
    where fraction_i = fraction of tokens routed to expert i
          weight_i   = average gate weight for expert i

    Perfectly balanced -> loss = 1.0
    Fully collapsed   -> loss = N (num_experts)

    Multiply by a small coefficient (e.g. 0.01) and add to your
    PPO loss to keep experts alive.
    """
    # TODO: Come back to this loss and see if we want to change this
    num_experts = gate_weights.shape[-1]

    # Fraction of batch each expert handles (hard assignment by argmax)
    expert_assignments = gate_weights.argmax(dim=-1)  # [batch]
    fractions = torch.zeros(num_experts, device=gate_weights.device)
    for i in range(num_experts):
        fractions[i] = (expert_assignments == i).float().mean()

    # Average gate probability per expert
    mean_weights = gate_weights.mean(dim=0)  # [num_experts]

    return num_experts * (fractions * mean_weights).sum()

# ─── Quick Test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    batch_size = 4096   # typical for GPU-parallel RL
    obs_dim = 48        # e.g. a quadruped observation
    act_dim = 12        # e.g. 12 joint targets

    moe = MixtureOfExperts(
        input_dim=obs_dim,
        output_dim=act_dim,
        num_experts=4,
        expert_hidden_dims=[256, 128],
        gate_hidden_dims=[64],
        activation="elu",
    )

    x = torch.randn(batch_size, obs_dim)
    y = moe(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {sum(p.numel() for p in moe.parameters()):,}")

    # Check expert utilization
    util = moe.get_expert_utilization(x)
    print(f"Expert weights: {util['mean_weights']}")
    print(f"Gate entropy:   {util['entropy']:.3f}")

    # Load balancing loss
    gate_w = moe.gate(x)
    lb_loss = load_balancing_loss(gate_w)
    print(f"Load-balance loss: {lb_loss:.3f}  (1.0 = perfect)")