from typing import Literal

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlRNNModelCfg,
)

##
# MLP -> MLP Multiskill Distillation
##
@configclass
class G1MulitskillMLP2MLPDistillationRunner(RslRlDistillationRunnerCfg):
    """Distillation runner config for the G1 walk-run task.
        Uses an MLP Teacher and an MLP Student.
    """

    num_steps_per_env = 120
    max_iterations = 5000
    save_interval = 200
    experiment_name = "g1_multiskill_distillation"
    logger = "wandb"
    wandb_project = "robot_rl_distillation"
    obs_groups = {"student": ["student"], "teacher": ["policy"]}

    # Teacher checkpoint loading — these specify where to find the pre-trained teacher.
    # Can be overridden via CLI: --teacher_experiment, --teacher_run, --teacher_checkpoint
    teacher_experiment_name: str = "g1"
    teacher_load_run: str = ".*"
    teacher_load_checkpoint: str = "model_.*.pt"

    student = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],    # NOTE: Trying to increase the MLP size to see if that will help
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=5.0e-4,
        gradient_length=1,  # No need for BPTT with MLPs — update every step
        loss_type="huber",
    )

##
# MoE -> MLP Walk-Run Distillation
##
@configclass
class G1MultiskillMoE2MLPDistillationRunnerCfg(G1MulitskillMLP2MLPDistillationRunner):
    """Distillation runner config for the G1 walk-run task."""

    # Teacher checkpoint loading — these specify where to find the pre-trained teacher.
    # Can be overridden via CLI: --teacher_experiment, --teacher_run, --teacher_checkpoint
    teacher_experiment_name: str = "g1_multiskill_arch"
    teacher_load_run: str = ".*"
    teacher_load_checkpoint: str = "model_.*.pt"

    student = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    teacher = RslRlMLPModelCfg(
        class_name="robot_rl.network.moe_network:MoEModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )


##
# MoE -> MoE Walk-Run Distillation
##
@configclass
class G1MultiskillMoE2MoEDistillationRunnerCfg(G1MulitskillMLP2MLPDistillationRunner):
    """Distillation runner config for the G1 walk-run task."""

    # Teacher checkpoint loading — these specify where to find the pre-trained teacher.
    # Can be overridden via CLI: --teacher_experiment, --teacher_run, --teacher_checkpoint
    teacher_experiment_name: str = "g1_multiskill_arch"
    teacher_load_run: str = ".*"
    teacher_load_checkpoint: str = "model_.*.pt"

    student = RslRlMLPModelCfg(
        class_name="robot_rl.network.moe_network:MoEModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    teacher = RslRlMLPModelCfg(
        class_name="robot_rl.network.moe_network:MoEModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="robot_rl.network.moe_distillation:MoEDistillation",
        num_learning_epochs=2,
        learning_rate=5.0e-4,
        gradient_length=1,
        loss_type="huber",
    )

##
# MLP -> MoE Walk-Run Distillation
##
@configclass
class G1MultiskillMLP2MoEDistillationRunnerCfg(G1MulitskillMLP2MLPDistillationRunner):
    """Distillation runner config for the G1 walk-run task."""

    # Teacher checkpoint loading — these specify where to find the pre-trained teacher.
    # Can be overridden via CLI: --teacher_experiment, --teacher_run, --teacher_checkpoint
    teacher_experiment_name: str = "g1"
    teacher_load_run: str = ".*"
    teacher_load_checkpoint: str = "model_.*.pt"

    student = RslRlMLPModelCfg(
        class_name="robot_rl.network.moe_network:MoEModel",
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="robot_rl.network.moe_distillation:MoEDistillation",
        num_learning_epochs=2,
        learning_rate=5.0e-4,
        gradient_length=1,
        loss_type="huber",
    )


##
# MLP -> LSTM Walk-Run Distillation
##
@configclass
class G1MultiskillMLP2LSTMDistillationRunnerCfg(G1MulitskillMLP2MLPDistillationRunner):
    """Distillation runner config for the G1 walk-run task.
    Uses an MLP Teacher and an LSTM Student.
    """

    student = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=5.0e-4,
        gradient_length=15,  # BPTT window for LSTM temporal learning
        loss_type="huber",
    )
    # TODO: Try an arch where an LSTM goes into an MLP but the observation also comes directly to the MLP. Can also try with an MoE

##
# MLP -> Transformer Walk-Run Distillation
##
@configclass
class RslRlCausalTransformerModelCfg(RslRlMLPModelCfg):
    """Configuration for the Causal Transformer model."""
    class_name: str = "robot_rl.network.transformer_network:CausalTransformerModel"
    single_obs_dim: int = 72
    history_length: int = 10
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.0

@configclass
class G1MultiskillMLP2TransformerDistillationRunnerCfg(G1MulitskillMLP2MLPDistillationRunner):
    """Distillation runner config for the G1 walk-run task.
    Uses an MLP Teacher and an LSTM Student.
    """

    student = RslRlCausalTransformerModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
        obs_normalization=False,
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=5.0e-4,
        gradient_length=15,  # BPTT window for LSTM temporal learning
        loss_type="huber",
    )


# TODO: Need to test all of this:
##
# Hybrid DAgger + PPO Algorithm Config
##
@configclass
class HybridDistillationAlgorithmCfg:
    """Configuration for the hybrid DAgger + PPO distillation algorithm.

    Combines behavior cloning with PPO using a curriculum that shifts from
    DAgger-dominated to PPO-dominated training.
    """

    class_name: str = "robot_rl.network.hybrid_distillation:HybridDistillation"
    """The algorithm class name."""

    # Distillation params
    loss_type: Literal["mse", "huber"] = "mse"
    """Loss type for the behavior cloning term."""

    behavior_loss_coef: float = 10.0
    """Scaling coefficient for the DAgger behavior loss."""

    # PPO params
    num_learning_epochs: int = 2
    """Number of learning epochs per update."""

    num_mini_batches: int = 4
    """Number of mini-batches per epoch."""

    learning_rate: float = 3e-4
    """Learning rate for the optimizer."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """Optimizer type."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    clip_param: float = 0.2
    """PPO clipping parameter."""

    gamma: float = 0.99
    """Discount factor."""

    lam: float = 0.95
    """GAE lambda."""

    value_loss_coef: float = 1.0
    """Value function loss coefficient."""

    entropy_coef: float = 0.001
    """Entropy bonus coefficient."""

    use_clipped_value_loss: bool = True
    """Whether to use clipped value loss."""

    schedule: str = "adaptive"
    """Learning rate schedule type."""

    desired_kl: float = 0.01
    """Target KL divergence for adaptive LR."""

    # Curriculum params
    curriculum_end_iteration: int = 10000
    """Iteration at which lambda_D reaches its minimum value."""

    min_dagger_weight: float = 0.1
    """Minimum DAgger weight (lambda_D floor). Never drops to zero."""


##
# Hybrid DAgger + PPO Distillation
##
@configclass
class G1HybridDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Hybrid DAgger+PPO distillation runner config for the G1 walk-run task."""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 500
    experiment_name = "g1_hybrid_distillation"
    logger = "wandb"
    wandb_project = "robot_rl_distillation"
    obs_groups = {"student": ["student"], "teacher": ["policy"], "critic": ["policy"]}

    # Teacher checkpoint loading
    teacher_experiment_name: str = "g1"
    teacher_load_run: str = ".*"
    teacher_load_checkpoint: str = "model_.*.pt"

    student = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1),
    )
    teacher = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="log"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = HybridDistillationAlgorithmCfg()