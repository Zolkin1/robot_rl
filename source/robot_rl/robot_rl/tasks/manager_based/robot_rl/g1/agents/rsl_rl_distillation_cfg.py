from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)

##
# MLP -> MLP Multiskill Distillation
##
@configclass
class G1nMulitskillMLP2MLPDistillationRunner(RslRlDistillationRunnerCfg):
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
class G1MultiskillMoE2MLPDistillationRunnerCfg(G1nMulitskillMLP2MLPDistillationRunner):
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
class G1MultiskillMoE2MoEDistillationRunnerCfg(G1nMulitskillMLP2MLPDistillationRunner):
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
        learning_rate=1.0e-3,
        gradient_length=15,
    )

##
# MLP -> MoE Walk-Run Distillation
##
@configclass
class G1MultiskillMLP2MoEDistillationRunnerCfg(G1nMulitskillMLP2MLPDistillationRunner):
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
        learning_rate=1.0e-3,
        gradient_length=15,
    )