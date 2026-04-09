from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
)


@configclass
class G1WalkRunDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distillation runner config for the G1 walk-run task."""

    num_steps_per_env = 120
    max_iterations = 300
    save_interval = 50
    experiment_name = "g1_walk_run_distillation"
    logger = "wandb"
    wandb_project = "robot_rl_distillation"
    obs_groups = {"student": ["student"], "teacher": ["policy"]}

    # Teacher checkpoint loading — these specify where to find the pre-trained teacher.
    # Can be overridden via CLI: --teacher_experiment, --teacher_run, --teacher_checkpoint
    teacher_experiment_name: str = "g1_multiskill_arch"
    teacher_load_run: str = ".*"
    teacher_load_checkpoint: str = "model_.*.pt"

    # TODO: Should also try an MoE here at some point
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
