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
    wandb_project = "robot_rl"
    obs_groups = {"student": ["student"], "teacher": ["policy"]}
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
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.0),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )
