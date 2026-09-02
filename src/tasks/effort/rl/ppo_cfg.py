"""Shared residual-effort PPO runner configs."""

from dataclasses import dataclass

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class ResidualMlpModelCfg(RslRlModelCfg):
  """RSL-RL MLP config with residual-centered actor initialization."""

  class_name: str = "src.tasks.effort.rl.models:ResidualMlpModel"
  output_gain: float = 0.01


@dataclass
class ResidualMhaModelCfg(ResidualMlpModelCfg):
  """RSL-RL config for the native time-major 4+1 MHA actor."""

  class_name: str = "src.tasks.effort.rl.models:ResidualMhaModel"
  history_length: int = 5
  num_heads: int = 4
  encoder_hidden_dim: int = 256
  learnable_pos_embedding: bool = True


def _ppo_algorithm_cfg() -> RslRlPpoAlgorithmCfg:
  return RslRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.01,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
  )


def make_effort_ppo_runner_cfg(
  experiment_name: str,
  *,
  mha: bool = False,
  max_iterations: int | None = None,
) -> RslRlOnPolicyRunnerCfg:
  """Create a residual-effort PPO runner, optionally with a 5-frame MHA actor."""
  if mha:
    actor: RslRlModelCfg = ResidualMhaModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.3,
        "std_type": "log",
      },
      output_gain=0.01,
      history_length=5,
      num_heads=4,
      encoder_hidden_dim=256,
      learnable_pos_embedding=True,
    )
    iterations = 10_000 if max_iterations is None else max_iterations
  else:
    actor = ResidualMlpModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.3,
        "std_type": "log",
      },
      output_gain=0.01,
    )
    iterations = 50_000 if max_iterations is None else max_iterations

  return RslRlOnPolicyRunnerCfg(
    actor=actor,
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=_ppo_algorithm_cfg(),
    experiment_name=experiment_name,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=iterations,
  )
