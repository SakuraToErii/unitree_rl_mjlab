"""RL configuration for Unitree G1-23DOF residual-effort task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from src.tasks.effort.rl.ppo_cfg import ResidualMhaModelCfg, ResidualMlpModelCfg

# Plant saturates at |action| = 4 (0.25×Y1). Cap σ so entropy cannot
# inflate unused dimensions past that range. Do not clip actions.
_EFFORT_DISTRIBUTION_CFG = {
  "class_name": "GaussianDistribution",
  "init_std": 1.0,
  "std_type": "scalar",
  "std_range": (1e-3, 2.0),
}


def unitree_g1_23dof_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create residual-effort PPO runner configuration for Unitree G1-23DOF."""
  return RslRlOnPolicyRunnerCfg(
    actor=ResidualMlpModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg=dict(_EFFORT_DISTRIBUTION_CFG),
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.001,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_23dof_effort",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=50_000,
  )


def unitree_g1_23dof_ppo_mha_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create residual-effort MHA PPO runner configuration for Unitree G1-23DOF."""
  return RslRlOnPolicyRunnerCfg(
    actor=ResidualMhaModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg=dict(_EFFORT_DISTRIBUTION_CFG),
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.001,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_23dof_effort_mha",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
