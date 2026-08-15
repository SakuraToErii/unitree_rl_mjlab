"""RL configuration for the TK3 physical-motion-generator teacher."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def tk3_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the PPO runner configuration for TienKung 3 tracking."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
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
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      # 100 Hz policy: match ~50 Hz physical discount / GAE horizons.
      gamma=0.995,
      lam=0.975,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="tk3_ghost",
    save_interval=100,
    # 48 * 0.01 s ≈ 0.48 s, same physical rollout as 24 steps at 50 Hz.
    num_steps_per_env=48,
    # 50k * 48 = 2.4M common control steps; command noise fades to zero
    # by step 1.2M in the environment configuration.
    max_iterations=50_000,
    logger="tensorboard",
    clip_actions=100.0,
  )
