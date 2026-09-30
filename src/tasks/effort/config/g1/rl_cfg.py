"""RL configuration for Unitree G1 residual-effort task."""

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from src.tasks.effort.config.g1.robot_cfg import get_g1_effort_robot_cfg
from src.tasks.effort.rl.ppo_cfg import (
  ResidualCrossAttnModelCfg,
  ResidualMhaModelCfg,
  ResidualMlpModelCfg,
)

_ACTOR_TERM_WIDTHS = {
  "base_ang_vel": 3,
  "projected_gravity": 3,
  "command": 3,
  "phase": 2,
}

# Plant saturates at |action| = 4 (0.25×Y1). Cap σ so entropy cannot
# inflate unused dimensions past that range. Do not clip actions.
_EFFORT_DISTRIBUTION_CFG = {
  "class_name": "GaussianDistribution",
  "init_std": 1.0,
  "std_type": "scalar",
  "std_range": (1e-3, 2.0),
}


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create residual-effort PPO runner configuration for Unitree G1."""
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
    experiment_name="g1_effort",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=50_000,
  )


def unitree_g1_ppo_mha_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create residual-effort MHA PPO runner configuration for Unitree G1."""
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
    experiment_name="g1_effort_mha",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10_000,
  )


def g1_actor_cross_attn_layout(
  env_cfg: ManagerBasedRlEnvCfg,
) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...], tuple[str, ...]]:
  """Name the actor terms that a cross-attention policy has to slice."""
  robot = Entity(get_g1_effort_robot_cfg())
  _, action_names = robot.find_joints(env_cfg.actions["joint_effort"].actuator_names)
  term_dims: list[tuple[str, int]] = []
  joint_pos_names: tuple[str, ...] | None = None
  for term_name, term_cfg in env_cfg.observations["actor"].terms.items():
    if term_name == "joint_pos":
      asset_cfg = term_cfg.params.get("asset_cfg")
      if asset_cfg is None:
        names = tuple(robot.joint_names)
      else:
        _, resolved = robot.find_joints(asset_cfg.joint_names)
        names = tuple(resolved)
      joint_pos_names = names
      term_dims.append((term_name, len(names)))
    elif term_name == "actions":
      term_dims.append((term_name, len(action_names)))
    elif term_name in _ACTOR_TERM_WIDTHS:
      term_dims.append((term_name, _ACTOR_TERM_WIDTHS[term_name]))
    else:
      raise ValueError(f"No feature width registered for actor term '{term_name}'.")
  if joint_pos_names is None:
    raise ValueError("Cross-attention actor observations have no joint_pos term.")
  return tuple(term_dims), joint_pos_names, tuple(action_names)


def unitree_g1_ppo_cross_attn_runner_cfg(
  env_cfg: ManagerBasedRlEnvCfg,
  *,
  use_bf16_autocast: bool = False,
) -> RslRlOnPolicyRunnerCfg:
  """Create residual-effort cross-attention PPO runner configuration for Unitree G1."""
  term_dims, joint_pos_names, action_names = g1_actor_cross_attn_layout(env_cfg)
  return RslRlOnPolicyRunnerCfg(
    actor=ResidualCrossAttnModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg=dict(_EFFORT_DISTRIBUTION_CFG),
      term_dims=term_dims,
      joint_pos_names=joint_pos_names,
      action_names=action_names,
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
      class_name=(
        "src.tasks.effort.rl.bf16_autocast:Bf16AutocastPPO"
        if use_bf16_autocast
        else "PPO"
      ),
    ),
    experiment_name="g1_effort_xattn",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
