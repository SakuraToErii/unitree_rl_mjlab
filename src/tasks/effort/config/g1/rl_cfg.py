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
  ResidualGroupTimeModelCfg,
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


def _residual_actor_cfg(
  actor_cfg_cls: type[ResidualMlpModelCfg],
) -> ResidualMlpModelCfg:
  """Create the residual actor shared by the MLP and MHA runners."""
  return actor_cfg_cls(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg=dict(_EFFORT_DISTRIBUTION_CFG),
  )


def _effort_runner_cfg(
  actor: RslRlModelCfg,
  experiment_name: str,
  max_iterations: int,
) -> RslRlOnPolicyRunnerCfg:
  """Assemble the PPO runner shared by every G1 effort actor."""
  return RslRlOnPolicyRunnerCfg(
    actor=actor,
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
    experiment_name=experiment_name,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=max_iterations,
  )


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create residual-effort PPO runner configuration for Unitree G1."""
  return _effort_runner_cfg(
    _residual_actor_cfg(ResidualMlpModelCfg), "g1_effort", 50_000
  )


def unitree_g1_ppo_mha_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create residual-effort MHA PPO runner configuration for Unitree G1."""
  return _effort_runner_cfg(
    _residual_actor_cfg(ResidualMhaModelCfg), "g1_effort_mha", 10_000
  )


def g1_actor_cross_attn_layout(
  env_cfg: ManagerBasedRlEnvCfg,
) -> tuple[tuple[tuple[str, int], ...], tuple[str, ...], tuple[str, ...]]:
  """Name the actor terms that a per-joint history policy has to slice."""
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
    raise ValueError("Actor observations have no joint_pos term.")
  return tuple(term_dims), joint_pos_names, tuple(action_names)


def unitree_g1_ppo_group_time_runner_cfg(
  env_cfg: ManagerBasedRlEnvCfg,
) -> RslRlOnPolicyRunnerCfg:
  """Create the conditional group-time PPO runner. Summary fusion is always on."""
  term_dims, joint_pos_names, action_names = g1_actor_cross_attn_layout(env_cfg)
  actor = ResidualGroupTimeModelCfg(
    hidden_dims=(320, 160),
    activation="elu",
    obs_normalization=True,
    distribution_cfg=dict(_EFFORT_DISTRIBUTION_CFG),
    history_length=5,
    num_heads=4,
    encoder_hidden_dim=64,
    joint_attention_dim=32,
    joint_attention_heads=4,
    joint_attention_output_dim=64,
    term_dims=term_dims,
    joint_pos_names=joint_pos_names,
    action_names=action_names,
  )
  return _effort_runner_cfg(actor, "g1_effort_group_time_fusion", 10_000)
