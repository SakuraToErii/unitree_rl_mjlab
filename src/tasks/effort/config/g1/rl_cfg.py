"""RL configuration for Unitree G1 residual-effort task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.effort.rl.ppo_cfg import make_effort_ppo_runner_cfg


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create residual-effort PPO runner configuration for Unitree G1."""
  return make_effort_ppo_runner_cfg("g1_effort")


def unitree_g1_ppo_mha_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create residual-effort MHA PPO runner configuration for Unitree G1."""
  return make_effort_ppo_runner_cfg("g1_effort_mha", mha=True)
