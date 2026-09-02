"""RL configuration for Unitree Go2 residual-effort task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.effort.rl.ppo_cfg import make_effort_ppo_runner_cfg


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create residual-effort PPO runner configuration for Unitree Go2."""
  return make_effort_ppo_runner_cfg("go2_effort")


def unitree_go2_ppo_mha_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create residual-effort MHA PPO runner configuration for Unitree Go2."""
  return make_effort_ppo_runner_cfg("go2_effort_mha", mha=True)
