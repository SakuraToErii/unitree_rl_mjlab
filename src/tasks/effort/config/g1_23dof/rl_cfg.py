"""RL configuration for Unitree G1-23DOF residual-effort task."""

from mjlab.rl import RslRlOnPolicyRunnerCfg

from src.tasks.effort.rl.ppo_cfg import make_effort_ppo_runner_cfg


def unitree_g1_23dof_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create residual-effort PPO runner configuration for Unitree G1-23DOF."""
  return make_effort_ppo_runner_cfg("g1_23dof_effort")


def unitree_g1_23dof_ppo_mha_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create residual-effort MHA PPO runner configuration for Unitree G1-23DOF."""
  return make_effort_ppo_runner_cfg("g1_23dof_effort_mha", mha=True)
