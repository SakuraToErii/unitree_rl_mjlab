"""Residual effort action for the Unitree Go2."""

from mjlab.envs.mdp.actions import JointEffortActionCfg

from src.assets.robots.unitree_go2.go2_constants import GO2_ARTICULATION
from src.tasks.effort.zero_pd import residual_effort_action_cfg


def go2_residual_effort_action_cfg() -> JointEffortActionCfg:
  """Create a 40% residual torque action around a zero baseline."""
  return residual_effort_action_cfg(GO2_ARTICULATION.actuators)
