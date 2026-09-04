"""Absolute effort action for the Unitree Go2."""

from mjlab.envs.mdp.actions import JointEffortActionCfg

from src.assets.robots.unitree_go2.go2_constants import GO2_ARTICULATION
from src.tasks.effort.zero_pd import absolute_effort_action_cfg


def go2_effort_action_cfg() -> JointEffortActionCfg:
  """Create ``tau = clip(action * fraction * effort_limit, ±effort_limit)``."""
  return absolute_effort_action_cfg(GO2_ARTICULATION.actuators)
