"""Absolute effort action for the Unitree G1-23DOF."""

from mjlab.envs.mdp.actions import JointEffortActionCfg

from src.assets.robots.unitree_g1.g1_23dof_constants import G1_23DOF_ARTICULATION
from src.tasks.effort.zero_pd import absolute_effort_action_cfg


def g1_23dof_effort_action_cfg() -> JointEffortActionCfg:
  """Create ``tau = clip(action * fraction * effort_limit, ±effort_limit)``."""
  return absolute_effort_action_cfg(G1_23DOF_ARTICULATION.actuators)
