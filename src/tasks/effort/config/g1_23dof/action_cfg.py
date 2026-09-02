"""Residual effort action for the Unitree G1-23DOF."""

from mjlab.envs.mdp.actions import JointEffortActionCfg

from src.assets.robots.unitree_g1.g1_23dof_constants import G1_23DOF_ARTICULATION
from src.tasks.effort.zero_pd import residual_effort_action_cfg


def g1_23dof_residual_effort_action_cfg() -> JointEffortActionCfg:
  """Create a 40% residual torque action around a zero baseline."""
  return residual_effort_action_cfg(G1_23DOF_ARTICULATION.actuators)
