"""Unitree G1-23DOF configuration for feed-forward torque control."""

from mjlab.entity import EntityCfg

from src.assets.robots.unitree_g1.g1_23dof_constants import get_g1_23dof_robot_cfg
from src.tasks.effort.zero_pd import with_zero_pd


def get_g1_23dof_effort_robot_cfg() -> EntityCfg:
  """Return a G1-23DOF config with zero-gain motors."""
  return with_zero_pd(get_g1_23dof_robot_cfg())
