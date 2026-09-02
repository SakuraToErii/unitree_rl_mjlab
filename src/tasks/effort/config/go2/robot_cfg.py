"""Unitree Go2 configuration for feed-forward torque control."""

from mjlab.entity import EntityCfg

from src.assets.robots.unitree_go2.go2_constants import get_go2_robot_cfg
from src.tasks.effort.zero_pd import with_zero_pd


def get_go2_effort_robot_cfg() -> EntityCfg:
  """Return a Go2 config with zero-gain motors."""
  return with_zero_pd(get_go2_robot_cfg())
