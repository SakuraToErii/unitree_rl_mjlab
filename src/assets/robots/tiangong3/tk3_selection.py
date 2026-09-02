"""Select a TK3 robot configuration for a registered task."""

from typing import Literal

from mjlab.entity import EntityCfg

from .tk3_constants import get_tk3_robot_cfg

FootCollision = Literal["xml", "sole"]


def select_tk3_robot_cfg(
  task_id: str,
  *,
  foot: FootCollision | None,
) -> EntityCfg | None:
  """Return a requested TK3 override, or ``None`` to keep the registered cfg.

  An omitted foot mode never changes the task's robot configuration. Any
  explicit foot mode is rejected for a non-TK3 task.
  """
  if foot is None:
    return None
  if foot not in ("xml", "sole"):
    raise ValueError(f"Unknown TK3 foot collision mode: {foot!r}.")
  if not task_id.startswith("TK3-"):
    raise ValueError(f"--foot is only supported for TK3 tasks, got {task_id}.")

  return get_tk3_robot_cfg(convex_sole=foot == "sole")
