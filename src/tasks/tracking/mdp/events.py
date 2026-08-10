from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.entity import Entity


def randomize_actuator_command_lag(
  asset: Entity,
  env_ids: torch.Tensor,
  lag_range: tuple[int, int],
) -> None:
  """Sample one actuator-command lag per environment after an entity reset."""
  min_lag, max_lag = lag_range
  if min_lag < 0 or max_lag < min_lag:
    raise ValueError(f"Invalid actuator command lag range: {lag_range}")

  lags = torch.randint(
    min_lag,
    max_lag + 1,
    (len(env_ids),),
    device=env_ids.device,
  )
  delayed_actuators = [actuator for actuator in asset.actuators if actuator.has_delay]
  if not delayed_actuators:
    raise ValueError("Entity has no delayed actuators")

  # Actuators fused into one MJLab delay group share a buffer. Reapplying the
  # same sampled tensor through the public API is harmless and also covers any
  # future actuator groups that use separate buffers.
  for actuator in delayed_actuators:
    actuator.set_lags(lags, env_ids)
