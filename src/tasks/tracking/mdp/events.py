from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def randomize_actuator_command_lag(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  lag_range: tuple[int, int],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Sample one actuator-command lag per environment for the next episode."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

  min_lag, max_lag = lag_range
  if min_lag < 0 or max_lag < min_lag:
    raise ValueError(f"Invalid actuator command lag range: {lag_range}")

  lags = torch.randint(
    min_lag,
    max_lag + 1,
    (len(env_ids),),
    device=env.device,
  )
  asset: Entity = env.scene[asset_cfg.name]
  delayed_actuators = [actuator for actuator in asset.actuators if actuator.has_delay]
  if not delayed_actuators:
    raise ValueError(f"Entity '{asset_cfg.name}' has no delayed actuators")

  # Actuators fused into one MJLab delay group share a buffer. Reapplying the
  # same sampled tensor through the public API is harmless and also covers any
  # future actuator groups that use separate buffers.
  for actuator in delayed_actuators:
    actuator.set_lags(lags, env_ids)
