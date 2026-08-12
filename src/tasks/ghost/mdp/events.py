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
  """域随机化：在实体 reset 后为每个环境采样一次执行器指令延迟（控制步数）。

  ``lag_range`` 为闭区间 ``[min_lag, max_lag]``，单位是控制步。
  仅作用于带 delay 缓冲的执行器；同属一个 MJLab delay 组的执行器共享缓冲。
  """
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

  # 融合进同一 delay 组的执行器共享缓冲；通过公开 API 重复写入同一采样结果是安全的，
  # 也兼容未来使用独立缓冲的执行器组。
  for actuator in delayed_actuators:
    actuator.set_lags(lags, env_ids)
