"""Ghost q_ref 残差设计的纯逻辑原型。

该原型用于验证：以 q_ref 为中心的有限位置残差能否覆盖所需物理修正，
以及目标函数何时会优先修复坏参考，而不是盲目精确跟踪。

三个标量关节/末端只是玩具模型，并非 TK3 运动学；它只隔离真实任务中的
两个关键决策：动作中心和物理质量相对风格误差的优先级。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import sqrt
from typing import Literal


Baseline = Literal["q0", "q_ref"]

ENDPOINT_NAMES = ("left foot", "right foot", "right fist/TCP")
Q_DEFAULT = (0.0, 0.0, 0.0)
Q_REFERENCE = (0.30, -0.20, 0.15)
REFERENCE_CLEARANCE_M = (-0.05, 0.03, 0.03)
ENDPOINT_GAIN_M_PER_RAD = (0.20, 0.15, 0.20)
ACTION_SCALE_RAD = 0.25
ACTION_LIMIT = 1.0
PENETRATION_GATE_M = 0.005
SUPPORT_GAP_GATE_M = 0.01


@dataclass(frozen=True)
class PrototypeState:
  baseline: Baseline = "q_ref"
  action: tuple[float, float, float] = (0.0, 0.0, 0.0)
  previous_target: tuple[float, float, float] = Q_REFERENCE
  selected_endpoint: int = 0
  tracking_weight: float = 1.0
  feasibility_weight: float = 0.1


@dataclass(frozen=True)
class Evaluation:
  target: tuple[float, float, float]
  endpoint_clearance_m: tuple[float, float, float]
  style_rmse_rad: float
  target_step_rmse_rad: float
  penetration_m: float
  support_gap_m: float
  tracking_cost: float
  feasibility_cost: float
  smoothness_cost: float
  total_cost: float
  quality_passed: bool


def action_target(state: PrototypeState) -> tuple[float, float, float]:
  """将限幅后的策略动作映射为关节位置目标。"""
  base = Q_REFERENCE if state.baseline == "q_ref" else Q_DEFAULT
  return tuple(
    q_base + ACTION_SCALE_RAD * max(-ACTION_LIMIT, min(ACTION_LIMIT, action))
    for q_base, action in zip(base, state.action, strict=True)
  )


def evaluate(state: PrototypeState) -> Evaluation:
  """计算跟踪代价、物理质量及不可妥协的质量门槛。"""
  target = action_target(state)
  clearance = tuple(
    z_ref + gain * (q_target - q_ref)
    for z_ref, gain, q_target, q_ref in zip(
      REFERENCE_CLEARANCE_M,
      ENDPOINT_GAIN_M_PER_RAD,
      target,
      Q_REFERENCE,
      strict=True,
    )
  )
  style_rmse = sqrt(
    sum(
      (target_i - ref_i) ** 2
      for target_i, ref_i in zip(target, Q_REFERENCE, strict=True)
    )
    / len(target)
  )
  target_step_rmse = sqrt(
    sum(
      (target_i - previous_i) ** 2
      for target_i, previous_i in zip(
        target, state.previous_target, strict=True
      )
    )
    / len(target)
  )
  penetration = max(max(-z, 0.0) for z in clearance)
  support_gap = max(max(z, 0.0) for z in clearance)

  tracking_cost = (style_rmse / ACTION_SCALE_RAD) ** 2
  penetration_cost = (
    max(penetration - 0.002, 0.0) / PENETRATION_GATE_M
  ) ** 2
  support_cost = (support_gap / SUPPORT_GAP_GATE_M) ** 2
  feasibility_cost = penetration_cost + support_cost
  smoothness_cost = 0.05 * (target_step_rmse / ACTION_SCALE_RAD) ** 2

  return Evaluation(
    target=target,
    endpoint_clearance_m=clearance,
    style_rmse_rad=style_rmse,
    target_step_rmse_rad=target_step_rmse,
    penetration_m=penetration,
    support_gap_m=support_gap,
    tracking_cost=tracking_cost,
    feasibility_cost=feasibility_cost,
    smoothness_cost=smoothness_cost,
    total_cost=(
      state.tracking_weight * tracking_cost
      + state.feasibility_weight * feasibility_cost
      + smoothness_cost
    ),
    quality_passed=(
      penetration <= PENETRATION_GATE_M
      and support_gap <= SUPPORT_GAP_GATE_M
    ),
  )


def reduce_state(state: PrototypeState, key: str) -> PrototypeState:
  """根据一次 TUI 按键纯函数式更新原型状态。"""
  if key in {"1", "2", "3"}:
    return replace(state, selected_endpoint=int(key) - 1)
  if key == "t":
    return replace(
      state,
      tracking_weight=_next_value(state.tracking_weight, (0.25, 1.0, 4.0)),
    )
  if key == "p":
    return replace(
      state,
      feasibility_weight=_next_value(
        state.feasibility_weight, (0.01, 0.1, 1.0, 10.0)
      ),
    )

  old_target = action_target(state)
  if key == "b":
    baseline: Baseline = "q0" if state.baseline == "q_ref" else "q_ref"
    return replace(state, baseline=baseline, previous_target=old_target)
  if key == "x":
    return replace(
      state,
      action=(0.0, 0.0, 0.0),
      previous_target=old_target,
    )
  if key == "c":
    corrected_action = tuple(
      max(
        -ACTION_LIMIT,
        min(
          ACTION_LIMIT,
          (q_ref - q_base - z_ref / gain) / ACTION_SCALE_RAD,
        ),
      )
      for q_ref, q_base, z_ref, gain in zip(
        Q_REFERENCE,
        Q_REFERENCE if state.baseline == "q_ref" else Q_DEFAULT,
        REFERENCE_CLEARANCE_M,
        ENDPOINT_GAIN_M_PER_RAD,
        strict=True,
      )
    )
    return replace(
      state,
      action=corrected_action,
      previous_target=old_target,
    )
  if key in {"j", "k"}:
    delta = -0.1 if key == "j" else 0.1
    action = list(state.action)
    index = state.selected_endpoint
    action[index] = max(-ACTION_LIMIT, min(ACTION_LIMIT, action[index] + delta))
    return replace(
      state,
      action=tuple(action),
      previous_target=old_target,
    )
  return state


def _next_value(current: float, values: tuple[float, ...]) -> float:
  index = min(range(len(values)), key=lambda i: abs(values[i] - current))
  return values[(index + 1) % len(values)]
