from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class JointPositionLimitAction(JointPositionAction):
  """Position action whose processed targets stay inside hard joint limits."""

  @property
  def processed_action(self) -> torch.Tensor:
    return self._processed_actions

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    limits = self._entity.data.joint_pos_limits[:, self.target_ids]
    self._processed_actions = torch.clamp(
      self._processed_actions,
      min=limits[..., 0],
      max=limits[..., 1],
    )


@dataclass(kw_only=True)
class JointPositionLimitActionCfg(JointPositionActionCfg):
  """Configuration for hard-limit-clipped joint position targets."""

  def build(self, env) -> JointPositionLimitAction:
    return JointPositionLimitAction(self, env)


class ReferenceJointPositionLimitAction(JointPositionLimitAction):
  """以当前运动参考为中心的关节位置残差动作。

  策略只负责输出 ``delta_q``，最终执行器目标为
  ``q_ref + scale * clamp(action)``。该动作与默认姿态偏置动作相互独立，
  避免两种动作语义被意外混用。
  """

  cfg: ReferenceJointPositionLimitActionCfg

  def __init__(
    self,
    cfg: ReferenceJointPositionLimitActionCfg,
    env: ManagerBasedRlEnv,
  ) -> None:
    super().__init__(cfg, env)
    self._motion_command = cast(
      MotionCommand, env.command_manager.get_term(cfg.command_name)
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw_actions[:] = actions
    # 先在策略空间限幅，再将残差映射到参考关节位置附近。
    bounded_actions = torch.clamp(
      actions,
      min=-self.cfg.residual_clip,
      max=self.cfg.residual_clip,
    )
    reference = self._motion_command.joint_pos[:, self.target_ids]
    self._processed_actions = reference + bounded_actions * self._scale

    # 最终位置目标仍需限制在硬限位内侧，给物理跟踪误差留出余量。
    limits = self._entity.data.joint_pos_limits[:, self.target_ids]
    lower = limits[..., 0]
    upper = limits[..., 1]
    mid = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * self.cfg.joint_limit_factor
    self._processed_actions = torch.clamp(
      self._processed_actions,
      min=mid - half,
      max=mid + half,
    )


@dataclass(kw_only=True)
class ReferenceJointPositionLimitActionCfg(JointPositionLimitActionCfg):
  """以运动参考为中心的残差位置目标配置。"""

  command_name: str = "motion"
  residual_clip: float = 1.0
  joint_limit_factor: float = 0.98
  use_default_offset: bool = False

  def __post_init__(self) -> None:
    super().__post_init__()
    if not 0.0 < self.residual_clip:
      raise ValueError("residual_clip must be positive.")
    if not 0.0 < self.joint_limit_factor <= 1.0:
      raise ValueError("joint_limit_factor must be in (0, 1].")

  def build(self, env) -> ReferenceJointPositionLimitAction:
    return ReferenceJointPositionLimitAction(self, env)
