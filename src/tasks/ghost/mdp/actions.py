from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


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
