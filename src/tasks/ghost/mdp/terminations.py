from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse, quat_error_magnitude

from .commands import MotionCommand
from .rewards import _get_body_indexes

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def bad_anchor_pos(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """根位置相对参考的欧氏距离超过阈值则终止。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
  )


def bad_anchor_pos_z_only(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """仅比较根高度（z）偏离是否超过阈值。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1])
    > threshold
  )


def bad_anchor_ori(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
  """用投影重力 z 分量差衡量根姿态倾倒程度，超过阈值则终止。"""
  asset: Entity = env.scene[asset_cfg.name]

  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  motion_projected_gravity_b = quat_apply_inverse(
    command.anchor_quat_w, asset.data.gravity_vec_w
  )

  robot_projected_gravity_b = quat_apply_inverse(
    command.robot_anchor_quat_w, asset.data.gravity_vec_w
  )

  return (
    motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]
  ).abs() > threshold


def bad_anchor_quat(
  env: ManagerBasedRlEnv, command_name: str, threshold: float
) -> torch.Tensor:
  """根姿态四元数误差幅值超过阈值则终止。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return (
    quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w)
    > threshold
  )


def bad_motion_body_pos(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """任一选定身体的相对位置误差超过阈值则终止（常用于末端）。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.norm(
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes],
    dim=-1,
  )
  return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
  env: ManagerBasedRlEnv,
  command_name: str,
  threshold: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """任一选定身体的相对高度误差超过阈值则终止。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  body_indexes = _get_body_indexes(command, body_names)
  error = torch.abs(
    command.body_pos_relative_w[:, body_indexes, -1]
    - command.robot_body_pos_w[:, body_indexes, -1]
  )
  return torch.any(error > threshold, dim=-1)


def nonfinite_robot_state(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """机器人状态出现 NaN/Inf 时终止。"""
  asset: Entity = env.scene[asset_cfg.name]
  state = torch.cat(
    [
      asset.data.root_link_pose_w,
      asset.data.root_link_vel_w,
      asset.data.joint_pos,
      asset.data.joint_vel,
    ],
    dim=-1,
  )
  return ~torch.isfinite(state).all(dim=-1)


def hard_joint_limit_violation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  tolerance: float = 1.0e-4,
) -> torch.Tensor:
  """任一关节越过硬限位（含容差）时终止。"""
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  limits = asset.data.joint_pos_limits[:, asset_cfg.joint_ids]
  below = joint_pos < limits[..., 0] - tolerance
  above = joint_pos > limits[..., 1] + tolerance
  return torch.any(below | above, dim=-1)
