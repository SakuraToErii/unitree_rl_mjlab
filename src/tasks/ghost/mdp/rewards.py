from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.actuator import BuiltinPositionActuator
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_error_magnitude,
  quat_inv,
  quat_mul,
  yaw_quat,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  """按 body 名筛选跟踪目标索引；``None`` 表示使用全部 body。"""
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """锚点（根）世界系位置跟踪奖励：误差平方经高斯核 ``exp(-e/std^2)`` 映射到 (0, 1]。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_xy_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """世界系根 XY 轨迹奖励；与 root-relative 身体形状奖励解耦。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(
      command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2]
    ),
    dim=-1,
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_z_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """世界系根高度奖励；单独低权重处理，允许动力学修正参考高度。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.square(
    command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  """锚点（根）世界系姿态跟踪奖励：四元数误差幅值经高斯核映射。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """根位置/heading 对齐后的身体形状奖励，不重复计算全局根平移。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  body_count = len(body_indexes)
  reference_heading = yaw_quat(command.anchor_quat_w)[:, None, :].expand(
    -1, body_count, -1
  )
  robot_heading = yaw_quat(command.robot_anchor_quat_w)[:, None, :].expand(
    -1, body_count, -1
  )
  reference_pos_h = quat_apply_inverse(
    reference_heading,
    command.body_pos_w[:, body_indexes] - command.anchor_pos_w[:, None, :],
  )
  robot_pos_h = quat_apply_inverse(
    robot_heading,
    command.robot_body_pos_w[:, body_indexes]
    - command.robot_anchor_pos_w[:, None, :],
  )
  error = torch.sum(
    torch.square(reference_pos_h - robot_pos_h),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """世界系身体位置跟踪奖励；使用未加噪的干净参考作为目标。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_w[:, body_indexes]
      - command.robot_body_pos_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """相对锚点对齐后的身体姿态跟踪奖励。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  body_count = len(body_indexes)
  reference_heading_inv = quat_inv(
    yaw_quat(command.anchor_quat_w)
  )[:, None, :].expand(-1, body_count, -1)
  robot_heading_inv = quat_inv(
    yaw_quat(command.robot_anchor_quat_w)
  )[:, None, :].expand(-1, body_count, -1)
  reference_quat_h = quat_mul(
    reference_heading_inv, command.body_quat_w[:, body_indexes]
  )
  robot_quat_h = quat_mul(
    robot_heading_inv, command.robot_body_quat_w[:, body_indexes]
  )
  error = (
    quat_error_magnitude(
      reference_quat_h,
      robot_quat_h,
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """世界系身体姿态跟踪奖励；使用未加噪的干净参考作为目标。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """世界系身体线速度跟踪奖励。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  """世界系身体角速度跟踪奖励。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_joint_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """关节位置跟踪奖励：匹配干净参考关节角。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.mean(
    torch.square(command.joint_pos - command.robot_joint_pos), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_joint_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """关节速度跟踪奖励：匹配干净参考关节角速度。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.mean(
    torch.square(command.joint_vel - command.robot_joint_vel), dim=-1
  )
  return torch.exp(-error / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """自碰撞惩罚项。

  若接触传感器提供力历史（``history_length > 0``），则统计任一接触力超过
  *force_threshold* 的子步数；否则回退为瞬时 ``found`` 接触计数。
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def undesired_ground_contact_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  allowed_body_names: tuple[str, ...],
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """非期望触地惩罚：统计允许末端以外、且受力超过阈值的地面接触数量。"""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  assert sensor.data.force is not None
  allowed = set(allowed_body_names)
  undesired_indexes = [
    index
    for index, body_name in enumerate(sensor.primary_names)
    if body_name not in allowed
  ]
  if not undesired_indexes:
    return torch.zeros(env.num_envs, device=env.device)
  force = sensor.data.force[:, undesired_indexes]
  return (torch.norm(force, dim=-1) > force_threshold).sum(dim=-1).float()
