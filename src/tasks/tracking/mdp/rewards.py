from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.actuator import BuiltinPositionActuator, XmlActuator
from mjlab.actuator.actuator import TransmissionType
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_pos_relative_w[:, body_indexes]
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
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
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
  """奖励关节位置与参考动作一致。"""
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
  """奖励关节速度与参考动作一致。"""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.mean(
    torch.square(command.joint_vel - command.robot_joint_vel), dim=-1
  )
  return torch.exp(-error / std**2)


class raw_action_torque_limit_penalty:
  """惩罚未裁剪关节动作隐含的力矩超限。

  这里使用当前策略输出的 raw action，而不是经过延迟或裁剪的位置目标。
  每次调用都重新读取运行时增益与力矩上限，使执行器域随机化能够反映到奖励中。
  运行时力矩上限已经包含资产配置中的 effort_limit 缩放，soft_ratio 会在此基础上
  再次缩小奖励的无惩罚区间。
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    action_name = cfg.params.get("action_name", "joint_pos")
    asset_cfg = cfg.params["asset_cfg"]

    asset: Entity = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, JointPositionAction):
      raise TypeError(
        "raw_action_torque_limit_penalty requires a JointPositionAction, "
        f"but action term '{action_name}' is {type(action_term).__name__}."
      )
    self._action_name = action_name

    all_joint_ids = torch.arange(
      asset.num_joints, device=env.device, dtype=torch.long
    )
    selected_joint_ids = all_joint_ids[asset_cfg.joint_ids]
    matches = selected_joint_ids[:, None] == action_term.target_ids[None, :]
    matched_joint_mask = torch.any(matches, dim=1)

    self._joint_ids = selected_joint_ids[matched_joint_mask]
    self._action_columns = torch.argmax(
      matches[matched_joint_mask].to(torch.long), dim=1
    )

    joint_to_ctrl_id: dict[int, int] = {}
    for actuator in asset.actuators:
      if actuator.transmission_type != TransmissionType.JOINT:
        continue
      is_position_actuator = isinstance(actuator, BuiltinPositionActuator) or (
        isinstance(actuator, XmlActuator)
        and actuator.command_field == "position"
      )
      if not is_position_actuator:
        continue

      for joint_id, ctrl_id in zip(
        actuator.target_ids.tolist(),
        actuator.global_ctrl_ids.tolist(),
        strict=True,
      ):
        if joint_id in joint_to_ctrl_id:
          raise ValueError(
            "raw_action_torque_limit_penalty found more than one position "
            f"actuator for joint '{asset.joint_names[joint_id]}'."
          )
        joint_to_ctrl_id[joint_id] = ctrl_id

    unsupported_joint_ids = [
      joint_id
      for joint_id in self._joint_ids.tolist()
      if joint_id not in joint_to_ctrl_id
    ]
    if unsupported_joint_ids:
      unsupported_joint_names = [
        asset.joint_names[joint_id] for joint_id in unsupported_joint_ids
      ]
      raise TypeError(
        "raw_action_torque_limit_penalty only supports MuJoCo position "
        "actuators, but the following selected joints use another "
        f"actuator type: {unsupported_joint_names}."
      )

    self._ctrl_ids = torch.tensor(
      [joint_to_ctrl_id[joint_id] for joint_id in self._joint_ids.tolist()],
      device=env.device,
      dtype=torch.long,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    soft_ratio: float = 1.0,
  ) -> torch.Tensor:
    del action_name  # 已在初始化阶段完成解析与校验。

    if not 0.0 < soft_ratio <= 1.0:
      raise ValueError(
        f"soft_ratio must be in the interval (0, 1], got {soft_ratio}."
      )
    if self._joint_ids.numel() == 0:
      return torch.zeros(env.num_envs, device=env.device)

    asset: Entity = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term(self._action_name)
    assert isinstance(action_term, JointPositionAction)

    raw_actions = action_term.raw_action[:, self._action_columns]
    scale = action_term.scale
    if isinstance(scale, torch.Tensor):
      scale = scale[:, self._action_columns]
    offset = action_term.offset
    if isinstance(offset, torch.Tensor):
      offset = offset[:, self._action_columns]

    # JointPositionAction 在应用目标位置前会扣除编码器偏置。
    raw_position_target = raw_actions * scale + offset
    raw_position_target -= asset.data.encoder_bias[:, self._joint_ids]

    stiffness = env.sim.model.actuator_gainprm[:, self._ctrl_ids, 0]
    damping = -env.sim.model.actuator_biasprm[:, self._ctrl_ids, 2]
    raw_torque = (
      stiffness
      * (raw_position_target - asset.data.joint_pos[:, self._joint_ids])
      - damping * asset.data.joint_vel[:, self._joint_ids]
    )

    force_range = env.sim.model.actuator_forcerange[:, self._ctrl_ids]
    torque_limit = torch.where(
      raw_torque >= 0.0,
      force_range[..., 1],
      -force_range[..., 0],
    )
    soft_torque_limit = torque_limit * soft_ratio
    excess = torch.clamp(torch.abs(raw_torque) - soft_torque_limit, min=0.0)
    normalized_excess = excess / soft_torque_limit.clamp_min(1.0e-6)
    return torch.amax(torch.square(normalized_excess), dim=1)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """惩罚机器人自碰撞。

  当传感器通过 ``history_length > 0`` 提供力历史时，统计任一接触力超过
  ``force_threshold`` 的物理子步数量；否则退化为当前时刻的 ``found`` 计数。
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
