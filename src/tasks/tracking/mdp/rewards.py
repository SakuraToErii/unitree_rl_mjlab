from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.actuator import BuiltinPositionActuator
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
  """Reward matching the reference joint positions."""
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
  """Reward matching the reference joint velocities."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.mean(
    torch.square(command.joint_vel - command.robot_joint_vel), dim=-1
  )
  return torch.exp(-error / std**2)


class raw_action_torque_limit_penalty:
  """Penalize large raw-action torque requests before the hard force limit.

  This is a policy-output regularizer, not a measurement of delayed/applied
  actuator torque. It supports MuJoCo built-in position actuators, whose joint
  transmission uses the unit gear created by MJLab.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    action_name = cfg.params.get("action_name", "joint_pos")
    asset_cfg = cfg.params.get("asset_cfg")
    if not isinstance(asset_cfg, SceneEntityCfg):
      raise TypeError("asset_cfg must be a SceneEntityCfg")

    soft_ratio = float(cfg.params.get("soft_ratio", 1.0))
    if not 0.0 < soft_ratio <= 1.0:
      raise ValueError(f"soft_ratio must be in (0, 1], got {soft_ratio}.")

    asset: Entity = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, JointPositionAction):
      raise TypeError(
        "raw_action_torque_limit_penalty requires a JointPositionAction, "
        f"but action term '{action_name}' is {type(action_term).__name__}."
      )

    all_joint_ids = torch.arange(
      asset.num_joints, device=env.device, dtype=torch.long
    )
    selected_joint_ids = all_joint_ids[asset_cfg.joint_ids]
    if selected_joint_ids.numel() == 0:
      raise ValueError("raw_action_torque_limit_penalty selected no joints")

    matches = selected_joint_ids[:, None] == action_term.target_ids[None, :]
    match_counts = matches.sum(dim=1)
    if torch.any(match_counts != 1):
      unmatched_names = [
        asset.joint_names[joint_id]
        for joint_id, count in zip(
          selected_joint_ids.tolist(), match_counts.tolist(), strict=True
        )
        if count != 1
      ]
      raise ValueError(
        "Each selected joint must map to exactly one action column; "
        f"invalid joints: {unmatched_names}."
      )

    joint_to_ctrl_id: dict[int, int] = {}
    for actuator in asset.actuators:
      if (
        not isinstance(actuator, BuiltinPositionActuator)
        or actuator.transmission_type != TransmissionType.JOINT
      ):
        continue
      for joint_id, ctrl_id in zip(
        actuator.target_ids.tolist(),
        actuator.global_ctrl_ids.tolist(),
        strict=True,
      ):
        if joint_id in joint_to_ctrl_id:
          raise ValueError(
            "More than one built-in position actuator controls joint "
            f"'{asset.joint_names[joint_id]}'."
          )
        joint_to_ctrl_id[joint_id] = ctrl_id

    unsupported_joint_ids = [
      joint_id
      for joint_id in selected_joint_ids.tolist()
      if joint_id not in joint_to_ctrl_id
    ]
    if unsupported_joint_ids:
      unsupported_joint_names = [
        asset.joint_names[joint_id] for joint_id in unsupported_joint_ids
      ]
      raise TypeError(
        "raw_action_torque_limit_penalty only supports MuJoCo built-in "
        f"joint-position actuators; unsupported joints: {unsupported_joint_names}."
      )

    self._asset = asset
    self._action_term = action_term
    self._soft_ratio = soft_ratio
    self._joint_ids = selected_joint_ids
    self._action_columns = torch.argmax(matches.to(torch.long), dim=1)
    self._ctrl_ids = torch.tensor(
      [joint_to_ctrl_id[joint_id] for joint_id in selected_joint_ids.tolist()],
      device=env.device,
      dtype=torch.long,
    )

    force_range = self._model_field(env, "actuator_forcerange")
    if torch.any(force_range[..., 0] >= 0.0) or torch.any(
      force_range[..., 1] <= 0.0
    ):
      raise ValueError(
        "Selected actuators must have negative/positive force limits."
      )

  def _model_field(
    self,
    env: ManagerBasedRlEnv,
    field_name: str,
  ) -> torch.Tensor:
    values = getattr(env.sim.model, field_name)
    if field_name in env.sim.expanded_fields:
      return values[:, self._ctrl_ids]
    return values[self._ctrl_ids].unsqueeze(0)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    action_name: str = "joint_pos",
    soft_ratio: float = 1.0,
  ) -> torch.Tensor:
    del asset_cfg, action_name, soft_ratio  # Resolved during initialization.

    raw_actions = self._action_term.raw_action[:, self._action_columns]
    scale = self._action_term.scale
    if isinstance(scale, torch.Tensor):
      scale = scale[:, self._action_columns]
    offset = self._action_term.offset
    if isinstance(offset, torch.Tensor):
      offset = offset[:, self._action_columns]

    position_target = raw_actions * scale + offset
    position_target -= self._asset.data.encoder_bias[:, self._joint_ids]

    gain = self._model_field(env, "actuator_gainprm")[..., 0]
    bias = self._model_field(env, "actuator_biasprm")
    requested_torque = (
      gain * position_target
      + bias[..., 1] * self._asset.data.joint_pos[:, self._joint_ids]
      + bias[..., 2] * self._asset.data.joint_vel[:, self._joint_ids]
    )

    force_range = self._model_field(env, "actuator_forcerange")
    torque_limit = torch.where(
      requested_torque >= 0.0,
      force_range[..., 1],
      -force_range[..., 0],
    )
    soft_limit = torque_limit * self._soft_ratio
    excess = torch.clamp(torch.abs(requested_torque) - soft_limit, min=0.0)
    normalized_excess = excess / soft_limit
    return torch.sum(torch.square(normalized_excess), dim=1)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
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
