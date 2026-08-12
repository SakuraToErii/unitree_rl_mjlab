from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
  yaw_quat,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def _command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
  return cast(MotionCommand, env.command_manager.get_term(command_name))


def _expand_heading(
  heading_quat_w: torch.Tensor, vector_w: torch.Tensor
) -> torch.Tensor:
  """Express arbitrary batched world vectors in each robot's yaw frame."""
  view_shape = (
    heading_quat_w.shape[0],
    *((1,) * (vector_w.ndim - 2)),
    4,
  )
  expanded_heading = heading_quat_w.view(view_shape).expand(
    *vector_w.shape[:-1], 4
  )
  return quat_apply_inverse(expanded_heading, vector_w)


def _rotation_6d(quat: torch.Tensor) -> torch.Tensor:
  matrix = matrix_from_quat(quat)
  return matrix[..., :2].reshape(*quat.shape[:-1], 6)


def _rotation_delta_6d(
  current_quat_w: torch.Tensor, target_quat_w: torch.Tensor
) -> torch.Tensor:
  """Target orientation relative to current orientation, in 6D form."""
  return _rotation_6d(quat_mul(quat_inv(current_quat_w), target_quat_w))


def robot_projected_gravity(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Gravity expressed in the robot anchor frame; absolute yaw is omitted."""
  command = _command(env, command_name)
  return quat_apply_inverse(
    command.robot_anchor_quat_w,
    command.robot.data.gravity_vec_w,
  )


def robot_root_height(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Privileged root height above the flat world plane."""
  command = _command(env, command_name)
  return command.robot_anchor_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3]


def robot_root_velocity_h(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Exact robot anchor linear/angular velocity in its heading frame."""
  command = _command(env, command_name)
  heading = yaw_quat(command.robot_anchor_quat_w)
  return torch.cat(
    [
      _expand_heading(heading, command.robot_anchor_lin_vel_w),
      _expand_heading(heading, command.robot_anchor_ang_vel_w),
    ],
    dim=-1,
  )


def current_motion_errors(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Current key-body orientation/velocity residuals missing from preview.

  Preview offset zero already supplies q/qdot, root pose/twist, and key-body
  position errors. Keeping only the remaining residuals avoids presenting the
  same goal twice.
  """
  command = _command(env, command_name)
  heading = yaw_quat(command.robot_anchor_quat_w)

  body_ori_error_6d = _rotation_delta_6d(
    command.robot_body_quat_w, command.noisy_body_quat_relative_w
  ).reshape(env.num_envs, -1)
  body_lin_vel_error_h = _expand_heading(
    heading,
    command.body_lin_vel_w - command.robot_body_lin_vel_w,
  ).reshape(env.num_envs, -1)
  body_ang_vel_error_h = _expand_heading(
    heading,
    command.body_ang_vel_w - command.robot_body_ang_vel_w,
  ).reshape(env.num_envs, -1)

  return torch.cat(
    [
      body_ori_error_6d,
      body_lin_vel_error_h,
      body_ang_vel_error_h,
    ],
    dim=-1,
  )


def future_motion_goal(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Horizon-major error-centric preview, clipped at the final motion frame.

  Every horizon contains q/qdot errors, root position/orientation/twist errors,
  and tracked-key-body position residuals. One held command-noise sample is
  shared by every horizon.
  """
  command = _command(env, command_name)
  horizon_count = len(command.cfg.preview_frame_offsets)
  heading = yaw_quat(command.robot_anchor_quat_w)

  robot_joint_pos = command.robot_joint_pos[:, None, :].expand(
    -1, horizon_count, -1
  )
  robot_joint_vel = command.robot_joint_vel[:, None, :].expand(
    -1, horizon_count, -1
  )
  robot_anchor_pos = command.robot_anchor_pos_w[:, None, :].expand(
    -1, horizon_count, -1
  )
  robot_anchor_quat = command.robot_anchor_quat_w[:, None, :].expand(
    -1, horizon_count, -1
  )
  robot_anchor_lin_vel = command.robot_anchor_lin_vel_w[:, None, :].expand(
    -1, horizon_count, -1
  )
  robot_anchor_ang_vel = command.robot_anchor_ang_vel_w[:, None, :].expand(
    -1, horizon_count, -1
  )

  reference_heading = yaw_quat(command.noisy_preview_anchor_quat_w)
  body_count = len(command.preview_body_names)
  reference_heading = reference_heading[:, :, None, :].expand(
    -1, -1, body_count, -1
  )
  robot_heading = heading[:, None, None, :].expand(
    -1, horizon_count, body_count, -1
  )
  reference_body_pos_h = quat_apply_inverse(
    reference_heading,
    command.preview_body_pos_w
    - command.noisy_preview_anchor_pos_w[:, :, None, :],
  )
  robot_body_pos_h = quat_apply_inverse(
    robot_heading,
    (
      command.robot_body_pos_w[:, command.preview_body_indexes]
      - command.robot_anchor_pos_w[:, None, :]
    )[:, None, :, :].expand(-1, horizon_count, -1, -1),
  )
  body_pos_error_h = (
    reference_body_pos_h - robot_body_pos_h
  ).reshape(env.num_envs, horizon_count, -1)

  per_horizon = torch.cat(
    [
      command.noisy_preview_joint_pos - robot_joint_pos,
      command.noisy_preview_joint_vel - robot_joint_vel,
      _expand_heading(
        heading, command.noisy_preview_anchor_pos_w - robot_anchor_pos
      ),
      _rotation_delta_6d(
        robot_anchor_quat, command.noisy_preview_anchor_quat_w
      ),
      _expand_heading(
        heading, command.preview_anchor_lin_vel_w - robot_anchor_lin_vel
      ),
      _expand_heading(
        heading, command.preview_anchor_ang_vel_w - robot_anchor_ang_vel
      ),
      body_pos_error_h,
    ],
    dim=-1,
  )
  return per_horizon.reshape(env.num_envs, -1)


def exact_joint_pos(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_pos[:, asset_cfg.joint_ids]


def exact_joint_vel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.joint_vel[:, asset_cfg.joint_ids]


def compact_ground_contacts(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  allowed_body_names: tuple[str, ...],
  force_scale: float = 100.0,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Allowed-end masks/forces plus compact non-end contact aggregates.

  The four allowed ends contribute mask + heading-frame 3D signed-log force.
  Other bodies contribute normalized active count, log maximum force, and the
  heading-frame signed-log resultant force. No geometry distance is observed.
  """
  if force_scale <= 0.0:
    raise ValueError("force_scale must be positive.")
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  assert sensor.data.found is not None
  assert sensor.data.force is not None

  name_to_index = {
    body_name: index for index, body_name in enumerate(sensor.primary_names)
  }
  missing_names = [
    body_name for body_name in allowed_body_names if body_name not in name_to_index
  ]
  if missing_names:
    raise ValueError(
      f"Ground-contact sensor is missing allowed bodies: {missing_names}."
    )
  allowed_indexes = [name_to_index[name] for name in allowed_body_names]
  undesired_indexes = [
    index
    for index, body_name in enumerate(sensor.primary_names)
    if body_name not in set(allowed_body_names)
  ]

  command = _command(env, command_name)
  heading = yaw_quat(command.robot_anchor_quat_w)
  contact = (sensor.data.found[:, allowed_indexes] > 0).float().reshape(
    env.num_envs, len(allowed_indexes), -1
  )
  contact = contact.amax(dim=-1)
  allowed_force_h = _expand_heading(
    heading, sensor.data.force[:, allowed_indexes]
  )
  allowed_force_log = torch.sign(allowed_force_h) * torch.log1p(
    torch.abs(allowed_force_h) / force_scale
  )

  undesired_force_w = sensor.data.force[:, undesired_indexes]
  undesired_force_mag = torch.norm(undesired_force_w, dim=-1)
  undesired_count = (
    (undesired_force_mag > force_threshold).sum(dim=-1, keepdim=True).float()
    / max(len(undesired_indexes), 1)
  )
  undesired_max_force = torch.log1p(
    undesired_force_mag.amax(dim=-1, keepdim=True) / force_scale
  )
  undesired_resultant_h = _expand_heading(
    heading, undesired_force_w.sum(dim=1)
  )
  undesired_resultant_log = torch.sign(undesired_resultant_h) * torch.log1p(
    torch.abs(undesired_resultant_h) / force_scale
  )

  return torch.cat(
    [
      contact,
      allowed_force_log.reshape(env.num_envs, -1),
      undesired_count,
      undesired_max_force,
      undesired_resultant_log,
    ],
    dim=-1,
  )


def normalized_actuator_force(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.normalized_actuator_force
