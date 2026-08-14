"""Adapt a foreign ONNX actor to the current Ghost environment."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from .mdp.commands import MotionCommand

ObsBuilder = Callable[["ManagerBasedRlEnv", "MotionCommand"], torch.Tensor]
NamedObservationBuilder = Callable[[tuple[str, ...]], torch.Tensor]


def parse_csv_floats(
  value: str,
  *,
  field_name: str = "values",
) -> np.ndarray:
  """Parse numeric metadata without silently discarding empty entries."""
  parts = tuple(item.strip() for item in value.split(","))
  empty = [index for index, item in enumerate(parts) if not item]
  if empty:
    raise ValueError(
      f"ONNX {field_name} metadata contains empty entries at indices {empty}."
    )
  try:
    return np.asarray([float(item) for item in parts], dtype=np.float32)
  except ValueError as error:
    raise ValueError(
      f"ONNX {field_name} metadata must contain comma-separated numbers."
    ) from error


def parse_csv_names(
  value: str | None,
  *,
  field_name: str = "names",
  required: bool = False,
) -> tuple[str, ...]:
  """Parse ordered metadata names without hiding malformed entries."""
  if value is None or not value.strip():
    if required:
      raise ValueError(f"ONNX {field_name} metadata is required and non-empty.")
    return ()
  names = tuple(item.strip() for item in value.split(","))
  empty = [index for index, name in enumerate(names) if not name]
  if empty:
    raise ValueError(
      f"ONNX {field_name} metadata contains empty entries at indices {empty}."
    )
  duplicates = sorted({name for name in names if names.count(name) > 1})
  if duplicates:
    raise ValueError(
      f"ONNX {field_name} metadata contains duplicate names: {duplicates}."
    )
  return names


def actor_observation_names(env: ManagerBasedRlEnv) -> tuple[str, ...]:
  """Return the ordered, concatenated actor-term layout exposed by MJLab."""
  manager = env.observation_manager
  if not manager.group_obs_concatenate.get("actor", False):
    raise ValueError("ONNX inference requires a concatenated actor observation.")
  names = manager.active_terms.get("actor")
  if not names:
    raise ValueError("Environment has no active actor observation terms.")
  return tuple(names)


def resolve_actor_observation(
  actor_obs: torch.Tensor,
  *,
  expected_dim: int,
  model_names: tuple[str, ...],
  current_names: tuple[str, ...],
  build_named: NamedObservationBuilder,
) -> torch.Tensor:
  """Validate an actor layout or rebuild it by the ONNX term names.

  Dimension equality alone is not layout compatibility: reordered or replaced
  terms can have the same width while changing the policy input semantics.
  """
  if actor_obs.ndim != 2:
    raise ValueError(
      f"ONNX actor observations must have shape [batch, D], got {actor_obs.shape}."
    )
  if not model_names:
    raise ValueError(
      "ONNX model has no observation_names metadata; its input layout cannot "
      "be verified safely."
    )
  if model_names == current_names:
    if actor_obs.shape[1] != expected_dim:
      raise ValueError(
        "ONNX observation_names match the environment, but the dimensions "
        f"differ ({expected_dim} expected, {actor_obs.shape[1]} produced)."
      )
    return actor_obs

  rebuilt = build_named(model_names)
  if rebuilt.ndim != 2 or rebuilt.shape[0] != actor_obs.shape[0]:
    raise ValueError(
      "Rebuilt ONNX observation must preserve the actor batch dimension; "
      f"got {tuple(rebuilt.shape)} from {tuple(actor_obs.shape)}."
    )
  if rebuilt.shape[1] != expected_dim:
    raise ValueError(
      f"Rebuilt ONNX observation has {rebuilt.shape[1]} dims, model expects "
      f"{expected_dim} (terms={model_names})."
    )
  return rebuilt


def convert_policy_actions(
  actions: torch.Tensor,
  *,
  source_scale: torch.Tensor,
  target_scale: torch.Tensor,
) -> torch.Tensor:
  """Map raw policy actions between two JointPositionAction scales."""
  return actions * source_scale / target_scale


def _command_obs(env: ManagerBasedRlEnv, command: MotionCommand) -> torch.Tensor:
  del env
  return command.command


def _motion_anchor_ori_b(
  env: ManagerBasedRlEnv, command: MotionCommand
) -> torch.Tensor:
  del env
  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  return matrix_from_quat(ori)[..., :2].reshape(ori.shape[0], -1)


def _motion_anchor_pos_b(
  env: ManagerBasedRlEnv, command: MotionCommand
) -> torch.Tensor:
  del env
  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  return pos.view(pos.shape[0], -1)


def _base_ang_vel(env: ManagerBasedRlEnv, command: MotionCommand) -> torch.Tensor:
  del env
  return quat_apply_inverse(
    command.robot_anchor_quat_w, command.robot_anchor_ang_vel_w
  )


def _base_lin_vel(env: ManagerBasedRlEnv, command: MotionCommand) -> torch.Tensor:
  del env
  return quat_apply_inverse(
    command.robot_anchor_quat_w, command.robot_anchor_lin_vel_w
  )


def _joint_pos(env: ManagerBasedRlEnv, command: MotionCommand) -> torch.Tensor:
  del env
  return command.robot_joint_pos - command.robot.data.default_joint_pos


def _joint_vel(env: ManagerBasedRlEnv, command: MotionCommand) -> torch.Tensor:
  del env
  return command.robot_joint_vel


_TERM_BUILDERS: dict[str, ObsBuilder] = {
  "command": _command_obs,
  "motion_anchor_ori_b": _motion_anchor_ori_b,
  "motion_anchor_pos_b": _motion_anchor_pos_b,
  "base_ang_vel": _base_ang_vel,
  "base_lin_vel": _base_lin_vel,
  "joint_pos": _joint_pos,
  "joint_vel": _joint_vel,
}


def assemble_named_observation(
  names: tuple[str, ...],
  parts: dict[str, torch.Tensor],
) -> torch.Tensor:
  missing = [name for name in names if name not in parts]
  if missing:
    raise ValueError(f"Cannot assemble ONNX observation; missing terms {missing}.")
  return torch.cat([parts[name] for name in names], dim=-1)


def build_tracking_observation(
  env: ManagerBasedRlEnv,
  command: MotionCommand,
  names: tuple[str, ...],
  *,
  last_action: torch.Tensor,
) -> torch.Tensor:
  """Rebuild a production tracking actor obs from Ghost simulator state."""
  parts: dict[str, torch.Tensor] = {"actions": last_action}
  unknown = [name for name in names if name not in parts and name not in _TERM_BUILDERS]
  if unknown:
    raise ValueError(
      "Foreign ONNX observation layout is not supported: "
      f"{unknown}. Known terms: {sorted(_TERM_BUILDERS)}."
    )
  for name in names:
    if name == "actions":
      continue
    parts[name] = _TERM_BUILDERS[name](env, command)
  return assemble_named_observation(names, parts)
