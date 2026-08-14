from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import mujoco
import numpy as np
import torch
from mjlab.sensor import ContactSensor

from .mdp.actions import JointPositionLimitAction
from .mdp.commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


_ALLOWED_CONTACT_BODIES = {
  "ankle_roll_l_link",
  "ankle_roll_r_link",
  "left_tcp_link",
  "right_tcp_link",
  "wrist_pitch_l_link",
  "wrist_pitch_r_link",
}
_FOOT_BODIES = ("ankle_roll_l_link", "ankle_roll_r_link")


def sha256_file(path: str | Path) -> str:
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
  return tensor.detach().cpu().numpy().copy()


def _longest_true_run(mask: np.ndarray) -> int:
  longest = 0
  current = 0
  for value in np.asarray(mask, dtype=bool):
    current = current + 1 if value else 0
    longest = max(longest, current)
  return longest


def _quat_rotate_inverse(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
  quat_xyz = quat_wxyz[1:]
  return (
    vector
    - 2.0 * quat_wxyz[0] * np.cross(quat_xyz, vector)
    + 2.0 * np.cross(quat_xyz, np.cross(quat_xyz, vector))
  )


@dataclass
class PhysicsRolloutRecorder:
  """Record post-forward simulator state at policy boundaries."""

  env: ManagerBasedRlEnv
  _samples: dict[str, list[np.ndarray]] = field(default_factory=dict, init=False)

  def __post_init__(self) -> None:
    if self.env.num_envs != 1:
      raise ValueError("Physics rollout requires exactly one environment.")

    self.robot = cast("Entity", self.env.scene["robot"])
    self.command = cast(
      MotionCommand, self.env.command_manager.get_term("motion")
    )
    self.action_term = cast(
      JointPositionLimitAction,
      self.env.action_manager.get_term("joint_pos"),
    )
    self.ground_contact = self._contact_sensor("ground_contact")
    self.ground_penetration = self._contact_sensor("ground_penetration")
    self.self_collision = self._contact_sensor("self_collision")

    self.body_names = tuple(self.robot.body_names)
    self.joint_names = tuple(self.robot.joint_names)
    self._contact_indexes = self._indexes_in_order(
      self.ground_contact.primary_names, self.body_names, "ground contact"
    )
    self._penetration_indexes = self._indexes_in_order(
      self.ground_penetration.primary_names, self.body_names, "ground penetration"
    )
    self._tracked_body_indexes = [
      self.body_names.index(name) for name in self.command.cfg.body_names
    ]

    ctrl_ids = self.robot.indexing.ctrl_ids.detach().cpu().numpy()
    force_range = self.env.sim.mj_model.actuator_forcerange[ctrl_ids]
    self.actuator_effort_limits = np.max(np.abs(force_range), axis=-1).astype(
      np.float32
    )
    self.joint_limits = _to_numpy(self.robot.data.joint_pos_limits[0])

  def _contact_sensor(self, name: str) -> ContactSensor:
    sensor = self.env.scene[name]
    if not isinstance(sensor, ContactSensor):
      raise TypeError(f"Scene entry {name!r} is not a ContactSensor.")
    return sensor

  @staticmethod
  def _indexes_in_order(
    available: list[str],
    requested: tuple[str, ...],
    label: str,
  ) -> list[int]:
    index_by_name = {name: index for index, name in enumerate(available)}
    missing = [name for name in requested if name not in index_by_name]
    if missing:
      raise ValueError(f"{label} sensor is missing robot bodies: {missing}.")
    return [index_by_name[name] for name in requested]

  def _append(self, name: str, value: np.ndarray | bool | int) -> None:
    array = np.asarray(value)
    self._samples.setdefault(name, []).append(array.copy())

  def capture(self, *, terminated: bool = False) -> None:
    """Capture the current post-forward state for environment zero."""
    origin = self.env.scene.env_origins[0]
    body_pos_w = self.robot.data.body_link_pos_w[0] - origin
    body_quat_w = self.robot.data.body_link_quat_w[0]
    body_lin_vel_w = self.robot.data.body_link_lin_vel_w[0]
    body_ang_vel_w = self.robot.data.body_link_ang_vel_w[0]

    assert self.ground_contact.data.found is not None
    assert self.ground_contact.data.force is not None
    assert self.ground_penetration.data.dist is not None
    assert self.self_collision.data.found is not None

    contact_mask = (
      self.ground_contact.data.found[0, self._contact_indexes] > 0
    )
    contact_force_w = self.ground_contact.data.force[
      0, self._contact_indexes
    ]
    contact_dist = self.ground_penetration.data.dist[
      0, self._penetration_indexes
    ]

    self._append("joint_pos", _to_numpy(self.robot.data.joint_pos[0]))
    self._append("joint_vel", _to_numpy(self.robot.data.joint_vel[0]))
    self._append("body_pos_w", _to_numpy(body_pos_w))
    self._append("body_quat_w", _to_numpy(body_quat_w))
    self._append("body_lin_vel_w", _to_numpy(body_lin_vel_w))
    self._append("body_ang_vel_w", _to_numpy(body_ang_vel_w))
    self._append("contact_mask", _to_numpy(contact_mask))
    self._append("contact_force_w", _to_numpy(contact_force_w))
    self._append("contact_dist", _to_numpy(contact_dist))
    self._append(
      "self_collision",
      bool(torch.any(self.self_collision.data.found[0] > 0).item()),
    )
    self._append(
      "reference_index", int(self.command.reference_index[0].item())
    )
    self._append("raw_action", _to_numpy(self.env.action_manager.action[0]))
    self._append(
      "processed_action", _to_numpy(self.action_term.processed_action[0])
    )
    self._append(
      "joint_pos_target", _to_numpy(self.robot.data.joint_pos_target[0])
    )
    self._append(
      "actuator_force", _to_numpy(self.robot.data.actuator_force[0])
    )
    self._append(
      "qfrc_actuator", _to_numpy(self.robot.data.qfrc_actuator[0])
    )
    self._append("termination", terminated)
    self._append(
      "source_anchor_pos_w",
      _to_numpy(self.command.anchor_pos_w[0] - origin),
    )
    self._append(
      "source_body_pos_w",
      _to_numpy(self.command.body_pos_w[0] - origin),
    )

  @property
  def frame_count(self) -> int:
    return len(self._samples.get("reference_index", ()))

  def state_is_finite(self) -> bool:
    """Check the simulator state explicitly when play has no terminations."""
    tensors = (
      self.robot.data.joint_pos,
      self.robot.data.joint_vel,
      self.robot.data.body_link_pos_w,
      self.robot.data.body_link_quat_w,
      self.robot.data.body_link_lin_vel_w,
      self.robot.data.body_link_ang_vel_w,
      self.robot.data.actuator_force,
    )
    return all(bool(torch.isfinite(value).all().item()) for value in tensors)

  def build_payload(self, provenance: dict[str, str]) -> dict[str, np.ndarray]:
    if self.frame_count == 0:
      raise RuntimeError("No rollout samples were recorded.")

    payload = {
      name: np.stack(values).astype(self._dtype_for(name), copy=False)
      for name, values in self._samples.items()
    }
    payload.update(
      {
        "fps": np.asarray([1.0 / self.env.step_dt], dtype=np.float64),
        "sim_dt": np.asarray(
          [self.env.cfg.sim.mujoco.timestep], dtype=np.float64
        ),
        "decimation": np.asarray([self.env.cfg.decimation], dtype=np.int64),
        "joint_names": np.asarray(self.joint_names, dtype=np.str_),
        "body_names": np.asarray(self.body_names, dtype=np.str_),
        "source_body_names": np.asarray(
          self.command.cfg.body_names, dtype=np.str_
        ),
        "actuator_names": np.asarray(
          [actuator.name for actuator in self.robot.spec.actuators],
          dtype=np.str_,
        ),
        "actuator_effort_limits": self.actuator_effort_limits,
      }
    )
    for key, value in provenance.items():
      payload[key] = np.asarray(value, dtype=np.str_)
    return payload

  @staticmethod
  def _dtype_for(name: str) -> np.dtype:
    if name in {"contact_mask", "self_collision", "termination"}:
      return np.dtype(np.bool_)
    if name == "reference_index":
      return np.dtype(np.int64)
    return np.dtype(np.float32)

  def _kinematic_consistency(
    self, payload: dict[str, np.ndarray]
  ) -> dict[str, float]:
    """Reconstruct saved states with native MuJoCo and compare link quantities."""
    model = self.env.sim.mj_model
    data = mujoco.MjData(model)
    body_ids = self.robot.indexing.body_ids.detach().cpu().numpy()
    joint_q_adrs = self.robot.indexing.joint_q_adr.detach().cpu().numpy()
    joint_v_adrs = self.robot.indexing.joint_v_adr.detach().cpu().numpy()
    free_q_adrs = self.robot.indexing.free_joint_q_adr.detach().cpu().numpy()
    free_v_adrs = self.robot.indexing.free_joint_v_adr.detach().cpu().numpy()
    root_body_id = self.robot.indexing.root_body_id
    root_index = self.body_names.index(self.command.cfg.anchor_body_name)

    max_pos_error = 0.0
    max_ori_error = 0.0
    max_lin_vel_error = 0.0
    max_ang_vel_error = 0.0
    for frame in range(payload["joint_pos"].shape[0]):
      data.qpos[:] = model.qpos0
      data.qvel[:] = 0.0
      root_quat = payload["body_quat_w"][frame, root_index]
      root_ang_vel_w = payload["body_ang_vel_w"][frame, root_index]
      data.qpos[free_q_adrs[:3]] = payload["body_pos_w"][frame, root_index]
      data.qpos[free_q_adrs[3:7]] = root_quat
      data.qvel[free_v_adrs[:3]] = payload["body_lin_vel_w"][frame, root_index]
      data.qvel[free_v_adrs[3:6]] = _quat_rotate_inverse(
        root_quat, root_ang_vel_w
      )
      data.qpos[joint_q_adrs] = payload["joint_pos"][frame]
      data.qvel[joint_v_adrs] = payload["joint_vel"][frame]
      mujoco.mj_forward(model, data)

      reconstructed_pos = data.xpos[body_ids]
      reconstructed_quat = data.xquat[body_ids]
      reconstructed_ang_vel = data.cvel[body_ids, :3]
      offset = data.subtree_com[root_body_id] - reconstructed_pos
      reconstructed_lin_vel = (
        data.cvel[body_ids, 3:]
        - np.cross(reconstructed_ang_vel, offset)
      )

      max_pos_error = max(
        max_pos_error,
        float(
          np.max(
            np.abs(reconstructed_pos - payload["body_pos_w"][frame])
          )
        ),
      )
      quat_dot = np.sum(
        reconstructed_quat * payload["body_quat_w"][frame], axis=-1
      )
      max_ori_error = max(
        max_ori_error,
        float(np.max(1.0 - np.minimum(1.0, np.abs(quat_dot)))),
      )
      max_lin_vel_error = max(
        max_lin_vel_error,
        float(
          np.max(
            np.abs(
              reconstructed_lin_vel - payload["body_lin_vel_w"][frame]
            )
          )
        ),
      )
      max_ang_vel_error = max(
        max_ang_vel_error,
        float(
          np.max(
            np.abs(
              reconstructed_ang_vel - payload["body_ang_vel_w"][frame]
            )
          )
        ),
      )

    return {
      "fk_position_error_max_m": max_pos_error,
      "fk_orientation_dot_error_max": max_ori_error,
      "fk_linear_velocity_error_max_m_s": max_lin_vel_error,
      "fk_angular_velocity_error_max_rad_s": max_ang_vel_error,
    }

  def quality_report(
    self,
    payload: dict[str, np.ndarray],
    *,
    expected_frame_count: int | None = None,
  ) -> dict[str, Any]:
    fps = float(payload["fps"][0])
    reference_index = payload["reference_index"]
    body_pos = payload["body_pos_w"]
    body_lin_vel = payload["body_lin_vel_w"]
    joint_pos = payload["joint_pos"]
    joint_vel = payload["joint_vel"]
    contact_mask = payload["contact_mask"]
    contact_dist = payload["contact_dist"]
    actuator_force = payload["actuator_force"]
    finite = all(
      np.isfinite(array).all()
      for array in payload.values()
      if np.issubdtype(array.dtype, np.number)
    )
    consistency = (
      self._kinematic_consistency(payload)
      if finite
      else {
        "fk_position_error_max_m": float("inf"),
        "fk_orientation_dot_error_max": float("inf"),
        "fk_linear_velocity_error_max_m_s": float("inf"),
        "fk_angular_velocity_error_max_rad_s": float("inf"),
      }
    )
    quaternion_norm_error = float(
      np.max(np.abs(np.linalg.norm(payload["body_quat_w"], axis=-1) - 1.0))
    )
    contiguous_reference = bool(
      np.array_equal(
        reference_index,
        np.arange(reference_index[0], reference_index[0] + len(reference_index)),
      )
    )
    expected_frames = (
      int(expected_frame_count)
      if expected_frame_count is not None
      else int(self.command.motion.time_step_total)
    )
    complete_clip = bool(
      reference_index[0] == 0
      and reference_index[-1] == expected_frames - 1
      and len(reference_index) == expected_frames
    )

    below = joint_pos < self.joint_limits[None, :, 0] - 1.0e-4
    above = joint_pos > self.joint_limits[None, :, 1] + 1.0e-4
    hard_limit_violations = int(np.count_nonzero(below | above))

    normalized_effort = np.abs(actuator_force) / np.maximum(
      self.actuator_effort_limits[None, :], 1.0e-6
    )
    effort_exceedances = int(np.count_nonzero(normalized_effort > 1.0 + 1.0e-5))
    saturation_ratio = np.mean(normalized_effort >= 0.999, axis=0)

    penetration = np.maximum(0.0, -contact_dist)
    max_penetration = float(np.max(penetration))
    deep_penetration_frames = int(
      np.count_nonzero(np.any(penetration > 0.005, axis=1))
    )

    root_index = self.body_names.index(self.command.cfg.anchor_body_name)
    root_velocity_z = body_lin_vel[:, root_index, 2]
    root_acceleration_z = (
      np.gradient(root_velocity_z, 1.0 / fps)
      if len(root_velocity_z) >= 2
      else np.zeros_like(root_velocity_z)
    )
    no_ground_contact = ~np.any(contact_mask, axis=1)
    unsupported_static = (
      no_ground_contact
      & (np.abs(root_velocity_z) < 0.1)
      & (np.abs(root_acceleration_z + 9.81) > 2.0)
    )
    unsupported_static_duration = _longest_true_run(unsupported_static) / fps

    root_error = np.linalg.norm(
      body_pos[:, root_index] - payload["source_anchor_pos_w"], axis=-1
    )
    tracked_actual = body_pos[:, self._tracked_body_indexes]
    body_error = np.linalg.norm(
      tracked_actual - payload["source_body_pos_w"], axis=-1
    )

    foot_indexes = [self.body_names.index(name) for name in _FOOT_BODIES]
    foot_contact = contact_mask[:, foot_indexes]
    foot_speed = np.linalg.norm(
      body_lin_vel[:, foot_indexes, :2], axis=-1
    )
    foot_slip_samples = foot_speed[foot_contact]
    mean_foot_slip = (
      float(np.mean(foot_slip_samples)) if foot_slip_samples.size else 0.0
    )

    if len(joint_vel) >= 2:
      joint_acceleration = np.gradient(joint_vel, 1.0 / fps, axis=0)
      joint_jerk = np.gradient(joint_acceleration, 1.0 / fps, axis=0)
    else:
      joint_jerk = np.zeros_like(joint_vel)
    joint_jerk_rms = float(np.sqrt(np.mean(np.square(joint_jerk))))

    allowed_indexes = {
      self.body_names.index(name)
      for name in _ALLOWED_CONTACT_BODIES
      if name in self.body_names
    }
    undesired_indexes = [
      index for index in range(len(self.body_names)) if index not in allowed_indexes
    ]
    undesired_contact_frames = int(
      np.count_nonzero(np.any(contact_mask[:, undesired_indexes], axis=1))
    )

    root_rmse = float(np.sqrt(np.mean(np.square(root_error))))
    root_p95 = float(np.percentile(root_error, 95))
    quality_passed = bool(
      finite
      and quaternion_norm_error <= 1.0e-3
      and consistency["fk_position_error_max_m"] <= 1.0e-4
      and consistency["fk_orientation_dot_error_max"] <= 1.0e-5
      and consistency["fk_linear_velocity_error_max_m_s"] <= 1.0e-3
      and consistency["fk_angular_velocity_error_max_rad_s"] <= 1.0e-3
      and contiguous_reference
      and complete_clip
      and hard_limit_violations == 0
      and effort_exceedances == 0
      and deep_penetration_frames == 0
      and unsupported_static_duration <= 1.0
      and root_rmse <= 0.15
      and root_p95 <= 0.25
      and not np.any(payload["termination"])
    )

    return {
      "quality_passed": quality_passed,
      "frame_count": len(reference_index),
      "expected_frame_count": expected_frames,
      "finite": bool(finite),
      "quaternion_norm_error_max": quaternion_norm_error,
      **consistency,
      "contiguous_reference": contiguous_reference,
      "complete_clip": complete_clip,
      "hard_joint_limit_violations": hard_limit_violations,
      "actuator_effort_exceedances": effort_exceedances,
      "actuator_saturation_ratio_by_name": {
        name: float(ratio)
        for name, ratio in zip(
          payload["actuator_names"].tolist(), saturation_ratio, strict=True
        )
      },
      "max_penetration_m": max_penetration,
      "deep_penetration_frames_gt_5mm": deep_penetration_frames,
      "unsupported_static_duration_s": float(unsupported_static_duration),
      "root_error_rmse_m": root_rmse,
      "root_error_p95_m": root_p95,
      "body_mpjpe_m": float(np.mean(body_error)),
      "joint_jerk_rms_rad_s3": joint_jerk_rms,
      "mean_contact_foot_slip_m_s": mean_foot_slip,
      "undesired_contact_frames": undesired_contact_frames,
      "self_collision_frames": int(np.count_nonzero(payload["self_collision"])),
      "termination_frames": int(np.count_nonzero(payload["termination"])),
    }


def save_rollout(
  output_file: str | Path,
  payload: dict[str, np.ndarray],
  metrics: dict[str, Any],
) -> tuple[Path, Path]:
  output_path = Path(output_file).expanduser().resolve()
  if output_path.suffix != ".npz":
    output_path = output_path.with_suffix(".npz")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output_path, **payload)

  metrics_path = output_path.with_suffix(".metrics.json")
  metrics_path.write_text(
    json.dumps(metrics, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
  )
  return output_path, metrics_path


class RolloutQualityError(RuntimeError):
  """A rejected rollout whose diagnostic artifacts were saved successfully."""

  def __init__(
    self, npz_path: Path, metrics_path: Path, reason: str
  ) -> None:
    self.npz_path = npz_path
    self.metrics_path = metrics_path
    self.reason = reason
    super().__init__(
      f"{reason}; failed rollout diagnostics were saved to {npz_path} "
      f"and {metrics_path}."
    )


def _failed_rollout_path(output_file: str | Path) -> Path:
  output_path = Path(output_file).expanduser().resolve()
  if output_path.suffix != ".npz":
    output_path = output_path.with_suffix(".npz")
  if output_path.stem.endswith(".failed"):
    return output_path
  return output_path.with_name(f"{output_path.stem}.failed.npz")


def save_rollout_checked(
  output_file: str | Path,
  payload: dict[str, np.ndarray],
  metrics: dict[str, Any],
  *,
  failure_reason: str | None = None,
) -> tuple[Path, Path]:
  """Save an accepted rollout or raise after saving ``*.failed`` diagnostics.

  The requested output path is reserved for a quality-passing artifact. Any
  early runtime failure or failed quality gate is written to a sibling
  ``<stem>.failed.npz`` and ``<stem>.failed.metrics.json`` before raising.
  """
  accepted = bool(metrics.get("quality_passed", False)) and failure_reason is None
  saved_metrics = dict(metrics)
  saved_metrics["artifact_status"] = "accepted" if accepted else "failed"
  if not accepted:
    failure_reason = failure_reason or "quality gates failed"
    saved_metrics["failure_reason"] = failure_reason
    output_file = _failed_rollout_path(output_file)

  npz_path, metrics_path = save_rollout(output_file, payload, saved_metrics)
  if not accepted:
    assert failure_reason is not None
    raise RolloutQualityError(npz_path, metrics_path, failure_reason)
  return npz_path, metrics_path
