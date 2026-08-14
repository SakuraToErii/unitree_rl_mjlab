"""Validated motion-array model shared by Ghost loading and chaining."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MOTION_ARRAY_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def _name_list(values: np.ndarray | None) -> tuple[str, ...] | None:
  if values is None:
    return None
  return tuple(str(name) for name in np.asarray(values).tolist())


def _reorder(
  array: np.ndarray,
  source_names: tuple[str, ...] | None,
  target_names: tuple[str, ...],
  axis: int,
  label: str,
) -> np.ndarray:
  if source_names is None:
    if array.shape[axis] != len(target_names):
      raise ValueError(
        f"{label} has {array.shape[axis]} entries but expected "
        f"{len(target_names)}; add names to the motion."
      )
    return array
  index_by_name = {name: index for index, name in enumerate(source_names)}
  missing = [name for name in target_names if name not in index_by_name]
  if missing:
    raise ValueError(f"{label} is missing required names: {missing}.")
  indexes = [index_by_name[name] for name in target_names]
  return np.take(array, indexes, axis=axis)


@dataclass(frozen=True)
class MotionClip:
  """Validated motion arrays plus metadata behind one small interface."""

  _payload: dict[str, np.ndarray]
  frame_count: int

  @classmethod
  def from_mapping(
    cls,
    motion: Mapping[str, np.ndarray],
    *,
    minimum_frames: int = 1,
    require_fps: bool = False,
  ) -> MotionClip:
    if minimum_frames < 1:
      raise ValueError("minimum_frames must be positive.")
    missing = [key for key in MOTION_ARRAY_KEYS if key not in motion]
    if missing:
      raise ValueError(f"Motion is missing required arrays: {missing}.")

    payload = {key: np.asarray(value) for key, value in motion.items()}
    joint_pos = payload["joint_pos"]
    joint_vel = payload["joint_vel"]
    body_pos_w = payload["body_pos_w"]
    body_quat_w = payload["body_quat_w"]
    body_lin_vel_w = payload["body_lin_vel_w"]
    body_ang_vel_w = payload["body_ang_vel_w"]

    if joint_pos.ndim != 2 or joint_vel.shape != joint_pos.shape:
      raise ValueError(
        "joint_pos and joint_vel must both have shape [T, num_joints]."
      )
    if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
      raise ValueError("body_pos_w must have shape [T, num_bodies, 3].")
    if body_lin_vel_w.shape != body_pos_w.shape:
      raise ValueError("body_lin_vel_w shape must match body_pos_w.")
    if body_ang_vel_w.shape != body_pos_w.shape:
      raise ValueError("body_ang_vel_w shape must match body_pos_w.")
    if body_quat_w.shape != (*body_pos_w.shape[:2], 4):
      raise ValueError("body_quat_w must have shape [T, num_bodies, 4].")

    frame_counts = {
      key: int(payload[key].shape[0]) for key in MOTION_ARRAY_KEYS
    }
    if len(set(frame_counts.values())) != 1:
      raise ValueError(
        f"Motion arrays have inconsistent frame counts: {frame_counts}."
      )
    frame_count = frame_counts["joint_pos"]
    if frame_count < minimum_frames:
      raise ValueError(
        f"Motion must contain at least {minimum_frames} frame(s), "
        f"got {frame_count}."
      )
    nonnumeric = [
      key
      for key in MOTION_ARRAY_KEYS
      if not np.issubdtype(payload[key].dtype, np.number)
    ]
    if nonnumeric:
      raise ValueError(f"Motion arrays must be numeric: {nonnumeric}.")
    nonfinite = [
      key for key in MOTION_ARRAY_KEYS if not np.isfinite(payload[key]).all()
    ]
    if nonfinite:
      raise ValueError(f"Motion arrays contain NaN or Inf: {nonfinite}.")
    quat_norm = np.linalg.norm(body_quat_w, axis=-1)
    if not np.allclose(quat_norm, 1.0, atol=1.0e-3):
      max_error = float(np.max(np.abs(quat_norm - 1.0)))
      raise ValueError(
        "body_quat_w contains non-unit quaternions "
        f"(max norm error {max_error:g})."
      )

    if require_fps and "fps" not in payload:
      raise ValueError("Motion is missing fps.")
    if "fps" in payload:
      fps_values = payload["fps"].reshape(-1)
      if fps_values.size != 1:
        raise ValueError("Motion fps must contain exactly one value.")
      fps = float(fps_values[0])
      if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Motion fps must be positive and finite, got {fps}.")

    cls._validate_names(payload, "joint_names", joint_pos.shape[1])
    cls._validate_names(payload, "body_names", body_pos_w.shape[1])
    return cls(payload, frame_count)

  @staticmethod
  def _validate_names(
    payload: Mapping[str, np.ndarray], key: str, expected_count: int
  ) -> None:
    if key not in payload:
      return
    raw_names = np.asarray(payload[key])
    if raw_names.ndim != 1:
      raise ValueError(f"Motion metadata {key!r} must be one-dimensional.")
    if raw_names.dtype.kind not in "SU":
      raise ValueError(f"Motion metadata {key!r} must contain strings.")
    values = tuple(str(name) for name in raw_names.tolist())
    if len(values) != expected_count:
      raise ValueError(
        f"Motion metadata {key!r} has {len(values)} names but expected "
        f"{expected_count}."
      )
    if len(values) != len(set(values)):
      raise ValueError(f"Motion metadata {key!r} contains duplicate names.")

  def to_dict(self, *, copy: bool = True) -> dict[str, np.ndarray]:
    if not copy:
      return dict(self._payload)
    return {key: value.copy() for key, value in self._payload.items()}

  def for_motion_loader(self) -> tuple[dict[str, np.ndarray], int]:
    """Return a >=2-frame loader payload without changing source coverage."""
    payload = self.to_dict()
    if self.frame_count == 1:
      for key in MOTION_ARRAY_KEYS:
        payload[key] = np.repeat(payload[key], 2, axis=0)
    return payload, self.frame_count

  def sliced(self, start: int, end: int | None) -> MotionClip:
    """Return an inclusive source-frame slice, including a one-frame slice."""
    last = self.frame_count - 1 if end is None else end
    if start < 0 or last < start or last >= self.frame_count:
      raise ValueError(
        f"Frame range [{start}, {last}] is outside [0, {self.frame_count - 1}]."
      )

    sliced: dict[str, np.ndarray] = {}
    for key, array in self._payload.items():
      is_frame_array = key in MOTION_ARRAY_KEYS or (
        array.ndim >= 1
        and array.shape[0] == self.frame_count
        and key != "fps"
        and array.dtype.kind not in "SUO"
      )
      sliced[key] = (
        array[start : last + 1].copy()
        if is_frame_array
        else array.copy()
      )
    return MotionClip.from_mapping(sliced)

  def aligned(
    self,
    *,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
  ) -> MotionClip:
    """Reorder joint/body axes to the environment's named layouts."""
    aligned = self.to_dict()
    source_joints = _name_list(aligned.get("joint_names"))
    source_bodies = _name_list(aligned.get("body_names"))
    for key in ("joint_pos", "joint_vel"):
      aligned[key] = _reorder(
        aligned[key], source_joints, joint_names, 1, key
      )
    for key in (
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    ):
      aligned[key] = _reorder(
        aligned[key], source_bodies, body_names, 1, key
      )
    aligned["joint_names"] = np.asarray(joint_names)
    aligned["body_names"] = np.asarray(body_names)
    return MotionClip.from_mapping(aligned)

  def frame(
    self,
    frame_index: int,
    *,
    joint_names: tuple[str, ...],
    body_names: tuple[str, ...],
  ) -> dict[str, np.ndarray]:
    """Project one frame onto explicit joint/body layouts."""
    if frame_index < 0 or frame_index >= self.frame_count:
      raise IndexError(
        f"Motion frame {frame_index} is outside [0, {self.frame_count - 1}]."
      )
    source_joints = _name_list(self._payload.get("joint_names")) or joint_names
    source_bodies = _name_list(self._payload.get("body_names")) or body_names
    frame: dict[str, np.ndarray] = {}
    for key in ("joint_pos", "joint_vel"):
      frame[key] = _reorder(
        self._payload[key][frame_index], source_joints, joint_names, 0, key
      )
    for key in (
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    ):
      frame[key] = _reorder(
        self._payload[key][frame_index], source_bodies, body_names, 0, key
      )
    return frame

  def with_start_frame(
    self, start_frame: Mapping[str, np.ndarray]
  ) -> MotionClip:
    """Overwrite frame zero with a validated handover pose."""
    overlaid = self.to_dict()
    for key in MOTION_ARRAY_KEYS:
      if key not in start_frame:
        raise ValueError(f"Start pose is missing {key}.")
      pose = np.asarray(start_frame[key])
      if pose.shape != overlaid[key][0].shape:
        raise ValueError(
          f"Start pose {key} has shape {pose.shape}, "
          f"motion frame has {overlaid[key][0].shape}."
        )
      overlaid[key][0] = pose
    return MotionClip.from_mapping(overlaid)


def load_motion_npz(path: str | Path) -> dict[str, np.ndarray]:
  motion_path = Path(path).expanduser().resolve()
  if not motion_path.is_file():
    raise FileNotFoundError(motion_path)
  with np.load(motion_path, allow_pickle=False) as data:
    payload = {key: np.asarray(data[key]) for key in data.files}
  try:
    return MotionClip.from_mapping(payload, require_fps=True).to_dict()
  except ValueError as error:
    raise ValueError(f"Invalid motion {motion_path}: {error}") from error


def slice_motion_frames(
  motion: Mapping[str, np.ndarray],
  start: int,
  end: int | None,
) -> dict[str, np.ndarray]:
  """Inclusive [start, end] slice of time-varying motion arrays."""
  return MotionClip.from_mapping(motion).sliced(start, end).to_dict()


def prepare_motion_for_loader(
  motion: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], int]:
  """Adapt a source clip to MotionLoader while preserving its frame count."""
  return MotionClip.from_mapping(motion).for_motion_loader()


def align_motion_layout(
  motion: Mapping[str, np.ndarray],
  *,
  joint_names: tuple[str, ...],
  body_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
  """Reorder a clip to the robot / tracking-body layout used by the env."""
  return MotionClip.from_mapping(motion).aligned(
    joint_names=joint_names, body_names=body_names
  ).to_dict()


def frame_from_rollout_payload(
  payload: Mapping[str, np.ndarray],
  frame_index: int,
  *,
  joint_names: tuple[str, ...],
  body_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
  """Project one recorded robot frame onto a motion-NPZ layout."""
  return MotionClip.from_mapping(payload).frame(
    frame_index, joint_names=joint_names, body_names=body_names
  )


def overlay_start_frame(
  motion: Mapping[str, np.ndarray],
  start_frame: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
  """Overwrite frame 0 with a robot pose; keep the rest of the clip."""
  return MotionClip.from_mapping(motion).with_start_frame(start_frame).to_dict()


def write_motion_npz(path: str | Path, motion: Mapping[str, np.ndarray]) -> Path:
  output = Path(path).expanduser().resolve()
  output.parent.mkdir(parents=True, exist_ok=True)
  payload = MotionClip.from_mapping(motion).to_dict(copy=False)
  np.savez_compressed(output, **payload)
  return output
