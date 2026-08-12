#!/usr/bin/env python3
"""Convert a GMR PKL/NPZ motion to the MJLab tracking NPZ format with MuJoCo.

The motion-editing options intentionally mirror the Isaac Lab version in
``cheer-beyondmimiclab/scripts/gmr_to_npz_inter.py``.  MuJoCo is used for
forward kinematics and body-velocity evaluation; Isaac Sim is not required.

Example:

  python scripts/gmr_to_npz_inter.py \
    --input_file /path/to/motion.pkl \
    --input_fps 30 \
    --frame_range 122 722 \
    --output_name dance1_subject2 \
    --output_dir datasets \
    --output_fps 50 \
    --rotate_z90

Input root quaternions are expected in xyzw order.  Output body quaternions
use MuJoCo/MJLab's wxyz order.
"""

from __future__ import annotations

import argparse
import re
import sys
import types
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from tqdm import tqdm

from src.assets.robots.tiangong3.tk3_constants import get_tk3_robot_cfg


# Joint layout produced by the 31-DoF GMR retargeter.  The local TK3 model has
# fixed head joints, so those two columns are dropped by name during conversion.
GMR_INPUT_JOINT_ORDER = (
  "hip_pitch_l_joint",
  "hip_roll_l_joint",
  "hip_yaw_l_joint",
  "knee_pitch_l_joint",
  "ankle_pitch_l_joint",
  "ankle_roll_l_joint",
  "hip_pitch_r_joint",
  "hip_roll_r_joint",
  "hip_yaw_r_joint",
  "knee_pitch_r_joint",
  "ankle_pitch_r_joint",
  "ankle_roll_r_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "head_yaw_joint_ctrl",
  "head_pitch_joint_ctrl",
  "shoulder_pitch_l_joint",
  "shoulder_roll_l_joint",
  "shoulder_yaw_l_joint",
  "elbow_pitch_l_joint",
  "elbow_yaw_l_joint",
  "wrist_pitch_l_joint",
  "wrist_roll_l_joint",
  "shoulder_pitch_r_joint",
  "shoulder_roll_r_joint",
  "shoulder_yaw_r_joint",
  "elbow_pitch_r_joint",
  "elbow_yaw_r_joint",
  "wrist_pitch_r_joint",
  "wrist_roll_r_joint",
)

SOLE_BODY_NAMES = ("ankle_roll_l_link", "ankle_roll_r_link")


def _ensure_numpy_core_compatibility() -> None:
  """Register numpy._core aliases when loading a newer NumPy pickle."""
  if "numpy._core" in sys.modules:
    return
  core_module = getattr(np, "core", None)
  if core_module is None:
    return
  shim = types.ModuleType("numpy._core")
  shim.__dict__.update(core_module.__dict__)
  sys.modules["numpy._core"] = shim
  for name in ("multiarray", "umath", "numerictypes", "_multiarray_umath"):
    submodule = getattr(core_module, name, None)
    if submodule is not None:
      sys.modules[f"numpy._core.{name}"] = submodule


def _load_motion_mapping(path: Path) -> dict[str, Any]:
  """Load either an NPZ archive or a pickled mapping accepted by np.load."""
  _ensure_numpy_core_compatibility()
  loaded = np.load(path, allow_pickle=True)
  if isinstance(loaded, np.lib.npyio.NpzFile):
    try:
      return {key: loaded[key] for key in loaded.files}
    finally:
      loaded.close()
  if isinstance(loaded, dict):
    return loaded
  if isinstance(loaded, np.ndarray) and loaded.dtype == object:
    value = loaded.item()
    if isinstance(value, dict):
      return value
  raise TypeError(
    f"{path} did not contain an NPZ archive or pickled dictionary."
  )


def _wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
  return np.asarray(quat)[..., (1, 2, 3, 0)]


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
  return np.asarray(quat)[..., (3, 0, 1, 2)]


def _normalize_quaternions(quat: np.ndarray, *, label: str) -> np.ndarray:
  quat = np.asarray(quat, dtype=np.float64)
  norms = np.linalg.norm(quat, axis=-1, keepdims=True)
  if np.any(norms < 1e-8):
    bad = np.flatnonzero(norms[:, 0] < 1e-8)
    raise ValueError(f"{label} contains zero quaternions at frames {bad.tolist()}.")
  return quat / norms


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
  values = np.asarray(values, dtype=np.float64)
  if values.shape[0] == 1:
    return np.zeros_like(values)
  return np.gradient(values, dt, axis=0)


def _so3_derivative(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
  """World-frame angular velocity, matching q_next * inverse(q_prev)."""
  rotations = R.from_quat(_wxyz_to_xyzw(quat_wxyz))
  count = len(rotations)
  angular_velocity = np.zeros((count, 3), dtype=np.float64)
  if count == 1:
    return angular_velocity
  if count == 2:
    omega = (rotations[1] * rotations[0].inv()).as_rotvec() / dt
    angular_velocity[:] = omega
    return angular_velocity

  centered = (rotations[2:] * rotations[:-2].inv()).as_rotvec() / (2.0 * dt)
  angular_velocity[1:-1] = centered
  # Preserve the source script's repeated centered endpoint samples.
  angular_velocity[0] = centered[0]
  angular_velocity[-1] = centered[-1]
  return angular_velocity


def _resolve_default_joint_pos(
  joint_names: list[str], joint_patterns: dict[str, float]
) -> np.ndarray:
  values = np.zeros(len(joint_names), dtype=np.float64)
  missing: list[str] = []
  for index, joint_name in enumerate(joint_names):
    matches = [
      value
      for pattern, value in joint_patterns.items()
      if re.fullmatch(pattern, joint_name)
    ]
    if not matches:
      missing.append(joint_name)
      continue
    values[index] = matches[-1]
  if missing:
    raise ValueError(f"TK3 initial-state config is missing joints: {missing}.")
  return values


def _build_robot() -> tuple[
  mujoco.MjModel,
  mujoco.MjData,
  list[str],
  list[str],
  np.ndarray,
  np.ndarray,
]:
  cfg = get_tk3_robot_cfg()
  if cfg.spec_fn is None:
    raise RuntimeError("TK3 EntityCfg does not provide a MuJoCo spec.")
  model = cfg.spec_fn().compile()
  data = mujoco.MjData(model)

  joint_names = [
    model.joint(joint_id).name
    for joint_id in range(model.njnt)
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
  ]
  body_names = [
    model.body(body_id).name
    for body_id in range(1, model.nbody)
  ]
  default_joint_pos = _resolve_default_joint_pos(
    joint_names, cfg.init_state.joint_pos
  )
  default_pose = np.concatenate(
    [
      np.asarray(cfg.init_state.pos, dtype=np.float64),
      np.asarray(cfg.init_state.rot, dtype=np.float64),
      default_joint_pos,
    ]
  )
  joint_limits = np.asarray(
    [
      model.jnt_range[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      ]
      for name in joint_names
    ],
    dtype=np.float64,
  )
  return model, data, joint_names, body_names, default_pose, joint_limits


def correct_root_pose_coupled(
  root_pos: np.ndarray,
  root_rot_xyzw: np.ndarray,
  target_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Apply a single world-frame translation and yaw correction."""
  root_pos = np.asarray(root_pos, dtype=np.float64)
  root_rot_xyzw = np.asarray(root_rot_xyzw, dtype=np.float64)
  target_pos = np.asarray(target_pos, dtype=np.float64)

  initial_pos = root_pos[0]
  initial_rot = R.from_quat(root_rot_xyzw[0])
  initial_yaw = initial_rot.as_euler("zyx", degrees=False)[0]
  correction_rot = R.from_euler("z", -initial_yaw)
  correction_translation = target_pos - correction_rot.apply(initial_pos)

  corrected_pos = correction_rot.apply(root_pos) + correction_translation
  corrected_rot = (correction_rot * R.from_quat(root_rot_xyzw)).as_quat()
  print(
    "[INFO] Coupled root correction: "
    f"yaw={np.degrees(-initial_yaw):.3f} deg, "
    f"translation={correction_translation}."
  )
  return corrected_pos, corrected_rot


class MotionLoader:
  """Load and edit a GMR motion before MuJoCo forward kinematics."""

  def __init__(
    self,
    motion_file: Path,
    input_fps: int,
    output_fps: int,
    frame_range: tuple[int, int] | None,
    joint_names: list[str],
    default_pose: np.ndarray,
    joint_limits: np.ndarray,
    *,
    knee_modify: bool = False,
    start_frames: int = 0,
    end_frames: int = 0,
    correct_root_pose: bool = False,
    rotate_z90: bool = False,
    hold_pose_frames: int = 0,
    hold_pose_start_frames: int = 0,
    reset_root_xy: bool = False,
    z_offset: float = 0.0,
    joint_filter_window: int = 0,
    joint_filter_polyorder: int = 2,
    hip_roll_r_offset: float = 0.0,
    hip_roll_l_offset: float = 0.0,
    joint_overrides: list[list[str]] | None = None,
    joint_delta_overrides: list[list[str]] | None = None,
    joint_limit_factor: float | None = None,
  ) -> None:
    if input_fps <= 0 or output_fps <= 0:
      raise ValueError("input_fps and output_fps must be positive.")
    if start_frames < 0 or end_frames < 0:
      raise ValueError("start_frames and end_frames must be non-negative.")
    if joint_limit_factor is not None and not 0.0 < joint_limit_factor <= 1.0:
      raise ValueError("joint_limit_factor must be in (0, 1].")

    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / input_fps
    self.output_dt = 1.0 / output_fps
    self.frame_range = frame_range
    self.joint_names = joint_names
    self.default_pose = np.asarray(default_pose, dtype=np.float64)
    self.joint_limits = np.asarray(joint_limits, dtype=np.float64)
    self.joint_limit_factor = joint_limit_factor
    if self.joint_limits.shape != (len(joint_names), 2):
      raise ValueError(
        f"joint_limits must have shape ({len(joint_names)}, 2), "
        f"got {self.joint_limits.shape}."
      )
    self.knee_modify = knee_modify
    self.hold_pose_frames = max(0, hold_pose_frames)
    self.hold_pose_start_frames = max(0, hold_pose_start_frames)
    self.joint_filter_window = joint_filter_window
    self.joint_filter_polyorder = joint_filter_polyorder
    self.joint_overrides = joint_overrides or []
    self.joint_delta_overrides = joint_delta_overrides or []

    self._load_motion()
    self._apply_joint_offset("hip_roll_r_joint", hip_roll_r_offset)
    self._apply_joint_offset("hip_roll_l_joint", hip_roll_l_offset)

    if correct_root_pose:
      target_pos = self.default_pose[:3].copy()
      # Match the original script: preserve the first frame's root height.
      target_pos[2] = self.root_pos_input[0, 2]
      corrected_pos, corrected_xyzw = correct_root_pose_coupled(
        self.root_pos_input,
        _wxyz_to_xyzw(self.root_quat_input),
        target_pos,
      )
      self.root_pos_input = corrected_pos
      self.root_quat_input = _xyzw_to_wxyz(corrected_xyzw)

    if rotate_z90:
      rotation = R.from_euler("z", 90.0, degrees=True)
      self.root_pos_input = rotation.apply(self.root_pos_input)
      corrected = rotation * R.from_quat(_wxyz_to_xyzw(self.root_quat_input))
      self.root_quat_input = _xyzw_to_wxyz(corrected.as_quat())
      print("[INFO] Applied a +90 degree world-frame Z rotation.")

    if reset_root_xy:
      offset = self.root_pos_input[0, :2].copy()
      self.root_pos_input[:, :2] -= offset
      print(f"[INFO] Reset first-frame root XY by subtracting {offset}.")

    if z_offset != 0.0:
      self.root_pos_input[:, 2] += z_offset
      print(f"[INFO] Applied root Z offset {z_offset:g} m.")

    if self.hold_pose_start_frames > 0:
      self._prepend_default_hold(start_frames)
      effective_start_frames = 0
    else:
      effective_start_frames = start_frames

    self._interpolate_start_end(effective_start_frames, end_frames)
    self._resample()
    self._apply_joint_overrides()
    self._append_hold_pose_frames()
    self._clip_joint_positions()
    self._compute_velocities()

  def _load_motion(self) -> None:
    payload = _load_motion_mapping(self.motion_file)
    missing = [key for key in ("root_pos", "root_rot", "dof_pos") if key not in payload]
    if missing:
      raise KeyError(f"Input motion is missing keys: {missing}.")

    root_pos = np.asarray(payload["root_pos"], dtype=np.float64)
    root_xyzw = np.asarray(payload["root_rot"], dtype=np.float64)
    dof_pos = np.asarray(payload["dof_pos"], dtype=np.float64)
    if root_pos.ndim != 2 or root_pos.shape[1] != 3:
      raise ValueError(f"root_pos must have shape (T, 3), got {root_pos.shape}.")
    if root_xyzw.ndim != 2 or root_xyzw.shape[1] != 4:
      raise ValueError(f"root_rot must have shape (T, 4), got {root_xyzw.shape}.")
    if dof_pos.ndim != 2:
      raise ValueError(f"dof_pos must have shape (T, D), got {dof_pos.shape}.")
    if not (len(root_pos) == len(root_xyzw) == len(dof_pos)):
      raise ValueError(
        "root_pos, root_rot, and dof_pos must have the same frame count."
      )

    dof_pos = self._match_dof_layout(dof_pos)
    if self.frame_range is not None:
      start_frame, end_frame = self.frame_range
      if start_frame < 1 or end_frame < start_frame or start_frame > len(root_pos):
        raise ValueError(
          f"frame_range must satisfy 1 <= START <= {len(root_pos)} and END >= START, "
          f"got {self.frame_range}."
        )
      if end_frame > len(root_pos):
        print(
          f"[INFO] Clamped frame_range END from {end_frame} to "
          f"{len(root_pos)} (last input frame)."
        )
        end_frame = len(root_pos)
      frame_slice = slice(start_frame - 1, end_frame)
      root_pos = root_pos[frame_slice]
      root_xyzw = root_xyzw[frame_slice]
      dof_pos = dof_pos[frame_slice]

    if len(root_pos) < 2:
      raise ValueError("At least two input frames are required.")

    dof_pos = self._maybe_filter_joint_positions(dof_pos)
    self.root_pos_input = root_pos.copy()
    self.root_quat_input = _xyzw_to_wxyz(
      _normalize_quaternions(root_xyzw, label="root_rot")
    )
    self.joint_pos_input = dof_pos.copy()
    self._update_input_timing()
    print(
      f"[INFO] Loaded {self.motion_file}: {self.input_frames} frames, "
      f"{self.duration:.3f} s at {self.input_fps} Hz."
    )

  def _match_dof_layout(self, dof_pos: np.ndarray) -> np.ndarray:
    target_count = len(self.joint_names)
    if dof_pos.shape[1] == target_count:
      return dof_pos
    if dof_pos.shape[1] != len(GMR_INPUT_JOINT_ORDER):
      raise ValueError(
        f"Input has {dof_pos.shape[1]} DOFs; expected {target_count} TK3 DOFs "
        f"or {len(GMR_INPUT_JOINT_ORDER)} GMR DOFs."
      )
    source_index = {
      name: index for index, name in enumerate(GMR_INPUT_JOINT_ORDER)
    }
    missing = [name for name in self.joint_names if name not in source_index]
    if missing:
      raise ValueError(f"Cannot remap GMR DOFs; missing joints: {missing}.")
    print(
      f"[INFO] Remapped {len(GMR_INPUT_JOINT_ORDER)} GMR DOFs to "
      f"{target_count} TK3 DOFs by joint name."
    )
    return dof_pos[:, [source_index[name] for name in self.joint_names]]

  def _maybe_filter_joint_positions(self, dof_pos: np.ndarray) -> np.ndarray:
    if self.motion_file.suffix.lower() != ".pkl" or self.joint_filter_window <= 2:
      return dof_pos
    if self.joint_filter_polyorder < 0:
      raise ValueError("joint_filter_polyorder must be non-negative.")

    window = min(self.joint_filter_window, len(dof_pos))
    if window % 2 == 0:
      window -= 1
    minimum = self.joint_filter_polyorder + 1
    if minimum % 2 == 0:
      minimum += 1
    window = max(window, minimum)
    if window > len(dof_pos):
      window = len(dof_pos) if len(dof_pos) % 2 else len(dof_pos) - 1
    if window < 3 or window <= self.joint_filter_polyorder:
      print("[WARN] Joint smoothing skipped: no valid Savitzky-Golay window.")
      return dof_pos

    print(
      f"[INFO] Smoothed PKL joints with window={window}, "
      f"polyorder={self.joint_filter_polyorder}."
    )
    return savgol_filter(
      dof_pos,
      window_length=window,
      polyorder=self.joint_filter_polyorder,
      axis=0,
      mode="interp",
    )

  def _apply_joint_offset(self, joint_name: str, offset: float) -> None:
    if offset == 0.0:
      return
    index = self.joint_names.index(joint_name)
    self.joint_pos_input[:, index] += offset
    print(f"[INFO] Applied {offset:.4f} rad to {joint_name}.")

  def _update_input_timing(self) -> None:
    self.input_frames = len(self.root_pos_input)
    self.duration = (self.input_frames - 1) * self.input_dt

  def _prepend_default_hold(self, transition_frames: int) -> None:
    original_pos = self.root_pos_input.copy()
    original_quat = self.root_quat_input.copy()
    original_joint = self.joint_pos_input.copy()

    if transition_frames > 0:
      transition_pos, transition_quat, transition_joint = self._start_end_arrays(
        transition_frames, 0
      )
      prefix_pos = transition_pos[:transition_frames]
      prefix_quat = transition_quat[:transition_frames]
      prefix_joint = transition_joint[:transition_frames]
    else:
      prefix_pos = np.empty((0, 3), dtype=np.float64)
      prefix_quat = np.empty((0, 4), dtype=np.float64)
      prefix_joint = np.empty((0, len(self.joint_names)), dtype=np.float64)

    hold_pos = np.repeat(
      self.default_pose[None, :3], self.hold_pose_start_frames, axis=0
    )
    hold_quat = np.repeat(
      self.default_pose[None, 3:7], self.hold_pose_start_frames, axis=0
    )
    hold_joint = np.repeat(
      self.default_pose[None, 7:], self.hold_pose_start_frames, axis=0
    )
    self.root_pos_input = np.concatenate(
      [hold_pos, prefix_pos, original_pos], axis=0
    )
    self.root_quat_input = np.concatenate(
      [hold_quat, prefix_quat, original_quat], axis=0
    )
    self.joint_pos_input = np.concatenate(
      [hold_joint, prefix_joint, original_joint], axis=0
    )
    self._update_input_timing()
    print(
      f"[INFO] Prepended {self.hold_pose_start_frames} default-pose frames "
      f"(transition={transition_frames} frames)."
    )

  @staticmethod
  def _lower_dof_interpolation(
    start_dof: np.ndarray, end_dof: np.ndarray, frame_count: int
  ) -> np.ndarray:
    """Interpolate legs while bending each knee to 1 rad near the midpoint."""
    if frame_count <= 0:
      return np.empty((0, 12), dtype=np.float64)
    midpoint = max(1, frame_count // 2)
    left = np.linspace(
      start_dof[:6], end_dof[:6], midpoint + 1, endpoint=False
    )[1:]
    right = np.linspace(
      start_dof[6:12], end_dof[6:12], midpoint + 1, endpoint=False
    )[1:]

    first_half = midpoint // 2
    knee = np.concatenate(
      [
        np.linspace(start_dof[3], 1.0, first_half + 1)[:-1],
        np.linspace(1.0, end_dof[3], midpoint - first_half + 1),
      ]
    )[:midpoint]
    left[:, 3] = knee
    knee = np.concatenate(
      [
        np.linspace(start_dof[9], 1.0, first_half + 1)[:-1],
        np.linspace(1.0, end_dof[9], midpoint - first_half + 1),
      ]
    )[:midpoint]
    right[:, 3] = knee

    left = np.concatenate(
      [left, np.repeat(left[-1:], frame_count - midpoint, axis=0)], axis=0
    )
    right = np.concatenate(
      [np.repeat(right[:1], frame_count - midpoint, axis=0), right], axis=0
    )
    return np.concatenate([left, right], axis=1)

  def _start_end_arrays(
    self, start_frames: int, end_frames: int
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base_pos = self.root_pos_input
    base_quat = self.root_quat_input
    dof_pos = self.joint_pos_input
    default_pos = self.default_pose[:3]
    default_quat = self.default_pose[3:7]
    default_dof = self.default_pose[7:]

    start_euler = R.from_quat(_wxyz_to_xyzw(base_quat[0])).as_euler("ZYX")
    end_euler = R.from_quat(_wxyz_to_xyzw(base_quat[-1])).as_euler("ZYX")
    default_euler = R.from_quat(_wxyz_to_xyzw(default_quat)).as_euler("ZYX")

    if start_frames > 0:
      start_pos = np.zeros((start_frames, 3), dtype=np.float64)
      start_pos[:, :2] = base_pos[0, :2]
      start_pos[:, 2] = np.linspace(
        default_pos[2], base_pos[0, 2], start_frames
      )
      key_rot = R.from_euler(
        "ZYX",
        [
          (start_euler[0], default_euler[1], default_euler[2]),
          start_euler,
        ],
      )
      start_rot = Slerp([0.0, 1.0], key_rot)(
        np.linspace(0.0, 1.0, start_frames)
      )
      start_quat = _xyzw_to_wxyz(start_rot.as_quat())
      if self.knee_modify:
        lower = self._lower_dof_interpolation(
          default_dof[:12], dof_pos[0, :12], start_frames
        )
        upper = np.linspace(
          default_dof[12:],
          dof_pos[0, 12:],
          start_frames + 1,
          endpoint=False,
        )[1:]
        start_dof = np.concatenate([lower, upper], axis=1)
      else:
        start_dof = np.linspace(
          default_dof,
          dof_pos[0],
          start_frames + 1,
          endpoint=False,
        )[1:]
    else:
      start_pos = np.empty((0, 3), dtype=np.float64)
      start_quat = np.empty((0, 4), dtype=np.float64)
      start_dof = np.empty((0, dof_pos.shape[1]), dtype=np.float64)

    if end_frames > 0:
      end_pos = np.zeros((end_frames, 3), dtype=np.float64)
      end_pos[:, :2] = base_pos[-1, :2]
      end_pos[:, 2] = np.linspace(
        base_pos[-1, 2], default_pos[2], end_frames + 1
      )[1:]
      key_rot = R.from_euler(
        "ZYX",
        [
          (end_euler[0], default_euler[1], default_euler[2]),
          end_euler,
        ],
      )
      end_rot = Slerp([0.0, 1.0], key_rot)(
        np.linspace(1.0, 0.0, end_frames)
      )
      end_quat = _xyzw_to_wxyz(end_rot.as_quat())
      if self.knee_modify:
        lower = self._lower_dof_interpolation(
          dof_pos[-1, :12], default_dof[:12], end_frames
        )
        upper = np.linspace(
          dof_pos[-1, 12:], default_dof[12:], end_frames + 1
        )[1:]
        end_dof = np.concatenate([lower, upper], axis=1)
      else:
        end_dof = np.linspace(
          dof_pos[-1], default_dof, end_frames + 1
        )[1:]
    else:
      end_pos = np.empty((0, 3), dtype=np.float64)
      end_quat = np.empty((0, 4), dtype=np.float64)
      end_dof = np.empty((0, dof_pos.shape[1]), dtype=np.float64)

    return (
      np.concatenate([start_pos, base_pos, end_pos], axis=0),
      np.concatenate([start_quat, base_quat, end_quat], axis=0),
      np.concatenate([start_dof, dof_pos, end_dof], axis=0),
    )

  def _interpolate_start_end(
    self, start_frames: int, end_frames: int
  ) -> None:
    (
      self.root_pos_input,
      self.root_quat_input,
      self.joint_pos_input,
    ) = self._start_end_arrays(start_frames, end_frames)
    self._update_input_timing()
    mode = "knee-modified" if self.knee_modify else "linear"
    print(
      f"[INFO] Added start/end transitions ({mode}): "
      f"start={start_frames}, end={end_frames}, "
      f"input frames={self.input_frames}."
    )

  def _resample(self) -> None:
    output_times = np.arange(0.0, self.duration, self.output_dt)
    if len(output_times) == 0:
      raise ValueError("The edited motion is shorter than one output timestep.")
    input_times = np.arange(self.input_frames, dtype=np.float64) * self.input_dt
    input_times[-1] = self.duration

    self.root_pos = np.stack(
      [
        np.interp(output_times, input_times, self.root_pos_input[:, axis])
        for axis in range(3)
      ],
      axis=1,
    )
    rotations = Slerp(
      input_times, R.from_quat(_wxyz_to_xyzw(self.root_quat_input))
    )(output_times)
    self.root_quat = _xyzw_to_wxyz(rotations.as_quat())
    self.joint_pos = np.stack(
      [
        np.interp(output_times, input_times, self.joint_pos_input[:, axis])
        for axis in range(self.joint_pos_input.shape[1])
      ],
      axis=1,
    )
    self.output_frames = len(output_times)
    print(
      f"[INFO] Resampled {self.input_frames} frames at {self.input_fps} Hz "
      f"to {self.output_frames} frames at {self.output_fps} Hz."
    )

  def _parse_override(
    self, override: list[str], flag: str
  ) -> tuple[str, int, int, int, float, int] | None:
    if len(override) not in (4, 5):
      print(f"[WARN] {flag} expects 4 or 5 values, got {override}; skipped.")
      return None
    joint_name = override[0]
    if joint_name not in self.joint_names:
      print(f"[WARN] Unknown joint {joint_name!r} in {flag}; skipped.")
      return None
    frame_start = int(override[1])
    frame_end = int(override[2])
    amount = float(override[3])
    transition = int(override[4]) if len(override) == 5 else 0
    start = max(0, frame_start - 1)
    end = min(self.output_frames, frame_end)
    if start >= end:
      print(
        f"[WARN] Empty {flag} frame range {frame_start}-{frame_end}; skipped."
      )
      return None
    return (
      joint_name,
      self.joint_names.index(joint_name),
      start,
      end,
      amount,
      max(0, transition),
    )

  def _apply_override_ramps(
    self,
    joint_index: int,
    start: int,
    end: int,
    flat_start: float,
    flat_end: float,
    transition: int,
  ) -> None:
    ramp_start = max(0, start - transition)
    count = start - ramp_start
    if count > 0:
      original = self.joint_pos[ramp_start, joint_index]
      alpha = np.linspace(0.0, 1.0, count + 1)[1:]
      self.joint_pos[ramp_start:start, joint_index] = (
        original + alpha * (flat_start - original)
      )

    ramp_end = min(self.output_frames, end + transition)
    count = ramp_end - end
    if count > 0:
      original = self.joint_pos[ramp_end - 1, joint_index]
      alpha = np.linspace(1.0, 0.0, count + 1)[1:]
      self.joint_pos[end:ramp_end, joint_index] = (
        original + alpha * (flat_end - original)
      )

  def _apply_joint_overrides(self) -> None:
    for override in self.joint_overrides:
      parsed = self._parse_override(override, "--joint_override")
      if parsed is None:
        continue
      joint_name, index, start, end, value, transition = parsed
      self.joint_pos[start:end, index] = value
      if transition > 0:
        self._apply_override_ramps(
          index, start, end, value, value, transition
        )
      print(
        f"[INFO] Set {joint_name} frames {start + 1}-{end} to "
        f"{value:.4f} rad (transition={transition})."
      )

    for override in self.joint_delta_overrides:
      parsed = self._parse_override(override, "--joint_override_delta")
      if parsed is None:
        continue
      joint_name, index, start, end, delta, transition = parsed
      self.joint_pos[start:end, index] += delta
      if transition > 0:
        self._apply_override_ramps(
          index,
          start,
          end,
          self.joint_pos[start, index],
          self.joint_pos[end - 1, index],
          transition,
        )
      print(
        f"[INFO] Added {delta:.4f} rad to {joint_name} frames "
        f"{start + 1}-{end} (transition={transition})."
      )

  def _append_hold_pose_frames(self) -> None:
    if self.hold_pose_frames <= 0:
      return
    self.root_pos = np.concatenate(
      [
        self.root_pos,
        np.repeat(self.root_pos[-1:], self.hold_pose_frames, axis=0),
      ],
      axis=0,
    )
    self.root_quat = np.concatenate(
      [
        self.root_quat,
        np.repeat(self.root_quat[-1:], self.hold_pose_frames, axis=0),
      ],
      axis=0,
    )
    self.joint_pos = np.concatenate(
      [
        self.joint_pos,
        np.repeat(self.joint_pos[-1:], self.hold_pose_frames, axis=0),
      ],
      axis=0,
    )
    self.output_frames = len(self.root_pos)
    print(f"[INFO] Appended {self.hold_pose_frames} last-pose frames.")

  def _clip_joint_positions(self) -> None:
    """Clip around each hard-limit midpoint by the requested range factor."""
    if self.joint_limit_factor is None:
      return
    lower = self.joint_limits[:, 0]
    upper = self.joint_limits[:, 1]
    midpoint = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower) * self.joint_limit_factor
    soft_lower = midpoint - half_range
    soft_upper = midpoint + half_range

    original = self.joint_pos.copy()
    self.joint_pos = np.clip(self.joint_pos, soft_lower, soft_upper)
    correction = np.abs(self.joint_pos - original)
    clipped = correction > 1e-12
    clipped_names = [
      name
      for index, name in enumerate(self.joint_names)
      if np.any(clipped[:, index])
    ]
    print(
      f"[INFO] Clipped joints to {100 * self.joint_limit_factor:g}% of "
      f"their midpoint-centered hard-limit ranges: "
      f"{int(np.count_nonzero(clipped))} samples across "
      f"{len(clipped_names)} joints, max correction="
      f"{float(np.max(correction)):.6f} rad."
    )
    if clipped_names:
      print(f"[INFO] Clipped joint names: {clipped_names}.")

  def _compute_velocities(self) -> None:
    self.root_lin_vel = _finite_difference(self.root_pos, self.output_dt)
    self.root_ang_vel = _so3_derivative(self.root_quat, self.output_dt)
    self.joint_vel = _finite_difference(self.joint_pos, self.output_dt)


def _sample_mujoco_kinematics(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  motion: MotionLoader,
  joint_names: list[str],
  body_names: list[str],
) -> dict[str, np.ndarray]:
  free_joint_ids = [
    joint_id
    for joint_id in range(model.njnt)
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
  ]
  if len(free_joint_ids) != 1:
    raise RuntimeError(
      f"Expected one TK3 free joint, found {len(free_joint_ids)}."
    )
  free_joint_id = free_joint_ids[0]
  free_qpos_adr = int(model.jnt_qposadr[free_joint_id])
  free_dof_adr = int(model.jnt_dofadr[free_joint_id])

  joint_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    for name in joint_names
  ]
  joint_qpos_adrs = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
  joint_dof_adrs = [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
  body_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    for name in body_names
  ]

  frame_count = motion.output_frames
  body_count = len(body_ids)
  body_pos = np.empty((frame_count, body_count, 3), dtype=np.float32)
  body_quat = np.empty((frame_count, body_count, 4), dtype=np.float32)
  body_lin_vel = np.empty((frame_count, body_count, 3), dtype=np.float32)
  body_ang_vel = np.empty((frame_count, body_count, 3), dtype=np.float32)
  spatial_velocity = np.empty(6, dtype=np.float64)

  for frame in tqdm(
    range(frame_count), desc="MuJoCo forward kinematics", unit="frame"
  ):
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[free_qpos_adr : free_qpos_adr + 3] = motion.root_pos[frame]
    data.qpos[free_qpos_adr + 3 : free_qpos_adr + 7] = motion.root_quat[frame]
    data.qvel[free_dof_adr : free_dof_adr + 3] = motion.root_lin_vel[frame]
    data.qvel[free_dof_adr + 3 : free_dof_adr + 6] = motion.root_ang_vel[frame]
    data.qpos[joint_qpos_adrs] = motion.joint_pos[frame]
    data.qvel[joint_dof_adrs] = motion.joint_vel[frame]
    mujoco.mj_forward(model, data)

    body_pos[frame] = data.xpos[body_ids]
    body_quat[frame] = data.xquat[body_ids]
    for body_index, body_id in enumerate(body_ids):
      mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id,
        spatial_velocity,
        0,
      )
      body_ang_vel[frame, body_index] = spatial_velocity[:3]
      body_lin_vel[frame, body_index] = spatial_velocity[3:]

  return {
    "joint_pos": motion.joint_pos.astype(np.float32),
    "joint_vel": motion.joint_vel.astype(np.float32),
    "body_pos_w": body_pos,
    "body_quat_w": body_quat,
    "body_lin_vel_w": body_lin_vel,
    "body_ang_vel_w": body_ang_vel,
  }


def _detect_feet_contacts(
  body_pos_w: np.ndarray,
  foot_indices: list[int],
  vel_threshold: float = 0.002,
  height_threshold: float = 0.12,
) -> np.ndarray:
  """Mirror the source script's low-motion plus low-height contact heuristic."""
  if body_pos_w.shape[0] < 2 or not foot_indices:
    return np.ones(
      (body_pos_w.shape[0], len(foot_indices)), dtype=np.float32
    )
  feet_pos = body_pos_w[:, foot_indices]
  feet_delta = np.diff(feet_pos, axis=0)
  feet_speed_sq = np.sum(feet_delta**2, axis=-1)
  feet_height = feet_pos[1:, :, 2]
  contacts = (
    (feet_speed_sq < vel_threshold) & (feet_height < height_threshold)
  ).astype(np.float32)
  return np.concatenate(
    [np.ones((1, contacts.shape[1]), dtype=np.float32), contacts], axis=0
  )


def convert_motion(args: argparse.Namespace) -> Path:
  input_path = Path(args.input_file).expanduser().resolve()
  if not input_path.exists():
    raise FileNotFoundError(input_path)

  model, data, joint_names, body_names, default_pose, joint_limits = _build_robot()
  print(
    f"[INFO] MuJoCo TK3 model: {len(joint_names)} joints, "
    f"{len(body_names)} bodies."
  )
  motion = MotionLoader(
    motion_file=input_path,
    input_fps=args.input_fps,
    output_fps=args.output_fps,
    frame_range=tuple(args.frame_range) if args.frame_range else None,
    joint_names=joint_names,
    default_pose=default_pose,
    joint_limits=joint_limits,
    knee_modify=args.knee_modify,
    start_frames=args.start_frames,
    end_frames=args.end_frames,
    correct_root_pose=args.correct_root_pose_coupled,
    rotate_z90=args.rotate_z90,
    hold_pose_frames=args.hold_pos,
    hold_pose_start_frames=args.hold_pos_start,
    reset_root_xy=args.reset_root_xy,
    z_offset=args.z_offset,
    joint_filter_window=args.joint_filter_window,
    joint_filter_polyorder=args.joint_filter_polyorder,
    hip_roll_r_offset=args.hip_roll_r_offset,
    hip_roll_l_offset=args.hip_roll_l_offset,
    joint_overrides=args.joint_override,
    joint_delta_overrides=args.joint_override_delta,
    joint_limit_factor=args.joint_limit_factor,
  )
  log = _sample_mujoco_kinematics(
    model, data, motion, joint_names, body_names
  )

  if args.ground_lowest_link:
    minimum_z = np.min(log["body_pos_w"][:, :, 2], axis=1)
    log["body_pos_w"][:, :, 2] -= minimum_z[:, None]
    print("[INFO] Put the lowest body origin at z=0 in every frame.")

  sole_indices = [
    body_names.index(name) for name in SOLE_BODY_NAMES if name in body_names
  ]
  if args.lift_sole_height is not None:
    if len(sole_indices) != len(SOLE_BODY_NAMES):
      raise RuntimeError(
        f"Could not resolve all sole bodies {SOLE_BODY_NAMES} in the TK3 model."
      )
    sole_z = log["body_pos_w"][:, sole_indices, 2]
    shifts = np.maximum(
      0.0, args.lift_sole_height - np.min(sole_z, axis=1)
    )
    log["body_pos_w"][:, :, 2] += shifts[:, None]
    print(
      f"[INFO] Sole-height correction lifted "
      f"{int(np.count_nonzero(shifts > 0))}/{len(shifts)} frames "
      f"(threshold={args.lift_sole_height:.4f} m, "
      f"max shift={float(np.max(shifts)):.4f} m)."
    )

  if args.detect_feet_contact:
    log["contact_mask"] = _detect_feet_contacts(
      log["body_pos_w"], sole_indices
    )
    print(f"[INFO] Added contact_mask with shape {log['contact_mask'].shape}.")

  log["fps"] = np.asarray([args.output_fps], dtype=np.int64)
  log["joint_names"] = np.asarray(joint_names, dtype=np.str_)
  log["body_names"] = np.asarray(body_names, dtype=np.str_)

  output_dir = Path(args.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  output_name = args.output_name
  if not output_name.endswith(".npz"):
    output_name += ".npz"
  output_path = output_dir / output_name
  np.savez(output_path, **log)
  print(f"[INFO] Motion saved to {output_path}.")
  return output_path


def _build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description=(
      "Convert a GMR PKL/NPZ motion to a TK3 tracking NPZ with MuJoCo."
    )
  )
  parser.add_argument("--input_file", required=True)
  parser.add_argument("--input_fps", type=int, default=30)
  parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help="1-indexed inclusive input frame range.",
  )
  parser.add_argument("--output_name", required=True)
  parser.add_argument("--output_dir", default="motion_data")
  parser.add_argument("--output_fps", type=int, default=50)
  parser.add_argument(
    "--robot",
    choices=("dex_evt", "tk3"),
    default="dex_evt",
    help="TK3 aliases retained for compatibility with the Isaac Lab script.",
  )
  parser.add_argument("--knee_modify", action="store_true")
  parser.add_argument("--start_frames", type=int, default=0)
  parser.add_argument("--end_frames", type=int, default=0)

  correction_group = parser.add_mutually_exclusive_group()
  correction_group.add_argument(
    "--correct_root_pose_coupled", action="store_true"
  )
  correction_group.add_argument("--rotate_z90", action="store_true")

  parser.add_argument("--hold_pos", type=int, default=0)
  parser.add_argument("--hold_pos_start", type=int, default=0)
  parser.add_argument("--reset_root_xy", action="store_true")
  parser.add_argument("--z_offset", type=float, default=0.0)
  parser.add_argument("--hip_roll_r_offset", type=float, default=0.0)
  parser.add_argument("--hip_roll_l_offset", type=float, default=0.0)
  parser.add_argument("--detect_feet_contact", action="store_true")
  parser.add_argument("--ground_lowest_link", action="store_true")
  parser.add_argument("--lift_sole_height", type=float)
  parser.add_argument("--joint_filter_window", type=int, default=0)
  parser.add_argument("--joint_filter_polyorder", type=int, default=2)
  parser.add_argument(
    "--joint_limit_factor",
    type=float,
    default=None,
    help=(
      "Clip joint positions to this midpoint-centered fraction of each "
      "MuJoCo hard-limit range; use 0.95 for the MJLab soft range."
    ),
  )
  parser.add_argument(
    "--joint_override",
    nargs="+",
    action="append",
    metavar="ARG",
    help=(
      "JOINT_NAME FRAME_START FRAME_END VALUE [TRANSITION_FRAMES], "
      "using 1-indexed inclusive output frames; repeatable."
    ),
  )
  parser.add_argument(
    "--joint_override_delta",
    nargs="+",
    action="append",
    metavar="ARG",
    help=(
      "JOINT_NAME FRAME_START FRAME_END DELTA [TRANSITION_FRAMES], "
      "using 1-indexed inclusive output frames; repeatable."
    ),
  )
  return parser


def main() -> None:
  convert_motion(_build_parser().parse_args())


if __name__ == "__main__":
  main()
