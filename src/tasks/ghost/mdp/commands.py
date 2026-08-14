from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import mujoco
import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from ..motion_clip import MOTION_ARRAY_KEYS, MotionClip
from .events import randomize_actuator_command_lag

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    body_indexes: torch.Tensor,
    device: str = "cpu",
    *,
    joint_names: tuple[str, ...] = (),
    body_names: tuple[str, ...] = (),
    expected_fps: float | None = None,
  ) -> None:
    with np.load(motion_file, allow_pickle=False) as data:
      payload = {key: np.asarray(data[key]) for key in data.files}
    clip = MotionClip.from_mapping(
      payload, minimum_frames=2, require_fps=True
    )
    arrays = clip.to_dict(copy=False)
    self.fps = float(arrays["fps"].reshape(-1)[0])
    if expected_fps is not None and not math.isclose(
      self.fps, expected_fps, rel_tol=1.0e-6, abs_tol=1.0e-6
    ):
      raise ValueError(
        f"Motion fps {self.fps:g} does not match environment frequency "
        f"{expected_fps:g} Hz."
      )

    joint_indexes = self._name_indexes(arrays, "joint_names", joint_names)
    body_indexes_by_name = self._name_indexes(arrays, "body_names", body_names)
    if joint_indexes is None and joint_names:
      if arrays["joint_pos"].shape[1] != len(joint_names):
        raise ValueError(
          f"Motion has {arrays['joint_pos'].shape[1]} joints but robot has "
          f"{len(joint_names)}. Add joint_names metadata to the motion file "
          "when their layouts differ."
        )
      joint_indexes = list(range(len(joint_names)))
    selected_body_indexes = (
      body_indexes_by_name
      if body_indexes_by_name is not None
      else body_indexes.detach().cpu().tolist()
    )
    joint_selection: list[int] | slice = (
      joint_indexes if joint_indexes is not None else slice(None)
    )
    selected = {
      "joint_pos": arrays["joint_pos"][:, joint_selection],
      "joint_vel": arrays["joint_vel"][:, joint_selection],
      "body_pos_w": arrays["body_pos_w"][:, selected_body_indexes],
      "body_quat_w": arrays["body_quat_w"][:, selected_body_indexes],
      "body_lin_vel_w": arrays["body_lin_vel_w"][:, selected_body_indexes],
      "body_ang_vel_w": arrays["body_ang_vel_w"][:, selected_body_indexes],
    }
    for key, array in selected.items():
      setattr(self, key, self._to_tensor(array, device))
    self.time_step_total = clip.frame_count

  def replace_arrays(
    self, arrays: Mapping[str, np.ndarray], device: str | None = None
  ) -> None:
    """Replace the loaded clip in-place. Arrays must already match robot layout."""
    device = device or str(self.joint_pos.device)
    clip = MotionClip.from_mapping(arrays, minimum_frames=2)
    payload = clip.to_dict(copy=False)
    expected_shapes = {
      key: tuple(getattr(self, key).shape[1:]) for key in MOTION_ARRAY_KEYS
    }
    mismatched = {
      key: (tuple(payload[key].shape[1:]), expected_shapes[key])
      for key in MOTION_ARRAY_KEYS
      if tuple(payload[key].shape[1:]) != expected_shapes[key]
    }
    if mismatched:
      raise ValueError(
        "Replacement motion does not match the loaded robot layout: "
        f"{mismatched}."
      )
    if "fps" in payload:
      fps = float(payload["fps"].reshape(-1)[0])
      if not math.isclose(fps, self.fps, rel_tol=1.0e-6, abs_tol=1.0e-6):
        raise ValueError(
          f"Replacement motion fps {fps:g} does not match loaded fps "
          f"{self.fps:g}."
        )
    for key in MOTION_ARRAY_KEYS:
      setattr(self, key, self._to_tensor(payload[key], device))
    self.time_step_total = clip.frame_count

  @staticmethod
  def _to_tensor(array: np.ndarray, device: str) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)

  @staticmethod
  def _name_indexes(
    data: Mapping[str, np.ndarray],
    key: str,
    requested_names: tuple[str, ...],
  ) -> list[int] | None:
    """Resolve requested names against optional NPZ layout metadata."""
    if not requested_names or key not in data:
      return None

    stored_names = tuple(str(name) for name in data[key].tolist())
    if len(stored_names) != len(set(stored_names)):
      raise ValueError(f"Motion metadata {key!r} contains duplicate names.")

    index_by_name = {name: index for index, name in enumerate(stored_names)}
    missing_names = [name for name in requested_names if name not in index_by_name]
    if missing_names:
      raise ValueError(
        f"Motion metadata {key!r} is missing required names: {missing_names}."
      )
    return [index_by_name[name] for name in requested_names]


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    self.motion = MotionLoader(
      self.cfg.motion_file,
      self.body_indexes,
      device=self.device,
      joint_names=tuple(self.robot.joint_names),
      body_names=self.cfg.body_names,
      expected_fps=1.0 / env.step_dt,
    )
    if not self.cfg.preview_frame_offsets:
      raise ValueError("preview_frame_offsets must contain at least one horizon.")
    if any(offset < 0 for offset in self.cfg.preview_frame_offsets):
      raise ValueError("preview_frame_offsets must be non-negative.")
    if self.cfg.preview_frame_offsets[0] != 0:
      raise ValueError("preview_frame_offsets must begin with the current frame (0).")
    if tuple(sorted(self.cfg.preview_frame_offsets)) != self.cfg.preview_frame_offsets:
      raise ValueError("preview_frame_offsets must be sorted in ascending order.")
    preview_body_names = self.cfg.preview_body_names or self.cfg.body_names
    missing_preview_bodies = [
      name for name in preview_body_names if name not in self.cfg.body_names
    ]
    if missing_preview_bodies:
      raise ValueError(
        "preview_body_names must be tracked motion bodies; missing "
        f"{missing_preview_bodies}."
      )
    self.preview_body_names = preview_body_names
    self.preview_body_indexes = torch.tensor(
      [self.cfg.body_names.index(name) for name in preview_body_names],
      dtype=torch.long,
      device=self.device,
    )
    if self.cfg.command_noise_resample_time_s <= 0.0:
      raise ValueError("command_noise_resample_time_s must be positive.")
    if (
      self.cfg.command_noise_anneal_end_step
      <= self.cfg.command_noise_anneal_start_step
    ):
      raise ValueError(
        "command_noise_anneal_end_step must be greater than "
        "command_noise_anneal_start_step."
      )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._preview_frame_offsets = torch.tensor(
      self.cfg.preview_frame_offsets,
      dtype=torch.long,
      device=self.device,
    )
    self._resampled_since_update = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._command_noise_time_left = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )
    self._noise_resampled_since_update = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._joint_pos_command_noise = torch.zeros(
      self.num_envs, len(self.robot.joint_names), device=self.device
    )
    self._joint_vel_command_noise = torch.zeros_like(self._joint_pos_command_noise)
    self._root_pos_command_noise = torch.zeros(
      self.num_envs, 3, device=self.device
    )
    self._root_ori_command_noise = torch.zeros(
      self.num_envs, 3, device=self.device
    )
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    motion_duration_s = self.motion.time_step_total * env.step_dt
    self.bin_count = max(1, math.ceil(motion_duration_s / self.cfg.adaptive_bin_s))
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._gui_slider = None
    self._gui_status = None
    self._gui_get_env_idx: Callable[[], int] | None = None
    self._gui_requested_frame = 0
    self._gui_updating = False
    self._gui_seek_pending = False
    self._gui_last_synced_frame = -1

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      [
        self.joint_pos + self._scaled_noise(self._joint_pos_command_noise),
        self.joint_vel + self._scaled_noise(self._joint_vel_command_noise),
      ],
      dim=1,
    )

  @property
  def command_noise_scale(self) -> float:
    """Global linear command-noise scale at the current control step."""
    step = self._env.common_step_counter
    start = self.cfg.command_noise_anneal_start_step
    end = self.cfg.command_noise_anneal_end_step
    progress = min(max((step - start) / (end - start), 0.0), 1.0)
    return 1.0 - progress if self.cfg.command_noise_enabled else 0.0

  def _scaled_noise(self, noise: torch.Tensor) -> torch.Tensor:
    return noise * self.command_noise_scale

  @staticmethod
  def _expand_env_tensor(
    value: torch.Tensor, target: torch.Tensor
  ) -> torch.Tensor:
    """Expand ``[B, D]`` values across target's intermediate dimensions."""
    view_shape = (
      value.shape[0],
      *((1,) * (target.ndim - 2)),
      value.shape[-1],
    )
    return value.view(view_shape).expand(*target.shape[:-1], value.shape[-1])

  @property
  def _root_orientation_noise_quat(self) -> torch.Tensor:
    noise = self._scaled_noise(self._root_ori_command_noise)
    return quat_from_euler_xyz(noise[:, 0], noise[:, 1], noise[:, 2])

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def noisy_anchor_pos_w(self) -> torch.Tensor:
    return self.anchor_pos_w + self._scaled_noise(self._root_pos_command_noise)

  @property
  def noisy_anchor_quat_w(self) -> torch.Tensor:
    return quat_mul(self._root_orientation_noise_quat, self.anchor_quat_w)

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def preview_indices(self) -> torch.Tensor:
    """Current/future reference indexes, clipped at the final motion frame."""
    return torch.clamp(
      self.time_steps[:, None] + self._preview_frame_offsets[None, :],
      max=self.motion.time_step_total - 1,
    )

  @property
  def noisy_preview_joint_pos(self) -> torch.Tensor:
    joint_pos = self.motion.joint_pos[self.preview_indices]
    noise = self._scaled_noise(self._joint_pos_command_noise)[:, None, :]
    return joint_pos + noise

  @property
  def noisy_preview_joint_vel(self) -> torch.Tensor:
    joint_vel = self.motion.joint_vel[self.preview_indices]
    noise = self._scaled_noise(self._joint_vel_command_noise)[:, None, :]
    return joint_vel + noise

  @property
  def preview_anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[
        self.preview_indices, self.motion_anchor_body_index
      ]
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def preview_anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[
      self.preview_indices, self.motion_anchor_body_index
    ]

  @property
  def noisy_preview_anchor_pos_w(self) -> torch.Tensor:
    return (
      self.preview_anchor_pos_w
      + self._scaled_noise(self._root_pos_command_noise)[:, None, :]
    )

  @property
  def noisy_preview_anchor_quat_w(self) -> torch.Tensor:
    anchor_quat = self.preview_anchor_quat_w
    delta = self._expand_env_tensor(
      self._root_orientation_noise_quat, anchor_quat
    )
    return quat_mul(delta, anchor_quat)

  @property
  def preview_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[
      self.preview_indices, self.motion_anchor_body_index
    ]

  @property
  def preview_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[
      self.preview_indices, self.motion_anchor_body_index
    ]

  @property
  def preview_body_pos_w(self) -> torch.Tensor:
    """Noisy future key-body targets in world coordinates."""
    body_pos = (
      self.motion.body_pos_w[self.preview_indices][
        :, :, self.preview_body_indexes
      ]
      + self._env.scene.env_origins[:, None, None, :]
    )
    anchor_pos = self.preview_anchor_pos_w[:, :, None, :]
    delta = self._expand_env_tensor(
      self._root_orientation_noise_quat,
      body_pos,
    )
    return self.noisy_preview_anchor_pos_w[:, :, None, :] + quat_apply(
      delta, body_pos - anchor_pos
    )

  @property
  def noisy_body_pos_relative_w(self) -> torch.Tensor:
    """Noisy current key-body shape aligned to the robot anchor and heading."""
    body_pos = self.body_pos_w
    root_noise = self._expand_env_tensor(
      self._root_orientation_noise_quat, body_pos
    )
    body_pos = self.noisy_anchor_pos_w[:, None, :] + quat_apply(
      root_noise, body_pos - self.anchor_pos_w[:, None, :]
    )
    delta_ori = yaw_quat(
      quat_mul(
        self.robot_anchor_quat_w, quat_inv(self.noisy_anchor_quat_w)
      )
    )
    expanded_delta = self._expand_env_tensor(
      delta_ori,
      body_pos,
    )
    return self.robot_anchor_pos_w[:, None, :] + quat_apply(
      expanded_delta, body_pos - self.noisy_anchor_pos_w[:, None, :]
    )

  @property
  def noisy_body_quat_relative_w(self) -> torch.Tensor:
    """Noisy current key-body orientations aligned to the robot heading."""
    body_quat = self.motion.body_quat_w[self.time_steps]
    root_noise = self._expand_env_tensor(
      self._root_orientation_noise_quat, body_quat
    )
    body_quat = quat_mul(root_noise, body_quat)
    delta_ori = yaw_quat(
      quat_mul(self.robot_anchor_quat_w, quat_inv(self.noisy_anchor_quat_w))
    )
    expanded_delta = self._expand_env_tensor(delta_ori, body_quat)
    return quat_mul(expanded_delta, body_quat)

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  @property
  def actuator_effort_limits(self) -> torch.Tensor:
    """Read the live per-environment force limits for this robot's controls."""
    force = self.robot.data.actuator_force
    force_range = self._env.sim.model.actuator_forcerange
    if force_range.ndim != 3 or force_range.shape[-1] != 2:
      raise ValueError(
        "Simulator actuator_forcerange must have shape [B, num_ctrl, 2], "
        f"got {tuple(force_range.shape)}."
      )
    ctrl_ids = self.robot.indexing.ctrl_ids.to(
      device=force_range.device, dtype=torch.long
    )
    limits = torch.amax(
      torch.abs(torch.index_select(force_range, 1, ctrl_ids)), dim=-1
    )
    if limits.shape[0] == 1 and force.shape[0] != 1:
      limits = limits.expand(force.shape[0], -1)
    if limits.shape != force.shape:
      raise ValueError(
        "Live actuator force-limit layout does not match robot forces: "
        f"limits={tuple(limits.shape)}, forces={tuple(force.shape)}."
      )
    return limits.to(device=force.device, dtype=force.dtype).clamp_min(1.0e-6)

  @property
  def normalized_actuator_force(self) -> torch.Tensor:
    return self.robot.data.actuator_force / self.actuator_effort_limits

  @property
  def reference_index(self) -> torch.Tensor:
    return self.time_steps

  @property
  def segment_end(self) -> torch.Tensor:
    return self.time_steps >= self.motion.time_step_total - 1

  def replace_motion(
    self, arrays: Mapping[str, np.ndarray], *, frame: int = 0
  ) -> None:
    """Swap the reference clip without resetting the robot.

    Used for ONNX chaining: keep the current simulated pose and point the
    command at a new (possibly start-overlaid) motion.
    """
    self.motion.replace_arrays(arrays, device=self.device)
    last_index = self.motion.time_step_total - 1
    if frame < 0 or frame > last_index:
      raise ValueError(f"Motion frame {frame} is outside [0, {last_index}].")
    self.time_steps[:] = frame
    # Next env.step should advance this clip, not hold the handover frame.
    self._resampled_since_update[:] = False
    env_ids = torch.arange(self.num_envs, device=self.device)
    self._set_body_targets(
      env_ids, self.robot_anchor_pos_w, self.robot_anchor_quat_w
    )

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    # Sample.
    sampling_probabilities = self.bin_failed_count.clamp_min(0.0) + 1.0e-6
    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
    min_probability = self.cfg.adaptive_min_factor / self.bin_count
    max_probability = min(
      1.0, self.cfg.adaptive_max_factor / self.bin_count
    )
    sampling_probabilities = sampling_probabilities.clamp(
      min=min_probability,
      max=max_probability,
    )
    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count
      * (self.motion.time_step_total - 1)
    ).long()

    # Update metrics.
    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _uniform_sampling(self, env_ids: torch.Tensor):
    self.time_steps[env_ids] = torch.randint(
      0, self.motion.time_step_total, (len(env_ids),), device=self.device
    )
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5  # No specific bin preference.

  def _resample_command(self, env_ids: torch.Tensor):
    self._resampled_since_update[env_ids] = True
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    reference_root_pos = self.body_pos_w[:, 0].clone()
    reference_root_ori = self.body_quat_w[:, 0].clone()
    root_pos = reference_root_pos.clone()
    root_ori = reference_root_ori.clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    self._reset_robot_and_randomize_actuator_command_lag(env_ids)

    # CommandManager evaluates terminations before the first command update after
    # reset. Seed the relative targets here so that they never contain the zero
    # initialization (or a target left over from the previous episode). Estimate
    # the randomized anchor by applying the sampled root transform; the regular
    # update below replaces this estimate with forward-kinematics data each step.
    root_delta_ori = quat_mul(
      root_ori[env_ids], quat_inv(reference_root_ori[env_ids])
    )
    estimated_anchor_pos = root_pos[env_ids] + quat_apply(
      root_delta_ori,
      self.anchor_pos_w[env_ids] - reference_root_pos[env_ids],
    )
    estimated_anchor_quat = quat_mul(
      root_delta_ori, self.anchor_quat_w[env_ids]
    )
    self._set_body_targets(
      env_ids,
      estimated_anchor_pos,
      estimated_anchor_quat,
    )
    self._resample_command_noise(env_ids, reset_timer=True)

  def _reset_robot_and_randomize_actuator_command_lag(
    self,
    env_ids: torch.Tensor,
  ) -> None:
    """重置机器人，并按配置采样执行器指令延迟。"""
    self.robot.reset(env_ids=env_ids)
    if self.cfg.actuator_command_lag_range is not None:
      randomize_actuator_command_lag(
        self.robot,
        env_ids,
        lag_range=self.cfg.actuator_command_lag_range,
      )

  def _set_body_targets(
    self,
    env_ids: torch.Tensor,
    robot_anchor_pos_w: torch.Tensor,
    robot_anchor_quat_w: torch.Tensor,
  ) -> None:
    """Align reference bodies to the robot anchor position and heading."""
    body_count = len(self.cfg.body_names)
    anchor_pos_w = self.anchor_pos_w[env_ids, None, :].repeat(1, body_count, 1)
    anchor_quat_w = self.anchor_quat_w[env_ids, None, :].repeat(1, body_count, 1)
    robot_anchor_pos_w = robot_anchor_pos_w[:, None, :].repeat(1, body_count, 1)
    robot_anchor_quat_w = robot_anchor_quat_w[:, None, :].repeat(1, body_count, 1)

    # Remove all global translation from body-shape targets. Global root XY/Z
    # are tracked by dedicated rewards; this target only preserves articulated
    # shape after aligning the reference and robot headings.
    delta_pos_w = robot_anchor_pos_w
    delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(anchor_quat_w)))

    self.body_quat_relative_w[env_ids] = quat_mul(
      delta_ori_w, self.body_quat_w[env_ids]
    )
    self.body_pos_relative_w[env_ids] = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w[env_ids] - anchor_pos_w
    )

  def _update_command(self):
    advance_env_ids = torch.where(~self._resampled_since_update)[0]
    self.time_steps[advance_env_ids] += 1
    self._resampled_since_update.zero_()
    resampled_env_ids = torch.where(
      self.time_steps >= self.motion.time_step_total
    )[0]
    if resampled_env_ids.numel() > 0:
      self._resample_command(resampled_env_ids)
      self._resampled_since_update[resampled_env_ids] = False
      # Refresh derived body poses after teleporting the resampled environments.
      self._env.sim.forward()

    update_env_ids = torch.arange(self.num_envs, device=self.device)
    self._set_body_targets(
      update_env_ids,
      self.robot_anchor_pos_w,
      self.robot_anchor_quat_w,
    )
    self._update_command_noise()

    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()
    self._sync_gui_progress()

  def create_gui(
    self,
    name: str,
    server: Any,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create a motion timeline that can respawn the robot at any frame."""
    del name
    self._gui_get_env_idx = get_env_idx
    total = self.motion.time_step_total
    self._gui_status = server.gui.add_html("")
    self._gui_slider = server.gui.add_slider(
      "Motion progress",
      min=0,
      max=total - 1,
      step=1,
      initial_value=0,
    )

    @self._gui_slider.on_update
    def _on_slider(event) -> None:
      if self._gui_updating:
        return
      self._gui_requested_frame = int(event.target.value)
      self._gui_seek_pending = True
      self._update_gui_status(self._gui_requested_frame)
      if on_change is not None:
        on_change()

    spawn_button = server.gui.add_button("Spawn at selected frame")

    @spawn_button.on_click
    def _on_spawn(_) -> None:
      if request_action is not None:
        request_action(
          "CUSTOM",
          {"type": "gui_reset", "all_envs": False},
        )

    self._update_gui_status(0)

  def _update_gui_status(self, frame: int) -> None:
    if self._gui_status is None:
      return
    total = self.motion.time_step_total
    fps = self.motion.fps
    self._gui_status.content = (
      f"<b>Frame:</b> {frame:,} / {total - 1:,}<br/>"
      f"<b>Time:</b> {frame / fps:.2f} s / {(total - 1) / fps:.2f} s"
    )

  def _sync_gui_progress(self) -> None:
    if self._gui_slider is None or self._gui_get_env_idx is None:
      return
    if self._gui_seek_pending:
      return
    env_idx = min(max(self._gui_get_env_idx(), 0), self.num_envs - 1)
    frame = int(self.time_steps[env_idx].item())
    if frame == self._gui_last_synced_frame:
      return
    sync_interval = max(1, round(self.motion.fps / 10.0))
    if (
      self._gui_last_synced_frame >= 0
      and frame > self._gui_last_synced_frame
      and frame - self._gui_last_synced_frame < sync_interval
    ):
      return
    self._gui_last_synced_frame = frame
    self._gui_requested_frame = frame
    self._gui_updating = True
    try:
      self._gui_slider.value = frame
      self._update_gui_status(frame)
    finally:
      self._gui_updating = False

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    """Respawn selected environments at the timeline's requested frame."""
    frame = min(
      max(self._gui_requested_frame, 0),
      self.motion.time_step_total - 1,
    )
    self.time_steps[env_ids] = frame
    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids].clone()
    limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clamp(
      joint_pos,
      min=limits[:, :, 0],
      max=limits[:, :, 1],
    )
    root_state = torch.cat(
      [
        self.body_pos_w[env_ids, 0],
        self.body_quat_w[env_ids, 0],
        self.body_lin_vel_w[env_ids, 0],
        self.body_ang_vel_w[env_ids, 0],
      ],
      dim=-1,
    )
    self.robot.write_joint_state_to_sim(
      joint_pos, joint_vel, env_ids=env_ids
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self._reset_robot_and_randomize_actuator_command_lag(env_ids)
    self._set_body_targets(
      env_ids,
      root_state[:, :3],
      root_state[:, 3:7],
    )
    self._gui_last_synced_frame = -1
    self._gui_seek_pending = False
    self._sync_gui_progress()
    return True

  def _resample_command_noise(
    self, env_ids: torch.Tensor, *, reset_timer: bool
  ) -> None:
    """Resample held observation-only command noise for selected environments."""
    if not self.cfg.command_noise_enabled:
      self._joint_pos_command_noise[env_ids] = 0.0
      self._joint_vel_command_noise[env_ids] = 0.0
      self._root_pos_command_noise[env_ids] = 0.0
      self._root_ori_command_noise[env_ids] = 0.0
      self._command_noise_time_left[env_ids] = float("inf")
      return

    if reset_timer:
      self._command_noise_time_left[env_ids] = (
        self.cfg.command_noise_resample_time_s
      )
      self._noise_resampled_since_update[env_ids] = True
    count = len(env_ids)
    self._joint_pos_command_noise[env_ids] = sample_uniform(
      self.cfg.command_joint_pos_noise_range[0],
      self.cfg.command_joint_pos_noise_range[1],
      (count, self._joint_pos_command_noise.shape[1]),
      device=self.device,
    )
    self._joint_vel_command_noise[env_ids] = sample_uniform(
      self.cfg.command_joint_vel_noise_range[0],
      self.cfg.command_joint_vel_noise_range[1],
      (count, self._joint_vel_command_noise.shape[1]),
      device=self.device,
    )
    self._root_pos_command_noise[env_ids] = sample_uniform(
      self.cfg.command_root_pos_noise_range[0],
      self.cfg.command_root_pos_noise_range[1],
      (count, 3),
      device=self.device,
    )
    self._root_ori_command_noise[env_ids] = sample_uniform(
      self.cfg.command_root_ori_noise_range[0],
      self.cfg.command_root_ori_noise_range[1],
      (count, 3),
      device=self.device,
    )

  def _update_command_noise(self) -> None:
    """Advance per-environment timers and hold each sample between expiries."""
    if not self.cfg.command_noise_enabled:
      return
    advance_env_ids = torch.where(~self._noise_resampled_since_update)[0]
    self._command_noise_time_left[advance_env_ids] -= self._env.step_dt
    self._noise_resampled_since_update.zero_()
    expired_env_ids = torch.where(self._command_noise_time_left <= 1.0e-6)[0]
    if expired_env_ids.numel() == 0:
      return
    self._resample_command_noise(expired_env_ids, reset_timer=False)
    self._command_noise_time_left[expired_env_ids] = (
      self.cfg.command_noise_resample_time_s
    )

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(qpos, model=self._ghost_model, label=f"ghost_{batch}")

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  entity_name: str
  # RSI：reset 时在参考位姿上叠加的根位置/姿态扰动范围（空 dict = 关闭）
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  # RSI：reset 时根线/角速度扰动范围
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  # RSI：reset 时关节位置扰动范围
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_bin_s: float = 1.0
  adaptive_min_factor: float = 0.75
  adaptive_max_factor: float = 100.0
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  # 域随机化：执行器指令延迟（控制步数）；None = 不随机化
  actuator_command_lag_range: tuple[int, int] | None = None
  # 100 Hz reference preview horizons. Indexing clips at the last motion frame.
  preview_frame_offsets: tuple[int, ...] = (0, 5, 10, 20)
  # Only bodies needed for anticipatory contact planning. Empty means all tracked.
  preview_body_names: tuple[str, ...] = ()
  # 域随机化：参考指令噪声（只污染观测目标，不污染奖励目标）
  command_noise_enabled: bool = True
  # Each environment holds one sample for this duration; a reset starts a new timer.
  command_noise_resample_time_s: float = 1.0
  # 50k iterations * 24 steps: linearly fade noise over the final 20%.
  command_noise_anneal_start_step: int = 960_000
  command_noise_anneal_end_step: int = 1_200_000
  command_joint_pos_noise_range: tuple[float, float] = (-0.01, 0.01)
  command_joint_vel_noise_range: tuple[float, float] = (-0.5, 0.5)
  command_root_pos_noise_range: tuple[float, float] = (-0.01, 0.01)
  command_root_ori_noise_range: tuple[float, float] = (-0.05, 0.05)

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)
