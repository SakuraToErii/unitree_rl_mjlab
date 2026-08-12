#!/usr/bin/env python3
"""Replay a TK3 motion NPZ with the Ghost sole; measure ground clearance.

Uses the same foot-plane signed-distance test as ground_npz_motion.py.
Tracked foot/hand collision geoms turn red below --clearance.  Labels show each
foot or hand's minimum signed distance to z=0 in millimeters.

Scene uses the same checker floor + directional light as scene_tiangong3.xml.

Example:
  python scripts/visualize_npz_foot_penetration.py \\
    --input datasets/1_1_padding.npz
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro
from mjlab.entity.entity import Entity
from mjlab.utils import spec_config as spec_cfg

from src.assets.robots.tiangong3.tk3_constants_ghost import (
  FOOT_GEOM_PATTERN,
  get_tk3_robot_cfg,
)

_PENETRATING = np.array([1.0, 0.15, 0.1, 0.95], dtype=np.float32)
_LABEL_OK = np.array([0.2, 0.85, 0.35, 0.95], dtype=np.float32)
_LABEL_BAD = np.array([1.0, 0.2, 0.15, 0.95], dtype=np.float32)
_IDENTITY_MAT = np.eye(3, dtype=np.float64).reshape(9)
_HAND_GEOM_PATTERN = re.compile(
  r"^(left|right)_tcp_link_collision(?:_[0-9]+)?$"
)

# Match scene_tiangong3.xml / mjlab default plane look.
_FLOOR_TEXTURE = spec_cfg.TextureCfg(
  name="groundplane",
  type="2d",
  builtin="checker",
  mark="edge",
  rgb1=(0.2, 0.3, 0.4),
  rgb2=(0.1, 0.2, 0.3),
  markrgb=(0.8, 0.8, 0.8),
  width=300,
  height=300,
)
_FLOOR_MATERIAL = spec_cfg.MaterialCfg(
  name="groundplane",
  texuniform=True,
  texrepeat=(5.0, 5.0),
  reflectance=0.2,
  texture="groundplane",
  geom_names_expr=(r"^terrain$",),
)
_SUN_LIGHT = spec_cfg.LightCfg(
  name="sun",
  pos=(1.0, 0.0, 3.5),
  dir=(0.0, 0.0, -1.0),
  type="directional",
)


def _build_model() -> tuple[mujoco.MjModel, mujoco.MjData, list[int], int, list[str]]:
  robot = Entity(get_tk3_robot_cfg())
  spec = robot.spec

  ground = spec.worldbody.add_geom()
  ground.name = "terrain"
  ground.type = mujoco.mjtGeom.mjGEOM_PLANE
  ground.size = [0.0, 0.0, 0.05]

  _FLOOR_TEXTURE.edit_spec(spec)
  _FLOOR_MATERIAL.edit_spec(spec)
  _SUN_LIGHT.edit_spec(spec)

  # Same headlight defaults as scene_tiangong3.xml.
  spec.visual.headlight.ambient[:] = (0.1, 0.1, 0.1)
  spec.visual.headlight.diffuse[:] = (0.6, 0.6, 0.6)
  spec.visual.headlight.specular[:] = (0.9, 0.9, 0.9)

  model = spec.compile()
  data = mujoco.MjData(model)

  foot_re = re.compile(FOOT_GEOM_PATTERN)
  foot_ids = [
    i
    for i in range(model.ngeom)
    if model.geom(i).name and foot_re.match(model.geom(i).name)
  ]
  plane_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  art_joints = [
    model.joint(i).name
    for i in range(model.njnt)
    if model.joint(i).type != mujoco.mjtJoint.mjJNT_FREE
  ]
  if not foot_ids:
    raise RuntimeError("No foot collision geoms matched FOOT_GEOM_PATTERN.")
  return model, data, foot_ids, plane_id, art_joints


def _load_motion_data(
  input_path: Path,
  art_joints: list[str],
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
  """Load motion arrays and map named Isaac Lab layouts to MuJoCo order."""
  with np.load(input_path, allow_pickle=False) as raw:
    fps = float(np.asarray(raw["fps"]).reshape(-1)[0])
    joint_pos = np.asarray(raw["joint_pos"], dtype=np.float32)
    body_pos_w = np.asarray(raw["body_pos_w"], dtype=np.float32)
    body_quat_w = np.asarray(raw["body_quat_w"], dtype=np.float32)

    if "joint_names" in raw.files:
      stored_joint_names = [str(name) for name in raw["joint_names"].tolist()]
      if len(stored_joint_names) != len(set(stored_joint_names)):
        raise ValueError("Motion joint_names contains duplicate names.")
      index_by_name = {
        name: index for index, name in enumerate(stored_joint_names)
      }
      missing = [name for name in art_joints if name not in index_by_name]
      if missing:
        raise ValueError(
          f"Motion joint_names is missing MuJoCo joints: {missing}."
        )
      joint_indexes = [index_by_name[name] for name in art_joints]
      joint_pos = joint_pos[:, joint_indexes]
      if joint_indexes != list(range(len(art_joints))):
        print("[INFO] Reordered NPZ joints from joint_names metadata.")
    elif joint_pos.shape[1] != len(art_joints):
      raise ValueError(
        f"Motion has {joint_pos.shape[1]} unnamed joints, but MuJoCo has "
        f"{len(art_joints)}; cannot determine their mapping."
      )

    pelvis_index = 0
    if "body_names" in raw.files:
      stored_body_names = [str(name) for name in raw["body_names"].tolist()]
      if len(stored_body_names) != len(set(stored_body_names)):
        raise ValueError("Motion body_names contains duplicate names.")
      if "pelvis" not in stored_body_names:
        raise ValueError("Motion body_names does not contain 'pelvis'.")
      pelvis_index = stored_body_names.index("pelvis")

  if not (
    joint_pos.shape[0] == body_pos_w.shape[0] == body_quat_w.shape[0]
  ):
    raise ValueError("Motion joint and body arrays have different frame counts.")
  return (
    fps,
    joint_pos,
    body_pos_w[:, pelvis_index],
    body_quat_w[:, pelvis_index],
  )


def _apply_frame(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  art_joints: list[str],
  root_pos: np.ndarray,
  root_quat_wxyz: np.ndarray,
  joint_pos: np.ndarray,
) -> None:
  data.qpos[0:3] = root_pos
  data.qpos[3:7] = root_quat_wxyz
  for joint_index, name in enumerate(art_joints):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[joint_id]] = joint_pos[joint_index]
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)


def _geom_distance(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  geom_id: int,
  plane_id: int,
) -> float:
  fromto = np.zeros(6)
  return float(mujoco.mj_geomDistance(model, data, geom_id, plane_id, 10.0, fromto))


def _split_feet(model: mujoco.MjModel, foot_ids: list[int]) -> dict[str, list[int]]:
  groups: dict[str, list[int]] = {"L": [], "R": []}
  for gid in foot_ids:
    name = model.geom(gid).name
    if "left" in name:
      groups["L"].append(gid)
    elif "right" in name:
      groups["R"].append(gid)
  return groups


def _find_hand_groups(model: mujoco.MjModel) -> dict[str, list[int]]:
  groups: dict[str, list[int]] = {"LH": [], "RH": []}
  for geom_id in range(model.ngeom):
    name = model.geom(geom_id).name or ""
    match = _HAND_GEOM_PATTERN.fullmatch(name)
    if match is None:
      continue
    groups["LH" if match.group(1) == "left" else "RH"].append(geom_id)
  missing = [label for label, geom_ids in groups.items() if not geom_ids]
  if missing:
    raise RuntimeError(f"No hand collision geoms found for: {missing}.")
  return groups


def _add_distance_label(
  scn: mujoco.MjvScene,
  pos: np.ndarray,
  dist_m: float,
  label: str,
  penetrating: bool,
) -> None:
  if scn.ngeom >= scn.maxgeom:
    return
  geom = scn.geoms[scn.ngeom]
  # Hover slightly above / beside the limb so the number stays readable.
  is_right = label.startswith("R")
  label_pos = pos + np.array([0.06 if is_right else -0.06, 0.0, 0.05])
  mujoco.mjv_initGeom(
    geom,
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=np.array([0.012, 0.0, 0.0]),
    pos=label_pos,
    mat=_IDENTITY_MAT,
    rgba=_LABEL_BAD if penetrating else _LABEL_OK,
  )
  # Signed distance: negative = into ground (penetration depth as negative mm).
  geom.label = f"{label}:{dist_m * 1000.0:+.1f}mm"
  scn.ngeom += 1


class _TimelineControl:
  """Small Tk timeline synchronized with the passive MuJoCo viewer."""

  def __init__(self, frame_count: int, fps: float, initial_frame: int) -> None:
    import tkinter as tk

    self._tk = tk
    self._frame_count = frame_count
    self._fps = fps
    self._requested_frame: int | None = None
    self._programmatic_update = False
    self._dragging = False
    self.playing = True
    self.is_open = True

    self._root = tk.Tk()
    self._root.title("Motion timeline")
    self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    controls = tk.Frame(self._root, padx=8, pady=8)
    controls.pack(fill=tk.BOTH, expand=True)
    self._play_button = tk.Button(
      controls, text="Pause", width=8, command=self._toggle_playback
    )
    self._play_button.pack(side=tk.LEFT, padx=(0, 8))

    self._scale = tk.Scale(
      controls,
      from_=0,
      to=frame_count - 1,
      orient=tk.HORIZONTAL,
      showvalue=False,
      resolution=1,
      length=640,
      command=self._on_seek,
    )
    self._scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self._scale.bind("<ButtonPress-1>", self._on_drag_start)
    self._scale.bind("<ButtonRelease-1>", self._on_drag_end)

    self._status = tk.Label(self._root, anchor="center", padx=8, pady=4)
    self._status.pack(fill=tk.X)
    self._root.bind("<space>", self._toggle_playback)
    self.update_frame(initial_frame)
    self.pump()

  def _on_seek(self, value: str) -> None:
    if self._programmatic_update:
      return
    self._requested_frame = int(round(float(value)))

  def _on_drag_start(self, _event: object) -> None:
    self._dragging = True

  def _on_drag_end(self, _event: object) -> None:
    self._dragging = False
    self._requested_frame = int(self._scale.get())

  def _toggle_playback(self, _event: object | None = None) -> None:
    self.playing = not self.playing
    self._play_button.configure(text="Pause" if self.playing else "Play")

  def _on_close(self) -> None:
    self.is_open = False

  def pump(self) -> None:
    if not self.is_open:
      return
    try:
      self._root.update_idletasks()
      self._root.update()
    except self._tk.TclError:
      self.is_open = False

  def consume_requested_frame(self) -> int | None:
    requested = self._requested_frame
    self._requested_frame = None
    if requested is None:
      return None
    return max(0, min(self._frame_count - 1, requested))

  def update_frame(self, frame: int) -> None:
    if not self.is_open:
      return
    if not self._dragging:
      self._programmatic_update = True
      self._scale.set(frame)
      self._programmatic_update = False
    current_time = frame / self._fps
    total_time = (self._frame_count - 1) / self._fps
    self._status.configure(
      text=(
        f"Frame {frame:,} / {self._frame_count - 1:,}    "
        f"{current_time:.2f}s / {total_time:.2f}s"
      )
    )

  def close(self) -> None:
    self.is_open = False
    try:
      self._root.destroy()
    except self._tk.TclError:
      pass


def _try_create_timeline(
  enabled: bool,
  frame_count: int,
  fps: float,
  initial_frame: int,
) -> _TimelineControl | None:
  if not enabled:
    return None
  try:
    return _TimelineControl(frame_count, fps, initial_frame)
  except Exception as exc:
    print(f"[WARN] Timeline unavailable ({exc}); continuing without it.")
    return None


def visualize(
  input: Path,
  clearance: float = 0.0,
  speed: float = 1.0,
  loop: bool = True,
  start_frame: int = 0,
  timeline: bool = True,
) -> None:
  """Replay NPZ and label foot/hand distance from the ground plane.

  Args:
    input: Motion NPZ to replay (read-only).
    clearance: Distance threshold in meters. Geoms with dist < clearance are
      marked red. Use 0.0 for true penetration; match grounding with -0.001.
    speed: Playback rate relative to NPZ fps.
    loop: Restart from start_frame when the clip ends.
    start_frame: Initial frame index.
    timeline: Open a draggable frame timeline with play/pause control.
  """
  input_path = input.expanduser().resolve()
  if not input_path.exists():
    raise FileNotFoundError(input_path)

  model, data, foot_ids, plane_id, art_joints = _build_model()
  foot_groups = _split_feet(model, foot_ids)
  hand_groups = _find_hand_groups(model)
  distance_groups = {**foot_groups, **hand_groups}
  tracked_geom_ids = [
    geom_id for geom_ids in distance_groups.values() for geom_id in geom_ids
  ]

  # Tracked collision geoms are group 3 (usually hidden); show them here.
  for gid in tracked_geom_ids:
    model.geom_group[gid] = 0

  fps, joint_pos, root_pos_w, root_quat_w = _load_motion_data(
    input_path, art_joints
  )

  n_frames = joint_pos.shape[0]
  if not (0 <= start_frame < n_frames):
    raise ValueError(f"start_frame={start_frame} out of range [0, {n_frames}).")

  nominal_rgba = {
    gid: model.geom_rgba[gid].copy() for gid in tracked_geom_ids
  }
  dt = 1.0 / max(fps * speed, 1e-6)
  frame = start_frame

  print(
    "\n".join(
      [
        f"input: {input_path}",
        f"frames: {n_frames} @ {fps:g} Hz (speed={speed:g})",
        f"clearance: {clearance:.4f} m  (geom red when dist < clearance)",
        f"foot geoms: {len(foot_ids)}",
        f"hand geoms: {sum(len(ids) for ids in hand_groups.values())}",
        "labels: L/R=feet, LH/RH=hands; signed distance in mm",
        "timeline: drag to seek; Space or button toggles play/pause",
        "close either window to exit",
      ]
    )
  )

  with mujoco.viewer.launch_passive(model, data) as viewer:
    timeline_control = _try_create_timeline(
      timeline, n_frames, fps, start_frame
    )
    # Show geom labels from user_scn markers.
    viewer.opt.label = mujoco.mjtLabel.mjLABEL_NONE
    last_reported_frame: int | None = None
    try:
      while viewer.is_running() and (
        timeline_control is None or timeline_control.is_open
      ):
        t0 = time.perf_counter()
        if timeline_control is not None:
          timeline_control.pump()
          requested_frame = timeline_control.consume_requested_frame()
          if requested_frame is not None:
            frame = requested_frame

        _apply_frame(
          model,
          data,
          art_joints,
          root_pos_w[frame],
          root_quat_w[frame],
          joint_pos[frame],
        )

        penetrations = 0
        dists: dict[int, float] = {}
        for gid in tracked_geom_ids:
          dist = _geom_distance(model, data, gid, plane_id)
          dists[gid] = dist
          if dist < clearance:
            model.geom_rgba[gid] = _PENETRATING
            penetrations += 1
          else:
            model.geom_rgba[gid] = nominal_rgba[gid]

        scn = viewer.user_scn
        scn.ngeom = 0
        group_min_dist: dict[str, float] = {}
        for label, gids in distance_groups.items():
          if not gids:
            continue
          # Worst (most negative) surface clearance for this foot or hand.
          worst = float(min(dists[g] for g in gids))
          group_min_dist[label] = worst
          pos = np.mean([data.geom_xpos[g] for g in gids], axis=0)
          _add_distance_label(
            scn,
            pos,
            worst,
            label,
            penetrating=worst < clearance,
          )

        if timeline_control is not None:
          timeline_control.update_frame(frame)
        viewer.sync()

        if (
          frame != last_reported_frame
          and frame % max(1, int(fps // 2)) == 0
        ):
          print(
            f"\rframe {frame:5d}/{n_frames}  "
            f"L:{group_min_dist['L'] * 1000.0:+.1f}mm  "
            f"R:{group_min_dist['R'] * 1000.0:+.1f}mm  "
            f"LH:{group_min_dist['LH'] * 1000.0:+.1f}mm  "
            f"RH:{group_min_dist['RH'] * 1000.0:+.1f}mm  "
            f"penetrating_geoms={penetrations}   ",
            end="",
            flush=True,
          )
          last_reported_frame = frame

        playing = timeline_control is None or timeline_control.playing
        if playing:
          frame += 1
          if frame >= n_frames:
            if not loop:
              break
            frame = start_frame

        elapsed = time.perf_counter() - t0
        target_dt = dt if playing else max(dt, 1.0 / 60.0)
        if elapsed < target_dt:
          time.sleep(target_dt - elapsed)
    finally:
      if timeline_control is not None:
        timeline_control.close()

  print()


if __name__ == "__main__":
  tyro.cli(visualize)
