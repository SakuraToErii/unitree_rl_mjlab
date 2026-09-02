#!/usr/bin/env python3
"""Replay two TK3 motion NPZs in one viewer, left/right, frame-synced.

The two robots share a clock: the same frame index is applied every
tick. A shorter clip holds its last pose. Playback dt follows the left NPZ fps.

Example::

  python scripts/visualize_npz_pair.py \\
    --left datasets/perfect1-1.npz \\
    --right datasets/perfect2-1.npz
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro
from mjlab.entity.entity import Entity

from src.assets.robots.tiangong3.tk3_constants import get_tk3_robot_cfg

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
  sys.path.insert(0, str(_SCRIPTS))

from visualize_npz_foot_penetration import (  # noqa: E402
  _FLOOR_MATERIAL,
  _FLOOR_TEXTURE,
  _IDENTITY_MAT,
  _SUN_LIGHT,
  _load_motion_data,
  _try_create_timeline,
)

_LEFT_TINT = np.array([0.35, 0.55, 0.85, 1.0], dtype=np.float32)
_RIGHT_TINT = np.array([0.90, 0.55, 0.25, 1.0], dtype=np.float32)
_LABEL_RGBA = np.array([0.95, 0.95, 0.95, 0.95], dtype=np.float32)


@dataclass(frozen=True)
class _Slot:
  prefix: str
  label: str
  free_qadr: int
  joint_qadrs: np.ndarray
  joint_pos: np.ndarray
  root_pos_w: np.ndarray
  root_quat_w: np.ndarray
  origin_offset: np.ndarray
  n_frames: int
  fps: float


def _robot_spec() -> mujoco.MjSpec:
  return Entity(get_tk3_robot_cfg(convex_sole=True)).spec


def _build_pair_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
  spec = mujoco.MjSpec()
  spec.modelname = "npz_pair"
  ground = spec.worldbody.add_geom()
  ground.name = "terrain"
  ground.type = mujoco.mjtGeom.mjGEOM_PLANE
  ground.size = [0.0, 0.0, 0.05]
  _FLOOR_TEXTURE.edit_spec(spec)
  _FLOOR_MATERIAL.edit_spec(spec)
  _SUN_LIGHT.edit_spec(spec)
  spec.visual.headlight.ambient[:] = (0.1, 0.1, 0.1)
  spec.visual.headlight.diffuse[:] = (0.6, 0.6, 0.6)
  spec.visual.headlight.specular[:] = (0.9, 0.9, 0.9)

  # Free joints write world qpos, so the left/right split is applied in
  # `_apply_slot` rather than on these frames.
  left_frame = spec.worldbody.add_frame(name="left_origin")
  right_frame = spec.worldbody.add_frame(name="right_origin")
  spec.attach(_robot_spec(), prefix="L/", frame=left_frame)
  spec.attach(_robot_spec(), prefix="R/", frame=right_frame)
  model = spec.compile()
  return model, mujoco.MjData(model)


def _free_qadr(model: mujoco.MjModel, prefix: str) -> int:
  for joint_id in range(model.njnt):
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
      continue
    if model.joint(joint_id).name.startswith(prefix):
      return int(model.jnt_qposadr[joint_id])
  raise RuntimeError(f"No free joint for prefix {prefix!r}.")


def _joint_qadrs(model: mujoco.MjModel, prefix: str, art_joints: list[str]) -> np.ndarray:
  addresses = []
  for name in art_joints:
    joint_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{name}"
    )
    if joint_id < 0:
      raise RuntimeError(f"Missing joint {prefix}{name}.")
    addresses.append(int(model.jnt_qposadr[joint_id]))
  return np.asarray(addresses, dtype=np.int32)


def _unprefixed_art_joints(model: mujoco.MjModel, prefix: str) -> list[str]:
  names: list[str] = []
  for joint_id in range(model.njnt):
    if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
      continue
    full = model.joint(joint_id).name
    if full.startswith(prefix):
      names.append(full[len(prefix) :])
  return names


def _tint_visual_geoms(model: mujoco.MjModel, prefix: str, rgba: np.ndarray) -> None:
  for geom_id in range(model.ngeom):
    name = model.geom(geom_id).name or ""
    if name.startswith(prefix) and int(model.geom_group[geom_id]) == 2:
      model.geom_rgba[geom_id] = rgba


def _load_slot(
  model: mujoco.MjModel,
  path: Path,
  prefix: str,
  label: str,
  art_joints: list[str],
  origin_offset: np.ndarray,
) -> _Slot:
  resolved = path.expanduser().resolve()
  if not resolved.is_file():
    raise FileNotFoundError(resolved)
  fps, joint_pos, root_pos_w, root_quat_w = _load_motion_data(resolved, art_joints)
  return _Slot(
    prefix=prefix,
    label=label,
    free_qadr=_free_qadr(model, prefix),
    joint_qadrs=_joint_qadrs(model, prefix, art_joints),
    joint_pos=joint_pos,
    root_pos_w=root_pos_w,
    root_quat_w=root_quat_w,
    origin_offset=np.asarray(origin_offset, dtype=np.float64),
    n_frames=int(joint_pos.shape[0]),
    fps=fps,
  )


def _apply_slot(model: mujoco.MjModel, data: mujoco.MjData, slot: _Slot, frame: int) -> None:
  index = min(frame, slot.n_frames - 1)
  qadr = slot.free_qadr
  data.qpos[qadr : qadr + 3] = slot.root_pos_w[index] + slot.origin_offset
  data.qpos[qadr + 3 : qadr + 7] = slot.root_quat_w[index]
  data.qpos[slot.joint_qadrs] = slot.joint_pos[index]


def _add_name_label(scn: mujoco.MjvScene, pos: np.ndarray, text: str) -> None:
  if scn.ngeom >= scn.maxgeom:
    return
  geom = scn.geoms[scn.ngeom]
  label_pos = np.asarray(pos, dtype=np.float64) + np.array([0.0, 0.0, 0.28])
  mujoco.mjv_initGeom(
    geom,
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=np.array([0.02, 0.0, 0.0]),
    pos=label_pos,
    mat=_IDENTITY_MAT,
    rgba=_LABEL_RGBA,
  )
  geom.label = text
  scn.ngeom += 1


def visualize(
  left: Path = Path("datasets/perfect1-1.npz"),
  right: Path = Path("datasets/perfect2-1.npz"),
  separation: float = 3.0,
  speed: float = 1.0,
  loop: bool = True,
  start_frame: int = 0,
  timeline: bool = True,
) -> None:
  """Play two NPZs side by side with a shared frame clock.

  Args:
    left: Clip placed at +Y (viewer left when looking along +X).
    right: Clip placed at -Y.
    separation: World-Y distance between the two attachment origins, metres.
    speed: Playback rate relative to the left clip fps.
    loop: Restart from start_frame after the longer clip ends.
    start_frame: Shared initial frame index.
    timeline: Open a draggable frame timeline with play/pause.
  """
  if separation <= 0.0:
    raise ValueError(f"separation must be positive, got {separation}.")

  model, data = _build_pair_model()
  art_joints = _unprefixed_art_joints(model, "L/")
  if art_joints != _unprefixed_art_joints(model, "R/"):
    raise RuntimeError("Left and right robots have different joint layouts.")

  half = 0.5 * separation
  left_slot = _load_slot(
    model, left, "L/", left.name, art_joints, np.array([0.0, half, 0.0])
  )
  right_slot = _load_slot(
    model, right, "R/", right.name, art_joints, np.array([0.0, -half, 0.0])
  )
  _tint_visual_geoms(model, "L/", _LEFT_TINT)
  _tint_visual_geoms(model, "R/", _RIGHT_TINT)

  n_frames = max(left_slot.n_frames, right_slot.n_frames)
  if not (0 <= start_frame < n_frames):
    raise ValueError(f"start_frame={start_frame} out of range [0, {n_frames}).")
  if abs(left_slot.fps - right_slot.fps) > 1.0e-6:
    print(
      f"[WARN] fps differs ({left_slot.fps:g} vs {right_slot.fps:g}); "
      "syncing by frame index, clock uses the left clip."
    )

  fps = left_slot.fps
  dt = 1.0 / max(fps * speed, 1.0e-6)
  frame = start_frame
  left_pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "L/pelvis")
  right_pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "R/pelvis")

  print(
    "\n".join(
      [
        f"left  (+Y): {left.expanduser().resolve()}  "
        f"{left_slot.n_frames} frames @ {left_slot.fps:g} Hz",
        f"right (-Y): {right.expanduser().resolve()}  "
        f"{right_slot.n_frames} frames @ {right_slot.fps:g} Hz",
        f"separation: {separation:g} m on world Y   synced frames: {n_frames}",
        "blue = left, orange = right; shorter clip holds its last pose",
        "timeline: drag to seek; Space toggles play/pause",
      ]
    )
  )

  with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = (0.0, 0.0, 0.9)
    viewer.cam.distance = 8.0
    viewer.cam.azimuth = 180.0
    viewer.cam.elevation = -15.0
    timeline_control = _try_create_timeline(timeline, n_frames, fps, start_frame)
    try:
      while viewer.is_running() and (
        timeline_control is None or timeline_control.is_open
      ):
        t0 = time.perf_counter()
        if timeline_control is not None:
          timeline_control.pump()
          requested = timeline_control.consume_requested_frame()
          if requested is not None:
            frame = requested

        data.qvel[:] = 0.0
        _apply_slot(model, data, left_slot, frame)
        _apply_slot(model, data, right_slot, frame)
        mujoco.mj_forward(model, data)

        scn = viewer.user_scn
        scn.ngeom = 0
        _add_name_label(scn, data.xpos[left_pelvis], left_slot.label)
        _add_name_label(scn, data.xpos[right_pelvis], right_slot.label)

        if timeline_control is not None:
          timeline_control.update_frame(frame)
        viewer.sync()

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


if __name__ == "__main__":
  tyro.cli(visualize)
