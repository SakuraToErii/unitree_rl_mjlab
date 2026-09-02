#!/usr/bin/env python3
"""Lift a TK3 tracking NPZ so selected collision geoms rest near the z=0 plane.

Per frame:
  1. Apply motion root + joints to the TK3 MuJoCo model.
  2. Measure min signed distance from the selected geoms to the ground plane.
  3. If they sit below --clearance, add a +Z lift to every body position.
     Joint angles / body quaternions are unchanged.
  4. Recompute body linear velocities from the corrected positions.

Foot-based lift is opt-in only. Raising the whole pose from the lower foot
unplants the other foot and can turn a plant into a shuffling step. For
sideflips / backflips, use --target hands so only TCP penetration drives a
whole-body +Z lift; joint angles are unchanged and aerial frames are left
alone (lift is never negative). Never overwrites the input NPZ.

Example:
  python src/tasks/tracking/scripts/ground_npz_motion.py \\
    --input datasets/1_1_padding.npz \\
    --output datasets/1_1_padding_grounded.npz

  python src/tasks/tracking/scripts/ground_npz_motion.py \\
    --input datasets/1-1_paddingv2.npz \\
    --output datasets/1-1_paddingv2_tcp8mm.npz \\
    --target hands \\
    --clearance -0.008
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np
import tyro
from tqdm import tqdm

from mjlab.entity.entity import Entity

from src.assets.robots.tiangong3.tk3_constants import (
  FOOT_GEOM_PATTERN,
  get_tk3_robot_cfg,
)

HAND_GEOM_PATTERN = r"^(left|right)_tcp_link_collision(?:_[0-9]+)?$"
GroundTarget = Literal["feet", "hands", "both"]


def _geom_ids_matching(model: mujoco.MjModel, pattern: str) -> list[int]:
  geom_re = re.compile(pattern)
  return [
    i
    for i in range(model.ngeom)
    if model.geom(i).name and geom_re.match(model.geom(i).name)
  ]


def _build_model(
  target: GroundTarget,
) -> tuple[mujoco.MjModel, mujoco.MjData, list[int], int, list[str]]:
  robot = Entity(get_tk3_robot_cfg())
  spec = robot.spec
  ground = spec.worldbody.add_geom()
  ground.name = "terrain"
  ground.type = mujoco.mjtGeom.mjGEOM_PLANE
  ground.size = [0.0, 0.0, 0.01]
  model = spec.compile()
  data = mujoco.MjData(model)

  foot_ids = _geom_ids_matching(model, FOOT_GEOM_PATTERN)
  hand_ids = _geom_ids_matching(model, HAND_GEOM_PATTERN)
  if target in ("feet", "both") and not foot_ids:
    raise RuntimeError("No foot collision geoms matched FOOT_GEOM_PATTERN.")
  if target in ("hands", "both") and not hand_ids:
    raise RuntimeError("No TCP collision geoms matched HAND_GEOM_PATTERN.")
  if target == "feet":
    tracked_ids = foot_ids
  elif target == "hands":
    tracked_ids = hand_ids
  else:
    tracked_ids = foot_ids + hand_ids

  plane_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
  art_joints = [
    model.joint(i).name
    for i in range(model.njnt)
    if model.joint(i).type != mujoco.mjtJoint.mjJNT_FREE
  ]
  return model, data, tracked_ids, plane_id, art_joints


def _min_geom_distance(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  geom_ids: list[int],
  plane_id: int,
) -> float:
  fromto = np.zeros(6)
  return float(
    min(
      mujoco.mj_geomDistance(model, data, gid, plane_id, 10.0, fromto)
      for gid in geom_ids
    )
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


def _finite_difference_vel(pos: np.ndarray, dt: float) -> np.ndarray:
  """Central differences; one-sided at the ends. pos: (T, ..., 3)."""
  vel = np.empty_like(pos)
  vel[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
  vel[0] = (pos[1] - pos[0]) / dt
  vel[-1] = (pos[-1] - pos[-2]) / dt
  return vel


def _motion_indexes(
  payload: dict[str, np.ndarray],
  art_joints: list[str],
) -> tuple[np.ndarray, int]:
  """Map stored joint columns to MuJoCo order and locate the pelvis body."""
  joint_pos = np.asarray(payload["joint_pos"], dtype=np.float32)
  if "joint_names" in payload:
    stored_joint_names = [str(name) for name in payload["joint_names"].tolist()]
    index_by_name = {name: index for index, name in enumerate(stored_joint_names)}
    missing = [name for name in art_joints if name not in index_by_name]
    if missing:
      raise ValueError(f"Motion joint_names is missing MuJoCo joints: {missing}.")
    joint_pos = joint_pos[:, [index_by_name[name] for name in art_joints]]
  elif joint_pos.shape[1] != len(art_joints):
    raise ValueError(
      f"Motion has {joint_pos.shape[1]} unnamed joints but robot has {len(art_joints)}."
    )

  pelvis_index = 0
  if "body_names" in payload:
    stored_body_names = [str(name) for name in payload["body_names"].tolist()]
    if "pelvis" not in stored_body_names:
      raise ValueError("Motion body_names does not contain 'pelvis'.")
    pelvis_index = stored_body_names.index("pelvis")
  return joint_pos, pelvis_index


def ground_motion(
  input_path: Path,
  output_path: Path,
  clearance: float = -0.001,
  *,
  target: GroundTarget = "feet",
  dry_run: bool = False,
  overwrite: bool = False,
) -> dict[str, float]:
  """Return summary stats; write grounded NPZ unless dry_run."""
  model, data, tracked_ids, plane_id, art_joints = _build_model(target)

  with np.load(input_path, allow_pickle=False) as raw:
    payload = {key: raw[key] for key in raw.files}

  fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
  dt = 1.0 / fps
  joint_pos, pelvis_index = _motion_indexes(payload, art_joints)
  body_pos_w = np.asarray(payload["body_pos_w"], dtype=np.float32).copy()
  body_quat_w = np.asarray(payload["body_quat_w"], dtype=np.float32)
  n_frames = joint_pos.shape[0]

  if body_pos_w.shape[1] != model.nbody - 1:
    raise ValueError(
      f"Motion has {body_pos_w.shape[1]} bodies but robot has {model.nbody - 1}."
    )

  lifts = np.zeros(n_frames, dtype=np.float32)
  dist_before = np.empty(n_frames, dtype=np.float32)
  label = "hand" if target == "hands" else "foot" if target == "feet" else "contact"

  for t in tqdm(range(n_frames), desc=f"Measuring {label} clearance", unit="frame"):
    _apply_frame(
      model,
      data,
      art_joints,
      body_pos_w[t, pelvis_index],
      body_quat_w[t, pelvis_index],
      joint_pos[t],
    )
    dist = _min_geom_distance(model, data, tracked_ids, plane_id)
    dist_before[t] = dist
    lifts[t] = max(0.0, clearance - dist)

  body_pos_w[..., 2] += lifts[:, None]

  # Recompute linear velocities from corrected positions. Angular velocities are
  # unchanged by a pure +Z translation; keep the originals.
  body_lin_vel_w = _finite_difference_vel(body_pos_w, dt).astype(np.float32)
  original_joint_pos = np.asarray(payload["joint_pos"], dtype=np.float32)
  joint_vel = _finite_difference_vel(
    original_joint_pos[..., None], dt
  ).astype(np.float32)[..., 0]

  # Spot-check after correction, including every lifted frame so short
  # acrobatic contacts cannot hide between a uniform subsample.
  dist_after = np.empty(n_frames, dtype=np.float32)
  lifted = np.flatnonzero(lifts > 1e-6)
  sample = np.unique(
    np.concatenate(
      [
        np.array([0, n_frames - 1], dtype=int),
        np.linspace(0, n_frames - 1, min(200, n_frames), dtype=int),
        lifted,
      ]
    )
  )
  for t in sample:
    _apply_frame(
      model,
      data,
      art_joints,
      body_pos_w[t, pelvis_index],
      body_quat_w[t, pelvis_index],
      joint_pos[t],
    )
    dist_after[t] = _min_geom_distance(model, data, tracked_ids, plane_id)
  checked = dist_after[sample]

  stats = {
    "frames": float(n_frames),
    "fps": fps,
    "clearance": clearance,
    "lift_mean": float(lifts.mean()),
    "lift_max": float(lifts.max()),
    "lift_frac_nonzero": float((lifts > 1e-6).mean()),
    "lift_frames": float(lifted.size),
    "dist_before_min": float(dist_before.min()),
    "dist_before_median": float(np.median(dist_before)),
    "dist_after_min_sampled": float(checked.min()),
    "dist_after_median_sampled": float(np.median(checked)),
  }

  cluster_lines = []
  if lifted.size:
    starts = [int(lifted[0])]
    ends = []
    for prev, cur in zip(lifted[:-1], lifted[1:]):
      if cur != prev + 1:
        ends.append(int(prev))
        starts.append(int(cur))
    ends.append(int(lifted[-1]))
    for start, end in zip(starts, ends):
      cluster_lines.append(
        f"  lift frames {start}-{end} "
        f"({(end - start + 1) / fps:.2f}s) "
        f"max={lifts[start : end + 1].max():.4f} m"
      )

  print(
    "\n".join(
      [
        f"input:  {input_path}",
        f"target: {target}",
        f"frames: {n_frames} @ {fps:g} Hz",
        f"clearance: {clearance:.4f} m",
        f"lift: mean={stats['lift_mean']:.4f} max={stats['lift_max']:.4f} "
        f"nonzero={int(stats['lift_frames'])} frames "
        f"({100 * stats['lift_frac_nonzero']:.1f}%)",
        f"{label} dist before: min={stats['dist_before_min']:.4f} "
        f"median={stats['dist_before_median']:.4f}",
        f"{label} dist after (sample): min={stats['dist_after_min_sampled']:.4f} "
        f"median={stats['dist_after_median_sampled']:.4f}",
        *cluster_lines,
      ]
    )
  )

  if dry_run:
    print("dry-run: not writing output")
    return stats

  if output_path.resolve() == input_path.resolve():
    raise ValueError(
      "Refusing to overwrite the input NPZ. Pass a different --output path "
      "(default is <input>_grounded.npz)."
    )
  if output_path.exists() and not overwrite:
    raise FileExistsError(
      f"Output already exists: {output_path}. Pass --overwrite or choose "
      "another path."
    )

  out = dict(payload)
  out["joint_pos"] = original_joint_pos
  out["joint_vel"] = joint_vel
  out["body_pos_w"] = body_pos_w
  out["body_quat_w"] = body_quat_w
  out["body_lin_vel_w"] = body_lin_vel_w
  # Keep original angular velocities: pure +Z lift does not rotate bodies.
  out["body_ang_vel_w"] = np.asarray(payload["body_ang_vel_w"], dtype=np.float32)
  out["fps"] = np.asarray(payload["fps"])

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output_path, **out)
  print(f"wrote:  {output_path}")
  return stats


def main(
  input: Path,
  output: Path | None = None,
  clearance: float = -0.001,
  target: GroundTarget = "feet",
  dry_run: bool = False,
  overwrite: bool = False,
) -> None:
  """Ground an NPZ motion so TK3 feet or TCP fists rest with a controlled sink.

  Never overwrites the input file. Defaults to writing <stem>_grounded.npz
  beside the source.

  Args:
    input: Source motion NPZ (left untouched).
    output: Destination NPZ. Defaults to <input>_grounded.npz.
    clearance: Minimum allowed geom-plane distance in meters. Negative values
      keep a controlled penetration (default -1 mm). Use -0.008 with
      --target hands so the deepest fist contact is 8 mm.
    target: Collision geoms that drive the per-frame +Z lift: feet, hands
      (TCP fist meshes), or both.
    dry_run: Measure and print stats without writing.
    overwrite: Replace an existing output NPZ. Still never overwrites input.
  """
  input_path = input.expanduser().resolve()
  if output is None:
    output_path = input_path.with_name(f"{input_path.stem}_grounded.npz")
  else:
    output_path = output.expanduser().resolve()
  ground_motion(
    input_path,
    output_path,
    clearance=clearance,
    target=target,
    dry_run=dry_run,
    overwrite=overwrite,
  )


if __name__ == "__main__":
  tyro.cli(main)
