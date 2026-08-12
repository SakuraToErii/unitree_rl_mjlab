#!/usr/bin/env python3
"""Lift a TK3 tracking NPZ so foot collision geoms rest near the z=0 plane.

Per frame:
  1. Apply motion root + joints to the TK3 MuJoCo model.
  2. Measure min signed distance from foot geoms to the ground plane.
  3. If the feet sit below --clearance, add a +Z lift to every body position.
     Joint angles / body quaternions are unchanged.
  4. Recompute body linear velocities from the corrected positions.

Default clearance is -1 mm so soft contacts engage immediately at reset.
Aerial frames are left alone (lift is never negative). Never overwrites the
input NPZ.

Example:
  python scripts/ground_npz_motion.py \\
    --input datasets/1_1_padding.npz \\
    --output datasets/1_1_padding_grounded.npz
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np
import tyro
from tqdm import tqdm

from mjlab.entity.entity import Entity

from src.assets.robots.tiangong3.tk3_constants import (
  FOOT_GEOM_PATTERN,
  get_tk3_robot_cfg,
)


def _build_model() -> tuple[mujoco.MjModel, mujoco.MjData, list[int], int, list[str]]:
  robot = Entity(get_tk3_robot_cfg())
  spec = robot.spec
  ground = spec.worldbody.add_geom()
  ground.name = "terrain"
  ground.type = mujoco.mjtGeom.mjGEOM_PLANE
  ground.size = [0.0, 0.0, 0.01]
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


def _min_foot_distance(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  foot_ids: list[int],
  plane_id: int,
) -> float:
  fromto = np.zeros(6)
  return float(
    min(
      mujoco.mj_geomDistance(model, data, gid, plane_id, 10.0, fromto)
      for gid in foot_ids
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


def ground_motion(
  input_path: Path,
  output_path: Path,
  clearance: float = -0.001,
  *,
  dry_run: bool = False,
  overwrite: bool = False,
) -> dict[str, float]:
  """Return summary stats; write grounded NPZ unless dry_run."""
  model, data, foot_ids, plane_id, art_joints = _build_model()

  with np.load(input_path, allow_pickle=False) as raw:
    payload = {key: raw[key] for key in raw.files}

  fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
  dt = 1.0 / fps
  joint_pos = np.asarray(payload["joint_pos"], dtype=np.float32)
  body_pos_w = np.asarray(payload["body_pos_w"], dtype=np.float32).copy()
  body_quat_w = np.asarray(payload["body_quat_w"], dtype=np.float32)
  n_frames = joint_pos.shape[0]

  if joint_pos.shape[1] != len(art_joints):
    raise ValueError(
      f"Motion has {joint_pos.shape[1]} joints but robot has {len(art_joints)}."
    )
  if body_pos_w.shape[1] != model.nbody - 1:
    raise ValueError(
      f"Motion has {body_pos_w.shape[1]} bodies but robot has {model.nbody - 1}."
    )

  lifts = np.zeros(n_frames, dtype=np.float32)
  dist_before = np.empty(n_frames, dtype=np.float32)

  for t in tqdm(range(n_frames), desc="Measuring foot clearance", unit="frame"):
    _apply_frame(
      model,
      data,
      art_joints,
      body_pos_w[t, 0],
      body_quat_w[t, 0],
      joint_pos[t],
    )
    dist = _min_foot_distance(model, data, foot_ids, plane_id)
    dist_before[t] = dist
    lifts[t] = max(0.0, clearance - dist)

  body_pos_w[..., 2] += lifts[:, None]

  # Recompute linear velocities from corrected positions. Angular velocities are
  # unchanged by a pure +Z translation; keep the originals.
  body_lin_vel_w = _finite_difference_vel(body_pos_w, dt).astype(np.float32)
  joint_vel = _finite_difference_vel(
    joint_pos[..., None], dt
  ).astype(np.float32)[..., 0]

  # Spot-check after correction.
  dist_after = np.empty(n_frames, dtype=np.float32)
  sample = np.unique(
    np.concatenate(
      [
        np.array([0, n_frames - 1], dtype=int),
        np.linspace(0, n_frames - 1, min(200, n_frames), dtype=int),
      ]
    )
  )
  for t in sample:
    _apply_frame(
      model,
      data,
      art_joints,
      body_pos_w[t, 0],
      body_quat_w[t, 0],
      joint_pos[t],
    )
    dist_after[t] = _min_foot_distance(model, data, foot_ids, plane_id)
  checked = dist_after[sample]

  stats = {
    "frames": float(n_frames),
    "fps": fps,
    "clearance": clearance,
    "lift_mean": float(lifts.mean()),
    "lift_max": float(lifts.max()),
    "lift_frac_nonzero": float((lifts > 1e-6).mean()),
    "dist_before_min": float(dist_before.min()),
    "dist_before_median": float(np.median(dist_before)),
    "dist_after_min_sampled": float(checked.min()),
    "dist_after_median_sampled": float(np.median(checked)),
  }

  print(
    "\n".join(
      [
        f"input:  {input_path}",
        f"frames: {n_frames} @ {fps:g} Hz",
        f"clearance: {clearance:.4f} m",
        f"lift: mean={stats['lift_mean']:.4f} max={stats['lift_max']:.4f} "
        f"nonzero={100 * stats['lift_frac_nonzero']:.1f}%",
        f"foot dist before: min={stats['dist_before_min']:.4f} "
        f"median={stats['dist_before_median']:.4f}",
        f"foot dist after (sample): min={stats['dist_after_min_sampled']:.4f} "
        f"median={stats['dist_after_median_sampled']:.4f}",
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
  out["joint_pos"] = joint_pos
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
  dry_run: bool = False,
  overwrite: bool = False,
) -> None:
  """Ground an NPZ motion so TK3 feet rest with a small soft-contact sink.

  Never overwrites the input file. Defaults to writing <stem>_grounded.npz
  beside the source.

  Args:
    input: Source motion NPZ (left untouched).
    output: Destination NPZ. Defaults to <input>_grounded.npz.
    clearance: Minimum allowed foot-plane distance in meters. Negative values
      keep a controlled penetration (default -1 mm).
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
    dry_run=dry_run,
    overwrite=overwrite,
  )


if __name__ == "__main__":
  tyro.cli(main)
