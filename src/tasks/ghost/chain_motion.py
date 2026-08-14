"""Plan chained Ghost ONNX clips and resolve their reference motions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from .motion_clip import (
  MOTION_ARRAY_KEYS,
  MotionClip,
  load_motion_npz,
)


@dataclass(frozen=True)
class ChainSegment:
  onnx: Path
  start: int = 0
  end: int | None = None
  motion: Path | None = None


def parse_segment_spec(spec: str) -> ChainSegment:
  """Parse ``onnx=PATH,start=0,end=10,motion=PATH`` (motion is optional)."""
  fields: dict[str, str] = {}
  for part in spec.split(","):
    piece = part.strip()
    if not piece:
      continue
    key, separator, value = piece.partition("=")
    if not separator or not key.strip() or not value.strip():
      raise ValueError(
        f"Invalid segment field {piece!r}; expected key=value in {spec!r}."
      )
    fields[key.strip()] = value.strip()

  unknown = sorted(set(fields) - {"onnx", "start", "end", "motion"})
  if unknown:
    raise ValueError(f"Unknown segment fields {unknown} in {spec!r}.")
  if "onnx" not in fields:
    raise ValueError(f"Segment spec is missing onnx=...: {spec!r}.")

  start = int(fields["start"]) if "start" in fields else 0
  end = int(fields["end"]) if "end" in fields else None
  motion = Path(fields["motion"]).expanduser() if "motion" in fields else None
  return ChainSegment(
    onnx=Path(fields["onnx"]).expanduser(),
    start=start,
    end=end,
    motion=motion,
  )


def _closed_range(
  start: int,
  end: int,
  *,
  label: str,
  minimum_frames: int = 1,
) -> tuple[int, int]:
  if start < 0 or end < start:
    raise ValueError(f"{label} frame range [{start}, {end}] is invalid.")
  if end - start + 1 < minimum_frames:
    raise ValueError(
      f"{label} [{start}, {end}] must contain at least "
      f"{minimum_frames} frames."
    )
  return start, end


def fill_coverage_plan(
  *,
  total_frames: int,
  default_onnx: Path,
  motion: Path,
  specialists: list[ChainSegment],
) -> list[ChainSegment]:
  """Fill every source frame: specialists keep ranges, default covers gaps."""
  if total_frames < 2:
    raise ValueError("Source motion must contain at least two frames.")
  last = total_frames - 1
  claimed: list[tuple[int, int, Path]] = []
  for spec in specialists:
    if spec.end is None:
      raise ValueError(
        "Specialist segments must set end= when using --default-onnx."
      )
    start, end = _closed_range(
      spec.start, spec.end, label=str(spec.onnx), minimum_frames=2
    )
    if end > last:
      raise ValueError(
        f"{spec.onnx} range [{start}, {end}] exceeds source last frame {last}."
      )
    claimed.append((start, end, spec.onnx))
  claimed.sort(key=lambda item: item[0])
  for (start_a, end_a, onnx_a), (start_b, end_b, onnx_b) in pairwise(claimed):
    if start_b <= end_a:
      raise ValueError(
        f"Overlapping specialist ranges [{start_a}, {end_a}] ({onnx_a.name}) "
        f"and [{start_b}, {end_b}] ({onnx_b.name})."
      )

  plan: list[ChainSegment] = []
  cursor = 0
  for start, end, onnx in claimed:
    if start > cursor:
      plan.append(
        ChainSegment(
          onnx=default_onnx, start=cursor, end=start - 1, motion=motion
        )
      )
    plan.append(ChainSegment(onnx=onnx, start=start, end=end, motion=motion))
    cursor = end + 1
  if cursor <= last:
    plan.append(
      ChainSegment(
        onnx=default_onnx, start=cursor, end=last, motion=motion
      )
    )
  if not plan:
    plan.append(
      ChainSegment(onnx=default_onnx, start=0, end=last, motion=motion)
    )
  return plan


def planned_frame_count(segments: list[ChainSegment]) -> int:
  """Inclusive coverage length; requires every segment to have a closed end."""
  total = 0
  for segment in segments:
    if segment.end is None:
      raise ValueError("planned_frame_count requires every segment to set end.")
    start, end = _closed_range(
      segment.start, segment.end, label=str(segment.onnx)
    )
    total += end - start + 1
  return total


def _csv_names(value: str, *, field_name: str) -> tuple[str, ...]:
  names = tuple(item.strip() for item in value.split(","))
  empty = [index for index, name in enumerate(names) if not name]
  if empty:
    raise ValueError(
      f"ONNX {field_name} contains empty entries at indices {empty}."
    )
  if len(names) != len(set(names)):
    raise ValueError(f"ONNX {field_name} contains duplicate names.")
  return names


def _initializer_map(model) -> dict[str, np.ndarray]:
  from onnx import numpy_helper

  arrays: dict[str, np.ndarray] = {}
  for initializer in model.graph.initializer:
    name = initializer.name.rsplit("/", 1)[-1]
    if name.endswith(".1") and name[:-2] in MOTION_ARRAY_KEYS:
      key = name[:-2]
    elif name in MOTION_ARRAY_KEYS:
      key = name
    else:
      continue
    arrays[key] = np.asarray(numpy_helper.to_array(initializer))
  return arrays


def extract_onnx_embedded_motion(onnx_path: str | Path) -> dict[str, np.ndarray]:
  """Read the motion clip baked into a Ghost motion-aware ONNX."""
  import onnx

  path = Path(onnx_path).expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(path)
  model = onnx.load(str(path), load_external_data=False)
  arrays = _initializer_map(model)
  missing = [key for key in MOTION_ARRAY_KEYS if key not in arrays]
  if missing:
    raise ValueError(
      f"ONNX {path} does not embed a motion clip (missing {missing}). "
      "Export a motion-aware Ghost policy, not actor-only policy.onnx."
    )

  metadata = {prop.key: prop.value for prop in model.metadata_props}
  payload: dict[str, np.ndarray] = {
    key: np.asarray(arrays[key], dtype=np.float32) for key in MOTION_ARRAY_KEYS
  }
  if "joint_names" in metadata:
    payload["joint_names"] = np.asarray(
      _csv_names(metadata["joint_names"], field_name="joint_names")
    )
  if "body_names" in metadata:
    payload["body_names"] = np.asarray(
      _csv_names(metadata["body_names"], field_name="body_names")
    )
  if "control_dt" in metadata:
    control_dt = float(metadata["control_dt"])
    if control_dt <= 0.0:
      raise ValueError(f"ONNX control_dt must be positive, got {control_dt}.")
    payload["fps"] = np.asarray([1.0 / control_dt], dtype=np.float64)
  else:
    payload["fps"] = np.asarray([100.0], dtype=np.float64)
  return MotionClip.from_mapping(payload, require_fps=True).to_dict()


def resolve_segment_motion(segment: ChainSegment) -> dict[str, np.ndarray]:
  """Use an explicit NPZ if given, otherwise the ONNX-embedded clip."""
  if segment.motion is not None:
    return load_motion_npz(segment.motion)
  return extract_onnx_embedded_motion(segment.onnx)
