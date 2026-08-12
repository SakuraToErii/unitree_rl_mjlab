"""Regression tests for named Isaac Lab NPZ visualization."""

import importlib.util
from pathlib import Path

import numpy as np

_SCRIPT_PATH = (
  Path(__file__).parents[1] / "scripts" / "visualize_npz_foot_penetration.py"
)
_SPEC = importlib.util.spec_from_file_location(
  "visualize_npz_foot_penetration", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_load_motion_data = _MODULE._load_motion_data
_build_model = _MODULE._build_model
_find_hand_groups = _MODULE._find_hand_groups


def test_load_motion_data_reorders_joints_and_resolves_pelvis(
  tmp_path: Path,
) -> None:
  motion_path = tmp_path / "isaac_order.npz"
  np.savez(
    motion_path,
    fps=np.asarray([100]),
    joint_pos=np.asarray([[20.0, 0.0, 10.0]], dtype=np.float32),
    joint_names=np.asarray(["j2", "j0", "j1"]),
    body_pos_w=np.asarray(
      [[[9.0, 9.0, 9.0], [1.0, 2.0, 3.0]]], dtype=np.float32
    ),
    body_quat_w=np.asarray(
      [[[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]],
      dtype=np.float32,
    ),
    body_names=np.asarray(["other", "pelvis"]),
  )

  fps, joint_pos, root_pos, root_quat = _load_motion_data(
    motion_path, ["j0", "j1", "j2"]
  )

  assert fps == 100.0
  np.testing.assert_array_equal(
    joint_pos, np.asarray([[0.0, 10.0, 20.0]], dtype=np.float32)
  )
  np.testing.assert_array_equal(
    root_pos, np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
  )
  np.testing.assert_array_equal(
    root_quat, np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
  )


def test_tk3_model_resolves_left_and_right_hand_collision_meshes() -> None:
  model, _, _, _, _ = _build_model()

  hand_groups = _find_hand_groups(model)
  hand_names = {
    label: [model.geom(geom_id).name for geom_id in geom_ids]
    for label, geom_ids in hand_groups.items()
  }

  assert hand_names == {
    "LH": ["left_tcp_link_collision_0"],
    "RH": ["right_tcp_link_collision_0"],
  }
