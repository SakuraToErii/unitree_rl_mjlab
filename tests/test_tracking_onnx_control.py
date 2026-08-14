"""Tests for tracking ONNX control overlays."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import onnx
import torch
from mjlab.actuator import BuiltinPositionActuator
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from onnx import helper

from src.tasks.tracking.onnx_control import OnnxControlOverlay


def _write_onnx(path: Path, metadata: list[tuple[str, str]]) -> None:
  graph = helper.make_graph([], "metadata_only", [], [])
  model = helper.make_model(graph)
  for key, value in metadata:
    model.metadata_props.append(onnx.StringStringEntryProto(key=key, value=value))
  onnx.save(model, path)


class _StubJointPositionAction(JointPositionAction):
  def __init__(
    self,
    *,
    scale: torch.Tensor | float | None = None,
    offset: torch.Tensor | float | None = None,
  ) -> None:
    self.cfg = JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=1.0,
      use_default_offset=True,
    )
    self.test_scale = torch.tensor([[0.3, 0.2], [0.3, 0.2]]) if scale is None else scale
    self.test_offset = torch.zeros((2, 2)) if offset is None else offset

  @property
  def scale(self) -> torch.Tensor | float:
    return self.test_scale

  @property
  def offset(self) -> torch.Tensor | float:
    return self.test_offset

  @property
  def target_ids(self) -> torch.Tensor:
    return torch.tensor([1, 0], dtype=torch.long)

  @property
  def target_names(self) -> list[str]:
    return ["joint_b", "joint_a"]


class _StubPositionActuator(BuiltinPositionActuator):
  @property
  def target_names(self) -> list[str]:
    return ["joint_a", "joint_b"]

  @property
  def global_ctrl_ids(self) -> torch.Tensor:
    return torch.tensor([2, 0], dtype=torch.long)


def _runtime_action(
  *,
  scale: torch.Tensor | float | None = None,
  offset: torch.Tensor | float | None = None,
) -> JointPositionAction:
  return _StubJointPositionAction(scale=scale, offset=offset)


def _position_actuator() -> BuiltinPositionActuator:
  return _StubPositionActuator.__new__(_StubPositionActuator)


def _runtime_env(action: JointPositionAction):
  robot = SimpleNamespace(
    actuators=[_position_actuator()],
    data=SimpleNamespace(default_joint_pos=torch.zeros((2, 3))),
  )
  model = SimpleNamespace(
    actuator_gainprm=torch.zeros((2, 3, 10)),
    actuator_biasprm=torch.zeros((2, 3, 10)),
    actuator_forcerange=torch.zeros((2, 3, 2)),
  )
  return SimpleNamespace(
    num_envs=2,
    device="cpu",
    scene={"robot": robot},
    action_manager=SimpleNamespace(get_term=lambda name: action),
    sim=SimpleNamespace(model=model),
  )


class TestOnnxControlParsing(unittest.TestCase):
  def test_reads_metadata_and_expands_scalars(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "policy.onnx"
      _write_onnx(
        path,
        [
          ("joint_names", "joint_a,joint_b"),
          ("joint_stiffness", "10"),
          ("joint_damping", "1,2"),
          ("action_scale", "0.25"),
          ("default_joint_pos", "0.1,0.2"),
          ("joint_effort_limit", "5"),
        ],
      )

      overlay = OnnxControlOverlay.from_onnx(path)

      self.assertEqual(overlay.source, path.resolve())
      self.assertEqual(overlay.joint_names, ("joint_a", "joint_b"))
      self.assertEqual(overlay.joint_stiffness, (10.0, 10.0))
      self.assertEqual(overlay.joint_damping, (1.0, 2.0))
      self.assertEqual(overlay.action_scale, (0.25, 0.25))
      self.assertEqual(overlay.default_joint_pos, (0.1, 0.2))
      self.assertEqual(overlay.joint_effort_limit, (5.0, 5.0))

  def test_rejects_invalid_metadata(self) -> None:
    cases = {
      "duplicate joints": (
        [("joint_names", "joint_a,joint_a")],
        "duplicate names",
      ),
      "wrong vector length": (
        [("joint_names", "joint_a,joint_b"), ("action_scale", "1,2,3")],
        "expected 2",
      ),
      "non-finite vector": (
        [("joint_names", "joint_a,joint_b"), ("action_scale", "nan,1")],
        "must be finite",
      ),
      "unpaired gains": (
        [("joint_names", "joint_a,joint_b"), ("joint_stiffness", "1")],
        "must be provided together",
      ),
      "negative stiffness": (
        [
          ("joint_names", "joint_a,joint_b"),
          ("joint_stiffness", "-1,2"),
          ("joint_damping", "1,2"),
        ],
        r"joint_stiffness\[0\] must be non-negative",
      ),
      "negative damping": (
        [
          ("joint_names", "joint_a,joint_b"),
          ("joint_stiffness", "1,2"),
          ("joint_damping", "-1,2"),
        ],
        r"joint_damping\[0\] must be non-negative",
      ),
      "zero action scale": (
        [("joint_names", "joint_a,joint_b"), ("action_scale", "0,1")],
        r"action_scale\[0\] must be non-zero",
      ),
      "negative effort": (
        [("joint_names", "joint_a,joint_b"), ("joint_effort_limit", "5,-1")],
        "must be non-negative",
      ),
      "empty joint": (
        [("joint_names", "joint_a,,joint_b")],
        "empty entries",
      ),
      "duplicate metadata key": (
        [("joint_names", "joint_a"), ("joint_names", "joint_b")],
        "duplicate key",
      ),
    }
    for name, (metadata, message) in cases.items():
      with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "policy.onnx"
        _write_onnx(path, metadata)
        with self.assertRaisesRegex(ValueError, message):
          OnnxControlOverlay.from_onnx(path)

  def test_accepts_zero_gains_and_negative_action_scale(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "policy.onnx"
      _write_onnx(
        path,
        [
          ("joint_names", "joint_a,joint_b"),
          ("joint_stiffness", "0"),
          ("joint_damping", "0"),
          ("action_scale", "-0.25,0.25"),
        ],
      )

      overlay = OnnxControlOverlay.from_onnx(path)

      self.assertEqual(overlay.joint_stiffness, (0.0, 0.0))
      self.assertEqual(overlay.joint_damping, (0.0, 0.0))
      self.assertEqual(overlay.action_scale, (-0.25, 0.25))


class TestOnnxControlApplication(unittest.TestCase):
  def test_applies_action_scale_to_cfg(self) -> None:
    overlay = OnnxControlOverlay(
      source=Path("policy.onnx"),
      joint_names=("joint_a", "joint_b"),
      action_scale=(0.2, 0.3),
    )
    action_cfg = JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=1.0,
      use_default_offset=True,
    )

    applied = overlay.apply_action_cfg(action_cfg)

    self.assertEqual(applied, ("action_scale",))
    self.assertEqual(action_cfg.scale, {"joint_a": 0.2, "joint_b": 0.3})

  def test_maps_runtime_params_by_joint_name(self) -> None:
    overlay = OnnxControlOverlay(
      source=Path("policy.onnx"),
      joint_names=("joint_a", "joint_b"),
      joint_stiffness=(10.0, 20.0),
      joint_damping=(1.0, 2.0),
      action_scale=(0.2, 0.3),
      default_joint_pos=(0.1, 0.2),
      joint_effort_limit=(5.0, 6.0),
    )
    action = _runtime_action()
    env = _runtime_env(action)

    applied = overlay.apply_runtime(env)

    self.assertEqual(applied, ("Kp/Kd", "effort_limit", "default_joint_pos"))
    model = env.sim.model
    torch.testing.assert_close(
      model.actuator_gainprm[:, [2, 0], 0],
      torch.tensor([[10.0, 20.0], [10.0, 20.0]]),
    )
    torch.testing.assert_close(
      model.actuator_biasprm[:, [2, 0], 1],
      torch.tensor([[-10.0, -20.0], [-10.0, -20.0]]),
    )
    torch.testing.assert_close(
      model.actuator_biasprm[:, [2, 0], 2],
      torch.tensor([[-1.0, -2.0], [-1.0, -2.0]]),
    )
    torch.testing.assert_close(
      model.actuator_forcerange[:, [2, 0]],
      torch.tensor([[[-5.0, 5.0], [-6.0, 6.0]], [[-5.0, 5.0], [-6.0, 6.0]]]),
    )
    torch.testing.assert_close(
      env.scene["robot"].data.default_joint_pos[:, :2],
      torch.tensor([[0.1, 0.2], [0.1, 0.2]]),
    )
    torch.testing.assert_close(
      action.offset,
      torch.tensor([[0.2, 0.1], [0.2, 0.1]]),
    )

  def test_requires_action_scale_before_env_construction(self) -> None:
    overlay = OnnxControlOverlay(
      source=Path("policy.onnx"),
      joint_names=("joint_a", "joint_b"),
      action_scale=(0.2, 0.3),
    )
    action = _runtime_action(scale=0.25)

    with self.assertRaisesRegex(ValueError, "apply_action_cfg"):
      overlay.apply_runtime(_runtime_env(action))

  def test_rejects_scalar_offset_without_partial_update(self) -> None:
    overlay = OnnxControlOverlay(
      source=Path("policy.onnx"),
      joint_names=("joint_a", "joint_b"),
      default_joint_pos=(0.1, 0.2),
    )
    action = _runtime_action(offset=0.0)
    env = _runtime_env(action)

    with self.assertRaisesRegex(TypeError, "scalar offset"):
      overlay.apply_runtime(env)

    torch.testing.assert_close(
      env.scene["robot"].data.default_joint_pos,
      torch.zeros((2, 3)),
    )


if __name__ == "__main__":
  unittest.main()
