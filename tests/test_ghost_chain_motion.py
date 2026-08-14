"""Stitch and overlay helpers for chained Ghost ONNX rollouts."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.tasks.ghost.chain_motion import (
  ChainSegment,
  extract_onnx_embedded_motion,
  fill_coverage_plan,
  parse_segment_spec,
  planned_frame_count,
)
from src.tasks.ghost.motion_clip import (
  align_motion_layout,
  frame_from_rollout_payload,
  overlay_start_frame,
  prepare_motion_for_loader,
  slice_motion_frames,
)


def _clip(
  *, frames: int = 4, joints: int = 2, bodies: int = 2
) -> dict[str, np.ndarray]:
  return {
    "fps": np.asarray([100.0]),
    "joint_names": np.asarray(["j1", "j0"]),
    "body_names": np.asarray(["b1", "b0"]),
    "joint_pos": np.arange(frames * joints, dtype=np.float32).reshape(frames, joints),
    "joint_vel": np.arange(frames * joints, dtype=np.float32).reshape(frames, joints)
    + 10.0,
    "body_pos_w": np.arange(frames * bodies * 3, dtype=np.float32).reshape(
      frames, bodies, 3
    ),
    "body_quat_w": np.tile(
      np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32), (frames, bodies, 1)
    ),
    "body_lin_vel_w": np.zeros((frames, bodies, 3), dtype=np.float32),
    "body_ang_vel_w": np.zeros((frames, bodies, 3), dtype=np.float32),
  }


class ChainMotionTest(unittest.TestCase):
  def test_parse_segment_spec_reads_optional_motion_and_frame_range(self) -> None:
    segment = parse_segment_spec(
      "onnx=/tmp/a.onnx,start=3,end=9,motion=/tmp/ref.npz"
    )
    self.assertEqual(segment.onnx, Path("/tmp/a.onnx"))
    self.assertEqual(segment.motion, Path("/tmp/ref.npz"))
    self.assertEqual(segment.start, 3)
    self.assertEqual(segment.end, 9)

  def test_fill_coverage_plan_inserts_default_onnx_in_the_gaps(self) -> None:
    plan = fill_coverage_plan(
      total_frames=20,
      default_onnx=Path("default.onnx"),
      motion=Path("source.npz"),
      specialists=[
        ChainSegment(onnx=Path("spec.onnx"), start=5, end=8),
        ChainSegment(onnx=Path("spec.onnx"), start=12, end=15),
      ],
    )
    self.assertEqual(
      [(item.onnx.name, item.start, item.end) for item in plan],
      [
        ("default.onnx", 0, 4),
        ("spec.onnx", 5, 8),
        ("default.onnx", 9, 11),
        ("spec.onnx", 12, 15),
        ("default.onnx", 16, 19),
      ],
    )
    self.assertTrue(all(item.motion == Path("source.npz") for item in plan))
    self.assertEqual(planned_frame_count(plan), 20)

  def test_fill_coverage_plan_rejects_overlapping_specialists(self) -> None:
    with self.assertRaisesRegex(ValueError, "Overlapping"):
      fill_coverage_plan(
        total_frames=20,
        default_onnx=Path("default.onnx"),
        motion=Path("source.npz"),
        specialists=[
          ChainSegment(onnx=Path("a.onnx"), start=3, end=10),
          ChainSegment(onnx=Path("b.onnx"), start=8, end=12),
        ],
      )

  def test_fill_coverage_plan_supports_single_frame_default_ranges(self) -> None:
    cases = {
      "first": (
        [ChainSegment(onnx=Path("spec.onnx"), start=1, end=4)],
        [("default.onnx", 0, 0), ("spec.onnx", 1, 4)],
      ),
      "middle": (
        [
          ChainSegment(onnx=Path("a.onnx"), start=0, end=1),
          ChainSegment(onnx=Path("b.onnx"), start=3, end=4),
        ],
        [("a.onnx", 0, 1), ("default.onnx", 2, 2), ("b.onnx", 3, 4)],
      ),
      "tail": (
        [ChainSegment(onnx=Path("spec.onnx"), start=0, end=3)],
        [("spec.onnx", 0, 3), ("default.onnx", 4, 4)],
      ),
    }
    for label, (specialists, expected) in cases.items():
      with self.subTest(label=label):
        plan = fill_coverage_plan(
          total_frames=5,
          default_onnx=Path("default.onnx"),
          motion=Path("source.npz"),
          specialists=specialists,
        )
        self.assertEqual(
          [(item.onnx.name, item.start, item.end) for item in plan], expected
        )
        self.assertEqual(planned_frame_count(plan), 5)

  def test_slice_is_inclusive_and_keeps_static_metadata(self) -> None:
    sliced = slice_motion_frames(_clip(), 1, 2)
    np.testing.assert_array_equal(sliced["joint_pos"], _clip()["joint_pos"][1:3])
    np.testing.assert_array_equal(sliced["joint_names"], _clip()["joint_names"])
    np.testing.assert_array_equal(sliced["fps"], [100.0])

  def test_single_frame_slice_is_padded_only_for_motion_loader(self) -> None:
    sliced = slice_motion_frames(_clip(), 2, 2)
    self.assertEqual(sliced["joint_pos"].shape[0], 1)

    loader_motion, source_frame_count = prepare_motion_for_loader(sliced)

    self.assertEqual(source_frame_count, 1)
    self.assertEqual(loader_motion["joint_pos"].shape[0], 2)
    for key in (
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    ):
      np.testing.assert_array_equal(loader_motion[key][0], loader_motion[key][1])
    np.testing.assert_array_equal(loader_motion["joint_names"], sliced["joint_names"])

  def test_overlay_replaces_only_the_first_frame(self) -> None:
    motion = slice_motion_frames(_clip(), 1, 3)
    start = {
      "joint_pos": np.array([9.0, 8.0], dtype=np.float32),
      "joint_vel": np.array([1.0, 2.0], dtype=np.float32),
      "body_pos_w": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
      "body_quat_w": np.tile(
        np.array((0.0, 1.0, 0.0, 0.0), dtype=np.float32), (2, 1)
      ),
      "body_lin_vel_w": np.ones((2, 3), dtype=np.float32),
      "body_ang_vel_w": np.full((2, 3), 0.5, dtype=np.float32),
    }
    overlaid = overlay_start_frame(motion, start)
    np.testing.assert_array_equal(overlaid["joint_pos"][0], [9.0, 8.0])
    np.testing.assert_array_equal(overlaid["joint_pos"][1:], motion["joint_pos"][1:])
    np.testing.assert_array_equal(overlaid["body_pos_w"][0], start["body_pos_w"])
    np.testing.assert_array_equal(
      overlaid["body_pos_w"][1:], motion["body_pos_w"][1:]
    )

  def test_align_and_pose_projection_follow_target_names(self) -> None:
    aligned = align_motion_layout(
      _clip(), joint_names=("j0", "j1"), body_names=("b0", "b1")
    )
    np.testing.assert_array_equal(aligned["joint_pos"][0], [1.0, 0.0])
    np.testing.assert_array_equal(
      aligned["body_pos_w"][0],
      np.array([[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]], dtype=np.float32),
    )

    payload = {
      "joint_names": np.asarray(["j0", "j1", "j2"]),
      "body_names": np.asarray(["root", "b0", "b1"]),
      "joint_pos": np.array([[10.0, 20.0, 30.0]], dtype=np.float32),
      "joint_vel": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
      "body_pos_w": np.arange(9, dtype=np.float32).reshape(1, 3, 3),
      "body_quat_w": np.tile(
        np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float32), (1, 3, 1)
      ),
      "body_lin_vel_w": np.zeros((1, 3, 3), dtype=np.float32),
      "body_ang_vel_w": np.zeros((1, 3, 3), dtype=np.float32),
    }
    pose = frame_from_rollout_payload(
      payload, 0, joint_names=("j1", "j0"), body_names=("b1", "b0")
    )
    np.testing.assert_array_equal(pose["joint_pos"], [20.0, 10.0])
    np.testing.assert_array_equal(
      pose["body_pos_w"],
      np.array([[6.0, 7.0, 8.0], [3.0, 4.0, 5.0]], dtype=np.float32),
    )

  def test_extract_onnx_embedded_motion_reads_buffers_and_metadata(self) -> None:
    class DummyMotionOnnx(torch.nn.Module):
      def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
          "joint_pos",
          torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=torch.float32),
        )
        self.register_buffer(
          "joint_vel",
          torch.tensor([[10.0, 11.0], [12.0, 13.0], [14.0, 15.0]], dtype=torch.float32),
        )
        self.register_buffer(
          "body_pos_w", torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
        )
        quat = torch.zeros(3, 2, 4)
        quat[..., 0] = 1.0
        self.register_buffer("body_quat_w", quat)
        self.register_buffer(
          "body_lin_vel_w",
          torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) + 100.0,
        )
        self.register_buffer(
          "body_ang_vel_w",
          torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) + 200.0,
        )

      def forward(self, obs, time_step):
        index = torch.clamp(time_step.long().squeeze(-1), max=2)
        return (
          torch.zeros(obs.shape[0], 2),
          self.joint_pos[index],
          self.joint_vel[index],
          self.body_pos_w[index],
          self.body_quat_w[index],
          self.body_lin_vel_w[index],
          self.body_ang_vel_w[index],
        )

    with tempfile.TemporaryDirectory() as tmp:
      onnx_path = Path(tmp) / "clip.onnx"
      model = DummyMotionOnnx().eval()
      torch.onnx.export(
        model,
        (torch.zeros(1, 2), torch.zeros(1, 1)),
        str(onnx_path),
        input_names=["obs", "time_step"],
        output_names=[
          "actions",
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
        ],
        opset_version=18,
        dynamo=False,
      )
      import onnx
      from mjlab.rl.exporter_utils import attach_metadata_to_onnx

      attach_metadata_to_onnx(
        str(onnx_path),
        {
          "joint_names": ["j0", "j1"],
          "body_names": ["pelvis", "torso"],
          "control_dt": 0.01,
        },
      )
      loaded = onnx.load(str(onnx_path), load_external_data=False)
      self.assertTrue(any(prop.key == "joint_names" for prop in loaded.metadata_props))

      motion = extract_onnx_embedded_motion(onnx_path)
      np.testing.assert_array_equal(motion["joint_pos"][1], [2.0, 3.0])
      np.testing.assert_array_equal(motion["joint_names"], ["j0", "j1"])
      np.testing.assert_array_equal(motion["body_names"], ["pelvis", "torso"])
      np.testing.assert_allclose(motion["fps"], [100.0])

  def test_tracking_obs_concat_matches_foreign_onnx_layout(self) -> None:
    from src.tasks.ghost.onnx_obs import (
      assemble_named_observation,
      convert_policy_actions,
      resolve_actor_observation,
    )

    parts = {
      "command": torch.arange(4, dtype=torch.float32).view(1, 4),
      "motion_anchor_ori_b": torch.full((1, 6), 0.5),
      "base_ang_vel": torch.tensor([[1.0, 2.0, 3.0]]),
      "joint_pos": torch.tensor([[0.1, 0.2]]),
      "joint_vel": torch.tensor([[0.3, 0.4]]),
      "actions": torch.tensor([[7.0, 8.0]]),
    }
    obs = assemble_named_observation(
      (
        "command",
        "motion_anchor_ori_b",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
      ),
      parts,
    )
    self.assertEqual(tuple(obs.shape), (1, 19))
    torch.testing.assert_close(obs[0, :4], torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(obs[0, -2:], torch.tensor([7.0, 8.0]))

    converted = convert_policy_actions(
      torch.tensor([[1.0, 2.0]]),
      source_scale=torch.tensor([0.05, 0.10]),
      target_scale=torch.tensor([0.25, 0.25]),
    )
    torch.testing.assert_close(converted, torch.tensor([[0.2, 0.8]]))

    actor_obs = torch.tensor([[1.0, 2.0, 3.0]])
    direct = resolve_actor_observation(
      actor_obs,
      expected_dim=3,
      model_names=("joint_pos", "actions"),
      current_names=("joint_pos", "actions"),
      build_named=lambda names: self.fail(f"unexpected rebuild for {names}"),
    )
    self.assertIs(direct, actor_obs)

  def test_same_sized_different_observation_layout_is_rebuilt(self) -> None:
    from src.tasks.ghost.onnx_obs import resolve_actor_observation

    actor_obs = torch.tensor([[1.0, 2.0, 3.0]])
    rebuilt = torch.tensor([[4.0, 5.0, 6.0]])
    requested_names: list[tuple[str, ...]] = []

    resolved = resolve_actor_observation(
      actor_obs,
      expected_dim=3,
      model_names=("command", "actions"),
      current_names=("joint_pos", "actions"),
      build_named=lambda names: requested_names.append(names) or rebuilt,
    )

    self.assertIs(resolved, rebuilt)
    self.assertEqual(requested_names, [("command", "actions")])

  def test_observation_without_layout_metadata_fails_even_when_size_matches(
    self,
  ) -> None:
    from src.tasks.ghost.onnx_obs import resolve_actor_observation

    with self.assertRaisesRegex(ValueError, "observation_names"):
      resolve_actor_observation(
        torch.zeros(1, 3),
        expected_dim=3,
        model_names=(),
        current_names=("joint_pos", "actions"),
        build_named=lambda names: torch.zeros(1, 3),
      )

  def test_csv_names_reject_empty_and_duplicate_entries(self) -> None:
    from src.tasks.ghost.onnx_obs import parse_csv_names

    for raw, message in (("j0,,j1", "empty"), ("j0,j0", "duplicate")):
      with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, message):
        parse_csv_names(raw, field_name="joint_names", required=True)

  def test_csv_floats_reject_empty_and_nonnumeric_entries(self) -> None:
    from src.tasks.ghost.onnx_obs import parse_csv_floats

    for raw, message in (("0.1,,0.2", "empty"), ("0.1,nope", "numbers")):
      with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, message):
        parse_csv_floats(raw, field_name="action_scale")

  def test_failed_quality_is_saved_only_as_failed_diagnostics(self) -> None:
    from src.tasks.ghost.rollout import (
      RolloutQualityError,
      save_rollout_checked,
    )

    with tempfile.TemporaryDirectory() as tmp:
      requested = Path(tmp) / "result.npz"
      with self.assertRaises(RolloutQualityError) as raised:
        save_rollout_checked(
          requested,
          {"joint_pos": np.zeros((1, 2), dtype=np.float32)},
          {"quality_passed": False, "frame_count": 1},
        )

      self.assertFalse(requested.exists())
      self.assertEqual(raised.exception.npz_path.name, "result.failed.npz")
      self.assertTrue(raised.exception.npz_path.is_file())
      saved_metrics = json.loads(
        raised.exception.metrics_path.read_text(encoding="utf-8")
      )
      self.assertEqual(saved_metrics["artifact_status"], "failed")
      self.assertEqual(saved_metrics["failure_reason"], "quality gates failed")

      accepted = Path(tmp) / "accepted.npz"
      npz_path, metrics_path = save_rollout_checked(
        accepted,
        {"joint_pos": np.zeros((1, 2), dtype=np.float32)},
        {"quality_passed": True, "frame_count": 1},
      )
      self.assertEqual(npz_path, accepted)
      self.assertTrue(npz_path.is_file())
      accepted_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
      self.assertEqual(accepted_metrics["artifact_status"], "accepted")
      self.assertNotIn("failure_reason", accepted_metrics)

  def test_single_frame_segment_captures_exactly_one_source_frame(self) -> None:
    from scripts.chain_ghost_onnx import _capture_until_segment_end

    command = SimpleNamespace(
      motion=SimpleNamespace(time_step_total=2),
      reference_index=torch.zeros(1, dtype=torch.long),
    )

    class FakeRecorder:
      def __init__(self) -> None:
        self.command = command
        self.frame_count = 0

      def capture(self, *, terminated: bool = False) -> None:
        del terminated
        self.frame_count += 1

      def state_is_finite(self) -> bool:
        return True

    class FakeEnv:
      def __init__(self) -> None:
        self.step_count = 0

      def get_observations(self):
        return {"actor": torch.zeros(1, 1)}

      def step(self, action):
        del action
        self.step_count += 1
        command.reference_index += 1
        return self.get_observations(), None, torch.tensor([False]), None

    env = FakeEnv()
    recorder = FakeRecorder()
    segment_ids: list[int] = []
    failure = _capture_until_segment_end(
      env,
      lambda obs: torch.zeros(1, 2),
      recorder,
      segment_ids,
      segment_index=3,
      source_frame_count=1,
    )

    self.assertIsNone(failure)
    self.assertEqual(env.step_count, 0)
    self.assertEqual(recorder.frame_count, 1)
    self.assertEqual(segment_ids, [3])

  def test_chain_model_provenance_covers_shared_robot_spec(self) -> None:
    from scripts.chain_ghost_onnx import _model_provenance
    from src.assets.robots.tiangong3.tk3_spec import TK3_SPEC_CONFIG
    from src.tasks.ghost.rollout import sha256_file

    provenance = _model_provenance()

    self.assertEqual(provenance["model_spec_config"], str(TK3_SPEC_CONFIG))
    self.assertEqual(
      provenance["model_spec_config_sha256"], sha256_file(TK3_SPEC_CONFIG)
    )

  def test_onnx_policy_accepts_verified_layout_and_public_scalar_scale(self) -> None:
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx

    from src.tasks.ghost.onnx_policy import OnnxGhostPolicy

    class Actor(torch.nn.Module):
      def forward(self, obs):
        return obs[:, :2]

    with tempfile.TemporaryDirectory() as tmp:
      model_path = Path(tmp) / "actor.onnx"
      torch.onnx.export(
        Actor().eval(),
        (torch.zeros(1, 3),),
        str(model_path),
        input_names=["obs"],
        output_names=["actions"],
        opset_version=18,
        dynamo=False,
      )
      attach_metadata_to_onnx(
        str(model_path),
        {
          "observation_names": ["joint_pos", "actions"],
          "action_scale": 0.25,
          "joint_names": ["j0", "j1"],
        },
      )

      action_term = SimpleNamespace(
        scale=0.25, action_dim=2, target_names=["j0", "j1"]
      )
      action_manager = SimpleNamespace(
        action=torch.zeros(1, 2),
        get_term=lambda name: action_term,
      )
      observation_manager = SimpleNamespace(
        active_terms={"actor": ["joint_pos", "actions"]},
        group_obs_concatenate={"actor": True},
      )
      env = SimpleNamespace(
        action_manager=action_manager,
        observation_manager=observation_manager,
      )
      policy = OnnxGhostPolicy(
        model_path,
        SimpleNamespace(reference_index=torch.zeros(1, dtype=torch.long)),
        device="cpu",
        env=env,
      )

      actions = policy({"actor": torch.tensor([[1.0, 2.0, 3.0]])})
      torch.testing.assert_close(actions, torch.tensor([[1.0, 2.0]]))

  def test_onnx_policy_requires_matching_joint_names_metadata(self) -> None:
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx

    from src.tasks.ghost.onnx_policy import OnnxGhostPolicy

    class Actor(torch.nn.Module):
      def forward(self, obs):
        return obs[:, :2]

    action_term = SimpleNamespace(
      scale=0.25, action_dim=2, target_names=["j0", "j1"]
    )
    env = SimpleNamespace(
      action_manager=SimpleNamespace(
        action=torch.zeros(1, 2), get_term=lambda name: action_term
      ),
      observation_manager=SimpleNamespace(
        active_terms={"actor": ["joint_pos", "actions"]},
        group_obs_concatenate={"actor": True},
      ),
    )
    command = SimpleNamespace(reference_index=torch.zeros(1, dtype=torch.long))

    with tempfile.TemporaryDirectory() as tmp:
      for label, joint_names, message in (
        ("missing", None, "joint_names.*required"),
        ("wrong-order", "j1,j0", "do not match"),
      ):
        with self.subTest(label=label):
          model_path = Path(tmp) / f"{label}.onnx"
          torch.onnx.export(
            Actor().eval(),
            (torch.zeros(1, 3),),
            str(model_path),
            input_names=["obs"],
            output_names=["actions"],
            opset_version=18,
            dynamo=False,
          )
          metadata: dict[str, list[str] | str | float] = {
            "observation_names": ["joint_pos", "actions"],
            "action_scale": 0.25,
          }
          if joint_names is not None:
            metadata["joint_names"] = joint_names
          attach_metadata_to_onnx(str(model_path), metadata)

          with self.assertRaisesRegex(ValueError, message):
            OnnxGhostPolicy(
              model_path, command, device="cpu", env=env
            )

  def test_normalized_effort_uses_live_batched_force_ranges(self) -> None:
    from src.tasks.ghost.mdp.commands import MotionCommand

    force_range = torch.tensor(
      [
        [[-2.0, 2.0], [-3.0, 3.0], [-5.0, 5.0]],
        [[-4.0, 4.0], [-6.0, 6.0], [-10.0, 10.0]],
      ]
    )
    command = MotionCommand.__new__(MotionCommand)
    command._env = SimpleNamespace(
      sim=SimpleNamespace(model=SimpleNamespace(actuator_forcerange=force_range))
    )
    command.robot = SimpleNamespace(
      indexing=SimpleNamespace(ctrl_ids=torch.tensor([2, 0])),
      data=SimpleNamespace(
        actuator_force=torch.tensor([[2.5, 1.0], [5.0, 2.0]])
      ),
    )

    torch.testing.assert_close(
      command.normalized_actuator_force,
      torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
    )

    force_range[:, 2] = torch.tensor([[-2.5, 2.5], [-5.0, 5.0]])
    force_range[:, 0] = 0.0
    command.robot.data.actuator_force[:, 1] = 0.0
    updated = command.normalized_actuator_force
    torch.testing.assert_close(updated[:, 0], torch.ones(2))
    torch.testing.assert_close(updated[:, 1], torch.zeros(2))
    self.assertTrue(bool(torch.isfinite(updated).all().item()))


if __name__ == "__main__":
  unittest.main()
