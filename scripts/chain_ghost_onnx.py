"""Chain multiple Ghost ONNX policies into one physics NPZ.

每个 ``--segment`` 指定一个 ONNX 及其在参考 motion 上的闭区间帧。
``--motion-file`` 是整条参考 NPZ；未写 ``motion=`` 的 segment 都用它。
``--default-onnx`` 负责所有没被 ``--segment`` 覆盖的帧，按时间自动插进缺口。

交接时：
  1. 把前一段结束时的机器人姿态写成下一段参考 NPZ 同格式的一帧；
  2. 覆盖下一段 slice 的第 0 帧；
  3. 立刻把当前姿态记为下一段的第 0 输出帧（同 q、新 command index），
     再 step。这样输出与 source 1:1，交接处是 10 ms hold，不丢帧。
仿真不 reset，机器人从上一姿态继续。

Example::

  python scripts/chain_ghost_onnx.py TK3-Ghost-Tracking-QRef-Prototype \\
    --output-file datasets/chained.npz \\
    --motion-file datasets/1-1_paddingv2_tcp8mm.npz \\
    --default-onnx resume.onnx \\
    --segment "onnx=specialist.onnx,start=3251,end=3593" \\
    --segment "onnx=specialist.onnx,start=7318,end=7998"
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import tyro
import tyro.conf
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends

from src.assets.robots.tiangong3.tk3_constants_ghost import (
  TK3_GHOST_CONFIG,
  TK3_XML,
)
from src.assets.robots.tiangong3.tk3_spec import TK3_SPEC_CONFIG
from src.tasks.ghost.chain_motion import (
  ChainSegment,
  fill_coverage_plan,
  parse_segment_spec,
  resolve_segment_motion,
)
from src.tasks.ghost.mdp import MotionCommand, MotionCommandCfg
from src.tasks.ghost.motion_clip import (
  align_motion_layout,
  frame_from_rollout_payload,
  load_motion_npz,
  overlay_start_frame,
  prepare_motion_for_loader,
  slice_motion_frames,
  write_motion_npz,
)
from src.tasks.ghost.onnx_policy import OnnxGhostPolicy
from src.tasks.ghost.rollout import (
  PhysicsRolloutRecorder,
  RolloutQualityError,
  save_rollout_checked,
  sha256_file,
)


@dataclass(frozen=True)
class ChainConfig:
  output_file: str
  segment: Annotated[list[str], tyro.conf.UseAppendAction]
  motion_file: str | None = None
  default_onnx: str | None = None
  device: str | None = None
  seed: int = 0


def _model_provenance() -> dict[str, str]:
  """Fingerprint every source file that defines the chained robot plant."""
  return {
    "model_xml": str(TK3_XML),
    "model_xml_sha256": sha256_file(TK3_XML),
    "model_config": str(TK3_GHOST_CONFIG),
    "model_config_sha256": sha256_file(TK3_GHOST_CONFIG),
    "model_spec_config": str(TK3_SPEC_CONFIG),
    "model_spec_config_sha256": sha256_file(TK3_SPEC_CONFIG),
  }


def _prepare_play_env_cfg(task_id: str, motion_file: Path, cfg: ChainConfig):
  env_cfg = load_env_cfg(task_id, play=True)
  motion_cfg = env_cfg.commands.get("motion")
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise TypeError(f"Task {task_id!r} is not a ghost motion-tracking task.")
  motion_cfg.motion_file = str(motion_file)
  motion_cfg.sampling_mode = "start"
  motion_cfg.command_noise_enabled = False
  motion_cfg.actuator_command_lag_range = None
  env_cfg.scene.num_envs = 1
  env_cfg.seed = cfg.seed
  env_cfg.auto_reset = False
  # TK3 play configs intentionally have no termination terms. Chaining uses
  # explicit finite-state checks while stepping and the final quality gate.
  return env_cfg


def _load_onnx_policy(
  onnx_path: Path,
  command: MotionCommand,
  env: ManagerBasedRlEnv,
  device: str,
  time_step_offset: int,
) -> OnnxGhostPolicy:
  policy = OnnxGhostPolicy(onnx_path, command, device=device, env=env)
  policy.set_time_step_offset(time_step_offset)
  return policy


def _capture_until_segment_end(
  env: RslRlVecEnvWrapper,
  policy: OnnxGhostPolicy,
  recorder: PhysicsRolloutRecorder,
  segment_ids: list[int],
  segment_index: int,
  source_frame_count: int,
) -> str | None:
  """Record every command frame in the current clip, including the handover pose.

  After a splice the robot has not stepped yet: capturing now writes the source
  frame that the previous segment ended on, under the new command index. Skipping
  that sample drops one frame per handover and breaks 1:1 alignment with source.
  """
  if source_frame_count < 1:
    raise ValueError("A chain segment must cover at least one source frame.")
  if source_frame_count > recorder.command.motion.time_step_total:
    raise ValueError(
      f"Cannot capture {source_frame_count} source frames from a "
      f"{recorder.command.motion.time_step_total}-frame loader clip."
    )
  recorder.capture()
  segment_ids.append(segment_index)
  with torch.inference_mode():
    obs = env.get_observations()
    for local_frame in range(1, source_frame_count):
      action = policy(obs)
      if not bool(torch.isfinite(action).all().item()):
        return (
          f"segment {segment_index} produced a non-finite action before "
          f"local frame {local_frame}"
        )
      obs, _, done, _ = env.step(action)
      is_done = bool(done[0].item())
      finite_state = recorder.state_is_finite()
      recorder.capture(terminated=is_done or not finite_state)
      segment_ids.append(segment_index)
      actual_frame = int(recorder.command.reference_index[0].item())
      if actual_frame != local_frame:
        return (
          f"segment {segment_index} reference index jumped from expected "
          f"{local_frame} to {actual_frame}"
        )
      if not finite_state:
        return (
          f"segment {segment_index} reached a non-finite simulator state at "
          f"local frame {local_frame}"
        )
      if is_done:
        return (
          f"segment {segment_index} received an unexpected done signal at "
          f"local frame {actual_frame}"
        )
  return None


def _resolve_segments(cfg: ChainConfig) -> list[ChainSegment]:
  specialists = [parse_segment_spec(spec) for spec in cfg.segment]
  shared_motion = (
    Path(cfg.motion_file).expanduser() if cfg.motion_file is not None else None
  )
  if shared_motion is not None and not shared_motion.is_file():
    raise FileNotFoundError(shared_motion)

  if cfg.default_onnx is not None:
    if shared_motion is None:
      raise ValueError("--default-onnx requires --motion-file.")
    default_onnx = Path(cfg.default_onnx).expanduser()
    if not default_onnx.is_file():
      raise FileNotFoundError(default_onnx)
    total_frames = int(load_motion_npz(shared_motion)["joint_pos"].shape[0])
    segments = fill_coverage_plan(
      total_frames=total_frames,
      default_onnx=default_onnx,
      motion=shared_motion,
      specialists=specialists,
    )
  else:
    if not specialists:
      raise ValueError("Provide at least one --segment.")
    segments = [
      ChainSegment(
        onnx=spec.onnx,
        start=spec.start,
        end=spec.end,
        motion=spec.motion or shared_motion,
      )
      for spec in specialists
    ]

  for segment in segments:
    if not segment.onnx.is_file():
      raise FileNotFoundError(segment.onnx)
  return segments


def run_chain(task_id: str, cfg: ChainConfig) -> tuple[Path, Path]:
  segments = _resolve_segments(cfg)
  segment_motions = [
    slice_motion_frames(
      resolve_segment_motion(segment), segment.start, segment.end
    )
    for segment in segments
  ]
  expected_frames = sum(
    int(motion["joint_pos"].shape[0]) for motion in segment_motions
  )
  for index, segment in enumerate(segments):
    print(
      f"[INFO] Plan {index}: {segment.onnx.name} "
      f"[{segment.start}, {segment.end if segment.end is not None else 'end'}]"
    )

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  agent_cfg = load_rl_cfg(task_id)

  first_motion = segment_motions[0]
  first_loader_motion, _ = prepare_motion_for_loader(first_motion)
  with tempfile.TemporaryDirectory() as tmp:
    first_npz = write_motion_npz(
      Path(tmp) / "segment0.npz", first_loader_motion
    )
    env_cfg = _prepare_play_env_cfg(task_id, first_npz, cfg)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    try:
      command = raw_env.command_manager.get_term("motion")
      if not isinstance(command, MotionCommand):
        raise TypeError("Ghost chain rollout requires a MotionCommand instance.")
      joint_names = tuple(command.robot.joint_names)
      body_names = tuple(command.cfg.body_names)
      recorder = PhysicsRolloutRecorder(raw_env)
      segment_ids: list[int] = []
      failure_reason: str | None = None

      for index, (segment, sliced) in enumerate(
        zip(segments, segment_motions, strict=True)
      ):
        aligned = align_motion_layout(
          sliced, joint_names=joint_names, body_names=body_names
        )
        if index > 0:
          if recorder.frame_count == 0:
            raise RuntimeError(
              "Cannot handover before the first segment records a frame."
            )
          partial = recorder.build_payload({})
          start_pose = frame_from_rollout_payload(
            partial,
            recorder.frame_count - 1,
            joint_names=joint_names,
            body_names=body_names,
          )
          aligned = overlay_start_frame(aligned, start_pose)
        loader_motion, source_frame_count = prepare_motion_for_loader(aligned)
        command.replace_motion(loader_motion, frame=0)

        policy = _load_onnx_policy(
          segment.onnx,
          command,
          raw_env,
          device,
          time_step_offset=segment.start,
        )
        print(
          f"[INFO] Segment {index}: onnx={segment.onnx.name} "
          f"frames={segment.start}:{segment.end if segment.end is not None else 'end'} "
          f"source_len={source_frame_count} "
          f"loader_len={command.motion.time_step_total}"
        )
        failure_reason = _capture_until_segment_end(
          env,
          policy,
          recorder,
          segment_ids,
          index,
          source_frame_count,
        )
        if failure_reason is not None:
          print(f"[WARN] {failure_reason}.")
          break

      provenance = {
        "task_id": task_id,
        "generator_checkpoint": ",".join(str(segment.onnx) for segment in segments),
        **_model_provenance(),
      }
      payload = recorder.build_payload(provenance)
      payload["reference_index"] = np.arange(len(segment_ids), dtype=np.int64)
      payload["segment_id"] = np.asarray(segment_ids, dtype=np.int64)
      metrics = recorder.quality_report(
        payload, expected_frame_count=expected_frames
      )
      metrics["chained_segment_count"] = len(segments)

      output_path = Path(cfg.output_file).expanduser().resolve()
      try:
        npz_path, metrics_path = save_rollout_checked(
          output_path,
          payload,
          metrics,
          failure_reason=failure_reason,
        )
      except RolloutQualityError as error:
        print(f"[INFO] Failed chained rollout: {error.npz_path}")
        print(f"[INFO] Failure report: {error.metrics_path}")
        print(f"[INFO] Frames: {payload['joint_pos'].shape[0]}")
        raise
      print(f"[INFO] Chained physics rollout: {npz_path}")
      print(f"[INFO] Quality report: {metrics_path}")
      print(f"[INFO] Frames: {payload['joint_pos'].shape[0]}")
      return npz_path, metrics_path
    finally:
      env.close()


def main() -> None:
  import mjlab.tasks

  import src.tasks  # noqa: F401

  task_id, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(list_tasks()),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  cfg = tyro.cli(
    ChainConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {task_id}",
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )
  run_chain(task_id, cfg)


if __name__ == "__main__":
  main()
