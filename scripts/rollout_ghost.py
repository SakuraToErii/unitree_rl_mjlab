"""Generate a physics-consistent NPZ from a trained TK3 ghost policy."""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from src.assets.robots.tiangong3.tk3_constants_ghost import (
  TK3_GHOST_CONFIG,
  TK3_XML,
)
from src.tasks.ghost.mdp import MotionCommand, MotionCommandCfg
from src.tasks.ghost.onnx_policy import OnnxGhostPolicy
from src.tasks.ghost.rollout import (
  PhysicsRolloutRecorder,
  save_rollout,
  sha256_file,
)


@dataclass(frozen=True)
class RolloutConfig:
  checkpoint_file: str
  motion_file: str
  output_file: str
  device: str | None = None
  seed: int = 0


def run_rollout(task_id: str, cfg: RolloutConfig) -> tuple[Path, Path]:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  checkpoint_path = Path(cfg.checkpoint_file).expanduser().resolve()
  motion_path = Path(cfg.motion_file).expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(checkpoint_path)
  if not motion_path.is_file():
    raise FileNotFoundError(motion_path)

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  motion_cfg = env_cfg.commands.get("motion")
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise TypeError(f"Task {task_id!r} is not a ghost motion-tracking task.")
  motion_cfg.motion_file = str(motion_path)
  motion_cfg.sampling_mode = "start"
  motion_cfg.command_noise_enabled = False
  motion_cfg.actuator_command_lag_range = None
  env_cfg.scene.num_envs = 1
  env_cfg.seed = cfg.seed
  env_cfg.auto_reset = False

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  try:
    command = raw_env.command_manager.get_term("motion")
    if not isinstance(command, MotionCommand):
      raise TypeError("Ghost rollout requires a MotionCommand instance.")
    if checkpoint_path.suffix.lower() == ".onnx":
      policy = OnnxGhostPolicy(
        checkpoint_path,
        command,
        device=device,
      )
    else:
      runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
      runner = runner_cls(env, asdict(agent_cfg), device=device)
      runner.load(
        str(checkpoint_path),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
      policy = runner.get_inference_policy(device=device)
    obs = env.get_observations()
    recorder = PhysicsRolloutRecorder(raw_env)

    if int(recorder.command.reference_index[0].item()) != 0:
      raise RuntimeError("Deterministic rollout did not initialize at frame zero.")
    recorder.capture()

    terminated_early = False
    with torch.inference_mode():
      while not bool(recorder.command.segment_end[0].item()):
        action = policy(obs)
        obs, _, done, _ = env.step(action)
        is_done = bool(done[0].item())
        recorder.capture(terminated=is_done)
        if is_done:
          terminated_early = True
          break

    provenance = {
      "task_id": task_id,
      "source_motion": str(motion_path),
      "source_motion_sha256": sha256_file(motion_path),
      "generator_checkpoint": str(checkpoint_path),
      "generator_checkpoint_sha256": sha256_file(checkpoint_path),
      "model_xml": str(TK3_XML),
      "model_xml_sha256": sha256_file(TK3_XML),
      "model_config": str(TK3_GHOST_CONFIG),
      "model_config_sha256": sha256_file(TK3_GHOST_CONFIG),
    }
    payload = recorder.build_payload(provenance)
    metrics = recorder.quality_report(payload)

    output_path = Path(cfg.output_file).expanduser().resolve()
    if terminated_early:
      suffix = output_path.suffix if output_path.suffix else ".npz"
      output_path = output_path.with_name(f"{output_path.stem}.failed{suffix}")
    npz_path, metrics_path = save_rollout(output_path, payload, metrics)

    print(f"[INFO] Physics rollout: {npz_path}")
    print(f"[INFO] Quality report: {metrics_path}")
    print(f"[INFO] Quality passed: {metrics['quality_passed']}")
    if terminated_early:
      raise RuntimeError(
        "Ghost rollout terminated before the source motion ended; "
        f"partial diagnostics were saved to {npz_path}."
      )
    if not metrics["quality_passed"]:
      raise RuntimeError(
        f"Rollout completed but failed quality gates; inspect {metrics_path}."
      )
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
    RolloutConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {task_id}",
    config=mjlab.TYRO_FLAGS,
  )
  run_rollout(task_id, cfg)


if __name__ == "__main__":
  main()
