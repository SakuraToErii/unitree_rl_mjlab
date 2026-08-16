"""将确定性的 Ghost rollout 录制为标准 motion NPZ。"""

from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab
import mujoco
import numpy as np
import torch
import tyro
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from src.assets.robots.tiangong3.tk3_constants_ghost import FOOT_GEOM_PATTERN
from src.assets.robots.tiangong3.tk3_selection import (
  FootCollision,
  select_tk3_robot_cfg,
)
from src.tasks.ghost.checkpoint_params import read_run_action_params
from src.tasks.ghost.mdp import MotionCommand, MotionCommandCfg
from src.tasks.ghost.rollout import PhysicsRolloutRecorder

_MOTION_ARRAY_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)
_FOOT_BODY_NAMES = ("ankle_roll_l_link", "ankle_roll_r_link")


@dataclass(frozen=True)
class RecordConfig:
  checkpoint_file: str
  motion_file: str
  output_file: str
  foot: FootCollision = "xml"
  """对地计算和 rollout 共同使用的脚部碰撞模型。"""
  initial_foot_penetration_m: float = 0.003
  """第 0 帧最深脚底的目标穿地量；正数表示位于地面以下。"""
  restore_action_params: bool = True
  """从 checkpoint 训练参数恢复动作限幅、缩放和残差限幅。"""
  device: str | None = None
  seed: int = 0


def _load_frame_zero(
  motion_path: Path,
  articulation_joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  with np.load(motion_path, allow_pickle=False) as motion:
    stored_joint_names = [str(name) for name in motion["joint_names"].tolist()]
    stored_body_names = [str(name) for name in motion["body_names"].tolist()]
    if len(stored_joint_names) != len(set(stored_joint_names)):
      raise ValueError("Motion joint_names contains duplicates.")
    if "pelvis" not in stored_body_names:
      raise ValueError("Motion body_names does not contain 'pelvis'.")

    joint_index = {name: index for index, name in enumerate(stored_joint_names)}
    missing = [name for name in articulation_joint_names if name not in joint_index]
    if missing:
      raise ValueError(f"Motion is missing robot joints: {missing}.")
    pelvis_index = stored_body_names.index("pelvis")
    root_pos = np.asarray(
      motion["body_pos_w"][0, pelvis_index], dtype=np.float64
    ).copy()
    root_quat = np.asarray(
      motion["body_quat_w"][0, pelvis_index], dtype=np.float64
    ).copy()
    joint_pos = np.asarray(
      motion["joint_pos"][0, [joint_index[name] for name in articulation_joint_names]],
      dtype=np.float64,
    )
  return root_pos, root_quat, joint_pos


def _initial_foot_distance(motion_path: Path, robot_cfg) -> float:
  """计算第 0 帧脚底到 z=0 地面的最小有符号距离。"""
  robot = Entity(robot_cfg)
  spec = robot.spec
  terrain = spec.worldbody.add_geom()
  terrain.name = "terrain"
  terrain.type = mujoco.mjtGeom.mjGEOM_PLANE
  terrain.size = (0.0, 0.0, 0.05)
  model = spec.compile()
  data = mujoco.MjData(model)

  articulation_joint_names = [
    model.joint(index).name
    for index in range(model.njnt)
    if model.joint(index).type != mujoco.mjtJoint.mjJNT_FREE
  ]
  root_pos, root_quat, joint_pos = _load_frame_zero(
    motion_path, articulation_joint_names
  )

  free_joint_ids = [
    index
    for index in range(model.njnt)
    if model.joint(index).type == mujoco.mjtJoint.mjJNT_FREE
  ]
  if len(free_joint_ids) != 1:
    raise RuntimeError(f"Expected one free joint, found {len(free_joint_ids)}.")
  free_qpos_adr = model.jnt_qposadr[free_joint_ids[0]]
  data.qpos[free_qpos_adr : free_qpos_adr + 3] = root_pos
  data.qpos[free_qpos_adr + 3 : free_qpos_adr + 7] = root_quat
  for index, name in enumerate(articulation_joint_names):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[model.jnt_qposadr[joint_id]] = joint_pos[index]
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)

  foot_pattern = re.compile(FOOT_GEOM_PATTERN)
  foot_geom_ids = [
    index
    for index in range(model.ngeom)
    if model.geom(index).name
    and foot_pattern.fullmatch(model.geom(index).name)
  ]
  if not foot_geom_ids:
    raise RuntimeError("No foot collision geoms matched FOOT_GEOM_PATTERN.")
  terrain_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "terrain"
  )
  fromto = np.zeros(6, dtype=np.float64)
  return min(
    float(
      mujoco.mj_geomDistance(
        model, data, geom_id, terrain_id, 10.0, fromto
      )
    )
    for geom_id in foot_geom_ids
  )


def _apply_saved_action_params(checkpoint_path: Path, env_cfg, agent_cfg) -> None:
  action_clip, action_scale, residual_clip = read_run_action_params(checkpoint_path)
  joint_pos_action = env_cfg.actions.get("joint_pos")
  if joint_pos_action is None or not hasattr(joint_pos_action, "scale"):
    raise TypeError("Ghost task has no configurable joint_pos action scale.")
  agent_cfg.clip_actions = action_clip
  joint_pos_action.scale = action_scale
  if residual_clip is not None:
    if not hasattr(joint_pos_action, "residual_clip"):
      raise TypeError("Ghost task does not support the saved residual action clip.")
    joint_pos_action.residual_clip = residual_clip
  print(
    "[INFO] Restored action params: "
    f"clip={action_clip:g}, scale={action_scale}, "
    f"residual_clip={residual_clip}"
  )


def _source_compatible_payload(
  payload: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
  fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
  rounded_fps = round(fps)
  if not np.isclose(fps, rounded_fps, atol=1e-9):
    raise ValueError(f"Motion schema requires integer fps, got {fps}.")
  return {
    "fps": np.asarray([rounded_fps], dtype=np.int64),
    **{
      key: np.asarray(payload[key], dtype=np.float32)
      for key in _MOTION_ARRAY_KEYS
    },
    "joint_names": np.asarray(payload["joint_names"], dtype=np.str_),
    "body_names": np.asarray(payload["body_names"], dtype=np.str_),
  }


def _save_motion(output_file: str | Path, payload: dict[str, np.ndarray]) -> Path:
  output_path = Path(output_file).expanduser().resolve()
  if output_path.suffix != ".npz":
    output_path = output_path.with_suffix(".npz")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(output_path, **payload)
  return output_path


def run_record(task_id: str, cfg: RecordConfig) -> Path:
  configure_torch_backends()
  if cfg.initial_foot_penetration_m < 0.0:
    raise ValueError("initial_foot_penetration_m must be non-negative.")

  checkpoint_path = Path(cfg.checkpoint_file).expanduser().resolve()
  motion_path = Path(cfg.motion_file).expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(checkpoint_path)
  if checkpoint_path.suffix.lower() != ".pt":
    raise ValueError("Ghost NPZ recording currently requires a .pt checkpoint.")
  if not motion_path.is_file():
    raise FileNotFoundError(motion_path)

  robot_cfg = select_tk3_robot_cfg(task_id, foot=cfg.foot)
  if robot_cfg is None:
    raise ValueError(f"Task {task_id!r} is not a TK3 task.")
  initial_distance = _initial_foot_distance(motion_path, robot_cfg)
  target_distance = -cfg.initial_foot_penetration_m
  height_shift = target_distance - initial_distance
  print(
    "[INFO] Frame-zero grounding: "
    f"before={initial_distance * 1000:+.3f} mm, "
    f"target={target_distance * 1000:+.3f} mm, "
    f"root_shift={height_shift * 1000:+.3f} mm"
  )

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  motion_cfg = env_cfg.commands.get("motion")
  if not isinstance(motion_cfg, MotionCommandCfg):
    raise TypeError(f"Task {task_id!r} is not a ghost motion-tracking task.")
  env_cfg.scene.entities["robot"] = robot_cfg
  motion_cfg.motion_file = str(motion_path)
  motion_cfg.sampling_mode = "start"
  motion_cfg.command_noise_enabled = False
  motion_cfg.actuator_command_lag_range = None
  motion_cfg.initial_root_z_offset = height_shift
  env_cfg.scene.num_envs = 1
  env_cfg.seed = cfg.seed
  env_cfg.auto_reset = False
  if cfg.restore_action_params:
    _apply_saved_action_params(checkpoint_path, env_cfg, agent_cfg)

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  try:
    # 环境构造结束时仍可能显示实体默认状态；显式 reset 才会应用出生高度修正，
    # 并在采集第一帧观测前刷新接触传感器。
    raw_env.reset()
    command = raw_env.command_manager.get_term("motion")
    if not isinstance(command, MotionCommand):
      raise TypeError("Ghost recording requires a MotionCommand instance.")
    if int(command.reference_index[0].item()) != 0:
      raise RuntimeError("Ghost recording did not initialize at frame zero.")

    penetration_sensor = raw_env.scene["ground_penetration"]
    foot_indexes = [
      penetration_sensor.primary_names.index(name) for name in _FOOT_BODY_NAMES
    ]
    actual_distance = float(
      torch.amin(penetration_sensor.data.dist[0, foot_indexes]).item()
    )
    if not np.isclose(actual_distance, target_distance, atol=5e-4):
      raise RuntimeError(
        "Frame-zero grounding verification failed: "
        f"expected {target_distance:g} m, measured {actual_distance:g} m."
      )
    print(
      f"[INFO] Verified simulated frame-zero foot distance: "
      f"{actual_distance * 1000:+.3f} mm"
    )

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
    recorder.capture()

    terminated_early = False
    with torch.inference_mode():
      while not bool(command.segment_end[0].item()):
        action = policy(obs)
        obs, _, done, _ = env.step(action)
        is_done = bool(done[0].item()) or not recorder.state_is_finite()
        recorder.capture(terminated=is_done)
        if is_done:
          terminated_early = True
          break

    if terminated_early:
      raise RuntimeError("Ghost recording terminated before the source motion ended.")

    diagnostic_payload = recorder.build_payload({})
    motion_payload = _source_compatible_payload(diagnostic_payload)
    npz_path = _save_motion(cfg.output_file, motion_payload)
    print(f"[INFO] Recorded motion: {npz_path}")
    return npz_path
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
    RecordConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {task_id}",
    config=mjlab.TYRO_FLAGS,
  )
  run_record(task_id, cfg)


if __name__ == "__main__":
  main()
