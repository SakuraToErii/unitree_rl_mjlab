import math
import os
from pathlib import Path
from typing import cast

import torch
import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.utils.os import dump_yaml
from rsl_rl.env.vec_env import VecEnv
from torch import nn

from src.tasks.ghost.mdp import MotionCommand


def build_beyond_mimic_config(
  *,
  onnx_filename: str,
  body_names: list[str],
  anchor_body: str,
  num_obs: int,
  joint_names: list[str],
  joint_stiffness: list[float],
  joint_damping: list[float],
  num_actions: int,
  physical_dt: float,
  decimation: int,
) -> dict[str, object]:
  """Build the uncommented deployment config consumed by BeyondMimic."""
  motor_nums = len(joint_names)
  if anchor_body not in body_names:
    raise ValueError(f"Anchor body {anchor_body!r} is missing from body_names")
  if num_obs <= 0 or num_actions <= 0:
    raise ValueError("num_obs and num_actions must be positive")
  if motor_nums != num_actions:
    raise ValueError(
      f"Expected one policy action per motor, got {num_actions} actions and "
      f"{motor_nums} motors"
    )
  if len(joint_stiffness) != motor_nums or len(joint_damping) != motor_nums:
    raise ValueError("Kp/Kd lengths must match the natural joint order")
  if physical_dt <= 0.0 or decimation <= 0:
    raise ValueError("physical_dt and decimation must be positive")

  return {
    "onnx_path": Path(onnx_filename).name,
    "warm_start_time": 0.0,
    "body_names": body_names,
    "anchor_body": anchor_body,
    "num_obs": num_obs,
    "locked_joint_map": list(range(num_actions)),
    "kps": joint_stiffness,
    "kds": joint_damping,
    "num_actions": num_actions,
    "motor_nums": motor_nums,
    "hold_final_reference": False,
    "physical_dt": physical_dt,
    "decimation": decimation,
  }


class _OnnxMotionModel(nn.Module):
  """ONNX-exportable model that wraps the policy and bundles motion reference data."""

  def __init__(self, actor, motion):
    super().__init__()
    self.policy = actor.as_onnx(verbose=False)
    self.register_buffer("joint_pos", motion.joint_pos.to("cpu"))
    self.register_buffer("joint_vel", motion.joint_vel.to("cpu"))
    self.register_buffer("body_pos_w", motion.body_pos_w.to("cpu"))
    self.register_buffer("body_quat_w", motion.body_quat_w.to("cpu"))
    self.register_buffer("body_lin_vel_w", motion.body_lin_vel_w.to("cpu"))
    self.register_buffer("body_ang_vel_w", motion.body_ang_vel_w.to("cpu"))
    self.time_step_total: int = self.joint_pos.shape[0]  # type: ignore[index]

  def forward(self, x, time_step):
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=self.time_step_total - 1
    )
    return (
      self.policy(x),
      self.joint_pos[time_step_clamped],  # type: ignore[index]
      self.joint_vel[time_step_clamped],  # type: ignore[index]
      self.body_pos_w[time_step_clamped],  # type: ignore[index]
      self.body_quat_w[time_step_clamped],  # type: ignore[index]
      self.body_lin_vel_w[time_step_clamped],  # type: ignore[index]
      self.body_ang_vel_w[time_step_clamped],  # type: ignore[index]
    )


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  def export_motion_policy_to_onnx(
    self, path: str, filename: str = "policy.onnx", verbose: bool = False
  ) -> None:
    os.makedirs(path, exist_ok=True)
    cmd = cast(MotionCommand, self.env.unwrapped.command_manager.get_term("motion"))
    model = _OnnxMotionModel(self.alg.get_policy(), cmd.motion)
    model.to("cpu")
    model.eval()
    obs = torch.zeros(1, model.policy.input_size)
    time_step = torch.zeros(1, 1)
    torch.onnx.export(
      model,
      (obs, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
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
      dynamic_axes={},
      dynamo=False,
    )

  def _get_nominal_metadata(self, run_name: str) -> dict[str, list | str | float]:
    """Return deployment metadata from the unrandomized host model."""
    env = self.env.unwrapped
    metadata = get_base_metadata(env, run_name)
    motion_term = cast(MotionCommand, env.command_manager.get_term("motion"))
    metadata.update(
      {
        "anchor_body_name": motion_term.cfg.anchor_body_name,
        "body_names": list(motion_term.cfg.body_names),
      }
    )

    robot = env.scene["robot"]
    joint_name_to_ctrl_id = {
      actuator.target.split("/")[-1]: actuator.id
      for actuator in robot.spec.actuators
    }
    ctrl_ids = [joint_name_to_ctrl_id[name] for name in robot.joint_names]
    host_model = env.sim.mj_model
    metadata["joint_effort_limit"] = [
      max(abs(float(value)) for value in host_model.actuator_forcerange[ctrl_id])
      for ctrl_id in ctrl_ids
    ]
    sim_dt = float(env.cfg.sim.mujoco.timestep)
    metadata.update(
      {
        "control_dt": sim_dt * env.cfg.decimation,
        "sim_dt": sim_dt,
        "decimation": str(env.cfg.decimation),
      }
    )
    return metadata

  def export_motion_policy_bundle(
    self,
    path: str,
    filename: str = "policy.onnx",
    *,
    run_name: str = "local",
  ) -> Path:
    """Export a motion-aware ONNX policy with nominal deployment metadata."""
    self.export_motion_policy_to_onnx(path, filename)
    onnx_path = Path(path) / filename
    attach_metadata_to_onnx(
      str(onnx_path), self._get_nominal_metadata(run_name)
    )
    return onnx_path

  def export_beyond_mimic_config(
    self,
    path: str | Path,
    *,
    onnx_filename: str = "policy.onnx",
    physical_dt: float,
    decimation: int,
  ) -> Path:
    """Export a policy-consistent BeyondMimic deployment YAML."""
    metadata = self._get_nominal_metadata("local")
    policy_model = self.alg.get_policy().as_onnx(verbose=False)
    control_dt = float(metadata["control_dt"])
    if not math.isclose(
      physical_dt * decimation, control_dt, rel_tol=1.0e-6, abs_tol=1.0e-9
    ):
      raise ValueError(
        "Deployment timing must match the trained policy period: "
        f"{physical_dt} * {decimation} != {control_dt}"
      )

    config = build_beyond_mimic_config(
      onnx_filename=onnx_filename,
      body_names=cast(list[str], metadata["body_names"]),
      anchor_body=cast(str, metadata["anchor_body_name"]),
      num_obs=int(policy_model.input_size),
      joint_names=cast(list[str], metadata["joint_names"]),
      joint_stiffness=cast(list[float], metadata["joint_stiffness"]),
      joint_damping=cast(list[float], metadata["joint_damping"]),
      num_actions=len(cast(list[float], metadata["action_scale"])),
      physical_dt=physical_dt,
      decimation=decimation,
    )
    output_path = Path(path)
    dump_yaml(output_path, config)
    return output_path

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, _ = self._get_export_paths(path)
    try:
      run_name: str = (
        wandb.run.name
        if self.logger.logger_type in ("wandb", "WandbLogWriter") and wandb.run
        else "local"
      )  # type: ignore[assignment]
      onnx_path = self.export_motion_policy_bundle(
        str(policy_dir), filename, run_name=run_name
      )
      self.export_policy_to_onnx(str(policy_dir), "policy.onnx")
      if (
        self.logger.logger_type in ("wandb", "WandbLogWriter")
        and self.cfg["upload_model"]
      ):
        wandb.save(str(onnx_path), base_path=str(policy_dir))
        if self.registry_name is not None and wandb.run is not None:
          wandb.run.use_artifact(self.registry_name)
          self.registry_name = None
    except Exception as error:
      print(f"[WARN] ONNX export failed (training continues): {error}")
