import torch
import wandb
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import BaseAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
)
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.utils.lab_api.string import resolve_matching_names_values


def _first_env_value(value: torch.Tensor | float) -> list | float:
  """Convert an action parameter to the metadata representation."""
  if isinstance(value, torch.Tensor):
    value = value[0] if value.ndim > 0 else value
    return value.detach().cpu().tolist()
  return value


def _action_offset_metadata(action: BaseAction) -> list | float:
  """Return one offset value per controlled action dimension."""
  offset = _first_env_value(action.offset)
  if isinstance(offset, (int, float)):
    return [float(offset)] * action.action_dim
  return offset


def _action_clip_metadata(action: BaseAction) -> list[list[float]]:
  """Return resolved processed-action clipping bounds in action order."""
  clip = getattr(action, "_clip", None)
  if isinstance(clip, torch.Tensor):
    if (
      clip.ndim != 3
      or clip.shape[0] == 0
      or clip.shape[1] != action.action_dim
      or clip.shape[2] != 2
    ):
      raise ValueError(
        "Effort action clip has unexpected shape "
        f"{tuple(clip.shape)} for {action.action_dim} actions."
      )
    return clip[0].detach().cpu().tolist()

  cfg_clip = action.cfg.clip
  if cfg_clip is None:
    raise ValueError("Effort action metadata requires configured action clipping.")
  _, _, clip_values = resolve_matching_names_values(
    cfg_clip, action.target_names
  )
  return [list(bounds) for bounds in clip_values]


def get_effort_metadata(
  env: ManagerBasedRlEnv, run_path: str
) -> dict[str, list | str | float]:
  """Build ONNX metadata for an affine joint-effort policy."""
  robot: Entity = env.scene["robot"]
  action = env.action_manager.get_term("joint_effort")
  if not isinstance(action, BaseAction):
    raise TypeError(
      "Effort policy export requires a BaseAction-compatible joint_effort term, "
      f"received {type(action).__name__}."
    )

  # Build the same joint-name-to-control-ID mapping used by the tracking exporter.
  joint_name_to_ctrl_id = {
    actuator.target.split("/")[-1]: actuator.id
    for actuator in robot.spec.actuators
  }
  ctrl_ids_natural = [
    joint_name_to_ctrl_id[joint_name]
    for joint_name in robot.joint_names
    if joint_name in joint_name_to_ctrl_id
  ]
  host_model = env.sim.mj_model

  observation_term_scale: list = []
  observation_term_flatten_history_dim: list = []
  observation_term_history_length: list = []
  observation_term_clip: list = []
  observation_names = env.observation_manager.active_terms["actor"]
  for active_term in observation_names:
    cfg = env.observation_manager.get_term_cfg("actor", active_term)

    if cfg.scale is None:
      observation_term_scale.append(1.0)
    else:
      raw_scale = cfg.scale
      scale = (
        raw_scale.cpu().tolist() if isinstance(raw_scale, torch.Tensor) else raw_scale
      )
      observation_term_scale.append(scale)

    raw_clip = cfg.clip
    if raw_clip is None:
      observation_term_clip.append([float("-inf"), float("inf")])
    else:
      observation_term_clip.append(list(raw_clip))

    observation_term_flatten_history_dim.append(cfg.flatten_history_dim)
    observation_term_history_length.append(cfg.history_length)

  return {
    "run_path": run_path,
    "joint_names": list(robot.joint_names),
    "joint_stiffness": [0.0] * len(ctrl_ids_natural),
    "joint_damping": [0.0] * len(ctrl_ids_natural),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "command_names": list(env.command_manager.active_terms),
    "observation_names": observation_names,
    "observation_terms_scale": observation_term_scale,
    "observation_terms_flatten_history_dim": observation_term_flatten_history_dim,
    "observation_terms_history_length": observation_term_history_length,
    "observation_terms_clip": observation_term_clip,
    "action_term": "joint_effort",
    "action_names": action.target_names,
    "action_scale": _first_env_value(action.scale),
    "action_offset": _action_offset_metadata(action),
    "action_clip": _action_clip_metadata(action),
    "joint_effort_limit": [
      max(abs(float(value)) for value in host_model.actuator_forcerange[ctrl_id])
      for ctrl_id in ctrl_ids_natural
    ],
  }


class EffortOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_dir, filename, onnx_path = self._get_export_paths(path)
    try:
      self.export_policy_to_onnx(str(policy_dir), filename)
      run_name: str = (
        wandb.run.name
        if self.logger.logger_type in ("wandb", "WandbLogWriter") and wandb.run
        else "local"
      )  # type: ignore[assignment]
      metadata = get_effort_metadata(self.env.unwrapped, run_name)
      attach_metadata_to_onnx(str(onnx_path), metadata)
      if (
        self.logger.logger_type in ("wandb", "WandbLogWriter")
        and self.cfg["upload_model"]
      ):
        wandb.save(str(onnx_path), base_path=str(policy_dir))
    except Exception as error:
      print(f"[WARN] ONNX export failed (training continues): {error}")
