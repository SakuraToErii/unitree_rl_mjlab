"""ONNX Runtime adapter for Ghost policy inference inside MJLab rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from .onnx_obs import (
  actor_observation_names,
  build_tracking_observation,
  convert_policy_actions,
  parse_csv_floats,
  parse_csv_names,
  resolve_actor_observation,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class OnnxGhostPolicy:
  """Expose a motion-aware or actor-only ONNX model as a torch policy callable."""

  def __init__(
    self,
    model_path: str | Path,
    command: Any,
    *,
    device: str,
    env: ManagerBasedRlEnv | None = None,
  ) -> None:
    try:
      import onnxruntime as ort
    except ModuleNotFoundError as error:
      raise ModuleNotFoundError(
        "ONNX inference requires `onnxruntime`; install it in the project venv."
      ) from error

    self.command = command
    self.env = env if env is not None else self._legacy_command_env(command)
    self.device = torch.device(device)
    providers = ort.get_available_providers()
    requested_providers: list[str] = []
    if self.device.type == "cuda" and "CUDAExecutionProvider" in providers:
      requested_providers.append("CUDAExecutionProvider")
    requested_providers.append("CPUExecutionProvider")
    self.session = ort.InferenceSession(
      str(Path(model_path).expanduser().resolve()),
      providers=requested_providers,
    )
    self.inputs = {
      model_input.name: model_input for model_input in self.session.get_inputs()
    }
    if "obs" not in self.inputs:
      raise ValueError(f"ONNX model inputs do not contain 'obs': {list(self.inputs)}.")
    obs_shape = self.inputs["obs"].shape
    self.expected_obs_dim = int(obs_shape[-1])
    output_names = [output.name for output in self.session.get_outputs()]
    self.action_output = "actions" if "actions" in output_names else output_names[0]
    self.time_step_offset = 0
    metadata = self.session.get_modelmeta().custom_metadata_map
    self.observation_names = parse_csv_names(
      metadata.get("observation_names"),
      field_name="observation_names",
      required=True,
    )
    self.current_observation_names = actor_observation_names(self.env)
    self._validate_action_names(metadata)
    self._env_action_scale = self._env_action_scale_tensor()
    self._onnx_action_scale = self._load_action_scale(metadata)
    self._adapt_actions = not torch.allclose(
      self._onnx_action_scale, self._env_action_scale, atol=1.0e-3
    )
    self._logged_obs_adapt = False
    if self._adapt_actions:
      print(
        "[INFO] Rescaling ONNX actions "
        f"({Path(model_path).name}) onto the env action term."
      )

  @staticmethod
  def _legacy_command_env(command: Any) -> ManagerBasedRlEnv:
    """Compatibility adapter for callers that have not passed ``env`` yet."""
    try:
      return command._env
    except AttributeError as error:
      raise TypeError(
        "OnnxGhostPolicy requires env=... when the command does not expose "
        "MJLab's legacy _env attribute."
      ) from error

  def set_time_step_offset(self, offset: int) -> None:
    """Map env-local frame 0 onto this ONNX clip's original start index."""
    if offset < 0:
      raise ValueError(f"time_step_offset must be non-negative, got {offset}.")
    self.time_step_offset = int(offset)

  def _load_action_scale(self, metadata: dict[str, str]) -> torch.Tensor:
    raw = metadata.get("action_scale")
    if not raw:
      return self._env_action_scale
    parsed = torch.as_tensor(
      parse_csv_floats(raw, field_name="action_scale"),
      dtype=torch.float32,
      device=self.device,
    )
    if parsed.numel() == 1:
      parsed = parsed.expand_as(self._env_action_scale)
    if parsed.shape != self._env_action_scale.shape:
      raise ValueError(
        f"ONNX action_scale has shape {tuple(parsed.shape)}, "
        f"environment has {tuple(self._env_action_scale.shape)}."
      )
    if not bool(torch.isfinite(parsed).all().item()) or bool(
      torch.any(parsed == 0.0).item()
    ):
      raise ValueError("ONNX action_scale must contain finite, non-zero values.")
    return parsed

  def _validate_action_names(self, metadata: dict[str, str]) -> None:
    model_names = parse_csv_names(
      metadata.get("joint_names"), field_name="joint_names", required=True
    )
    action_term = self.env.action_manager.get_term("joint_pos")
    try:
      current_names = tuple(action_term.target_names)
    except AttributeError as error:
      raise TypeError(
        "Cannot verify ONNX joint_names: the joint_pos action term has no "
        "public target_names property."
      ) from error
    if not current_names or any(not name for name in current_names):
      raise ValueError(
        "Environment joint_pos target_names must be non-empty strings."
      )
    if len(current_names) != len(set(current_names)):
      raise ValueError("Environment joint_pos target_names contain duplicates.")
    if model_names != current_names:
      raise ValueError(
        "ONNX joint_names do not match the joint_pos action layout: "
        f"model={model_names}, environment={current_names}."
      )

  def _env_action_scale_tensor(self) -> torch.Tensor:
    action_term = self.env.action_manager.get_term("joint_pos")
    try:
      scale = action_term.scale
      action_dim = int(action_term.action_dim)
    except AttributeError as error:
      raise TypeError(
        "ONNX policy requires a joint_pos action term with public scale and "
        "action_dim properties."
      ) from error
    if isinstance(scale, torch.Tensor):
      if scale.ndim == 2:
        scale = scale[0]
      elif scale.ndim > 2:
        raise ValueError(f"Environment action scale has shape {tuple(scale.shape)}.")
    else:
      scale = torch.as_tensor(scale, dtype=torch.float32)
    scale = torch.as_tensor(
      scale, dtype=torch.float32, device=self.device
    ).reshape(-1)
    if scale.numel() == 1:
      scale = scale.expand(action_dim)
    if scale.numel() != action_dim:
      raise ValueError(
        f"Environment action scale has {scale.numel()} values for "
        f"{action_dim} actions."
      )
    if not bool(torch.isfinite(scale).all().item()) or bool(
      torch.any(scale == 0.0).item()
    ):
      raise ValueError(
        "Environment action scale must contain finite, non-zero values."
      )
    return scale

  def _motion_time_steps(self) -> torch.Tensor:
    if hasattr(self.command, "reference_index"):
      return self.command.reference_index
    if hasattr(self.command, "time_steps"):
      return self.command.time_steps
    raise TypeError(
      "ONNX time_step input requires a motion command with reference_index "
      "or time_steps."
    )

  def _last_action_for_onnx(self) -> torch.Tensor:
    last_action = self.env.action_manager.action
    if not self._adapt_actions:
      return last_action
    return convert_policy_actions(
      last_action,
      source_scale=self._env_action_scale,
      target_scale=self._onnx_action_scale,
    )

  def _actor_observation(self, observations: Any) -> torch.Tensor:
    actor_obs = observations["actor"]
    if actor_obs.ndim != 2 or actor_obs.shape[0] != 1:
      raise ValueError(
        "ONNX play requires actor observations with shape [1, D]."
      )
    adapted = resolve_actor_observation(
      actor_obs,
      expected_dim=self.expected_obs_dim,
      model_names=self.observation_names,
      current_names=self.current_observation_names,
      build_named=self._build_named_observation,
    )
    if adapted is not actor_obs and not self._logged_obs_adapt:
      print(
        f"[INFO] Adapted Ghost obs {actor_obs.shape[1]} -> ONNX "
        f"{self.expected_obs_dim} ({','.join(self.observation_names)})."
      )
      self._logged_obs_adapt = True
    return adapted

  def _build_named_observation(self, names: tuple[str, ...]) -> torch.Tensor:
    from .mdp.commands import MotionCommand as GhostMotionCommand

    if not isinstance(self.command, GhostMotionCommand):
      raise TypeError(
        "Observation adaptation is only available when a foreign ONNX policy "
        "runs against a Ghost MotionCommand."
      )
    return build_tracking_observation(
      self.env,
      self.command,
      names,
      last_action=self._last_action_for_onnx(),
    )

  def __call__(self, observations: Any) -> torch.Tensor:
    actor_obs = self._actor_observation(observations)
    feed: dict[str, np.ndarray] = {
      "obs": actor_obs.detach().cpu().numpy().astype(np.float32, copy=False)
    }
    if "time_step" in self.inputs:
      feed["time_step"] = (
        (self._motion_time_steps() + self.time_step_offset)[:, None]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
      )
    actions = torch.as_tensor(
      self.session.run([self.action_output], feed)[0],
      dtype=torch.float32,
      device=self.device,
    )
    if actions.shape != (actor_obs.shape[0], self._env_action_scale.numel()):
      raise ValueError(
        f"ONNX returned actions with shape {tuple(actions.shape)}, expected "
        f"({actor_obs.shape[0]}, {self._env_action_scale.numel()})."
      )
    if not self._adapt_actions:
      return actions
    return convert_policy_actions(
      actions,
      source_scale=self._onnx_action_scale,
      target_scale=self._env_action_scale,
    )
