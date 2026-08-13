"""ONNX Runtime adapter for Ghost policy inference inside MJLab rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
  from .mdp.commands import MotionCommand


class OnnxGhostPolicy:
  """Expose a motion-aware or actor-only ONNX model as a torch policy callable."""

  def __init__(
    self,
    model_path: str | Path,
    command: MotionCommand,
    *,
    device: str,
  ) -> None:
    try:
      import onnxruntime as ort
    except ModuleNotFoundError as error:
      raise ModuleNotFoundError(
        "ONNX inference requires `onnxruntime`; install it in the project venv."
      ) from error

    self.command = command
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
    self.inputs = {model_input.name: model_input for model_input in self.session.get_inputs()}
    if "obs" not in self.inputs:
      raise ValueError(f"ONNX model inputs do not contain 'obs': {list(self.inputs)}.")
    obs_shape = self.inputs["obs"].shape
    self.expected_obs_dim = int(obs_shape[-1])
    output_names = [output.name for output in self.session.get_outputs()]
    self.action_output = "actions" if "actions" in output_names else output_names[0]

  def __call__(self, observations: Any) -> torch.Tensor:
    actor_obs = observations["actor"]
    if actor_obs.ndim != 2 or actor_obs.shape[0] != 1:
      raise ValueError(
        "Static Ghost ONNX rollout requires actor observations with shape [1, D]."
      )
    if actor_obs.shape[1] != self.expected_obs_dim:
      raise ValueError(
        f"ONNX expects {self.expected_obs_dim} observations, "
        f"environment produced {actor_obs.shape[1]}."
      )
    feed: dict[str, np.ndarray] = {
      "obs": actor_obs.detach().cpu().numpy().astype(np.float32, copy=False)
    }
    if "time_step" in self.inputs:
      feed["time_step"] = (
        self.command.reference_index[:, None]
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
      )
    actions = self.session.run([self.action_output], feed)[0]
    return torch.as_tensor(actions, dtype=torch.float32, device=self.device)
