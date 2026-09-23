"""CUDA bf16 autocast for G1 cross-attention PPO network matmuls."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict


class Bf16AutocastModel(nn.Module):
  """Wrap an RSL-RL model so CUDA matmuls run under bf16 autocast.

  Parameters stay fp32. The tensor that updates the action distribution, and
  critic values stored for returns, are cast back to fp32 so log_prob, entropy,
  KL, and the PPO loss remain fp32. On CPU, autocast is disabled.
  """

  def __init__(self, model: MLPModel) -> None:
    super().__init__()
    self.model = model

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    model = self.model
    if masks is not None and not model.is_recurrent:
      obs = unpad_trajectories(obs, masks)

    param = next(model.parameters())
    with torch.autocast(
      device_type="cuda",
      dtype=torch.bfloat16,
      enabled=param.is_cuda,
    ):
      latent = model.get_latent(obs, masks, hidden_state)
      mlp_output = model.mlp(latent)
    mlp_output = mlp_output.float()

    if model.distribution is not None:
      if stochastic_output:
        model.distribution.update(mlp_output)
        return model.distribution.sample()
      return model.distribution.deterministic_output(mlp_output)
    return mlp_output

  def __getattr__(self, name: str):
    try:
      return super().__getattr__(name)
    except AttributeError:
      return getattr(self._modules["model"], name)


class Bf16AutocastPPO(PPO):
  """PPO that autocasts actor/critic network matmuls in bf16 on CUDA."""

  @staticmethod
  def construct_algorithm(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str
  ) -> PPO:
    alg = PPO.construct_algorithm(obs, env, cfg, device)
    alg.actor = Bf16AutocastModel(alg.actor)
    alg.critic = Bf16AutocastModel(alg.critic)
    return alg
