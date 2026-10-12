"""RSL-RL models for residual effort policies."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.models import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict


def _initialize_output_layer(module: nn.Module, gain: float) -> None:
  """Initialize the final linear layer around the residual-action origin."""
  linear_layers = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
  if not linear_layers:
    raise ValueError("The policy head must contain an nn.Linear output layer.")
  output_layer = linear_layers[-1]
  nn.init.orthogonal_(output_layer.weight, gain=gain)
  nn.init.zeros_(output_layer.bias)


class ResidualMlpModel(MLPModel):
  """MLP policy with a small-gain orthogonal action output layer."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    output_gain: float = 0.01,
    **unused: object,
  ) -> None:
    del unused
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    _initialize_output_layer(self.mlp, output_gain)


def _sinusoidal_position_encoding(length: int, dim: int) -> torch.Tensor:
  encoding = torch.zeros(1, length, dim)
  position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
  div_term = torch.exp(
    torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10_000.0) / dim)
  )
  encoding[0, :, 0::2] = torch.sin(position * div_term)
  encoding[0, :, 1::2] = torch.cos(position * div_term[: encoding[0, :, 1::2].shape[1]])
  return encoding


class HistoryMhaEncoder(nn.Module):
  """Encode past observations with a pre-norm self-attention residual block."""

  def __init__(
    self,
    input_dim: int,
    history_length: int,
    hidden_dim: int,
    num_heads: int,
    activation: str,
    learnable_pos_embedding: bool,
  ) -> None:
    super().__init__()
    self.input_projection = nn.Linear(input_dim, hidden_dim)
    self.activation = resolve_nn_activation(activation)
    positional_encoding = _sinusoidal_position_encoding(history_length, hidden_dim)
    if learnable_pos_embedding:
      self.positional_embedding = nn.Parameter(positional_encoding)
    else:
      self.register_buffer("positional_embedding", positional_encoding)
    self.pre_norm = nn.LayerNorm(hidden_dim)
    self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
    self.output_norm = nn.LayerNorm(hidden_dim)

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    hidden = self.activation(self.input_projection(observations))
    hidden = hidden + self.positional_embedding[:, : hidden.shape[1]]
    normalized = self.pre_norm(hidden)
    attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
    hidden = self.output_norm(hidden + attended)
    return hidden[:, -1]


class ResidualMhaModel(MLPModel):
  """4+1 MHA policy over native time-major mjlab observation history."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    output_gain: float = 0.01,
    history_length: int = 5,
    num_heads: int = 4,
    encoder_hidden_dim: int = 256,
    learnable_pos_embedding: bool = True,
    **unused: object,
  ) -> None:
    del unused
    if history_length < 2:
      raise ValueError("ResidualMhaModel requires at least two observation frames.")
    if encoder_hidden_dim % num_heads != 0:
      raise ValueError("encoder_hidden_dim must be divisible by num_heads.")

    self.history_length = history_length
    self.encoder_hidden_dim = encoder_hidden_dim
    self.single_step_dim = 0
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )

    self.history_encoder = HistoryMhaEncoder(
      input_dim=self.single_step_dim,
      history_length=history_length - 1,
      hidden_dim=encoder_hidden_dim,
      num_heads=num_heads,
      activation=activation,
      learnable_pos_embedding=learnable_pos_embedding,
    )
    self.current_projection = nn.Linear(self.single_step_dim, encoder_hidden_dim)
    self.current_activation = resolve_nn_activation(activation)
    self.current_norm = nn.LayerNorm(encoder_hidden_dim)
    _initialize_output_layer(self.mlp, output_gain)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    sequence = torch.cat(
      [cast(torch.Tensor, obs[group]) for group in self.obs_groups], dim=-1
    )
    sequence = self.obs_normalizer(sequence)
    history_embedding = self.history_encoder(sequence[:, :-1])
    current_embedding = self.current_projection(sequence[:, -1])
    current_embedding = self.current_activation(current_embedding)
    current_embedding = self.current_norm(current_embedding)
    return torch.cat((current_embedding, history_embedding), dim=-1)

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      sequence = torch.cat(
        [cast(torch.Tensor, obs[group]) for group in self.obs_groups], dim=-1
      )
      flat_sequence = sequence.reshape(-1, self.single_step_dim)
      self.obs_normalizer.update(flat_sequence)  # type: ignore[union-attr]

  def as_jit(self) -> nn.Module:
    return _TorchResidualMhaModel(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxResidualMhaModel(self, verbose)

  def _get_obs_dim(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
  ) -> tuple[list[str], int]:
    active_obs_groups = obs_groups[obs_set]
    single_step_dim = 0
    for obs_group in active_obs_groups:
      shape = obs[obs_group].shape
      if len(shape) != 3:
        raise ValueError(
          "ResidualMhaModel expects [batch, history, features] observations, "
          f"received {shape} for '{obs_group}'."
        )
      if shape[-2] != self.history_length:
        raise ValueError(
          f"Expected {self.history_length} history frames for '{obs_group}', "
          f"received {shape[-2]}."
        )
      single_step_dim += shape[-1]
    self.single_step_dim = single_step_dim
    return active_obs_groups, single_step_dim

  def _get_latent_dim(self) -> int:
    return 2 * self.encoder_hidden_dim


class _ResidualMhaExportCore(nn.Module):
  def __init__(self, model: ResidualMhaModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.history_encoder = copy.deepcopy(model.history_encoder)
    self.current_projection = copy.deepcopy(model.current_projection)
    self.current_activation = copy.deepcopy(model.current_activation)
    self.current_norm = copy.deepcopy(model.current_norm)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is None:
      self.deterministic_output = nn.Identity()
    else:
      self.deterministic_output = model.distribution.as_deterministic_output_module()

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    sequence = self.obs_normalizer(observations)
    history_embedding = self.history_encoder(sequence[:, :-1])
    current_embedding = self.current_projection(sequence[:, -1])
    current_embedding = self.current_activation(current_embedding)
    current_embedding = self.current_norm(current_embedding)
    latent = torch.cat((current_embedding, history_embedding), dim=-1)
    return self.deterministic_output(self.mlp(latent))


class _TorchResidualMhaModel(_ResidualMhaExportCore):
  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxResidualMhaModel(_ResidualMhaExportCore):
  is_recurrent = False

  def __init__(self, model: ResidualMhaModel, verbose: bool) -> None:
    super().__init__(model)
    self.verbose = verbose
    self.history_length = model.history_length
    self.single_step_dim = model.single_step_dim

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.history_length, self.single_step_dim),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


@dataclass(frozen=True)
class _JointLayout:
  """Where joint, IMU, and context features sit in one flat actor frame."""

  joint_names: tuple[str, ...]
  joint_pos: slice
  actions: slice
  position_index: torch.Tensor
  missing: torch.Tensor
  imu_index: torch.Tensor
  context_index: torch.Tensor
  feature_dim: int


def _resolve_cross_attn_layout(
  term_dims: object,
  joint_pos_names: object,
  action_names: object,
) -> _JointLayout:
  """Map a flat actor frame onto one token per controlled joint."""
  if not term_dims or not joint_pos_names or not action_names:
    raise ValueError("The actor observation layout is missing.")

  slices: dict[str, slice] = {}
  offset = 0
  for item in term_dims:  # type: ignore[union-attr]
    name, width = item
    term_name = str(name)
    term_width = int(width)
    if term_width < 1:
      raise ValueError(f"Observation term '{term_name}' must have a positive width.")
    if term_name in slices:
      raise ValueError(f"Duplicate observation term '{term_name}'.")
    slices[term_name] = slice(offset, offset + term_width)
    offset += term_width

  for required in ("joint_pos", "actions", "command"):
    if required not in slices:
      raise ValueError(f"Actor layout is missing the '{required}' term.")

  observed_joints = tuple(str(name) for name in joint_pos_names)  # type: ignore[union-attr]
  controlled_joints = tuple(str(name) for name in action_names)  # type: ignore[union-attr]
  if len(set(observed_joints)) != len(observed_joints):
    raise ValueError("joint_pos_names must be unique.")
  if len(set(controlled_joints)) != len(controlled_joints):
    raise ValueError("action_names must be unique.")

  joint_pos_slice = slices["joint_pos"]
  action_slice = slices["actions"]
  if joint_pos_slice.stop - joint_pos_slice.start != len(observed_joints):
    raise ValueError("joint_pos width must match joint_pos_names.")
  if action_slice.stop - action_slice.start != len(controlled_joints):
    raise ValueError("actions width must match action_names.")

  unknown = [name for name in observed_joints if name not in controlled_joints]
  if unknown:
    raise ValueError(
      "Observed joints missing from the action vector: " + ", ".join(unknown) + "."
    )

  # Joints without a position encoder point one past the last observed joint,
  # which the slicer fills with zeros.
  observed_lookup = {name: index for index, name in enumerate(observed_joints)}
  position_index = torch.tensor(
    [observed_lookup.get(name, len(observed_joints)) for name in controlled_joints],
    dtype=torch.long,
  )
  missing = (position_index == len(observed_joints)).float()

  imu_features = [
    index
    for name in ("base_ang_vel", "projected_gravity")
    if name in slices
    for index in range(slices[name].start, slices[name].stop)
  ]
  joint_features = {
    *range(joint_pos_slice.start, joint_pos_slice.stop),
    *range(action_slice.start, action_slice.stop),
  }
  context_features = [index for index in range(offset) if index not in joint_features]
  return _JointLayout(
    joint_names=controlled_joints,
    joint_pos=joint_pos_slice,
    actions=action_slice,
    position_index=position_index,
    missing=missing,
    imu_index=torch.tensor(imu_features, dtype=torch.long),
    context_index=torch.tensor(context_features, dtype=torch.long),
    feature_dim=offset,
  )


class _JointHistorySlicer(nn.Module):
  """Cut a history window into per-joint position and torque histories."""

  def __init__(self, layout: _JointLayout) -> None:
    super().__init__()
    self.joint_pos_start = int(layout.joint_pos.start)
    self.joint_pos_end = int(layout.joint_pos.stop)
    self.action_start = int(layout.actions.start)
    self.action_end = int(layout.actions.stop)
    self.register_buffer("position_index", layout.position_index)

  def forward(self, window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    joint_pos = window[..., self.joint_pos_start : self.joint_pos_end]
    positions = F.pad(joint_pos, (0, 1)).index_select(-1, self.position_index)
    return positions, window[..., self.action_start : self.action_end]


class _MaskedRunningNorm(nn.Module):
  """Per-feature mean and variance. Masked-off features stay at mean 0 and read out 0."""

  def __init__(self, dim: int, eps: float = 1e-2) -> None:
    super().__init__()
    if dim < 1:
      raise ValueError("A running norm needs at least one feature.")
    self.eps = eps
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.zeros(dim))

  def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    normalized = (values - self.mean) / (torch.sqrt(self.var) + self.eps)
    return normalized * valid

  @torch.no_grad()
  def update(self, values: torch.Tensor, valid: torch.Tensor) -> None:
    if not self.training:
      return
    flat = values.detach().reshape(-1, self.mean.shape[0])
    if valid.dim() == 1:
      mask = valid.detach().reshape(1, -1).expand(flat.shape[0], -1).float()
    else:
      mask = valid.detach().reshape(-1, self.mean.shape[0]).float()
    count_x = mask.sum(dim=0)
    active = count_x > 0
    denom = count_x.clamp(min=1)
    mean_x = (flat * mask).sum(dim=0) / denom
    centered = (flat - mean_x) * mask
    var_x = centered.square().sum(dim=0) / denom
    count = self.count + count_x
    rate = torch.where(active, count_x / count.clamp(min=1), torch.zeros_like(count_x))
    delta = mean_x - self.mean
    new_mean = self.mean + rate * delta
    new_var = (self.var + rate * (var_x - self.var + delta * (mean_x - new_mean))).clamp(min=0)
    self.mean.copy_(torch.where(active, new_mean, self.mean))
    self.var.copy_(torch.where(active, new_var, self.var))
    self.count.copy_(torch.where(active, count, self.count))


class _CurrentQueryTimeEncoder(nn.Module):
  """Project a window, then let the current frame retrieve all frames including itself.

  Attention runs at ``hidden_dim``. The optional lift to ``output_dim`` happens
  after the residual and output LayerNorm, with no extra activation.
  """

  def __init__(
    self,
    input_dim: int,
    hidden_dim: int,
    num_heads: int,
    activation: str,
    output_dim: int | None = None,
    need_weights: bool = False,
  ) -> None:
    super().__init__()
    if hidden_dim % num_heads != 0:
      raise ValueError("Attention hidden_dim must be divisible by num_heads.")
    if output_dim is None:
      output_dim = hidden_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    self.need_weights = need_weights
    self.projection = nn.Linear(input_dim, hidden_dim)
    self.activation = resolve_nn_activation(activation)
    self.query_norm = nn.LayerNorm(hidden_dim)
    self.memory_norm = nn.LayerNorm(hidden_dim)
    self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
    self.output_norm = nn.LayerNorm(hidden_dim)
    self.output_projection = (
      nn.Identity() if output_dim == hidden_dim else nn.Linear(hidden_dim, output_dim)
    )

  def _hidden(self, features: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    hidden = self.activation(self.projection(features))
    return hidden + position[:, : features.shape[1]]

  def forward(self, features: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    hidden = self._hidden(features, position)
    query = hidden[:, -1:]
    memory = self.memory_norm(hidden)
    attended, _weights = self.attention(
      self.query_norm(query),
      memory,
      memory,
      need_weights=self.need_weights,
    )
    summary = self.output_norm(query + attended).squeeze(1)
    return self.output_projection(summary)

  def attention_weights(self, features: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Softmax of the current frame over the window, shaped [batch, heads, 1, time]."""
    hidden = self._hidden(features, position)
    query = hidden[:, -1:]
    memory = self.memory_norm(hidden)
    _attended, weights = self.attention(
      self.query_norm(query),
      memory,
      memory,
      need_weights=True,
      average_attn_weights=False,
    )
    return weights


class _SummaryFusion(nn.Module):
  """One residual self-attention over joint summaries and the IMU summary."""

  def __init__(self, hidden_dim: int, num_heads: int) -> None:
    super().__init__()
    self.pre_norm = nn.LayerNorm(hidden_dim)
    self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
    self.output_norm = nn.LayerNorm(hidden_dim)

  def forward(
    self, joints: torch.Tensor, imu: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = self._tokens(joints, imu)
    normed = self.pre_norm(tokens)
    attended, _ = self.attention(normed, normed, normed, need_weights=False)
    mixed = self.output_norm(tokens + attended)
    return mixed[:, :-1], mixed[:, -1]

  def attention_weights(self, joints: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
    """Softmax over joint summaries plus the IMU summary, shaped [batch, heads, tokens, tokens]."""
    tokens = self.pre_norm(self._tokens(joints, imu))
    _attended, weights = self.attention(
      tokens, tokens, tokens, need_weights=True, average_attn_weights=False
    )
    return weights

  @staticmethod
  def _tokens(joints: torch.Tensor, imu: torch.Tensor) -> torch.Tensor:
    return torch.cat((joints, imu.unsqueeze(1)), dim=1)


class _GroupTimeCore(nn.Module):
  """Per-joint (q_k, τ_{k-1}, m_k) retrieval, plus a separate IMU stream.

  Frame k already stores last_action, which is the torque applied before q_k.
  A later execution delay belongs in that stored torque, not in a second shift here.
  """

  def __init__(
    self,
    layout: _JointLayout,
    history_length: int,
    hidden_dim: int,
    num_heads: int,
    activation: str,
    cross_joint_fusion: bool,
    joint_attention_dim: int,
    joint_attention_heads: int,
    joint_attention_output_dim: int,
  ) -> None:
    super().__init__()
    self.slicer = _JointHistorySlicer(layout)
    self.cross_joint_fusion = cross_joint_fusion
    self.joint_attention_dim = joint_attention_dim
    self.imu_hidden_dim = hidden_dim
    imu_index = layout.imu_index.to(dtype=torch.long)
    imu_set = {int(index) for index in imu_index.tolist()}
    context_index = [
      int(index) for index in layout.context_index.tolist() if int(index) not in imu_set
    ]
    if not context_index:
      raise ValueError("Group-time actor observations have no command features.")
    if cross_joint_fusion and imu_index.numel() and joint_attention_output_dim != hidden_dim:
      raise ValueError(
        "Cross-joint fusion attends over joint and IMU summaries, so "
        "joint_attention_output_dim must equal the IMU encoder width."
      )
    self.register_buffer("valid_joint", (1.0 - layout.missing).to(dtype=torch.float32))
    self.register_buffer("imu_index", imu_index)
    self.register_buffer("context_index", torch.tensor(context_index, dtype=torch.long))
    self.joint_pos_norm = _MaskedRunningNorm(len(layout.joint_names))
    self.joint_torque_norm = _MaskedRunningNorm(len(layout.joint_names))
    self.context_norm = _MaskedRunningNorm(len(context_index))
    self.joint_encoder = _CurrentQueryTimeEncoder(
      3,
      joint_attention_dim,
      joint_attention_heads,
      activation,
      output_dim=joint_attention_output_dim,
      need_weights=True,
    )
    if imu_index.numel():
      self.imu_norm = _MaskedRunningNorm(int(imu_index.numel()))
      self.imu_encoder = _CurrentQueryTimeEncoder(
        int(imu_index.numel()) + 1,
        hidden_dim,
        num_heads,
        activation,
        need_weights=True,
      )
    else:
      self.imu_norm = None
      self.imu_encoder = None
    self.fusion = (
      _SummaryFusion(joint_attention_output_dim, num_heads) if cross_joint_fusion else None
    )
    if joint_attention_dim == hidden_dim:
      self.positional_embedding = nn.Parameter(
        _sinusoidal_position_encoding(history_length, hidden_dim)
      )
    else:
      self.joint_positional_embedding = nn.Parameter(
        _sinusoidal_position_encoding(history_length, joint_attention_dim)
      )
      if self.imu_encoder is not None:
        self.imu_positional_embedding = nn.Parameter(
          _sinusoidal_position_encoding(history_length, hidden_dim)
        )

  def update(self, sequence: torch.Tensor) -> None:
    positions, torques = self.slicer(sequence)
    self.joint_pos_norm.update(positions, self.valid_joint)
    self.joint_torque_norm.update(torques, self.valid_joint)
    context = sequence.index_select(-1, self.context_index)
    self.context_norm.update(context, torch.ones(self.context_index.numel(), device=sequence.device))
    if self.imu_norm is not None:
      imu = sequence.index_select(-1, self.imu_index)
      self.imu_norm.update(imu, torch.ones(self.imu_index.numel(), device=sequence.device))

  def _windows(
    self, sequence: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    positions, torques = self.slicer(sequence)
    valid = self.valid_joint.view(1, 1, -1)
    position_hat = self.joint_pos_norm(positions, valid)
    # Torque is the applied command, so a missing encoder does not zero it.
    torque_hat = self.joint_torque_norm(torques, torch.ones_like(self.valid_joint))
    mask = valid.expand(sequence.shape[0], sequence.shape[1], -1)
    joint_features = torch.stack((position_hat, torque_hat, mask), dim=-1)
    batch, time, joints, _channels = joint_features.shape
    flat_joints = joint_features.permute(0, 2, 1, 3).reshape(batch * joints, time, 3)
    if self.imu_encoder is None or self.imu_norm is None:
      imu_window = None
    else:
      imu = self.imu_norm(
        sequence.index_select(-1, self.imu_index), torch.ones(1, device=sequence.device)
      )
      imu_mask = torch.ones(batch, time, 1, device=sequence.device, dtype=sequence.dtype)
      imu_window = torch.cat((imu, imu_mask), dim=-1)
    return flat_joints, imu_window, position_hat, mask, torque_hat

  def _summaries(
    self, sequence: torch.Tensor, flat_joints: torch.Tensor, imu_window: torch.Tensor | None
  ) -> tuple[torch.Tensor, torch.Tensor]:
    joints = int(self.valid_joint.shape[0])
    summaries = self.joint_encoder(flat_joints, self._position("joint"))
    summaries = summaries.reshape(sequence.shape[0], joints, -1)
    if imu_window is None or self.imu_encoder is None:
      imu_summary = sequence.new_zeros(sequence.shape[0], summaries.shape[-1])
    else:
      imu_summary = self.imu_encoder(imu_window, self._position("imu"))
    return summaries, imu_summary

  def _position(self, stream: str) -> torch.Tensor:
    shared = getattr(self, "positional_embedding", None)
    if shared is not None:
      return shared
    if stream == "joint":
      return self.joint_positional_embedding
    return self.imu_positional_embedding

  def forward(self, sequence: torch.Tensor) -> torch.Tensor:
    flat_joints, imu_window, position_hat, mask, torque_hat = self._windows(sequence)
    summaries, imu_summary = self._summaries(sequence, flat_joints, imu_window)
    if self.fusion is not None:
      summaries, imu_summary = self.fusion(summaries, imu_summary)

    context = self.context_norm(
      sequence[:, -1].index_select(-1, self.context_index),
      torch.ones(self.context_index.numel(), device=sequence.device),
    )
    return torch.cat(
      (
        summaries.flatten(start_dim=1),
        imu_summary,
        position_hat[:, -1],
        mask[:, -1],
        torque_hat[:, -1],
        context,
      ),
      dim=-1,
    )

  def attention_weights(
    self, sequence: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Return time, IMU-time, and fusion softmax maps.

    Time is ``[batch, heads, joints, frames]``. IMU time is
    ``[batch, heads, 1, frames]`` when the actor has an IMU stream. Fusion is
    ``[batch, heads, tokens, tokens]`` over the joint summaries plus IMU.
    """
    flat_joints, imu_window, _position_hat, _mask, _torque_hat = self._windows(sequence)
    raw = self.joint_encoder.attention_weights(flat_joints, self._position("joint"))
    batch = sequence.shape[0]
    time = sequence.shape[1]
    joints = int(self.valid_joint.shape[0])
    time_weights = raw.reshape(batch, joints, -1, time).permute(0, 2, 1, 3).contiguous()
    imu_weights = None
    if imu_window is not None and self.imu_encoder is not None:
      imu_weights = self.imu_encoder.attention_weights(imu_window, self._position("imu"))
    fusion_weights = None
    if self.fusion is not None:
      summaries, imu_summary = self._summaries(sequence, flat_joints, imu_window)
      fusion_weights = self.fusion.attention_weights(summaries, imu_summary)
    return time_weights, imu_weights, fusion_weights


class ResidualGroupTimeModel(MLPModel):
  """Current-frame retrieval over per-joint torque-response history.

  The optional fusion layer is a single residual self-attention after the
  summaries exist. It does not replace the per-joint time retrieval.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (320, 160),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    output_gain: float = 0.01,
    history_length: int = 8,
    num_heads: int = 4,
    encoder_hidden_dim: int = 64,
    joint_attention_dim: int = 32,
    joint_attention_heads: int = 4,
    joint_attention_output_dim: int = 64,
    cross_joint_fusion: bool = False,
    term_dims: object = None,
    joint_pos_names: object = None,
    action_names: object = None,
    **unused: object,
  ) -> None:
    del unused
    if history_length < 2:
      raise ValueError("ResidualGroupTimeModel requires at least two observation frames.")
    if encoder_hidden_dim % num_heads != 0:
      raise ValueError("encoder_hidden_dim must be divisible by num_heads.")
    if joint_attention_dim % joint_attention_heads != 0:
      raise ValueError("joint_attention_dim must be divisible by joint_attention_heads.")
    layout = _resolve_cross_attn_layout(term_dims, joint_pos_names, action_names)
    self.history_length = history_length
    self.encoder_hidden_dim = encoder_hidden_dim
    self.joint_attention_dim = joint_attention_dim
    self.joint_attention_output_dim = joint_attention_output_dim
    self.has_imu = bool(layout.imu_index.numel())
    self.cross_joint_fusion = cross_joint_fusion
    self.feature_dim = layout.feature_dim
    self.joint_names = layout.joint_names
    imu_set = {int(index) for index in layout.imu_index.tolist()}
    self._context_dim = sum(
      1 for index in layout.context_index.tolist() if int(index) not in imu_set
    )
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    self.obs_normalizer = nn.Identity()
    self.core = _GroupTimeCore(
      layout,
      history_length=history_length,
      hidden_dim=encoder_hidden_dim,
      num_heads=num_heads,
      activation=activation,
      cross_joint_fusion=cross_joint_fusion,
      joint_attention_dim=joint_attention_dim,
      joint_attention_heads=joint_attention_heads,
      joint_attention_output_dim=joint_attention_output_dim,
    )
    _initialize_output_layer(self.mlp, output_gain)

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    sequence = torch.cat(
      [cast(torch.Tensor, obs[group]) for group in self.obs_groups], dim=-1
    )
    return self.core(sequence)

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      sequence = torch.cat(
        [cast(torch.Tensor, obs[group]) for group in self.obs_groups], dim=-1
      )
      self.core.update(sequence)

  @torch.no_grad()
  def attention_weights(self, obs: TensorDict) -> dict[str, torch.Tensor | tuple[str, ...]]:
    """Softmax maps for the group-time actor.

    ``time`` is ``[batch, heads, joints, frames]`` with ``time_rows`` and
    ``time_columns``. ``imu_time`` is present when the actor has an IMU stream.
    ``fusion`` is ``[batch, heads, tokens, tokens]`` when that layer is enabled,
    with ``fusion_names`` for both axes.
    """
    sequence = torch.cat(
      [cast(torch.Tensor, obs[group]) for group in self.obs_groups], dim=-1
    )
    time_weights, imu_weights, fusion_weights = self.core.attention_weights(sequence)
    saved: dict[str, torch.Tensor | tuple[str, ...]] = {
      "time": time_weights,
      "time_rows": self.joint_names,
      "time_columns": tuple(
        "t" if offset == 0 else f"t-{offset}"
        for offset in range(self.history_length - 1, -1, -1)
      ),
    }
    if imu_weights is not None:
      saved["imu_time"] = imu_weights
    if fusion_weights is not None:
      saved["fusion"] = fusion_weights
      saved["fusion_names"] = self.joint_names + ("imu",)
    return saved

  def as_jit(self) -> nn.Module:
    return _TorchResidualGroupTimeModel(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxResidualGroupTimeModel(self, verbose)

  def _get_obs_dim(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
  ) -> tuple[list[str], int]:
    active_obs_groups = obs_groups[obs_set]
    single_step_dim = 0
    for obs_group in active_obs_groups:
      shape = obs[obs_group].shape
      if len(shape) != 3:
        raise ValueError(
          "ResidualGroupTimeModel expects [batch, history, features] observations, "
          f"received {shape} for '{obs_group}'."
        )
      if shape[-2] != self.history_length:
        raise ValueError(
          f"Expected {self.history_length} history frames for '{obs_group}', "
          f"received {shape[-2]}."
        )
      single_step_dim += shape[-1]
    if single_step_dim != self.feature_dim:
      raise ValueError(
        f"Observation width {single_step_dim} does not match the actor layout "
        f"width {self.feature_dim}."
      )
    return active_obs_groups, single_step_dim

  def _get_latent_dim(self) -> int:
    joints = len(self.joint_names)
    imu_width = (
      self.encoder_hidden_dim if self.has_imu else self.joint_attention_output_dim
    )
    return (
      joints * self.joint_attention_output_dim
      + imu_width
      + 3 * joints
      + self._context_dim
    )


class _ResidualGroupTimeExportCore(nn.Module):
  def __init__(self, model: ResidualGroupTimeModel) -> None:
    super().__init__()
    self.core = copy.deepcopy(model.core)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is None:
      self.deterministic_output = nn.Identity()
    else:
      self.deterministic_output = model.distribution.as_deterministic_output_module()

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    return self.deterministic_output(self.mlp(self.core(observations)))


class _TorchResidualGroupTimeModel(_ResidualGroupTimeExportCore):
  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxResidualGroupTimeModel(_ResidualGroupTimeExportCore):
  is_recurrent = False

  def __init__(self, model: ResidualGroupTimeModel, verbose: bool) -> None:
    super().__init__(model)
    self.verbose = verbose
    self.history_length = model.history_length
    self.feature_dim = model.feature_dim

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.history_length, self.feature_dim),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
