"""RSL-RL models for residual effort policies."""

from __future__ import annotations

import copy
import math
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


def _resolve_cross_attn_layout(
  term_dims: object,
  joint_pos_names: object,
  action_names: object,
) -> tuple[dict[str, slice], tuple[str, ...], torch.Tensor, int]:
  """Map a flat actor frame onto joint-position and action slices."""
  if not term_dims or not joint_pos_names or not action_names:
    raise ValueError("ResidualCrossAttnModel requires an actor observation layout.")

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

  for required in ("joint_pos", "actions"):
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

  action_lookup = {name: index for index, name in enumerate(controlled_joints)}
  missing = [name for name in observed_joints if name not in action_lookup]
  if missing:
    raise ValueError(
      "Observed joints missing from the action vector: " + ", ".join(missing) + "."
    )
  action_index = torch.tensor(
    [action_lookup[name] for name in observed_joints], dtype=torch.long
  )
  return slices, observed_joints, action_index, offset


class _FrameEncoder(nn.Module):
  """Project one vector with a linear layer, activation, and layer norm."""

  def __init__(self, input_dim: int, hidden_dim: int, activation: str) -> None:
    super().__init__()
    self.projection = nn.Linear(input_dim, hidden_dim)
    self.activation = resolve_nn_activation(activation)
    self.norm = nn.LayerNorm(hidden_dim)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    return self.norm(self.activation(self.projection(features)))


class _JointHistoryEncoder(_FrameEncoder):
  """Encode one joint's past position and action, plus a learned joint id."""

  def __init__(
    self,
    history_span: int,
    num_joints: int,
    embedding_dim: int,
    hidden_dim: int,
    activation: str,
  ) -> None:
    super().__init__(2 * history_span + embedding_dim, hidden_dim, activation)
    self.embedding = nn.Parameter(torch.empty(num_joints, embedding_dim))
    nn.init.normal_(self.embedding, std=0.02)

  def forward(  # type: ignore[override]
    self, joint_pos: torch.Tensor, joint_action: torch.Tensor
  ) -> torch.Tensor:
    position = joint_pos.transpose(1, 2)
    action = joint_action.transpose(1, 2)
    identity = self.embedding.unsqueeze(0).expand(position.shape[0], -1, -1)
    token = torch.cat((position, action, identity), dim=-1)
    return super().forward(token)


class _JointHistorySlicer(nn.Module):
  """Cut past frames into per-joint position and same-named action history."""

  def __init__(
    self,
    joint_pos_slice: slice,
    action_slice: slice,
    action_index: torch.Tensor,
  ) -> None:
    super().__init__()
    self.joint_pos_start = int(joint_pos_slice.start)
    self.joint_pos_end = int(joint_pos_slice.stop)
    self.action_start = int(action_slice.start)
    self.action_end = int(action_slice.stop)
    self.register_buffer("action_index", action_index)

  def forward(self, past: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    joint_pos = past[..., self.joint_pos_start : self.joint_pos_end]
    actions = past[..., self.action_start : self.action_end]
    return joint_pos, actions.index_select(dim=-1, index=self.action_index)


class _JointCrossAttention(nn.Module):
  """Attend from the current frame to each observed joint's history."""

  def __init__(
    self,
    feature_dim: int,
    history_span: int,
    num_joints: int,
    joint_embedding_dim: int,
    hidden_dim: int,
    num_heads: int,
    activation: str,
    joint_pos_slice: slice,
    action_slice: slice,
    action_index: torch.Tensor,
  ) -> None:
    super().__init__()
    if hidden_dim % num_heads != 0:
      raise ValueError("hidden_dim must be divisible by num_heads.")
    self.num_heads = num_heads
    self.head_dim = hidden_dim // num_heads
    self.scale = self.head_dim**-0.5
    self.query_encoder = _FrameEncoder(feature_dim, hidden_dim, activation)
    self.slicer = _JointHistorySlicer(joint_pos_slice, action_slice, action_index)
    self.joint_encoder = _JointHistoryEncoder(
      history_span, num_joints, joint_embedding_dim, hidden_dim, activation
    )
    self.q_proj = nn.Linear(hidden_dim, hidden_dim)
    self.k_proj = nn.Linear(hidden_dim, hidden_dim)
    self.v_proj = nn.Linear(hidden_dim, hidden_dim)
    self.out_proj = nn.Linear(hidden_dim, hidden_dim)

  def forward(self, sequence: torch.Tensor) -> torch.Tensor:
    query, keys = self._encode(sequence)
    q, k, v = self._project_qkv(query, keys)
    attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    attended = (
      attended.transpose(1, 2).contiguous().view(query.shape[0], -1)
    )
    attended = self.out_proj(attended)
    return torch.cat((query, attended), dim=-1)

  def attention_weights(self, sequence: torch.Tensor) -> torch.Tensor:
    """Return per-head softmax weights over joints, shaped [batch, heads, joints]."""
    query, keys = self._encode(sequence)
    q, k, _ = self._project_qkv(query, keys)
    scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
    return torch.softmax(scores, dim=-1).squeeze(2)

  def _encode(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    query = self.query_encoder(sequence[:, -1])
    joint_pos, joint_action = self.slicer(sequence[:, :-1])
    keys = self.joint_encoder(joint_pos, joint_action)
    return query, keys

  def _project_qkv(
    self, query: torch.Tensor, keys: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, num_joints, _ = keys.shape
    q = self.q_proj(query).view(batch, 1, self.num_heads, self.head_dim).transpose(1, 2)
    k = (
      self.k_proj(keys)
      .view(batch, num_joints, self.num_heads, self.head_dim)
      .transpose(1, 2)
    )
    v = (
      self.v_proj(keys)
      .view(batch, num_joints, self.num_heads, self.head_dim)
      .transpose(1, 2)
    )
    return q, k, v


class ResidualCrossAttnModel(MLPModel):
  """Cross-attention actor: current frame queries per-joint history."""

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
    joint_embedding_dim: int = 16,
    term_dims: object = None,
    joint_pos_names: object = None,
    action_names: object = None,
    **unused: object,
  ) -> None:
    del unused
    if history_length < 2:
      raise ValueError("ResidualCrossAttnModel requires at least two observation frames.")
    if encoder_hidden_dim % num_heads != 0:
      raise ValueError("encoder_hidden_dim must be divisible by num_heads.")
    if joint_embedding_dim < 1:
      raise ValueError("joint_embedding_dim must be positive.")

    slices, observed_joints, action_index, feature_dim = _resolve_cross_attn_layout(
      term_dims, joint_pos_names, action_names
    )
    self.history_length = history_length
    self.encoder_hidden_dim = encoder_hidden_dim
    self.feature_dim = feature_dim
    self.joint_names = observed_joints
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
    self.cross_attention = _JointCrossAttention(
      feature_dim=feature_dim,
      history_span=history_length - 1,
      num_joints=len(observed_joints),
      joint_embedding_dim=joint_embedding_dim,
      hidden_dim=encoder_hidden_dim,
      num_heads=num_heads,
      activation=activation,
      joint_pos_slice=slices["joint_pos"],
      action_slice=slices["actions"],
      action_index=action_index,
    )
    _initialize_output_layer(self.mlp, output_gain)

  def attention_weights(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    """Return current-frame attention over observed joints, shaped [batch, heads, joints]."""
    del masks, hidden_state
    sequence = torch.cat(
      [cast(torch.Tensor, obs[group]) for group in self.obs_groups], dim=-1
    )
    return self.cross_attention.attention_weights(self.obs_normalizer(sequence))

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
    return self.cross_attention(self.obs_normalizer(sequence))

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      sequence = torch.cat(
        [cast(torch.Tensor, obs[group]) for group in self.obs_groups], dim=-1
      )
      flat_sequence = sequence.reshape(-1, self.feature_dim)
      self.obs_normalizer.update(flat_sequence)  # type: ignore[union-attr]

  def as_jit(self) -> nn.Module:
    return _TorchResidualCrossAttnModel(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxResidualCrossAttnModel(self, verbose)

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
          "ResidualCrossAttnModel expects [batch, history, features] observations, "
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
    return 2 * self.encoder_hidden_dim


class _ResidualCrossAttnExportCore(nn.Module):
  def __init__(self, model: ResidualCrossAttnModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.cross_attention = copy.deepcopy(model.cross_attention)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is None:
      self.deterministic_output = nn.Identity()
    else:
      self.deterministic_output = model.distribution.as_deterministic_output_module()

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    latent = self.cross_attention(self.obs_normalizer(observations))
    return self.deterministic_output(self.mlp(latent))


class _TorchResidualCrossAttnModel(_ResidualCrossAttnExportCore):
  @torch.jit.export
  def reset(self) -> None:
    pass


class _OnnxResidualCrossAttnModel(_ResidualCrossAttnExportCore):
  is_recurrent = False

  def __init__(self, model: ResidualCrossAttnModel, verbose: bool) -> None:
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
