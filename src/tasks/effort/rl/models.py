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


class _FrameEncoder(nn.Module):
  """Project one vector with a linear layer, activation, and layer norm."""

  def __init__(self, input_dim: int, hidden_dim: int, activation: str) -> None:
    super().__init__()
    self.projection = nn.Linear(input_dim, hidden_dim)
    self.activation = resolve_nn_activation(activation)
    self.norm = nn.LayerNorm(hidden_dim)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    return self.norm(self.activation(self.projection(features)))


def _token_encoder(input_dim: int, hidden_dim: int, activation: str) -> nn.Sequential:
  """Project raw token features; the attention block normalizes them."""
  return nn.Sequential(
    nn.Linear(input_dim, hidden_dim), resolve_nn_activation(activation)
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


class _JointAttentionBlock(nn.Module):
  """Pre-norm cross-attention with a residual on the queries, then an MLP."""

  def __init__(self, hidden_dim: int, num_heads: int, activation: str) -> None:
    super().__init__()
    self.query_norm = nn.LayerNorm(hidden_dim)
    self.memory_norm = nn.LayerNorm(hidden_dim)
    self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
    self.mlp_norm = nn.LayerNorm(hidden_dim)
    self.mlp = nn.Sequential(
      nn.Linear(hidden_dim, 2 * hidden_dim),
      resolve_nn_activation(activation),
      nn.Linear(2 * hidden_dim, hidden_dim),
    )

  def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    memory = self.memory_norm(memory)
    attended, _ = self.attention(
      self.query_norm(queries), memory, memory, need_weights=False
    )
    queries = queries + attended
    return queries + self.mlp(self.mlp_norm(queries))

  def attention_weights(
    self, queries: torch.Tensor, memory: torch.Tensor
  ) -> torch.Tensor:
    memory = self.memory_norm(memory)
    _, weights = self.attention(
      self.query_norm(queries),
      memory,
      memory,
      need_weights=True,
      average_attn_weights=False,
    )
    return weights


class _JointCrossAttention(nn.Module):
  """Let every joint query the past torques and positions of all joints and the IMU.

  A joint without a position encoder keeps its token: its position reads zero and
  its missing flag is set, so its torque history still reaches every query.
  """

  def __init__(
    self,
    layout: _JointLayout,
    history_span: int,
    joint_embedding_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_heads: int,
    activation: str,
  ) -> None:
    super().__init__()
    self.slicer = _JointHistorySlicer(layout)
    self.register_buffer("missing", layout.missing)
    self.register_buffer("context_index", layout.context_index)
    self.register_buffer("imu_index", layout.imu_index)
    self.joint_embedding = nn.Parameter(
      torch.empty(len(layout.joint_names), joint_embedding_dim)
    )
    nn.init.normal_(self.joint_embedding, std=0.02)

    identity_dim = joint_embedding_dim + 1
    self.query_encoder = _token_encoder(2 + identity_dim, hidden_dim, activation)
    self.history_encoder = _token_encoder(
      2 * history_span + identity_dim, hidden_dim, activation
    )
    self.context_encoder = _token_encoder(
      layout.context_index.numel(), hidden_dim, activation
    )
    imu_width = layout.imu_index.numel()
    self.imu_encoder = (
      _token_encoder(history_span * imu_width, hidden_dim, activation)
      if imu_width
      else None
    )
    self.block = _JointAttentionBlock(hidden_dim, num_heads, activation)
    self.output_norm = nn.LayerNorm(hidden_dim)
    self.output_projection = nn.Linear(hidden_dim, output_dim)
    self.current_encoder = _FrameEncoder(layout.feature_dim, hidden_dim, activation)

  def forward(self, sequence: torch.Tensor) -> torch.Tensor:
    queries, memory = self._encode(sequence)
    joints = self.output_projection(self.output_norm(self.block(queries, memory)))
    current = self.current_encoder(sequence[:, -1])
    # Flattening keeps one slot per joint, so the MLP head can still tell joints
    # apart and mix them. A follow-up variant is a head shared across joints,
    # tau_j = head(joints[:, j]), applied to the [batch, joints, output_dim] rows
    # in place of this flatten and the MLP. It is the strongest per-joint feedback
    # prior but has no coordination beyond this block, needs one action per joint
    # token (29 for G1), and its last layer should keep the small-gain initial
    # output of _initialize_output_layer.
    return torch.cat((current, joints.flatten(start_dim=1)), dim=-1)

  def attention_weights(self, sequence: torch.Tensor) -> torch.Tensor:
    """Return per-head softmax weights shaped [batch, heads, joints, tokens]."""
    queries, memory = self._encode(sequence)
    return self.block.attention_weights(queries, memory)

  def _encode(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    positions, torques = self.slicer(sequence)  # [batch, frames, joints]
    identity = torch.cat((self.joint_embedding, self.missing.unsqueeze(-1)), dim=-1)
    identity = identity.unsqueeze(0).expand(sequence.shape[0], -1, -1)

    current = torch.stack((positions[:, -1], torques[:, -1]), dim=-1)
    queries = self.query_encoder(torch.cat((current, identity), dim=-1))
    context = sequence[:, -1].index_select(-1, self.context_index)
    queries = queries + self.context_encoder(context).unsqueeze(1)

    history = torch.cat(
      (positions[:, :-1].transpose(1, 2), torques[:, :-1].transpose(1, 2)), dim=-1
    )
    tokens = [self.history_encoder(torch.cat((history, identity), dim=-1))]
    if self.imu_encoder is not None:
      imu = sequence[:, :-1].index_select(-1, self.imu_index).flatten(start_dim=1)
      tokens.append(self.imu_encoder(imu).unsqueeze(1))
    return queries, torch.cat(tokens, dim=1)


class ResidualCrossAttnModel(MLPModel):
  """Cross-attention actor: every joint queries past joint and IMU history.

  The per-joint outputs are flattened into the MLP head. See
  ``_JointCrossAttention.forward`` for the shared per-joint head alternative.
  """

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
    encoder_hidden_dim: int = 128,
    joint_embedding_dim: int = 16,
    joint_output_dim: int = 32,
    term_dims: object = None,
    joint_pos_names: object = None,
    action_names: object = None,
    **unused: object,
  ) -> None:
    del unused
    if history_length < 2:
      raise ValueError(
        "ResidualCrossAttnModel requires at least two observation frames."
      )
    if encoder_hidden_dim % num_heads != 0:
      raise ValueError("encoder_hidden_dim must be divisible by num_heads.")
    if joint_embedding_dim < 1:
      raise ValueError("joint_embedding_dim must be positive.")
    if joint_output_dim < 1:
      raise ValueError("joint_output_dim must be positive.")

    layout = _resolve_cross_attn_layout(term_dims, joint_pos_names, action_names)
    self.history_length = history_length
    self.encoder_hidden_dim = encoder_hidden_dim
    self.joint_output_dim = joint_output_dim
    self.feature_dim = layout.feature_dim
    self.joint_names = layout.joint_names
    self.token_names = layout.joint_names + (
      ("imu",) if layout.imu_index.numel() else ()
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
    self.cross_attention = _JointCrossAttention(
      layout,
      history_span=history_length - 1,
      joint_embedding_dim=joint_embedding_dim,
      hidden_dim=encoder_hidden_dim,
      output_dim=joint_output_dim,
      num_heads=num_heads,
      activation=activation,
    )
    _initialize_output_layer(self.mlp, output_gain)

  def attention_weights(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    """Return joint-query attention shaped [batch, heads, joints, tokens]."""
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
    return self.encoder_hidden_dim + len(self.joint_names) * self.joint_output_dim


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
