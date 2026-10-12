"""Effort-specific RSL-RL model configs.

Runner assembly (actor/critic/PPO hyperparameters) lives in each robot's
``rl_cfg.py``, matching the velocity tasks. These dataclasses only exist
because 0-PD uses a residual-centered output layer and optional history actors.
"""

from dataclasses import dataclass

from mjlab.rl import RslRlModelCfg


@dataclass
class ResidualMlpModelCfg(RslRlModelCfg):
  """RSL-RL MLP config with residual-centered actor initialization."""

  class_name: str = "src.tasks.effort.rl.models:ResidualMlpModel"
  output_gain: float = 0.01
  obs_normalization: bool = True


@dataclass
class ResidualMhaModelCfg(ResidualMlpModelCfg):
  """RSL-RL config for the native time-major 4+1 MHA actor."""

  class_name: str = "src.tasks.effort.rl.models:ResidualMhaModel"
  history_length: int = 5
  num_heads: int = 4
  encoder_hidden_dim: int = 256
  learnable_pos_embedding: bool = True


@dataclass
class ResidualGroupTimeModelCfg(ResidualMlpModelCfg):
  """RSL-RL config for conditional per-joint retrieval plus summary fusion."""

  class_name: str = "src.tasks.effort.rl.models:ResidualGroupTimeModel"
  hidden_dims: tuple[int, ...] = (320, 160)
  history_length: int = 5
  num_heads: int = 4
  encoder_hidden_dim: int = 64
  joint_attention_dim: int = 32
  joint_attention_heads: int = 4
  joint_attention_output_dim: int = 64
  term_dims: tuple[tuple[str, int], ...] = ()
  joint_pos_names: tuple[str, ...] = ()
  action_names: tuple[str, ...] = ()
