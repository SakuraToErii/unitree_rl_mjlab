"""Effort-specific RSL-RL model configs.

Runner assembly (actor/critic/PPO hyperparameters) lives in each robot's
``rl_cfg.py``, matching the velocity tasks. These dataclasses only exist
because 0-PD uses a residual-centered output layer and an optional MHA actor.
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
