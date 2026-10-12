"""Plot group-time attention saved by ``ResidualGroupTimeModel.attention_weights``.

The input is a ``torch.save`` dict. ``time`` has shape
``[heads, joints, frames]`` or ``[batch, heads, joints, frames]``, with
``time_rows`` and ``time_columns``. A batch axis is averaged. ``imu_time`` and
``fusion`` are written next to the time heatmap when the actor has them.
"""

import argparse
from pathlib import Path

import torch

from src.tasks.effort.rl.joint_attention import plot_group_time_attention


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("weights_path", type=Path)
  parser.add_argument("output_path", type=Path)
  args = parser.parse_args()
  saved = torch.load(args.weights_path, weights_only=False)
  plot_group_time_attention(saved, args.output_path)


if __name__ == "__main__":
  main()
