"""Plot saved per-joint attention over joint and IMU history tokens.

The input is a ``torch.save`` dict with ``weights`` of shape
``[heads, joints, tokens]`` or ``[batch, heads, joints, tokens]``, plus
``joint_names`` (rows) and ``token_names`` (columns) in the same order. Use
``model.attention_weights(obs)``, ``model.joint_names`` and ``model.token_names``.
A batch axis is averaged, so save one step when the figure should describe a
single moment.
"""

import argparse
from pathlib import Path

import torch

from src.tasks.effort.rl.joint_attention import plot_joint_attention


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("weights_path", type=Path)
  parser.add_argument("output_path", type=Path)
  args = parser.parse_args()
  saved = torch.load(args.weights_path, weights_only=False)
  plot_joint_attention(
    saved["weights"], saved["joint_names"], saved["token_names"], args.output_path
  )


if __name__ == "__main__":
  main()
