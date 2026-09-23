"""Plot a saved current-frame attention map over joint history.

The input is a ``torch.save`` dict with ``weights`` of shape ``[heads, joints]``
or ``[batch, heads, joints]``, and ``joint_names`` in that same joint order.
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
  plot_joint_attention(saved["weights"], saved["joint_names"], args.output_path)


if __name__ == "__main__":
  main()
