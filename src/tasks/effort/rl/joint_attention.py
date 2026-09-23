"""Plot current-frame attention over per-joint history.

AME draws one red intensity per key/value token (He et al., arXiv:2506.09588,
Figure 7). Heads are averaged so each observed joint gets that one weight.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def mean_joint_attention(weights: torch.Tensor) -> torch.Tensor:
  """Average a current-frame attention map down to one weight per joint."""
  if weights.ndim == 2:
    return weights.mean(dim=0)
  if weights.ndim == 3:
    return weights.mean(dim=(0, 1))
  raise ValueError(
    "Attention weights must be [heads, joints] or [batch, heads, joints], "
    f"received {tuple(weights.shape)}."
  )


def plot_joint_attention(
  weights: torch.Tensor,
  joint_names: Sequence[str],
  output_path: str | Path,
) -> None:
  """Write a one-row heatmap of attention from the current frame onto joints."""
  values = mean_joint_attention(weights).detach().cpu()
  names = tuple(joint_names)
  if values.shape != (len(names),):
    raise ValueError(
      f"Expected one weight for each of {len(names)} joints, received {tuple(values.shape)}."
    )

  row = values.numpy()[None, :]
  peak = float(values.max()) if values.numel() else 1.0
  fig, ax = plt.subplots(figsize=(max(8.0, 0.38 * len(names)), 2.6))
  image = ax.imshow(row, cmap="Reds", aspect="auto", vmin=0.0, vmax=max(peak, 1.0e-8))
  ax.set_yticks([0])
  ax.set_yticklabels(["current frame"])
  ax.set_xticks(range(len(names)))
  ax.set_xticklabels(names, rotation=90, fontsize=8)
  ax.set_xlabel("joint history")
  fig.colorbar(image, ax=ax, label="attention")
  fig.savefig(output_path, bbox_inches="tight")
  plt.close(fig)
