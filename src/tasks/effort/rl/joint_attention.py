"""Plot per-joint attention over joint and IMU history tokens."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def mean_joint_attention(weights: torch.Tensor) -> torch.Tensor:
  """Average attention over batch and heads down to a [joints, tokens] map."""
  if weights.ndim == 3:
    return weights.mean(dim=0)
  if weights.ndim == 4:
    return weights.mean(dim=(0, 1))
  raise ValueError(
    "Attention weights must be [heads, joints, tokens] or "
    f"[batch, heads, joints, tokens], received {tuple(weights.shape)}."
  )


def plot_joint_attention(
  weights: torch.Tensor,
  joint_names: Sequence[str],
  token_names: Sequence[str],
  output_path: str | Path,
) -> None:
  """Write a heatmap with one row per query joint and one column per token."""
  values = mean_joint_attention(weights).detach().cpu()
  rows = tuple(joint_names)
  columns = tuple(token_names)
  if values.shape != (len(rows), len(columns)):
    raise ValueError(
      f"Expected a {len(rows)} x {len(columns)} attention map, "
      f"received {tuple(values.shape)}."
    )

  peak = float(values.max()) if values.numel() else 1.0
  fig, ax = plt.subplots(
    figsize=(max(8.0, 0.38 * len(columns)), max(4.0, 0.3 * len(rows)))
  )
  image = ax.imshow(
    values.numpy(), cmap="Reds", aspect="auto", vmin=0.0, vmax=max(peak, 1.0e-8)
  )
  ax.set_yticks(range(len(rows)))
  ax.set_yticklabels(rows, fontsize=8)
  ax.set_xticks(range(len(columns)))
  ax.set_xticklabels(columns, rotation=90, fontsize=8)
  ax.set_xlabel("history token")
  ax.set_ylabel("query joint")
  fig.colorbar(image, ax=ax, label="attention")
  fig.savefig(output_path, bbox_inches="tight")
  plt.close(fig)
