"""Plot group-time attention: each joint over its frames, and optional fusion."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def mean_joint_attention(weights: torch.Tensor) -> torch.Tensor:
  """Average attention over batch and heads down to a [rows, columns] map."""
  if weights.ndim == 3:
    return weights.mean(dim=0)
  if weights.ndim == 4:
    return weights.mean(dim=(0, 1))
  raise ValueError(
    "Attention weights must be [heads, rows, columns] or "
    f"[batch, heads, rows, columns], received {tuple(weights.shape)}."
  )


def plot_joint_attention(
  weights: torch.Tensor,
  row_names: Sequence[str],
  column_names: Sequence[str],
  output_path: str | Path,
  *,
  xlabel: str = "history token",
  ylabel: str = "query joint",
) -> None:
  """Write a heatmap with one row per query and one column per key."""
  values = mean_joint_attention(weights).detach().cpu()
  rows = tuple(row_names)
  columns = tuple(column_names)
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
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  fig.colorbar(image, ax=ax, label="attention")
  fig.savefig(output_path, bbox_inches="tight")
  plt.close(fig)


def plot_group_time_attention(saved: dict[str, object], output_path: str | Path) -> None:
  """Write the time map, and IMU or fusion maps beside it when they are present."""
  path = Path(output_path)
  plot_joint_attention(
    saved["time"],  # type: ignore[arg-type]
    saved["time_rows"],  # type: ignore[arg-type]
    saved["time_columns"],  # type: ignore[arg-type]
    path,
    xlabel="frame",
    ylabel="joint",
  )
  if "imu_time" in saved:
    plot_joint_attention(
      saved["imu_time"],  # type: ignore[arg-type]
      ("imu",),
      saved["time_columns"],  # type: ignore[arg-type]
      path.with_name(f"{path.stem}_imu{path.suffix}"),
      xlabel="frame",
      ylabel="imu",
    )
  if "fusion" in saved:
    names = saved["fusion_names"]
    plot_joint_attention(
      saved["fusion"],  # type: ignore[arg-type]
      names,  # type: ignore[arg-type]
      names,  # type: ignore[arg-type]
      path.with_name(f"{path.stem}_fusion{path.suffix}"),
      xlabel="summary token",
      ylabel="query token",
    )
