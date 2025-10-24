from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_trend(
    stages: Sequence[int],
    pearson: Sequence[float],
    spearman: Sequence[float],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(stages, pearson, label="PCC", marker="o")
    plt.plot(stages, spearman, label="SCC", marker="s")
    plt.xlabel("MA-SPI Stage h")
    plt.ylabel("Correlation")
    plt.title("Reward Recovery Trend")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_reward_heatmap(
    matrix: np.ndarray,
    grid_size: int,
    path: Path,
    title: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    data = matrix.reshape(grid_size * grid_size, grid_size * grid_size)
    ax = sns.heatmap(data, cmap="viridis", cbar=True, vmin=vmin, vmax=vmax)
    ax.set_xlabel("Agent 2 State Index")
    ax.set_ylabel("Agent 1 State Index")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_reward_comparison(
    true_matrix: np.ndarray,
    predicted_matrix: np.ndarray,
    grid_size: int,
    output_dir: Path,
    prefix: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vmin = min(true_matrix.min(), predicted_matrix.min())
    vmax = max(true_matrix.max(), predicted_matrix.max())
    plot_reward_heatmap(
        true_matrix, grid_size, output_dir / f"{prefix}_true.png", f"{prefix} True Reward", vmin=vmin, vmax=vmax
    )
    plot_reward_heatmap(
        predicted_matrix,
        grid_size,
        output_dir / f"{prefix}_predicted.png",
        f"{prefix} Recovered Reward",
        vmin=vmin,
        vmax=vmax,
    )
