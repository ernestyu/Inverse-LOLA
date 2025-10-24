from __future__ import annotations

import numpy as np
from typing import Tuple


def _safe_std(arr: np.ndarray) -> float:
    return float(np.sqrt(np.maximum(np.var(arr), 1e-12)))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    numerator = float(np.sum((x - x_mean) * (y - y_mean)))
    denominator = _safe_std(x) * _safe_std(y) * (x.size)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _rankdata(arr: np.ndarray) -> np.ndarray:
    temp = arr.argsort()
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(len(arr))
    # Handle ties by averaging ranks
    _, inv, counts = np.unique(arr, return_inverse=True, return_counts=True)
    cumulative = np.cumsum(counts)
    cumulative = np.insert(cumulative, 0, 0)
    for i in range(len(counts)):
        mask = inv == i
        start = cumulative[i]
        end = cumulative[i + 1]
        ranks[mask] = (start + end - 1) / 2.0
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata(x)
    ry = _rankdata(y)
    return pearson_corr(rx, ry)


def flatten_reward_table(table: np.ndarray) -> np.ndarray:
    return table.reshape(-1)


def correlation_pair(pred: np.ndarray, true: np.ndarray) -> Tuple[float, float]:
    pred_flat = flatten_reward_table(pred)
    true_flat = flatten_reward_table(true)
    return pearson_corr(pred_flat, true_flat), spearman_corr(pred_flat, true_flat)
