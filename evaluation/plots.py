"""Plotting utilities for Project9 metrics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_metrics(env_tag: str, algorithm: str, seed: int):
    path = Path("outputs/metrics") / f"{env_tag}_{algorithm}_seed{seed}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_t1_kl(seed: int = 0):
    metrics = load_metrics("t1", "ma_lfl", seed)
    err_1a = metrics["err_1a"]
    err_1b = metrics["err_1b"]

    plt.figure()
    plt.bar(["err_1a", "err_1b"], [err_1a, err_1b])
    plt.ylabel("KL")
    plt.title(f"T1 GridWorld KL (seed={seed})")

    out_dir = Path("outputs/plots")
    ensure_dir(out_dir)
    out_path = out_dir / f"t1_kl_seed{seed}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[plots] saved {out_path}")


def plot_t2_kl_compare(seed: int = 0):
    metrics_ilola = load_metrics("t2", "ilola", seed)
    labels = ["ILOLA-err_1a", "ILOLA-err_1b"]
    values = [metrics_ilola["err_1a"], metrics_ilola["err_1b"]]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("KL")
    plt.xticks(rotation=25)
    plt.title(f"T2 MPE KL compare (seed={seed})")

    out_dir = Path("outputs/plots")
    ensure_dir(out_dir)
    out_path = out_dir / f"t2_kl_compare_seed{seed}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[plots] saved {out_path}")


def plot_t3_kl(seed: int = 0):
    metrics_ilola = load_metrics("t3", "ilola", seed)
    err_1b = metrics_ilola.get("err_1b", 0.0)

    plt.figure()
    plt.bar(["err_1b"], [err_1b])
    plt.ylabel("KL")
    plt.title(f"T3 MultiWalker KL (seed={seed})")

    out_dir = Path("outputs/plots")
    ensure_dir(out_dir)
    out_path = out_dir / f"t3_kl_seed{seed}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[plots] saved {out_path}")


def plot_t1_reward_heatmap():
    weights_dir = Path("outputs/weights")
    w_true_path = weights_dir / "t1_true.npy"
    w_hat_path = weights_dir / "t1_hat.npy"
    if not w_true_path.exists() or not w_hat_path.exists():
        print("[plots] t1 weight files not found; skip heatmap.")
        return
    w_true = np.load(w_true_path)
    w_hat = np.load(w_hat_path)

    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)

    im0 = axes[0].imshow(w_true[np.newaxis, :], aspect="auto")
    axes[0].set_title("T1 reward true")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(w_hat[np.newaxis, :], aspect="auto")
    axes[1].set_title("T1 reward recovered")
    plt.colorbar(im1, ax=axes[1])

    axes[1].set_xticks(range(len(w_true)))

    out_dir = Path("outputs/plots")
    ensure_dir(out_dir)
    out_path = out_dir / "t1_reward_heatmap.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[plots] saved {out_path}")


def plot_t3_induced_returns(induced_result: dict, seed: int = 0):
    curve = induced_result.get("R_induced_curve", [])
    if not curve:
        print("[plots] no induced returns to plot; skip.")
        return
    steps, R_induced = zip(*curve)
    R_random = induced_result.get("R_random", 0.0)
    R_expert = induced_result.get("R_expert", 0.0)

    plt.figure()
    plt.plot(steps, R_induced, label="R_induced")
    plt.axhline(R_random, linestyle="--", label="R_random")
    plt.axhline(R_expert, linestyle="--", label="R_expert")
    plt.xlabel("training step")
    plt.ylabel("return")
    plt.legend()
    plt.title(f"T3 MultiWalker induced returns (seed={seed})")

    out_dir = Path("outputs/plots")
    ensure_dir(out_dir)
    out_path = out_dir / f"t3_induced_returns_seed{seed}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[plots] saved {out_path}")
