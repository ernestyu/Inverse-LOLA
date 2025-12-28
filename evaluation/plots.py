"""Plotting utilities for Project9 metrics."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_t2_compare_lfl_ilola(seed: int = 0):
    try:
        metrics_lfl = load_metrics("t2", "ma_lfl", seed)
        metrics_ilola = load_metrics("t2", "ilola", seed)
    except FileNotFoundError:
        print("[plots] t2 metrics missing; skip mismatch compare plot.")
        return

    labels = ["MA-LfL 1a", "MA-LfL 1b", "I-LOLA 1a", "I-LOLA 1b"]
    values = [
        metrics_lfl.get("err_1a", 0.0),
        metrics_lfl.get("err_1b", 0.0),
        metrics_ilola.get("err_1a", 0.0),
        metrics_ilola.get("err_1b", 0.0),
    ]

    plt.figure()
    plt.bar(labels, values, color=["#4c72b0", "#55a868", "#c44e52", "#8172b3"])
    plt.ylabel("KL")
    plt.xticks(rotation=20)
    plt.title(f"T2 KL mismatch compare (seed={seed})")

    out_dir = Path("outputs/plots")
    ensure_dir(out_dir)
    out_path = out_dir / f"t2_kl_compare_mismatch_seed{seed}.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[plots] saved {out_path}")


def plot_t2_stageA_vs_stageB(seed: int = 0):
    try:
        m_stageA = load_metrics("t2", "ilogel_stageA", seed)
        m_stageB = load_metrics("t2", "ilola", seed)
        m_lfl = load_metrics("t2", "ma_lfl", seed)
    except FileNotFoundError:
        print("[plots] missing t2 metrics for stageA/stageB; skip plot.")
        return

    labels = ["StageA 1b", "StageB 1b", "MA-LfL 1b"]
    values = [
        m_stageA.get("err_1b", 0.0),
        m_stageB.get("err_1b", 0.0),
        m_lfl.get("err_1b", 0.0),
    ]

    plt.figure()
    plt.bar(labels, values, color=["#4c72b0", "#c44e52", "#55a868"])
    plt.ylabel("KL")
    plt.xticks(rotation=15)
    plt.title(f"T2 StageA vs StageB (seed={seed})")

    out_dir = Path("outputs/plots")
    ensure_dir(out_dir)
    out_path = out_dir / f"t2_stageA_vs_stageB_seed{seed}.png"
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


def plot_t2_induced_returns(induced_result: dict, seed: int = 0):
    curve = induced_result.get("R_induced_curve", [])
    if not curve:
        print("[plots] no t2 induced returns to plot; skip.")
        return
    # curve is list of (step, return)
    try:
        steps, returns = zip(*curve)
    except ValueError:
        # fallback if curve is a flat list of returns
        returns = list(curve)
        steps = list(range(1, len(returns) + 1))

    r_random = induced_result.get("R_random", None)
    r_expert = induced_result.get("R_expert", None)

    plt.figure()
    plt.plot(steps, returns, label="R_induced")
    if r_random is not None:
        plt.axhline(y=float(r_random), linestyle="--", label="R_random")
    if r_expert is not None:
        plt.axhline(y=float(r_expert), linestyle="--", label="R_ppo_best")
    plt.xlabel("training step")
    plt.ylabel("return")
    plt.title(f"T2 MPE induced returns (seed={seed})")
    plt.legend()

    out_dir = Path("outputs/plots")
    ensure_dir(out_dir)
    out_path = out_dir / f"t2_induced_returns_seed{seed}.png"
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


def plot_summary_err1b(summary_csv: str = "outputs/summary/metrics_summary.csv"):
    path = Path(summary_csv)
    if not path.exists():
        print("[plots] summary file not found; skip summary_err1b.")
        return
    df = pd.read_csv(path)
    if "err_1b" not in df.columns:
        print("[plots] err_1b column missing; skip summary_err1b.")
        return
    out_dir = Path("outputs/plots")
    ensure_dir(out_dir)

    labels_all = []
    means_all = []
    stds_all = []
    for task in sorted(df["task"].unique()):
        df_task = df[df["task"] == task]
        group_cols = ["algo"]
        if "mode" in df_task.columns:
            group_cols.append("mode")
        grouped = df_task.groupby(group_cols)["err_1b"].agg(["mean", "std"]).reset_index()
        labels = grouped.apply(lambda r: "-".join(str(r[c]) for c in group_cols if pd.notna(r[c])), axis=1)
        means = grouped["mean"].values
        stds = grouped["std"].fillna(0.0).values

        labels_all.extend([f"{task}:{lbl}" for lbl in labels])
        means_all.extend(means)
        stds_all.extend(stds)

        plt.figure(figsize=(8, 4))
        x = np.arange(len(labels))
        plt.bar(x, means, yerr=stds, capsize=4, color="#4c72b0")
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylabel("err_1b (mean ± std)")
        plt.title(f"{task} err_1b across seeds")
        plt.tight_layout()
        out_path = out_dir / f"summary_err1b_{task}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[plots] saved {out_path}")

    if labels_all:
        eps = 1e-12
        means_log = [m + eps for m in means_all]
        plt.figure(figsize=(10, 5))
        x = np.arange(len(labels_all))
        plt.bar(x, means_log, yerr=stds_all, capsize=4, color="#55a868")
        plt.xticks(x, labels_all, rotation=45, ha="right")
        plt.ylabel("err_1b (mean ± std)")
        plt.title("Summary err_1b across seeds (log scale)")
        plt.yscale("log")
        plt.tight_layout()
        out_path = out_dir / "summary_err1b.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[plots] saved {out_path}")


def plot_induced_summary(summary_csv: str = "outputs/summary/induced_summary.csv"):
    path = Path(summary_csv)
    if not path.exists():
        print("[plots] induced summary file not found; skip induced summary plot.")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("[plots] induced summary empty; skip.")
        return
    out_dir = Path("outputs/plots")
    ensure_dir(out_dir)
    metrics = ["R_random", "R_ppo_best", "R_induced_mean", "R_induced_last"]
    for task in sorted(df["task"].unique()):
        df_task = df[df["task"] == task]
        stats = {}
        for m in metrics:
            if m in df_task.columns and df_task[m].notna().any():
                stats[m] = (df_task[m].mean(), df_task[m].std())
        if not stats:
            continue
        labels = list(stats.keys())
        means = [stats[k][0] for k in labels]
        stds = [stats[k][1] for k in labels]
        plt.figure(figsize=(8, 4))
        x = np.arange(len(labels))
        plt.bar(x, means, yerr=stds, capsize=4, color="#c44e52")
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.ylabel("return (mean ± std)")
        plt.title(f"{task} induced returns summary")
        plt.tight_layout()
        out_path = out_dir / f"summary_induced_{task}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[plots] saved {out_path}")


def plot_t2_err1a_summary(summary_csv: str = "outputs/summary/metrics_summary.csv"):
    path = Path(summary_csv)
    if not path.exists():
        print("[plots] summary file not found; skip t2 err1a summary.")
        return
    df = pd.read_csv(path)
    df = df[df["task"] == "t2"]
    if df.empty or "err_1a" not in df.columns:
        print("[plots] t2 err_1a missing; skip.")
        return
    group_cols = ["algo"]
    if "mode" in df.columns:
        group_cols.append("mode")
    grouped = df.groupby(group_cols)["err_1a"].agg(["mean", "std"]).reset_index()
    labels = grouped.apply(lambda r: "-".join(str(r[c]) for c in group_cols if pd.notna(r[c])), axis=1)
    means = grouped["mean"].values
    stds = grouped["std"].fillna(0.0).values

    plt.figure(figsize=(8, 4))
    x = np.arange(len(labels))
        bars = plt.bar(x, means, yerr=stds, capsize=4, color="#4c72b0")
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel("err_1a (mean ± std)")
        plt.title("T2 err_1a across seeds")
        for rect, m, s in zip(bars, means, stds):
            plt.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                f"{m:.2e}±{s:.1e}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )
    plt.tight_layout()
    out_dir = Path("outputs/plots")
    ensure_dir(out_dir)
    out_path = out_dir / "t2_err1a_summary.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[plots] saved {out_path}")


def plot_t2_ratio_summary(summary_csv: str = "outputs/summary/metrics_summary.csv", eps: float = 1e-12):
    path = Path(summary_csv)
    if not path.exists():
        print("[plots] summary file not found; skip t2 ratio summary.")
        return
    df = pd.read_csv(path)
    df = df[df["task"] == "t2"]
    if df.empty or "err_1a" not in df.columns or "err_1b" not in df.columns:
        print("[plots] t2 ratio missing; skip.")
        return
    df = df.copy()
    df["ratio"] = df["err_1b"] / (df["err_1a"] + eps)
    group_cols = ["algo"]
    if "mode" in df.columns:
        group_cols.append("mode")
    grouped = df.groupby(group_cols)["ratio"].agg(["mean", "std"]).reset_index()
    labels = grouped.apply(lambda r: "-".join(str(r[c]) for c in group_cols if pd.notna(r[c])), axis=1)
    means = grouped["mean"].values
    stds = grouped["std"].fillna(0.0).values

    plt.figure(figsize=(8, 4))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=4, color="#8172b3")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("err_1b / err_1a (mean ± std, log scale)")
    plt.title("T2 ratio err_1b/err_1a across seeds")
    plt.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    plt.yscale("log")
    plt.tight_layout()
    out_dir = Path("outputs/plots")
    ensure_dir(out_dir)
    out_path = out_dir / "t2_ratio_summary.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"[plots] saved {out_path}")


def plot_induced_delta_summary(summary_csv: str = "outputs/summary/induced_summary.csv"):
    path = Path(summary_csv)
    if not path.exists():
        print("[plots] induced summary file not found; skip delta plot.")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("[plots] induced summary empty; skip delta plot.")
        return
    out_dir = Path("outputs/plots")
    ensure_dir(out_dir)
    for task in sorted(df["task"].unique()):
        df_task = df[df["task"] == task].copy()
        if df_task.empty:
            continue
        df_task["delta_ppo_best"] = df_task["R_ppo_best"] - df_task["R_random"]
        df_task["delta_induced_last"] = df_task["R_induced_last"] - df_task["R_random"]
        df_task["delta_induced_mean"] = df_task["R_induced_mean"] - df_task["R_random"]
        metrics = ["delta_ppo_best", "delta_induced_last", "delta_induced_mean"]
        labels = []
        means = []
        stds = []
        for m in metrics:
            if m in df_task.columns and df_task[m].notna().any():
                labels.append(m)
                means.append(df_task[m].mean())
                stds.append(df_task[m].std())
        if not labels:
            continue
        plt.figure(figsize=(8, 4))
        x = np.arange(len(labels))
        plt.bar(x, means, yerr=stds, capsize=4, color="#55a868")
        plt.xticks(x, labels, rotation=25, ha="right")
        plt.ylabel("Delta vs R_random (mean ± std)")
        plt.title(f"{task} induced deltas across seeds")
        plt.tight_layout()
        out_path = out_dir / f"summary_induced_delta_{task}.png"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"[plots] saved {out_path}")
