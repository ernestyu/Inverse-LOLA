"""Aggregate metrics and induced results across seeds into CSV/MD summaries."""
from __future__ import annotations

import json
import re
import csv
from pathlib import Path
from statistics import mean
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METRICS_DIR = Path("outputs/metrics")
INDUCED_DIR = Path("outputs/induced")
SUMMARY_DIR = Path("outputs/summary")
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)


def parse_seed(path: Path) -> int:
    m = re.search(r"_seed(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def parse_task_algo(path: Path) -> tuple[str, str]:
    stem = path.stem
    parts = stem.split("_")
    task = parts[0] if parts else "unknown"
    algo = "_".join(parts[1:]) if len(parts) > 1 else "unknown"
    return task, algo


def infer_seed_from_data_path(data_path: str) -> int | None:
    m = re.search(r"seed(\d+)", data_path)
    return int(m.group(1)) if m else None


def require_numeric(val, key: str, path: Path, required: bool = False):
    if val is None or (isinstance(val, float) and (val != val)):
        if required:
            raise SystemExit(f"[aggregate] missing required {key} in {path}")
        return None
    try:
        return float(val)
    except Exception:
        raise SystemExit(f"[aggregate] non-numeric {key} in {path}: {val}")


def aggregate_metrics() -> list[dict]:
    rows = []
    for p in METRICS_DIR.glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        seed_file = parse_seed(p)
        seed_data = data.get("seed", seed_file)
        seed_inferred = data.get("data_seed_inferred", seed_data)
        data_path = data.get("data_path", "")
        seed_from_path = infer_seed_from_data_path(str(data_path)) if data_path else seed_inferred
        if not all(s == seed_data for s in [seed_file, seed_data, seed_inferred, seed_from_path]):
            raise SystemExit(f"[aggregate] seed mismatch in {p}: file={seed_file}, data={seed_data}, inferred={seed_inferred}, from_path={seed_from_path}")

        task, _ = parse_task_algo(p)
        algo = data.get("algorithm", "unknown")
        rows.append(
            {
                "task": task,
                "algo": algo,
                "seed": int(seed_data),
                "err_1a": require_numeric(data.get("err_1a"), "err_1a", p),
                "err_1b": require_numeric(data.get("err_1b"), "err_1b", p, required=True),
                "err_1a_xphase": require_numeric(data.get("err_1a_xphase"), "err_1a_xphase", p),
                "err_1b_xphase": require_numeric(data.get("err_1b_xphase"), "err_1b_xphase", p),
                "err_1a_holdout": require_numeric(data.get("err_1a_holdout"), "err_1a_holdout", p),
                "err_1b_holdout": require_numeric(data.get("err_1b_holdout"), "err_1b_holdout", p),
                "final_loss": require_numeric(data.get("final_loss"), "final_loss", p),
                "omega_norm": require_numeric(data.get("omega_norm"), "omega_norm", p),
                "mode": data.get("mode", ""),
                "run_id": p.name,
                "data_path": data_path,
            }
        )
    return rows


def aggregate_induced() -> list[dict]:
    rows = []
    for p in INDUCED_DIR.glob("*.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        seed_file = parse_seed(p)
        seed_data = data.get("data_seed", data.get("seed", seed_file))
        if seed_data is None:
            seed_data = seed_file
        task, _ = parse_task_algo(p)
        curve = data.get("R_induced_curve") or []
        r_last = curve[-1][1] if curve and isinstance(curve[0], (list, tuple)) else (curve[-1] if curve else None)
        r_mean = (
            mean([c[1] if isinstance(c, (list, tuple)) else c for c in curve]) if curve else None
        )
        rows.append(
            {
                "task": task,
                "algo": "induced",
                "seed": int(seed_data),
                "R_random": require_numeric(data.get("R_random"), "R_random", p),
                "R_ppo_best": require_numeric(data.get("R_ppo_best", data.get("R_expert")), "R_ppo_best", p),
                "R_induced_last": require_numeric(r_last, "R_induced_last", p),
                "R_induced_mean": require_numeric(r_mean, "R_induced_mean", p),
                "run_id": p.name,
            }
        )
    return rows


def write_csv_md(rows: list[dict], fname: str) -> None:
    if not rows:
        return
    csv_path = SUMMARY_DIR / f"{fname}.csv"
    md_path = SUMMARY_DIR / f"{fname}.md"
    keys = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    md_lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for r in rows:
        md_lines.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[summary] wrote {csv_path} and {md_path}")


def main() -> None:
    metric_rows = aggregate_metrics()
    induced_rows = aggregate_induced()
    write_csv_md(metric_rows, "metrics_summary")
    write_csv_md(induced_rows, "induced_summary")
    # stats summary
    try:
        import pandas as pd
        from evaluation.plots import (
            plot_summary_err1b,
            plot_induced_summary,
            plot_t2_err1a_summary,
            plot_t2_ratio_summary,
            plot_induced_delta_summary,
        )
    except Exception as e:
        print(f"[summary] skip stats/plots (missing dependency/path): {e}")
        return

    def _group_stats(rows: list[dict], val_cols: list[str], fname: str):
        if not rows:
            return
        df = pd.DataFrame(rows)
        group_cols = ["task", "algo"]
        if "mode" in df.columns:
            group_cols.append("mode")
        stats = df.groupby(group_cols)[val_cols].agg(["mean", "std"])
        stats.columns = ["_".join(col).strip() for col in stats.columns.values]
        stats = stats.reset_index()
        csv_path = SUMMARY_DIR / f"{fname}.csv"
        md_path = SUMMARY_DIR / f"{fname}.md"
        stats.to_csv(csv_path, index=False)
        md_lines = ["| " + " | ".join(stats.columns) + " |", "| " + " | ".join(["---"] * len(stats.columns)) + " |"]
        for _, row in stats.iterrows():
            md_lines.append("| " + " | ".join(str(row[c]) for c in stats.columns) + " |")
        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        print(f"[summary] wrote {csv_path} and {md_path}")

    _group_stats(metric_rows, ["err_1b", "err_1a"], "metrics_stats")
    _group_stats(induced_rows, ["R_induced_last", "R_induced_mean", "R_random", "R_ppo_best"], "induced_stats")
    plot_summary_err1b()
    plot_induced_summary()
    plot_t2_err1a_summary()
    plot_t2_ratio_summary()
    plot_induced_delta_summary()


if __name__ == "__main__":
    main()
