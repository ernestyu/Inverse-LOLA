"""Automatic Markdown report generation."""
from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf


def load_config(config_path: str):
    return OmegaConf.load(config_path)


def load_metrics_file(env_tag: str, algorithm: str, seed: int):
    path = Path("outputs/metrics") / f"{env_tag}_{algorithm}_seed{seed}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_report_t1(config_path: str, seed: int = 0):
    cfg = load_config(config_path)
    metrics = load_metrics_file("t1", "ma_lfl", seed)

    lines = []
    lines.append(f"# T1 GridWorld 报告 (seed={seed})")
    lines.append("")
    lines.append("## 实验配置")
    lines.append(f"- config: `{config_path}`")
    lines.append("- 算法: MA-LfL")
    lines.append("")
    lines.append("## 关键指标")
    lines.append(f"- err_1a = {metrics.get('err_1a', 0.0):.3e}")
    lines.append(f"- err_1b = {metrics.get('err_1b', 0.0):.3e}")
    diag = metrics.get("diagnostics", {})
    if "weight_pcc" in diag:
        lines.append(f"- weight_pcc = {diag['weight_pcc']:.3f}")
    if "weight_rmse" in diag:
        lines.append(f"- weight_rmse = {diag['weight_rmse']:.3f}")
    lines.append("")
    lines.append("## 图表")
    lines.append("")
    lines.append("1. KL 对比图：")
    lines.append("   ![](../plots/t1_kl_seed0.png)")
    lines.append("")
    lines.append("2. 奖励热力图：")
    lines.append("   ![](../plots/t1_reward_heatmap.png)")
    lines.append("")

    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"report_t1_seed{seed}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] saved {out_path}")


def generate_report_t2(config_path: str, seed: int = 0):
    cfg = load_config(config_path)
    metrics_ilola = load_metrics_file("t2", "ilola", seed)

    lines = []
    lines.append(f"# T2 MPE 报告 (seed={seed})")
    lines.append("")
    lines.append("## 实验配置")
    lines.append(f"- config: `{config_path}`")
    lines.append("- 算法: I-LOLA (Stage B)")
    lines.append("")
    lines.append("## 关键指标")
    lines.append(f"- err_1a = {metrics_ilola.get('err_1a', 0.0):.3e}")
    lines.append(f"- err_1b = {metrics_ilola.get('err_1b', 0.0):.3e}")
    lines.append(f"- final_loss = {metrics_ilola.get('final_loss', 0.0):.3e}")
    lines.append(f"- omega_norm = {metrics_ilola.get('omega_norm', 0.0):.3e}")
    lines.append("")
    lines.append("## 图表")
    lines.append("")
    lines.append("1. KL 对比图：")
    lines.append("   ![](../plots/t2_kl_compare_seed0.png)")
    lines.append("")
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"report_t2_seed{seed}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] saved {out_path}")


def generate_report_t3(config_path: str, seed: int = 0):
    cfg = load_config(config_path)
    metrics_ilola = load_metrics_file("t3", "ilola", seed)

    lines = []
    lines.append(f"# T3 MultiWalker 报告 (seed={seed})")
    lines.append("")
    lines.append("## 实验配置")
    lines.append(f"- config: `{config_path}`")
    lines.append("- 场景: 连续动作协作（MultiWalker）")
    lines.append("- 算法: I-LOLA (Stage B) + Monte Carlo KL 评估")
    lines.append("")
    lines.append("## 关键指标")
    lines.append(f"- err_1a = {metrics_ilola.get('err_1a', 0.0):.3e}  (连续动作下暂无 W_true 基线，固定为 0)")
    lines.append(f"- err_1b = {metrics_ilola.get('err_1b', 0.0):.3e}  (Monte Carlo KL，states≈256，action_samples=8)")
    lines.append(f"- final_loss = {metrics_ilola.get('final_loss', 0.0):.3e}")
    lines.append(f"- omega_norm = {metrics_ilola.get('omega_norm', 0.0):.3e}")
    lines.append("")
    lines.append("## 图表")
    lines.append("")
    lines.append("1. KL 柱状图：")
    lines.append("   ![](../plots/t3_kl_seed0.png)")
    lines.append("")
    lines.append("2. 诱导策略回报曲线（如已生成）：")
    lines.append("   ![](../plots/t3_induced_returns_seed0.png)")
    lines.append("")
    out_dir = Path("outputs/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"report_t3_seed{seed}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[report] saved {out_path}")


__all__ = ["generate_report_t1", "generate_report_t2", "generate_report_t3"]
