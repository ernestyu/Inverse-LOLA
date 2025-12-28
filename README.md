# Project 9 — Multi-Agent Learning from Learners & I-LOLA

本仓库完成了 MA-LfL 与 I-LOLA 在三类多智能体环境上的复现与对比（T1 GridWorld，T2 MPE simple_spread，T3 SISL multiwalker_v9），并提供从数据生成、指标计算、诱导训练到多 seed 汇总与自动报告的全流程脚本。

---

## 环境准备

```bash
python -m venv .venv
.\.venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
python check_env.py              # 应输出 “Phase 0 OK”
```

所有脚本均使用当前激活的虚拟环境（`sys.executable`），无需手动指定解释器路径。

---

## 快速全流程（推荐，5 seeds）

1) **批量运行 T1/T2/T3**（默认 seeds = 0..4）：
```bash
python scripts/run_all_seeds.py
```
   - 每个 seed 会依次运行：数据生成 → T1 MA-LfL → T2 StageA/StageB/mismatch/induced → T3 ilola/induced。
   - 所有指标/权重/诱导结果分别落盘到 `outputs/metrics`、`outputs/weights`、`outputs/induced`。

2) **汇总表与统计图**（无需重跑实验）：
```bash
python scripts/aggregate_results.py
```
   - 生成汇总表：`outputs/summary/metrics_summary.csv|md`、`outputs/summary/induced_summary.csv|md`、均值/方差表 `metrics_stats.*`、`induced_stats.*`。
   - 生成统计图：
     - `outputs/plots/summary_err1b_t1.png`、`summary_err1b_t2.png`、`summary_err1b_t3.png`
     - `outputs/plots/summary_err1b_all_log.png`（全任务 log 轴概览）
     - `outputs/plots/t2_err1a_summary.png`、`t2_ratio_summary.png`（T2 1a/1b 比值对比）
     - `outputs/plots/summary_induced_t2.png`、`summary_induced_t3.png`
     - `outputs/plots/summary_induced_delta_t2.png`、`summary_induced_delta_t3.png`

3) **报告生成（可选）**：
```bash
python scripts/run_phase3_plots_reports.py
```
   - 生成分任务报告：`outputs/reports/report_t{1,2,3}_seed*.md`
   - 根目录提供综合稿件：`FINAL_REPORT.md` / `FINAL_REPORT.pdf`

---

## 单个 seed 调试（示例 seed=0）

T1：
```bash
python -m runners.gen_data --config configs/t1_gridworld.yaml --seed 0
python -m runners.run_ma_lfl --config configs/t1_gridworld.yaml --seed 0
```

T2：
```bash
python -m runners.gen_data --config configs/t2_mpe_simple_spread.yaml --seed 0
python scripts/run_t2_stageA_ilogel_eval.py --seed 0      # Stage A
python scripts/run_t2_ilogel_eval.py --seed 0              # Stage B (I-LOLA)
python scripts/run_t2_ma_lfl_mismatch.py --seed 0 --xphase --holdout 0.5
python scripts/run_t2_induced.py --seed 0                  # 诱导训练，包含 R_random / R_ppo_best / 曲线
```

T3：
```bash
python -m runners.gen_data --config configs/t3_multiwalker.yaml --seed 0
python scripts/run_t3_ilola_eval.py --seed 0
python scripts/run_t3_induced.py --seed 0                  # 写出 R_ppo_best、随机/诱导曲线
```

调试辅助：`scripts/debug_t1_ma_spi.py`、`scripts/debug_t2_ppo.py`、`scripts/debug_t3_ppo.py`、`scripts/debug_feature_maps.py`。

---

## 结果目录速览

- `outputs/data/`：各任务的 LearningPhaseData（文件名含 seed）。
- `outputs/metrics/`：指标 JSON（含 `data_path`、`data_seed_inferred`、`mode`）。
- `outputs/induced/`：诱导训练结果（含 `R_random`、`R_ppo_best`、`R_induced_curve`）。
- `outputs/summary/`：多 seed 汇总表与均值/方差表。
- `outputs/plots/`：
  - KL 对比与热力图（按任务/seed）
  - 多 seed 统计图（err_1b, err_1a, 1b/1a 比值）
  - 诱导回报汇总与相对提升（delta）
- `FINAL_REPORT.md` / `.pdf`：整合论文式报告；`Project 9 实验与工程进展报告.md` 为过程记录。

---

## 关键结论（摘要版）

- **T1（匹配 MA-SPI）**：MA-LfL 1a/1b 均贴近零，验证匹配设定下的正确性。
- **T2（PPO 学习者，错配）**：StageA/StageB 的 1b 与 1a 同量级，但 mismatch 出现“1a 高、1b 极小”，表明 KL 指标在错配下退化；诱导回报与随机基线相比无显著提升。
- **T3（连续动作）**：I-LOLA 1b 在 1e-2 量级，诱导回报与随机/PPO 基线相当，提供连续动作场景的参考基线。

更多细节、图表与讨论见 `FINAL_REPORT.md` 与 `outputs/plots`。

---

## 常见问题

1. **缺包/找不到模块**：确认已激活 `.venv` 并执行 `pip install -r requirements.txt`。所有脚本使用 `sys.executable`，无需手动改路径。
2. **汇总缺柱子/空图**：确保 metrics JSON 中 `algorithm`、`mode` 存在，且未把 seed 拼入 `algo`；缺失时重跑对应脚本或补写字段后再运行 `scripts/aggregate_results.py`。
3. **数据串用防护**：所有脚本会检查 `--seed` 与 `data_path` 中解析的 seed 是否一致；若不一致会直接报错，避免跨 seed 混淆。握有自定义数据时请保持文件名包含 `seed{N}`。

---

## 贡献与扩展

欢迎基于当前框架扩展：
- 强化 T2 PPO 训练强度，检验更强行为策略下的 reward recovery；
- 增加新环境或特征映射，对比更丰富的错配场景；
- 将统计图与报告进一步包装成 CLI 或 CI 流水线，便于复现与评分。收到问题可直接提 Issue/PR。
