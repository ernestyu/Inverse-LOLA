下面是一份你可以直接复制到新对话里的“交接文档”。它尽量只保留关键信息：我们做到了哪里、现在卡在哪、下一步要做什么、每一步怎么验收。你把它贴进新对话，新的助手就能快速接手，不需要再翻很长的历史。

---

# Project 9 交接文档（给新对话 / 新助手）

## 0. 项目目标（执行计划的核心）

项目要完成两条主线，并用三层环境 T1/T2/T3 做验证：

1. **基线复现与边界验证**：MA-LfL（假设学习者是 MA-SPI）

   * 在匹配数据（MA-SPI 生成）上应表现好（sanity）
   * 在错配数据（PPO 生成）上应失败，失败原因应归因于模型错配 + 计算不可行（T3）

2. **新算法 I-LOLA**：把 LOGEL 思路扩展到多智能体并引入 LOLA 耦合动态（实现上用黑盒优化 CMA-ES + 一阶 LOLA 模拟器）

   * 在 PPO 数据上可运行、可评估，并在 T2/T3 给出“比错配基线更可信”的结果
   * 用“分解式预测误差”指标：1a（模型天花板，用真奖励）与 1b（恢复奖励）

最终交付：可复现实验链路 + 自动 plots/report + 用 checklist 跑出最终结果。

---

## 1. 当前工程状态（已完成的部分）

### 1.1 基础设施（Phase 0）已完成

* Windows + venv 环境 OK
* 模块导入检查脚本 `check_env.py` 通过
* 目录结构、配置系统、输出路径已固定

### 1.2 数据生成（Phase 1）已完成

* **T1 GridWorld（MA-SPI 轨道）**：`runners.gen_data --config configs/t1_gridworld.yaml`
* **T2 MPE simple_spread（PPO 轨道）**：`configs/t2_mpe_simple_spread.yaml`
* **T3 MultiWalker（PPO 轨道）**：`configs/t3_multiwalker.yaml`
  输出为 `outputs/data/t{1,2,3}/...pkl`，并有 debug 脚本验证结构正确。

### 1.3 核心算法链路（Phase 2）已跑通

* **T1 MA-LfL（匹配 sanity）**：1a≈1b，KL 很小（预期）
* **I-LOGEL / I-LOLA**：CMA-ES 能在 T2/T3 产出 (\hat W)，并保存 metrics/weights

注意：之前修过 feature map/rbf 等问题，现已能产生合理 feature range（不是全 0）。

### 1.4 Phase 3（metrics/plots/report/induced）基本跑通

* `evaluation/metrics.py`：支持离散 KL（T1/T2）与 T3 Monte Carlo KL（连续）
* `evaluation/plots.py`：生成 t1_kl、t1 reward heatmap、t2 compare、t3 kl、t3 induced returns 等
* `evaluation/report.py`：输出 `outputs/reports/report_t{1,2,3}_seed0.md`
* induced 最小版本已跑通：T2/T3 可以输出 induced curve（仍偏简化，expert baseline 有时是 NaN 或 placeholder，后来修到不再触发 numpy mean empty slice warning）

---

## 2. 当前关键问题（需要在新对话继续解决）

### 2.1 关键矛盾：T2 的 MA-LfL mismatch 结果出现“1b≈0”

我们开始做 **Step 5.1：T2 mismatch 对比**，即在 PPO 数据上运行 MA-LfL（错模型）作为失败证据。

现象：

* `MA-LfL mismatch err_1a = 0.301886`（大，合理：用真奖励的模型天花板误差很高）
* 但 `MA-LfL mismatch err_1b ≈ 1e-11`（几乎 0，不符合“错配应失败”的直觉）
* debug 检查一显示：不是短路（不共享内存），而是 **(\pi_{t+1}) 本身接近均匀**，并且 `pi_hat_hatW` 与 `pi_real` 极接近（max diff 1e-6）。
* `w_hat_norm` 很小（0.010），`w_true_norm` 大约 9（注：此处“w_true”并非环境真实奖励，只是实现里某个参考权重/诊断值）

检查一输出（已确认）：

* `pi_real[:5] ≈ [0.2004,0.2003,0.2003,0.1996,0.1994]`（高熵接近均匀）
* `pi_ref_true` 很尖（不均匀）
* `pi_hat_hatW` 与 `pi_real` 仅差 1e-6
* 因此 KL( real || hat ) 自然会到 1e-11 量级

结论：现在不像是实现 bug，而更像是 **证据口径不够强**：在同一批 state 上，MA-LfL 可以通过“输出接近均匀”的 (\hat W) 拟合一步，从而 KL 很小。

### 2.2 需要把 “mismatch 失败证据”补强成可写进最终报告的形式

单步同批 state 的 1b 不能作为失败证据。需要改成：

* **留出评估（holdout states）**：train 用一半 states 拟合 (\hat W)，test 在另一半算 KL
* **跨时间泛化（cross-phase）**：用 (0→1) 拟合的 (\hat W) 去预测 (1→2) 的下一步
* **任务层证据（induced compare）**：用 mismatch 的 (\hat W) 做 induced，和 I-LOLA 的 induced 对比（mismatch 应更接近 random）

---

## 3. 新对话下一步要做什么（明确的行动清单）

### 3.1 立刻要做：解释 “1b≈0”的原因（可选但推荐）

在 mismatch 脚本里加两项诊断：

* `H(pi_real_next)`（熵）
* `KL(pi_real_next || Uniform)`（离均匀程度）
  验收：如果 `H` 接近 `log(5)` 且 `KL_to_uniform` 很小，说明 “pi_real 高熵” 是根因。

### 3.2 必做：把 mismatch 的评估改成“留出/泛化”

优先顺序建议：先做 cross-phase（更强），再做 holdout（更干净）。

**A) Cross-phase（强证据）**

* fit: 用 phase 0→1 数据拟合 (\hat W_0)
* test: 固定 (\hat W_0)，用 phase 1 的 (\pi_1) lookahead 预测 (\pi_2)，算 KL
  输出：`err_1b_xphase`
  验收预期：`err_1b_xphase` 不应再是 1e-11；应显著大于 I-LOLA 的 1b，且更接近 MA-LfL 的 1a（或至少明显上升）。

**B) Holdout states（补充证据）**

* 对同一对 (t→t+1)，用 states batch split：train half 拟合 (\hat W)，test half 算 KL
  输出：`err_1b_train`, `err_1b_holdout`
  验收预期：holdout KL 明显大于 train KL（或至少不再接近 0）。

### 3.3 必做：mismatch 的 induced 对比（任务层证据）

* 读取 mismatch 输出的 (\hat W)
* 在同一训练预算下做 induced（和 I-LOLA 一样的流程）
* 生成 compare 图：`t2_induced_compare_seed0.png`
  验收预期：mismatch 的 induced 更接近 random，I-LOLA 更接近 expert。

### 3.4 更新 plots/report

* plots：新增或更新 `t2_kl_compare_mismatch_seed0.png`，包含新的 `err_1b_xphase` 或 `holdout` 结果
* report：在 T2 报告里加入“为什么单步 1b 可很小（高熵策略）”以及“泛化/诱导失败才是错配证据”的段落

---

## 4. 运行入口与现有关键输出文件

### 常用命令

* 数据：

  * `python -m runners.gen_data --config configs/t2_mpe_simple_spread.yaml --seed 0`
* I-LOLA（T2/T3）：

  * `python scripts/run_t2_ilogel_eval.py`
  * `python scripts/run_t3_ilola_eval.py`
* MA-LfL mismatch（当前）：

  * `python scripts/run_t2_ma_lfl_mismatch.py --seed 0 [--debug]`
* 统一出图与报告：

  * `python scripts/run_phase3_plots_reports.py`
* induced（当前最小版本）：

  * `python scripts/run_t2_induced.py`
  * `python scripts/run_t3_induced.py`

### 关键输出目录

* `outputs/data/`：序列数据 pkl
* `outputs/metrics/`：t1/t2/t3 json
* `outputs/weights/`：保存权重
* `outputs/plots/`：png
* `outputs/reports/`：md
* `outputs/induced/`：induced json

---

## 5. 当前 5.1 的状态判定（给新助手的结论）

* **工程产物齐全**：mismatch metrics/json、mismatch plot、t2 report 都能生成
* **但科学结论尚未达标**：单步同批 state 的 `MA-LfL mismatch 1b≈0` 不能作为“错配失败证据”，必须改成 holdout/cross-phase/induced compare 才能写入最终报告

---

## 6. 需要新助手特别注意的点（避免重复走弯路）

1. 不要把“单步拟合成功（1b小）”当成错配成功；错配要看**泛化**和**任务表现**。
2. mismatch 的 (\hat W) 很小并不奇怪：高熵策略下，输出近似均匀的策略很容易。
3. 所有 KL 对比必须保证同一批 states（对 1a/1b）；但对 mismatch 证据，需要刻意加入 holdout/cross-phase。
4. PettingZoo `mpe` 有 deprecation warning（mpe2），目前可忽略，不要让 `-W error` 阻塞主链路。

---

