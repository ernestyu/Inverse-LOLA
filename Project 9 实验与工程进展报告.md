# Project 9 实验与工程进展报告（代码框架版）

**状态概览（截至当前代码）**

本报告总结目前仓库中已经实现和跑通的内容，并对照《Project 9 执行计划》（v3.0 Final）说明哪些目标已经满足，哪些仍然是后续工作的重点。最后给出一份带有操作步骤的 checklist，方便后续按条逐项完成。

---

## 1. 项目目标回顾（简要）

Project 9 的总体目标可以概括为三点：

1. 在多智能体设定下，严格复现并“解剖” MA-LfL，明确其在假设成立与失配时的边界。
2. 设计并实现一个基于策略梯度学习者的逆向算法 I-LOLA，作为 LOGEL 的多智能体扩展版本。
3. 在三个等级环境上进行系统对比：
   - T1：GridWorld（两智能体，离散动作，MA-SPI 匹配设定）。
   - T2：MPE simple_spread（多智能体，离散动作，PPO 学习者）。
   - T3：MultiWalker（多智能体，连续动作，PPO 学习者，MA-LFL 被视为计算上不可行）。

核心评估指标按执行计划分为两个层次：

- 指标 1：策略预测误差，进一步拆分为  
  1a）“固有基线误差”，1b）“使用恢复奖励后的误差”。  
- 指标 2：诱导策略性能，使用恢复的奖励从头训练新策略，并与随机策略和专家策略对比。

---

## 2. 已完成的工程模块与实验闭环

这一部分总结目前代码已经构成的闭环，按 T1/T2/T3 分别说明。

### 2.1 T1 GridWorld：MA-SPI + MA-LFL + I-LOGEL / I-LOLA

当前状态：

- `data_gen.ma_spi.gridworld_sampler` 可以在 3×3 GridWorld 上生成多阶段 MA-SPI 轨迹，并保存到 `outputs/data/t1/...pkl`。
- `runners.run_ma_lfl` 在 T1 上运行完 MA-LFL，输出
  - 1a 和 1b 的 KL，数量级约为 \(9\times10^{-6}\)；
  - 特征维度信息和奖励权重诊断指标（PCC 为负、RMSE 在 0.26 左右）。
- `scripts/run_t1_ilogel_sanity.py` 已在合成数据和真实 MA-SPI 数据上验证了 I-LOGEL：
  - 合成数据上，恢复的奖励方向与真值余弦相似度接近 1；
  - 实际 T1 数据上可以得到合理的权重范数和损失曲线。
- `scripts/run_t1_ilola_eval.py`（或等价逻辑）已在 T1 上测试 I-LOLA 的一小步版本，并输出 1a/1b 指标。
- `evaluation.metrics` 和 `evaluation.plots` 支持：
  - T1 上的 KL 直方图；
  - T1 上“真实奖励 vs 恢复奖励”的热力图。
- `evaluation.report` 会自动读取 T1 的指标和图表，生成 `outputs/reports/report_t1_seed0.md`。

这意味着：对于 T1，**从数据生成 → 奖励恢复 → 指标计算 → 图表与报告** 的完整代码路径已经跑通。

### 2.2 T2 MPE simple_spread：PPO + I-LOLA + induced 训练

当前状态：

- `runners.gen_data` 在 `configs/t2_mpe_simple_spread.yaml` 下，可以通过自实现的 PPO 训练 MPE simple_spread，并把多个更新阶段的参数和轨迹打包到 `outputs/data/t2/...pkl`。
- `models.feature_maps` 中针对 T2 的特征映射已经修复，维度为 23，数值范围合理。
- `scripts/run_t2_ilogel_eval.py` 实现了：
  - 读取 T2 数据；
  - 使用 I-LOLA 的阶段 B（CMA-ES）优化，得到一组奖励权重 `w_hat`；
  - 计算 T2 对应的 1a 和 1b：
    - 1a 原始值约 \(1.0\times10^{-6}\)；
    - 1b 原始值约 \(1.1\times10^{-6}\)（最新版本）；
  - 输出额外诊断信息，例如 \(\|w_{\text{true}}\|\)、\(\|w_{\text{hat}}\|\)、\(\|\theta_{\text{hat,true}} - \theta_{\text{hat,pred}}\|\)。
- `evaluation.metrics` 中 T2 的 KL 计算已经集成到 `run_t2_ilogel_eval.py`，并把结果写入 `outputs/metrics/t2_ilola_seed0.json`。
- `evaluation.induced_train` 中已经实现了 T2 的最小诱导训练版本：
  - 使用 `w_hat` 构造奖励网络；
  - 根据现有配置训练一个新策略；
  - 随机策略回报 `R_random` 会被估计；
  - 诱导策略回报曲线 `R_induced_curve`（训练步数 → 平均回报）已经写入 `outputs/induced/t2_induced_seed0.json`。
- `evaluation.plots` 可以绘制：
  - T2 上 I-LOLA 1a/1b KL 比较的柱状图；
  - T3 的 induced 曲线也复用同一接口。
- `evaluation.report` 已经支持生成 `report_t2_seed0.md`，自动插入 KL 图和关键信息。

这一部分说明：在 T2 上，**I-LOLA 的阶段 B + 指标 1 + 指标 2（诱导训练）** 已有一个可运行闭环。  
目前缺少的是：**MA-LFL 在 T2 上的完整基线实现和 I-LOGEL 阶段 A 与阶段 B 的系统对比**。

### 2.3 T3 MultiWalker：PPO + I-LOLA + induced 训练（连续动作）

当前状态：

- `runners.gen_data` 在 `configs/t3_multiwalker.yaml` 下，可以通过自实现 PPO 训练 MultiWalker，并保存多阶段参数到 `outputs/data/t3/...pkl`。
- `scripts/run_t3_ilola_eval.py` 已经实现：
  - 从 T3 PPO 轨迹中提取数据；
  - 使用 I-LOLA 阶段 B（CMA-ES）优化奖励权重；
  - 使用采样的方式估计连续动作上的 KL（指标 1b 的 Monte Carlo 版本）：
    - 最新版本输出中，1b 约为 \(5\times10^{-4}\)，同时打印了某个状态下的样本 KL（约 0.0035）；
  - 把结果写入 `outputs/metrics/t3_ilola_seed0.json`。
- 在 T3 上不会尝试 MA-LFL，这与执行计划中“视为计算不可行，不作为实现目标”的设定一致。
- `evaluation.induced_train` 已经支持 T3：
  - 使用 T3 的 `w_hat` 训练一个新策略；
  - 估计随机策略回报 `R_random` 和诱导策略回报序列 `R_induced_curve`；
  - 结果写入 `outputs/induced/t3_induced_seed0.json`。
- `evaluation.plots` 已可以绘制 T3 的 induced 曲线，例如当前的曲线大致在随机基线附近小幅波动。
- `evaluation.report` 已经生成 `report_t3_seed0.md`，包含 KL 和 induced 曲线的信息。

综合来看，在 T3 上，**“PPO 轨道 + I-LOLA + 连续动作 KL（Monte Carlo）+ 诱导训练”** 这条路径已经跑通，足以支撑“可行性示范”和“MA-LFL 在此设定下计算不可行”的结论。

---

## 3. 当前结果的初步解读（对照执行计划）

这一部分不是最终论文结论，只是对目前单个 seed 的结果做一个“方向性”对照。

### 3.1 T1：匹配设定下 MA-LFL 的表现

在 T1 上：

- 1a 和 1b 的 KL 都在 \(10^{-5}\) 量级且非常接近，这符合执行计划中“匹配设定下 MA-LFL 能恢复奖励”的预期。
- 奖励热力图表明：真实奖励只在一个格子非零，而恢复奖励呈现一条从某一侧向目标格子逐渐递增的形状。这体现了**奖励整形等价类**的现象：策略行为几乎一致，但奖励权重不一定点对点重合。
- 权重 PCC 为负，RMSE 不小，但对比 1a/1b 接近这一事实，可以在报告中解释为“形状不同但策略等价”。

总体上，T1 已经可以支撑执行计划里的第一步结论：**在假设完全匹配的小环境下，MA-LFL 和 I-LOLA 都可以达到很低的策略预测误差**。

### 3.2 T2：失配设定下 I-LOLA 的表现

在 T2 上：

- I-LOLA 的 1a/1b 都在 \(10^{-6}\) 量级，且差距不大。这说明在当前设置下，基于策略梯度的 I-LOLA 可以在 PPO 数据上达到与“理想基线”接近的预测误差。
- 诱导训练的结果显示，使用恢复奖励重新训练得到的策略，其回报大致略优于随机策略，曲线有一定波动，说明训练仍然受超参数和样本数影响。
- 由于 MA-LFL 基线在 T2 上尚未系统实现，目前还不能展示“MA-LFL 失配 → 1b 明显劣于 I-LOLA”的对比图，这是后续实验的重要部分。

从执行计划视角看，T2 已经完成了“在失配设定下跑通 I-LOLA 的代码和指标”，但**尚未完成“与 MA-LFL 的系统对比”这一关键实验任务**。

### 3.3 T3：连续控制下 I-LOLA 的可行性

在 T3 上：

- I-LOLA 已经可以在 MultiWalker 上稳定运行，并输出有限的 KL 和诱导训练曲线，这表明在计算上是可行的。
- 目前 1b 约为 \(5\times 10^{-4}\)。由于缺少 T3 上的“理想基线”（1a），这一数值主要作为绝对参考，而非相对差距。
- 诱导训练的曲线与随机基线非常接近，略有起伏，说明目前的特征设计和训练预算下，恢复的奖励对最终行为的影响还比较有限。

这一部分基本实现了执行计划中的要求：**展示 I-LOLA 在连续控制环境下的工程可行性，同时保留 MA-LFL 在该设定下“计算不可行”的负面结果。**

---

## 4. 与执行计划的差距与后续方向概览

对照《Project 9 执行计划》和《工程实施计划》，当前代码框架已经满足：

1. T1：MA-SPI + MA-LFL + I-LOGEL + I-LOLA 的完整闭环；
2. T2/T3：基于 PPO 的数据生成和 I-LOLA 的 Stage B 优化；
3. 三个环境的指标 1（1a/1b）、部分指标 2（诱导训练）、以及自动图表和 Markdown 报告生成。

仍然有几个重要方向尚未完成或只做了最小版本：

1. **T2 上 MA-LFL 与 I-LOLA 的系统对比**：需要在同一数据集上跑 MA-LFL，并将 1a/1b 与 I-LOLA 的结果并列展示。
2. **Stage A（I-LOGEL）与 Stage B（I-LOLA）的对比**：特别是在 T2 上，需要验证“联合优化（Stage B）优于独立近似（Stage A）”这一关键主张。
3. **多 seed 实验与统计结果**：当前基本都是 seed 0，需要多 seed 重复和均值 ± 置信区间。
4. **T2.5 扩展实验（可选）**：执行计划中提出的简单扩展版本，目前尚未在代码中体现。
5. **文档与论文写作**：需要把上述结果整理成正式论文式结构，并补充理论分析与图表说明。

下面给出一份 checklist，分条列出这些后续工作，并给出建议的具体操作步骤。

---

## 5. 后续工作 checklist 与操作步骤

### 5.1 T2：补齐 MA-LFL 基线并与 I-LOLA 对比

**目标**：在 T2 上获得 MA-LFL 的 1a/1b，并与当前 I-LOLA 的结果并列，验证“模型失配导致 MA-LFL 表现较差”的假设。

**建议步骤：**

1. 在 `runners.run_ma_lfl` 中，增加对 `env=mpe_simple_spread` 的分支：
   - 复用 T1 的接口，但改为从 `outputs/data/t2/...pkl` 读取 PPO 轨迹；
   - 注意把 MA-LFL 所需的 MA-SPI 相关参数用“近似”的方式代替（这一点在论文中可以解释为“刻意使用错误模型”）。
2. 扩展 `evaluation.metrics`：
   - 为 T2 的 MA-LFL 输出单独的 `err_1a`, `err_1b`；
   - 把结果写入 `outputs/metrics/t2_ma_lfl_seedX.json`。
3. 在 `evaluation.plots` 新增一个函数：
   - 读取 `t2_ma_lfl_seed0.json` 和 `t2_ilola_seed0.json`；
   - 画出四根柱子：`LFL-1a`, `LFL-1b`, `ILOLA-1a`, `ILOLA-1b`。
4. 更新 `evaluation.report` 中的 T2 报告：
   - 插入新的对比图；
   - 在文字说明中强调：MA-LFL 使用 MA-SPI 模型，在 PPO 数据上存在结构性失配。

完成后，你就有了“失配设定下 MA-LFL vs I-LOLA”的核心对比图。

---

### 5.2 Stage A vs Stage B：I-LOGEL 与 I-LOLA 的对比（重点在 T2）

**目标**：验证“联合优化（I-LOLA 阶段 B）优于独立近似（阶段 A）”。

**建议步骤：**

1. 在 `inverse.ilola.stage_a_independent` 的接口上，再包装一个运行脚本，例如 `scripts/run_t2_stageA_eval.py`：
   - 读取 T2 数据；
   - 使用 Stage A 得到 `w_hat_stageA`；
   - 调用与 I-LOLA 相同的 metrics 函数，输出 1a/1b。
2. 在 `evaluation.metrics` 中，为 Stage A 添加单独的 `err_1a_stageA`, `err_1b_stageA` 字段。
3. 在 `evaluation.plots` 中：
   - 增加一个“StageA vs StageB”的对比图（至少在 T2 上）；
   - 可以画三组柱子：StageA-1b, StageB-1b, LFL-1b。
4. 在 T2 报告中增加一小节文字说明，解释 Stage A 的近似假设，以及 Stage B 利用 CMA-ES 进行联合优化的优势。

这样可以直接回应执行计划中对“Stage B 带来额外收益”的论断。

---

### 5.3 多 seed 实验与统计结果

**目标**：从“单次跑通”升级为“可复现的统计结论”，符合论文级别的严谨性。

**建议步骤：**

1. 在 `scripts/` 目录下新增简单的多 seed 脚本，例如：
   - `scripts/run_all_t1_seeds.py`
   - `scripts/run_all_t2_seeds.py`
   - `scripts/run_all_t3_seeds.py`
   这些脚本循环不同的 `--seed` 值，依次调用 `runners.gen_data`、`run_ma_lfl`、`run_t2_ilogel_eval` 等。
2. 编写一个汇总脚本，例如 `scripts/aggregate_metrics.py`：
   - 扫描 `outputs/metrics` 中的 `*_seed*.json`；
   - 把每个环境、每个算法、每个 seed 的 1a/1b 汇总为一个表格；
   - 计算均值和标准差，写入 `outputs/metrics_summary/*.csv`。
3. 在 `evaluation.plots` 中增加“均值±标准差”的可视化函数，用于画误差条形图或带阴影的曲线。
4. 在各环境的报告中，增加一节“Across seeds”的统计结果。

---

### 5.4 T2.5 扩展实验（可选）

**目标**：在 MPE 上构造一个略复杂的扩展版，用来展示脚手架的可扩展性。

**建议步骤：**

1. 在 `configs/` 中新增 `t25_mpe_extended.yaml`：
   - 在 T2 基础上调整智能体数量、观测裁剪或 landmark 的生成方式；
   - 保持接口与 T2 相同（方便复用代码）。
2. 在 `envs/` 中视情况增加一个轻微改造后的包装器，或直接在配置中使用 PettingZoo 提供的其它变体。
3. 复用 T2 的数据生成、I-LOLA 和 induced 训练脚本，只修改 `--config`。
4. 为 T2.5 增加单独的报告和图表，以展示：
   - 1a/1b 的数值；
   - induced 曲线的形状；
   - 与 T2 的对比。

---

### 5.5 文档与论文写作整理

**目标**：把工程结果转化为论文或项目报告级别的文字稿。

**建议步骤：**

1. 在仓库根目录编写一个面向读者的 `README.md`：
   - 简要说明项目目的；
   - 给出“从零到结果”的最小命令序列（例如只跑 T1 和 T2）；
   - 指出所有重要输出目录（data, metrics, plots, reports）。
2. 在 `outputs/reports/` 中，把当前的 `report_tX_seed0.md` 视为“附录级自动报告”，在此基础上写一个高层次的 `project9_main_report.md`：
   - 结构可以按执行计划的四个部分来写；
   - 中间用图片链接引用 `outputs/plots` 里的图。
3. 把执行计划中的理论分析片段（MA-LFL 边界、I-LOLA 设计动机等）抽取出来，整理成论文式的“Related Work / Problem Setup / Method / Experiments”结构。

---

### 5.6 代码清理与复现环境固化

**目标**：让别人拉取仓库后可以按说明文档无障碍复现。

**建议步骤：**

1. 检查所有脚本的入口和默认参数：
   - 确保没有硬编码的绝对路径；
   - 确保默认 `--config` 指向仓库内部的 YAML。
2. 把当前 `.venv` 中的依赖整理为固定版本的 `requirements.txt`，已经基本完成，可以再检查一次是否有多余库。
3. 可选：写一个简单的 `run_all_minimal.sh` 或 `run_all_minimal.ps1`，串起：
   - `python check_env.py`
   - `python -m runners.gen_data ...`（T1/T2/T3）
   - `python scripts/run_tX_*.py`
   - `python scripts/run_phase3_plots_reports.py`
4. 若未来考虑交付给助教或评审，可以再补一个 `docker/Dockerfile`，记录完整环境。

---

## 6. 小结

从工程角度看，当前仓库已经完成了 Project 9 的**核心代码框架和单 seed 实验闭环**：

- T1：完全匹配设定下的 MA-LFL 和 I-LOLA；
- T2：PPO 轨道上的 I-LOLA + 诱导训练；
- T3：连续控制环境上的 I-LOLA 可行性示范。

接下来最重要的工作，不再是“能不能跑起来”，而是**系统地对比不同算法、在多个 seed 上收集稳定统计结果，并把这些结果写成清晰的理论和实验故事**。上面的 checklist 可以作为后半程的路线图，你可以按照自己的时间安排，先从 T2 的 MA-LFL 基线和 Stage A vs Stage B 对比开始，一步一步补齐整套故事。
