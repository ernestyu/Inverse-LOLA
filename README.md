---
title: "Project 9 - Multi-Agent RL"
emoji: "🤖"
colorFrom: "blue"
colorTo: "purple"
sdk: "gradio"
pinned: false
---
# Project 9 — Multi-Agent Learning from Learners and I-LOLA

本仓库是 Project 9 的工程实现代码，用来研究：

1. **MA-LfL**（Multi-Agent Learning from Learners）在多智能体环境中的适用边界；
2. **I-LOLA**（Inverse LOLA）这一基于策略梯度学习者的逆向算法，作为对 LOGEL 的多智能体扩展；
3. 在三个层次的环境上对比不同方法的表现：

   - **T1：GridWorld**（两智能体，离散动作，MA-SPI 匹配设定）  
   - **T2：MPE `simple_spread`**（多智能体，离散动作，PPO 学习者）  
   - **T3：SISL `multiwalker_v9`**（多智能体，连续动作，PPO 学习者）

仓库已经完成从数据生成、逆向算法、指标计算到图表和自动报告的基本闭环。下面说明如何安装环境、运行各阶段脚本，以及数据与结果的目录结构。

---

## 1. 环境准备

### 1.1 克隆仓库并创建虚拟环境

```bash
git clone <your_repo_url> Project9
cd Project9

# 创建并激活虚拟环境（Windows PowerShell）
python -m venv .venv
.\.venv\Scripts\activate
````

### 1.2 安装依赖

仓库中已经提供固定版本的依赖文件：

```bash
pip install -r requirements.txt
```

### 1.3 检查环境

运行环境检查脚本，确认依赖库和项目模块均能导入：

```bash
python check_env.py
```

输出中应包含：

* 各主要库版本（torch、numpy、pettingzoo、gymnasium 等）；
* 各模块导入状态 `OK`；
* 最后一行 `Phase 0 OK`。

如果某个模块导入失败，请先确认文件路径与包名是否保持仓库默认结构。

---

## 2. 目录结构

项目的核心目录大致如下（省略部分无关文件）：

```text
Project9/
├─ configs/              # 各环境的配置 (T1/T2/T3)
│   ├─ t1_gridworld.yaml
│   ├─ t2_mpe_simple_spread.yaml
│   └─ t3_multiwalker.yaml
├─ envs/                 # 环境封装 (GridWorld, MPE, MultiWalker)
├─ learners/
│   ├─ ma_spi/           # T1: MA-SPI 采样器
│   └─ ppo/              # T2/T3: PPO 训练脚本
├─ data_gen/
│   ├─ ma_spi/           # T1 数据生成
│   └─ custom_ppo/       # T2/T3 PPO 数据生成
│   └─ adapters.py       # 统一的 LearningPhaseData 数据结构
├─ models/
│   ├─ reward_nets.py    # 奖励网络 (线性 / RBF 等)
│   ├─ feature_maps.py   # 各环境的特征映射
│   └─ dynamics.py       # I-LOLA 前向模拟器
├─ inverse/
│   ├─ ma_lfl/           # MA-LfL 相关核心逻辑
│   └─ ilola/            # I-LOGEL (Stage A) 与 I-LOLA (Stage B)
├─ algorithms/
│   ├─ baseline_lfl.py   # 工程视角的 MA-LfL 封装
│   ├─ i_logel.py        # I-LOGEL 封装
│   └─ i_lola.py         # I-LOLA 封装
├─ evaluation/
│   ├─ metrics.py        # KL、PCC、RMSE 等指标
│   ├─ plots.py          # 绘制 KL、奖励热力图、诱导回报曲线
│   ├─ report.py         # 生成 Markdown 报告
│   └─ induced_train.py  # 使用恢复奖励进行诱导训练
├─ runners/
│   ├─ gen_data.py       # 统一数据生成入口
│   ├─ run_ma_lfl.py     # T1: MA-LfL 指标计算
│   ├─ run_ilogel.py     # I-LOGEL 评估入口 (如需要)
│   ├─ run_ilola.py      # 通用 I-LOLA 入口 (如需要)
│   ├─ run_induced.py    # 统一诱导训练入口 (如需要)
│   └─ ...
├─ scripts/              # 调试与单场景运行脚本
│   ├─ debug_t1_ma_spi.py
│   ├─ debug_t2_ppo.py
│   ├─ debug_t3_ppo.py
│   ├─ debug_feature_maps.py
│   ├─ run_t1_ilogel_sanity.py
│   ├─ run_t2_ilogel_eval.py
│   ├─ run_t2_induced.py
│   ├─ run_t3_ilola_eval.py
│   ├─ run_t3_induced.py
│   └─ run_phase3_plots_reports.py
└─ outputs/
    ├─ raw/              # 训练时保存的中间 PPO checkpoint 等
    ├─ data/             # LearningPhaseData 序列 (T1/T2/T3)
    ├─ metrics/          # 1a/1b 指标 JSON
    ├─ weights/          # 恢复的奖励权重
    ├─ induced/          # 诱导训练结果 JSON
    ├─ plots/            # 所有图表 (PNG)
    └─ reports/          # 自动生成的 Markdown 报告
```

---

## 3. 运行整体流程（单个 seed 示例）

下文以 `seed=0` 为例，展示从数据生成到报告生成的一条完整流水线。

### 3.1 T1：GridWorld（MA-SPI + MA-LfL + I-LOGEL）

#### 3.1.1 生成 T1 数据

```bash
python -m runners.gen_data --config configs/t1_gridworld.yaml --seed 0
```

生成文件将位于：

* `outputs/data/t1/t1_ma_spi_seed0_*.pkl`

可以用调试脚本查看结构：

```bash
python scripts/debug_t1_ma_spi.py
```

#### 3.1.2 运行 MA-LfL 并计算 KL

```bash
python -m runners.run_ma_lfl --config configs/t1_gridworld.yaml --seed 0
```

输出包括：

* 屏幕打印的 `err_1a`, `err_1b`, `feature_dim`；
* 指标 JSON：`outputs/metrics/t1_ma_lfl_seed0.json`；
* 权重文件：`outputs/weights/...`。

#### 3.1.3 I-LOGEL 与 I-LOLA 检查（可选）

用于确认 I-LOGEL 和 I-LOLA 在 T1 上的基本行为：

```bash
python scripts/run_t1_ilogel_sanity.py
# 以及其他与 T1 相关的 I-LOLA 脚本（若有）
```

### 3.2 T2：MPE simple_spread（PPO + I-LOLA + induced）

#### 3.2.1 生成 T2 PPO 数据

```bash
python -m runners.gen_data --config configs/t2_mpe_simple_spread.yaml --seed 0
```

生成文件：

* `outputs/data/t2/t2_ppo_seed0_*.pkl`

调试脚本：

```bash
python scripts/debug_t2_ppo.py
```

#### 3.2.2 检查特征映射（可选）

```bash
python scripts/debug_feature_maps.py
```

输出中会显示：

* T1/T2 对应的特征维度 `feat_dim`；
* 特征数值范围 `feat_range`。

#### 3.2.3 运行 I-LOLA 并计算 T2 的 KL

```bash
python scripts/run_t2_ilogel_eval.py
```

脚本主要步骤：

* 使用 CMA-ES 进行 I-LOLA Stage B 优化；
* 计算 1a 和 1b 的 KL（使用同一批状态和动作采样）；
* 将指标写入 `outputs/metrics/t2_ilola_seed0.json`；
* 将恢复的奖励权重写入 `outputs/weights/...`。

#### 3.2.4 T2 诱导训练

使用恢复的奖励，从头训练一个新策略，并与随机策略比较：

```bash
python scripts/run_t2_induced.py
```

输出包括：

* JSON：`outputs/induced/t2_induced_seed0.json`
  包含 `R_random` 以及 `R_induced_curve`（训练步 → 平均回报）。
* 图：`outputs/plots/t3_induced_returns_seed0.png`
  目前该图以统一样式绘制诱导训练曲线（曲线标题为 T3，但数据来自当前脚本，可以根据后续需要调整）。

### 3.3 T3：MultiWalker（PPO + I-LOLA + induced）

#### 3.3.1 生成 T3 PPO 数据

```bash
python -m runners.gen_data --config configs/t3_multiwalker.yaml --seed 0
```

生成文件：

* `outputs/data/t3/t3_ppo_seed0_*.pkl`

调试脚本：

```bash
python scripts/debug_t3_ppo.py
```

#### 3.3.2 运行 I-LOLA 并计算 T3 的 KL（Monte Carlo）

```bash
python scripts/run_t3_ilola_eval.py
```

脚本主要步骤：

* 使用 I-LOLA Stage B 在连续动作空间进行优化；
* 对若干状态采样动作，从高斯策略中估计 KL；
* 输出 `err_1b`（目前 1a 视为零基线，主要关注 1b 的绝对量级）；
* 保存 `outputs/metrics/t3_ilola_seed0.json` 和权重文件。

#### 3.3.3 T3 诱导训练

```bash
python scripts/run_t3_induced.py
```

输出包括：

* JSON：`outputs/induced/t3_induced_seed0.json`
  包含随机策略回报和诱导策略的训练曲线。
* 图：`outputs/plots/t3_induced_returns_seed0.png`。

---

## 4. 自动图表与报告生成

当 T1/T2/T3 的指标文件准备好后，可以一键生成图表和 Markdown 报告：

```bash
python scripts/run_phase3_plots_reports.py
```

脚本会执行：

* 绘制并保存：

  * `outputs/plots/t1_kl_seed0.png`
  * `outputs/plots/t1_reward_heatmap.png`
  * `outputs/plots/t2_kl_compare_seed0.png`
  * `outputs/plots/t3_kl_seed0.png`（若 T3 指标存在）
* 生成报告：

  * `outputs/reports/report_t1_seed0.md`
  * `outputs/reports/report_t2_seed0.md`
  * `outputs/reports/report_t3_seed0.md`

这些报告会自动读取对应指标 JSON 和图表文件，并生成简短的说明文字，适合作为论文或课程报告的附录材料。

---

## 5. 多 seed 实验（建议）

当前示例脚本主要使用 `seed=0`。如果希望得到更稳定的统计结论，可以自行编写简单的多 seed 脚本，例如：

```bash
# 示例：在 PowerShell 中运行多个 seed
$seeds = 0,1,2,3,4
foreach ($s in $seeds) {
    python -m runners.gen_data --config configs/t1_gridworld.yaml --seed $s
    python -m runners.run_ma_lfl --config configs/t1_gridworld.yaml --seed $s

    python -m runners.gen_data --config configs/t2_mpe_simple_spread.yaml --seed $s
    python scripts/run_t2_ilogel_eval.py  # 内部可固定 seed 或从环境读取

    python -m runners.gen_data --config configs/t3_multiwalker.yaml --seed $s
    python scripts/run_t3_ilola_eval.py
}
```

之后可以编写一个 `scripts/aggregate_metrics.py`（目前可以由使用者根据需要实现），将 `outputs/metrics` 中的多个 JSON 汇总成一个表格，计算均值和方差，用于绘制带误差条的图。

---

## 6. 常见问题与调试建议

1. **PettingZoo 的弃用警告**

   在导入 `pettingzoo.mpe` 时可能看到类似：

   > `DeprecationWarning: The environment pettingzoo.mpe has been moved to mpe2 ...`

   当前代码仍然使用 `mpe` 命名空间，警告不会影响运行。如果将来 PettingZoo 版本发生变动，可以在 `envs/mpe_simple_spread.py` 中调整导入为 `from pettingzoo.mpe import simple_spread_v3` 的新位置。

2. **KL 数值非常小**

   如果 1a 和 1b 都是 (10^{-6}) 量级，不代表没有效果，而是说明：

   * 当前采样批量较大，采样噪音很小；
   * 在这些设置下，模型已经非常接近观测策略，可以在报告中强调“接近基线误差”。

3. **诱导训练曲线波动较大**

   诱导训练使用的训练步数和 episode 数量目前设置较小，主要作为“冒烟测试”。如果需要更平滑曲线，可以在配置中增加训练步数或评估 episode 数量，但计算时间也会增加。

---

## 7. 贡献与扩展建议

如果希望在此基础上继续扩展，可以考虑：

* 在 T2 上补充 MA-LfL 基线，实现“错配模型 vs I-LOLA”的直接对比；
* 系统对比 Stage A（I-LOGEL）和 Stage B（I-LOLA）在 T2 上的表现；
* 增加一个 T2.5 扩展环境，用于进一步展示算法的可扩展性；
* 将当前自动报告整理为正式论文草稿，补充更多理论推导和实验分析。

---

## 8. 联系方式

如需在课程汇报或论文写作中使用本仓库的结果，可以在报告中简要说明：

* 使用多智能体 GridWorld、MPE `simple_spread` 与 SISL `multiwalker_v9` 作为测试环境；
* 复现 MA-LfL 的情况下，在匹配设定下得到接近零的策略预测误差；
* 设计并实现 I-LOLA，在 PPO 数据上恢复奖励并通过诱导训练进行验证。

