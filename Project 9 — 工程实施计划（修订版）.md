# Project 9 — 工程实施计划（修订版）

## 0\. 目标与范围

本计划将你的研究方案工程化，输出三类可重复的产物：
（1）数据生成与学习者（IPPO/MAPPO, MA-SPI for baseline）
（2）奖励反演（MA-LfL 基线、I-LOLA 阶段 A/B）
（3）评测与复现实验（T1→T2→T2.5→T3）。

**T3 将切换到连续控制环境（`sisl_multiwalker_v9`）**，以验证 I-LOLA 在策略梯度主场的绝对优势，并测试 MA-LfL 的理论边界。
**T2.5**（可选）则是在 T2（MPE `simple_spread`）基础上加难度（人数↑、地标漂移、观测裁剪等），用于测试同一脚手架的扩展性。

**验收标准（Definition of Done）**
1）能复现 T1 基线；
2）T2 上 I-LOLA-B 的“下一步策略预测误差”优于独立近似（阶段 A）；
3）**在 T3（`multiwalker`）上证明 I-LOLA 的可行性，而 MA-LfL 因理论限制无法直接适用**；
4）完整保存与复现实验所需的代码、配置、日志、模型、图表与报告。

-----

## 1\. 代码仓库结构与规范

```
project9/
├─ configs/
│  ├─ base.yaml                 # 全局默认超参
│  ├─ t1_gridworld.yaml         # T1 实验配置
│  ├─ t2_mpe_simple_spread.yaml # T2 实验配置
│  ├─ t25_mpe_extended.yaml     # [新增] T2.5 扩展配置
│  └─ t3_multiwalker.yaml       # [修改] T3 连续控制配置
├─ envs/
│  ├─ gridworld3x3.py           # T1 环境（两体，离散）
│  ├─ mpe_simple_spread_wrap.py # T2/T2.5: PettingZoo 包装器
│  └─ sisl_multiwalker_wrap.py  # [新增] T3 连续控制包装器
├─ learners/
│  ├─ ippo.py                   # 独立 PPO（数据生成轨道2）
│  ├─ mappo.py                  # MAPPO（可选，集中式价值评估）
│  └─ maspi.py                  # MA-SPI（数据生成轨道1，用于 LfL 基线）
├─ models/
│  ├─ policy_net.py             # 策略网络（离散/高斯）
│  ├─ value_net.py              # 价值网络 / critic
│  ├─ reward_net.py             # R_ω(s,a) 线性或轻量 MLP
│  └─ feature_maps.py           # ϕ(s,a) 特征（指示器/RBF/动作范数等）
├─ inverse/
│  ├─ common/
│  │  └─ policy_estimation.py   # [新增] 通用策略估计 (BC)，供双方共用
│  ├─ ma_lfl/
│  │  ├─ target_builder.py      # 由软改进关系式构造 Y_h
│  │  └─ reward_fit.py          # Eq.(12) 的联合优化（奖励+势函数）
│  └─ ilola/
│     ├─ stage_a_independent.py # I-LOLA 阶段A：独立近似
│     ├─ simulator.py           # 一阶 LOLA 前向模拟器 f (对称版)
│     └─ stage_b_fit.py         # I-LOLA 阶段B：无导数黑盒优化（CMA-ES）
├─ evals/
│  ├─ metrics.py                # 预测误差、任务性能、置信区间
│  ├─ plots.py                  # 曲线与热力图
│  └─ report.py                 # 生成 Markdown/HTML 报告
├─ runners/
│  ├─ gen_data.py               # 生成并落盘 θ/π/轨迹
│  ├─ run_ma_lfl.py             # 跑 MA-LfL 基线
│  ├─ run_ilola_a.py            # 跑 I-LOLA 阶段 A
│  └─ run_ilola_b.py            # 跑 I-LOLA 阶段 B
├─ scripts/
│  ├─ make_datasets.sh          # 批量生成数据集（不同 seeds）
│  ├─ run_all_t1.sh             # T1 全流程
│  ├─ run_all_t2.sh             # T2 全流程
│  ├─ run_all_t25.sh            # [新增] T2.5 全流程
│  └─ run_all_t3.sh             # [修改] T3 全流程
├─ outputs/                     # 日志、模型、图表与报告（gitignore）
├─ requirements.txt
├─ Makefile
└─ README.md
```

**代码风格**：Python 3.11，`ruff` + `black`，单元测试用 `pytest`。**代码注释与标识全用英文**，保证可读与可维护；论文文字性描述与报告按中文输出。

-----

## 2\. 环境准备与一键脚手架

**2.1 依赖安装**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt`（关键依赖）

```
torch>=2.2
numpy>=1.26
pettingzoo>=1.24
gymnasium>=0.29
matplotlib>=3.8
scipy>=1.12
cma>=3.3
tqdm>=4.66
omegaconf>=2.3
```

**2.2 统一脚手架**

`Makefile` 提供最小化工作流：

```makefile
SEED?=0
CONF?=configs/t2_mpe_simple_spread.yaml

init:
	python -c "import torch, numpy as np; print('OK')"

gen:
	python runners/gen_data.py --config $(CONF) --seed $(SEED)

lfl:
	python runners/run_ma_lfl.py --config $(CONF) --seed $(SEED)

ilola_a:
	python runners/run_ilola_a.py --config $(CONF) --seed $(SEED)

ilola_b:
	python runners/run_ilola_b.py --config $(CONF) --seed $(SEED)

# [新增] T2.5 和 T3 运行目标
run_t25:
	make gen CONF=configs/t25_mpe_extended.yaml
	make ilola_a CONF=configs/t25_mpe_extended.yaml
	make ilola_b CONF=configs/t25_mpe_extended.yaml

run_t3:
	make gen CONF=configs/t3_multiwalker.yaml
	make ilola_a CONF=configs/t3_multiwalker.yaml
	make ilola_b CONF=configs/t3_multiwalker.yaml

report:
	python evals/report.py --config $(CONF)
```

**2.3 配置模板（`configs/base.yaml`）**

```yaml
seed: 0
device: "cuda:0"

env:
  name: "mpe_simple_spread"   # t1: gridworld3x3, t2: mpe_simple_spread, t2.5: mpe_extended, t3: sisl_multiwalker_v9
  num_agents: 3
  episode_len: 200
  parallel_envs: 8

policy:                      # [新增] 策略头配置
  head: "categorical"      # categorical (离散) or gaussian (连续)

ppo:
  total_steps: 200000
  gamma: 0.99
  gae_lambda: 0.95
  lr: 3e-4
  entropy_coef: 0.01
  clip_coef: 0.2
  minibatch: 4
  update_epochs: 4

lfl:
  alpha_soft: 0.1
  kl_eps: 1e-6

ilola:
  stage: "B"         # A or B
  cma_sigma0: 0.5
  cma_maxiter: 100
  ridge_lambda: 1e-3
  grad_clip: 5.0

features:
  type: "rbf"        # t1: indicator, t2/t3: rbf
  rbf_num: 64
```

> 以上默认值与可执行流程，严格对应你计划书里对 T1/T2/T3 的定位与资源边界。

-----

## 3\. 数据生成（两条轨道）：可复现实验的基础

**目标**：产出两类“学习者数据”以驱动反演与评测：
**轨道1**（匹配前提）：MA-SPI 学习者；**轨道2**（失配前提）：PPO 学习者（IPPO/MAPPO）。

### 3.1 T1：Gridworld $3\times 3$

  * `envs/gridworld3x3.py`：确定性转移、两体、五动作（上/下/左/右/停），同质/异质奖励切换开关。
  * 轨道1（MA-SPI）：固定阶段策略 $\pi_h$ 采样 $D_h$，完成软 Bellman 与软改进；落盘 ${\pi_h}$、阶段轨迹。
  * 轨道2（IPPO）：每体独立策略，存 ${\theta_t}$、策略 logits、回放轨迹。
  * 最小可交付数据：$m=5$ 步邻接更新，每步 $n\in{50,100,200}$ 条轨迹作抽样曲线（用于“经验性收敛测试”）。

### 3.2 T2：MPE `simple_spread`（PettingZoo 包装）

  * `envs/mpe_simple_spread_wrap.py`：并行环境包装（AEC→Parallel），提供每体观测/动作，便于逐体统计与反演。
  * **[修改]** T2 的动作空间按 PettingZoo 版本与配置决定（可为离散或连续）。我们通过 `configs/t2...yaml` 中的 `policy.head` (categorical/gaussian) 来控制策略网络，确保适配。
  * 轨道1：如需，对 MA-SPI 做近似器版软 $Q$ 与软改进（成本较高，可仅做小样本对照）。
  * 轨道2：主通道——IPPO/MAPPO 产出 ${\theta_t}$、轨迹与策略快照；每 10 个 PPO 更新保存一次。
  * 数据预算：参考 T1 的“拐点”实证后定；初始建议 $m=10$、每步 $n=1{,}000$。

### 3.3 T3：连续控制压力测试 (multiwalker_v9)

  * **[修改]** `envs/sisl_multiwalker_wrap.py`：实现 PettingZoo SISL 的并行包装，返回每体 obs/act，动作为连续张量。
  * 轨道1：**不执行**。理由：MA-LfL/MA-SPI 在连续动作空间需要重度近似或无法自然定义，不作为主对照。
  * 轨道2：主通道——IPPO/MAPPO 产出 ${\theta_t}$。强制 `policy.head: gaussian`。
  * 数据预算：$m=10$、每步 $n=2{,}000$（可按算力调低为 $5\times 1{,}000$ 的烟囱测试）。
  * （原 T3 MPE 扩展内容移至 T2.5，在 `configs/t25_mpe_extended.yaml` 中定义，作为可选的扩展性实验）。

-----

## 4\. 反演模块一：MA-LfL 基线（匹配前提）

对应你计划书的“策略估计→目标构造 $Y_h$→奖励与势函数联合拟合”。实现要点如下。

### 4.1 策略估计（按阶段独立）

```
[修改] inverse/common/policy_estimation.py
```

  * 从 $D_h$ 用 MLE+熵正则拟合逐体策略 $\hat\pi_h^i$（离散：分类头；连续：高斯头）。
  * 概率裁剪 $\in[\epsilon,1]$，log-softmax 计算稳定；保存 logits 供后续 KL 与软改进使用。

### 4.2 目标构造 $Y_h$

```
inverse/ma_lfl/target_builder.py
```

  * 用软改进关系式把 ${\hat\pi_{h-1},\hat\pi_h}$ 转换为 $Y_h$（参考 Eq.(5)(6) 的离散化实现）。
  * 对 $s'$ 的期望：T1 为确定转移直接取值；T2/T3 用采样期望或小批蒙特卡洛。
  * KL 与熵项统一温度 $\alpha$；全流程保持同一 `alpha_soft`（配置项）。

### 4.3 联合拟合（奖励 + 势函数）

```
inverse/ma_lfl/reward_fit.py
```

  * 线性或浅层 MLP 奖励 $R_\omega(s,a)=\omega^\top \phi(s,a)$；阶段势函数 $g_{\psi_h}(s)$。
  * 损失：
    $$
    \mathcal{L}=\sum_{h}\sum_{(s,a,s')\in D_h}\Big(R_\omega(s,a)+g_{\psi_h}(s)-\gamma g_{\psi_h}(s')-Y_h\Big)^2+ \lambda\lVert \omega\rVert_2^2
    $$
    
  * 对 $g_{\psi_h}$ 施加零均值或幅度衰减稳定辨识；交替更新或同批联合更新均可。

> 注意：MA-LfL 主用于**轨道1**（匹配 MA-SPI）报告“能力上限”；在**轨道2**（PPO 数据）上作为“失配对照”。

-----

## 5\. 反演模块二：I-LOLA（阶段 A/B）

I-LOLA 是你的方法主线：先做“独立近似”（A），再做“一阶 LOLA 黑盒拟合”（B）。

### 5.1 阶段 A：独立近似（I-LOGEL）

```
inverse/ilola/stage_a_independent.py
```

  * 按体独立拟合：
    $$
    \theta^i_{t+1}\approx \theta^i_t+\alpha_t^i \nabla_{\theta_i}J_i(\theta_i,\theta_{-i,t};\omega_i),\quad
    \nabla_{\theta_i}J_i=\nabla_{\theta_i}\psi_i(\theta_i),\omega_i
    $$
    
    **[修改]** 通过 `inverse/common/policy_estimation.py` 行为克隆得到 $\hat\theta_t^i$；用回放估计 $\nabla_{\theta_i}\psi_i$ 的数值雅可比；
  * 交替闭式更新 ${\omega_i}$ 与 ${\alpha_t^i}$，岭回归稳条件数；梯度裁剪与下界 $\alpha\ge \varepsilon$。

### 5.2 阶段 B：一阶 LOLA 前向模拟 + 黑盒优化

```
inverse/ilola/simulator.py
```

  * 构造一阶耦合：
    $$
    \Delta\theta^B\approx \alpha^B\nabla_{\theta_B}J_B(\theta^B;\omega_B),\quad
    \hat\theta^A_{t+1}=\theta^A_t+\alpha^A\nabla_{\theta_A}J_A(\theta^A,\theta^B+\Delta\theta^B;\omega_A)
    $$
    对称地得到 $\hat\theta^B_{t+1}$。
  * `stage_b_fit.py`：把 $f(\theta_t; \omega)\mapsto \hat\theta_{t+1}$ 作为黑盒，目标最小化
    $$
    \sum_t\sum_i\lVert \hat\theta^i_{t+1}(\omega)-\theta^{i,\text{obs}}_{t+1}\rVert_2^2
    $$
    
    $$
  * **[新增]** 具体实现见 9.2 节，必须确保模拟器 $f$ 采用**对称的**一阶 LOLA 动态。
  * 求解器用 CMA-ES（`cma`），初始 $\omega$ 来自阶段 A 的解，设 `sigma0=0.5`、`maxiter=100`；必要时分块（两两子系统）拟合后合并。

> 该设计与你计划中“用 LOLA 的一阶耦合、避免全双层 BPTT、保证可交付”的取舍一致。

-----

## 6\. 评测与报告（整形鲁棒）

### 6.1 主指标：下一步策略预测误差（分解版）

```
evals/metrics.py
```

  * **(1a) 模型基线误差**：用“真实奖励”+“学习者模型（MA-SPI 或 I-LOLA）”从 $\pi_t$ 预测 $\hat\pi_{t+1}(\mathbf{W}^*)$，计算 $D\big(\pi_{t+1}\Vert \hat\pi_{t+1}(\mathbf{W}^*)\big)$；
  * **(1b) 反演误差**：把 $\hat{\mathbf{W}}$ 代入同一学习者模型，计算 $D\big(\pi_{t+1}\Vert \hat\pi_{t+1}(\hat{\mathbf{W}})\big)$；
  * **[修改]** 距离度量：离散用 $D_{\mathrm{KL}}$ (Categorical 分布)；连续（如 T3）用高斯解析 KL。
  * 预期：阶段 B 的 (1b) 接近 (1a)；MA-LfL 在轨道2会显著劣于 (1a)。

### 6.2 任务指标：诱导策略性能

  * 用各自 $\hat{\mathbf{W}}$ 重新训练 IPPO，报告最终回报均值与 $95%$ 置信区间；与专家性能对比。

### 6.3 报告生成

`evals/report.py` 汇总曲线（误差随步数、随样本量）、表格（任务性能）、热力图（T1 奖励对比），导出 `outputs/report.md` 与图像。

-----

## 7\. 实施里程碑（两周冲刺为单位）

**Sprint 1（周 1-2）：T1 完整闭环 + 管线固化**

  * 完成 Gridworld、MA-SPI 与 IPPO 数据生成；
  * 打通 MA-LfL（`common/policy_estimation`→目标→联合拟合），完成 T1 复现图表；
  * 打通 I-LOLA 阶段 A，输出对比；
  * 初版报告能生成。

**Sprint 2（周 3-4）：T2 主战场 + I-LOLA 阶段 B**

  * 接入 MPE `simple_spread` 并完成数据预算拐点试验；
  * **[修改]** 上线 I-LOLA 阶段 B（使用**对称**模拟器 + CMA-ES），稳定条件数，给出主指标优势；
  * 报告加入 T2 结果与误差分解。

**Sprint 3（周 5-6）：T3 连续控制 + 终版报告**

  * **[修改]** 接入 T3 `multiwalker` 环境 (`sisl_multiwalker_wrap.py`)。
  * 跑通 T3 的轨道2 数据生成 (IPPO + Gaussian head)。
  * 跑通 T3 上的 I-LOLA (A/B)，验证在连续控制下的可行性。
  * （可选）运行 T2.5 (MPE 扩展) 实验。
  * 消融：去掉耦合项（退化到 A）、改变样本量与学习率下界，验证鲁棒趋势。
  * 终版报告与复现实验脚本冻结。

-----

## 8\. 风险与回退策略（最小可交付切片）

  * **高方差的交叉灵敏度**：采用共享随机数、优势函数 baseline、轨迹数扩增、岭回归；必要时两两子系统分块拟合。
  * **CMA-ES 不收敛**：缩小搜索域、减参（冻结部分特征坐标）、用阶段 A 的 $\omega$ 作为多起点；条件数超阈则提升样本量或增大 `ridge_lambda`。
  * **算力不足**：先交付 T1 全闭环 + T2 小预算（$m=5,n=500$），报告主指标趋势与不确定性；T3 仅做烟囱测试。
    这些都与项目作业的“可交付优先”一致。

-----

## 9\. 关键实现片段（示例代码，可直接粘贴）

**9.1 策略网络（离散/高斯双头）— `models/policy_net.py`**

```python
import torch, torch.nn as nn

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, discrete=True):
        super().__init__()
        self.discrete = discrete
        hidden = 128
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )
        if discrete:
            # 用于 T1, T2 (离散版)
            self.logits = nn.Linear(hidden, act_dim)
        else:
            # 用于 T3 (multiwalker) 或 T2 (连续版)
            self.mu = nn.Linear(hidden, act_dim)
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        z = self.backbone(x)
        if self.discrete:
            return {"logits": self.logits(z)}
        return {"mu": self.mu(z), "log_std": self.log_std.clamp_(-5, 2)}
```

**9.2 I-LOLA 前向模拟器（对称版） — `inverse/ilola/simulator.py`**

```python
import torch

def ilola_step(theta_a, theta_b, omega_a, omega_b, alpha_a, alpha_b,
               gradA, gradB):
    """
    [修改] 实现对称的一阶 LOLA 动态。
    gradA/B: callables mapping (theta_self, theta_oppo, omega_self) -> grad tensor
    所有张量均为 torch.Tensor
    """
    # 1) 双方的“天真梯度”
    #    (注意：gradB 的输入也应是对称的 (self, oppo, omega_self))
    naive_dtheta_a = alpha_a * gradA(theta_a, theta_b, omega_a)
    naive_dtheta_b = alpha_b * gradB(theta_b, theta_a, omega_b)

    # 2) 双方的 LOLA 一阶校正：各自把“对手的天真步”代入自己的梯度
    next_a = theta_a + alpha_a * gradA(theta_a, theta_b + naive_dtheta_b, omega_a)
    next_b = theta_b + alpha_b * gradB(theta_b, theta_a + naive_dtheta_a, omega_b)

    return next_a, next_b
```

**9.3 CMA-ES 拟合入口 — `inverse/ilola/stage_b_fit.py`**

```python
import numpy as np, cma

def fit_omega(theta_seq, sim_fn, omega0, sigma0=0.5, maxiter=100):
    # theta_seq: list of (theta_a_t, theta_b_t, theta_a_tp1_obs, theta_b_tp1_obs)
    def loss_fn(omega_flat):
        omega_a, omega_b = unpack(omega_flat) # unpack 需要你自行实现
        err = 0.0
        for (ta, tb, ta1, tb1) in theta_seq:
            # sim_fn 内部应调用 9.2 中的 ilola_step
            hat_a1, hat_b1 = sim_fn(ta, tb, omega_a, omega_b)
            err += ((hat_a1 - ta1)**2).sum() + ((hat_b1 - tb1)**2).sum()
        return float(err / len(theta_seq))
    es = cma.CMAEvolutionStrategy(np.array(omega0), sigma0, {"maxiter": maxiter})
    es.optimize(loss_fn)
    return unpack(es.result.xbest)
```

> 代码注释与变量命名全英文、接口清晰，可直接落地在仓库中对应文件。

-----

## 10\. 运行清单（从零到报告）

1）初始化与冒烟测试：

```bash
make init
```

2）T1：生成两条轨道数据并反演、出报告：

```bash
make gen CONF=configs/t1_gridworld.yaml
make lfl CONF=configs/t1_gridworld.yaml
make ilola_a CONF=configs/t1_gridworld.yaml
make ilola_b CONF=configs/t1_gridworld.yaml
make report CONF=configs/t1_gridworld.yaml
```

3）T2：同流程（默认 `configs/t2_mpe_simple_spread.yaml`）：

```bash
make gen
make ilola_a
make ilola_b
make report
```

4）**[新增]** T2.5 扩展（可选）：

```bash
make run_t25
make report CONF=configs/t25_mpe_extended.yaml
```

5）**[修改]** T3 连续控制：

```bash
make run_t3
make report CONF=configs/t3_multiwalker.yaml
```

-----

## 11\. 文档与复现

  * `README.md`：写明一键命令与最小可交付切片（T1 全闭环 + T2 主结果）；
  * `outputs/`：保存所有中间件与终态，包含 `datasets/`、`models/`、`figures/`、`reports/`；
  * `scripts/`：提供多种 seeds 的批量脚本，自动汇总均值±置信区间。

-----

### 备注

  * 本工程计划严格对齐你在《Project9执行计划.md》中确立的技术路线与指标，并吸收了《engineering_guide.md》中“项目结构、阶段推进、结果与图表生成”的做法，确保“能跑、能评、能交”。
  * **[修订]** 本版本已根据评审建议，恢复 T3 为 `multiwalker` 连续控制，修正了 I-LOLA 模拟器的对称性，并重构了通用策略估计模块。