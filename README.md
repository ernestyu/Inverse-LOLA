# MA-LfL 复现

本仓库实现了论文 **"Multi-Agent Learning from Learners"**（多智能体从学习者中学习，简称 MA-LfL）的完整复现流程。代码遵循 `docs/engineering_guide.md` 中的工程蓝图，复现了 MA-SPI + MA-LfL 基线，包括策略估计、奖励恢复、相关性指标和诊断图。

## 环境设置

1. 创建并激活 Python 3.10+ 虚拟环境。
2. 安装 Python 依赖项：

   ~~~
   pip install -r requirements.txt
   ~~~

如果你计划使用支持 CUDA 的 GPU，请在安装 PyTorch 之前，根据你的 CUDA 版本调整 `requirements.txt` 中的 `--extra-index-url`。

## 配置

超参数和日志选项存储在 `config.yaml` 中。关键部分：

- `environment`: 网格大小、起始/目标位置和初始奖励族。
- `maspi`: MA-SPI 循环设置（迭代次数、回合长度、更新次数、随机种子）。
- `malfl`: 策略估计、目标计算以及联合奖励/势函数优化的超参数。
- `optimization`: MA-SPI 参与者（Actors）/评论者（Critics）的学习率。
- `logging`: 基本输出目录和数据保留标志。

默认设置遵循论文中的表 3。如果你想缩短运行时间或探索不同的机制，请调整 YAML 文件。

## 运行实验

使用 `main.py` 执行完整的流水线：

~~~
python main.py --config config.yaml --reward-families homogeneous heterogeneous
~~~

重要的命令行选项：

- `--output-dir`: 覆盖基本输出目录。
- `--reward-families`: 运行一个或多个奖励族（`homogeneous` 同质，`heterogeneous` 异质）。
- `--iterations / --episodes`: 覆盖 MA-SPI 迭代次数或回合数以进行实验。
- `--fast`: 缩减迭代次数/回合数，用于快速冒烟测试（例如 CI 运行）。

### 冒烟测试

要在不进行完整工作负载的情况下验证安装：

`ash
python main.py --fast --reward-families homogeneous
`

这将在几分钟内完成，并验证 MA-SPI、MA-LfL 和评估产物是否正确生成。

### 完整复现

为了忠实地复现论文结果，请运行两种奖励族：

`ash
python main.py --reward-families homogeneous heterogeneous
`

这将在`outputs/`目录下生成：

- `ma_spi/` — 在线策略数据集 $D_h$、策略 $\text{logits}$ 和训练损失。
- `ma_lfl/` — 策略估计产物、奖励目标、恢复的奖励网络和评估图表。
- `evaluation/` — 相关性指标、奖励趋势曲线（类似于图 3）以及与附录图表相当的热力图。
- `summary.json` — 每个实验的聚合指标。
- `cross_correlation.csv` — 表 2 风格的健全性检查，显示奖励族的正确配对上的最高相关性。

随机种子已固定以确保确定性运行。在重新运行之前，请删除相应的输出子目录，以避免结果混淆。

## 项目结构

`
├── algorithms/
│   ├── ma_spi.py        # 在线策略数据生成（算法 1）
│   └── ma_lfl.py        # 奖励推断、策略估计、指标（算法 2）
├── agents/
│   └── ma_spi_agent.py  # 具有共享 $\alpha$ 温度的软策略迭代智能体
├── environments/
│   └── gridworld.py     # 具有 $M_{hom} / M_{het}$ 奖励的确定性 $3 \times 3$ 网格世界
├── models/              # 策略、Q、奖励和势函数网络
├── data/structures.py   # 在线策略轨迹容器 ($D_h$)
├── evaluation/          # 相关性指标和绘图辅助函数
├── main.py              # 端到端编排和报告
├── config.yaml          # 默认超参数
└── outputs/             # 运行时创建，包含可复现的产物
`

## 注意事项

- 所有计算共享一个全局温度 $\alpha$，通过 `ExperimentConfig` 强制执行并传播到每个模块。
- 代码保存了完整的中间产物（策略、轨迹、奖励网络），以满足 `docs/engineering_guide.md` 中的可复现性清单要求。
- 交叉相关性诊断和趋势图提供了关于奖励恢复质量的即时反馈，与论文中的表 1/表 2 和图 3 相呼应。
