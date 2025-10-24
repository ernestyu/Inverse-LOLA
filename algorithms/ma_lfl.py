from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Dict, List, Sequence, Tuple

import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from config import ExperimentConfig
from data.structures import StageDataset
from environments.gridworld import GridWorld, RewardFamily
from evaluation.reporting import RewardEvaluationResult, evaluate_rewards
from models.policy_net import PolicyNetwork
from models.reward_net import JointRewardNetwork
from models.shaping_net import PotentialNetwork


# =========================
# Training knobs (no hidden scaling)
# =========================
WEIGHT_DECAY = 1e-5         # 轻量 L2，稳定训练
GRAD_CLIP_MAX_NORM = 1.0    # 梯度裁剪阈值
MAX_BATCH_SIZE = 1024       # 奖励学习阶段的最大 batch，避免 full-batch 的数值刚性


# ----------------------------
# Data containers
# ----------------------------

@dataclass
class EstimatedPolicy:
    model: PolicyNetwork
    logits: torch.Tensor
    probs: torch.Tensor
    training_loss: List[float]


@dataclass
class PolicyEstimationResult:
    policies: List[List[EstimatedPolicy]]  # [agent][stage]


@dataclass
class RewardModelBundle:
    reward_network: JointRewardNetwork
    # 共享势函数：列表只包含一个网络（保持与保存/载入逻辑兼容）
    potential_networks: List[PotentialNetwork]
    losses: List[float]


@dataclass
class RewardLearningResult:
    bundles: List[RewardModelBundle]
    predicted_reward_tables: List[torch.Tensor]  # [agent] -> (num_states, num_actions, num_actions)


@dataclass
class MALFLArtifacts:
    policy_estimation: PolicyEstimationResult
    reward_learning: RewardLearningResult
    targets: List[List[torch.Tensor | None]]  # [agent][stage]
    evaluation: RewardEvaluationResult | None = None

    def save(self, base_dir: Path) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)

        # save policy estimation
        policy_dir = base_dir / "policy_estimation"
        policy_dir.mkdir(exist_ok=True)
        for agent_id, stage_policies in enumerate(self.policy_estimation.policies):
            agent_dir = policy_dir / f"agent_{agent_id}"
            agent_dir.mkdir(exist_ok=True)
            for stage_idx, policy in enumerate(stage_policies):
                stage_dir = agent_dir / f"stage_{stage_idx:02d}"
                stage_dir.mkdir(exist_ok=True)
                torch.save(policy.model.state_dict(), stage_dir / "model.pt")
                torch.save(policy.logits.cpu(), stage_dir / "logits.pt")
                torch.save(policy.probs.cpu(), stage_dir / "probs.pt")
                np.save(stage_dir / "loss.npy", np.array(policy.training_loss, dtype=np.float32))

        # save targets
        targets_dir = base_dir / "targets"
        targets_dir.mkdir(exist_ok=True)
        for agent_id, agent_targets in enumerate(self.targets):
            agent_dir = targets_dir / f"agent_{agent_id}"
            agent_dir.mkdir(exist_ok=True)
            for stage_idx, tensor in enumerate(agent_targets):
                if tensor is None:
                    continue
                torch.save(tensor.cpu(), agent_dir / f"stage_{stage_idx:02d}.pt")

        # save reward models
        reward_dir = base_dir / "reward_learning"
        reward_dir.mkdir(exist_ok=True)
        for agent_id, bundle in enumerate(self.reward_learning.bundles):
            agent_dir = reward_dir / f"agent_{agent_id}"
            agent_dir.mkdir(exist_ok=True)
            torch.save(bundle.reward_network.state_dict(), agent_dir / "reward_model.pt")
            potentials_dir = agent_dir / "potentials"
            potentials_dir.mkdir(exist_ok=True)
            # 共享势函数：保存为 shared.pt
            if len(bundle.potential_networks) == 1:
                torch.save(bundle.potential_networks[0].state_dict(), potentials_dir / "shared.pt")
            else:
                for stage_idx, net in enumerate(bundle.potential_networks):
                    torch.save(net.state_dict(), potentials_dir / f"stage_{stage_idx:02d}.pt")
            np.save(agent_dir / "loss.npy", np.array(bundle.losses, dtype=np.float32))
            torch.save(self.reward_learning.predicted_reward_tables[agent_id].cpu(), agent_dir / "rewards.pt")

        # save evaluation
        if self.evaluation is not None:
            eval_dir = base_dir / "evaluation"
            eval_dir.mkdir(exist_ok=True)
            metrics_payload = {
                "agents": [
                    {"pearson": m.pearson, "spearman": m.spearman}
                    for m in self.evaluation.agent_metrics
                ],
                "mean": {
                    "pearson": self.evaluation.mean_metrics.pearson,
                    "spearman": self.evaluation.mean_metrics.spearman,
                },
                "trend": {
                    "stages": self.evaluation.trend_stages,
                    "pearson": self.evaluation.trend_pearson,
                    "spearman": self.evaluation.trend_spearman,
                    "figure": Path(self.evaluation.trend_path).name,
                },
                "heatmaps": {k: Path(v).name for k, v in self.evaluation.heatmap_paths.items()},
            }
            (eval_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))


# ----------------------------
# Policy estimation (per stage)
# ----------------------------

def estimate_policies(
    stage_datasets: Sequence[StageDataset],
    config: ExperimentConfig,
    device: torch.device,
) -> PolicyEstimationResult:
    num_agents = 2
    num_stages = len(stage_datasets)
    policies: List[List[EstimatedPolicy]] = [[None for _ in range(num_stages)] for _ in range(num_agents)]  # type: ignore

    baseline_loss = float(np.log(config.num_actions))

    for stage_idx, dataset in enumerate(stage_datasets):
        logging.info("[MA-LfL][Policy] Stage %d/%d - preparing data", stage_idx + 1, num_stages)

        tensors = dataset.concatenated(device=device)
        states = tensors.states
        actions_agent_0 = tensors.joint_actions[:, 0]
        actions_agent_1 = tensors.joint_actions[:, 1]

        for agent_id, actions in enumerate((actions_agent_0, actions_agent_1)):
            model = PolicyNetwork(config.num_states, config.num_actions).to(device)
            optimizer = optim.Adam(model.parameters(), lr=config.malfl.policy_estimation_lr)

            batch_size = min(config.malfl.policy_estimation_batch, states.size(0))
            epochs = config.malfl.policy_estimation_epochs
            losses: List[float] = []

            for epoch in range(epochs):
                perm = torch.randperm(states.size(0), device=device)
                epoch_loss = 0.0
                batches = 0

                for start in range(0, states.size(0), batch_size):
                    idx = perm[start:start + batch_size]
                    batch_states = states[idx]
                    batch_actions = actions[idx]

                    logits = model(batch_states)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_prob = dist.log_prob(batch_actions)
                    entropy = dist.entropy()

                    # maximize log_prob + coef * entropy  <=>  minimize negative
                    loss = -(log_prob.mean() + config.malfl.policy_entropy_coef * entropy.mean())

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += float(loss.item())
                    batches += 1

                losses.append(epoch_loss / max(1, batches))

                if (epoch + 1) % max(1, epochs // 5) == 0 or (epoch + 1) == epochs:
                    logging.info(
                        "[MA-LfL][Policy] Stage %d/%d - Agent %d: %d/%d epochs completed (loss=%.4f, baseline=%.4f)",
                        stage_idx + 1,
                        num_stages,
                        agent_id,
                        epoch + 1,
                        epochs,
                        losses[-1],
                        baseline_loss,
                    )

            # export policy over full state grid
            state_grid = torch.arange(config.num_states, device=device, dtype=torch.long)
            with torch.no_grad():
                logits = model(state_grid)
                probs = torch.softmax(logits, dim=-1)

            policies[agent_id][stage_idx] = EstimatedPolicy(
                model=model,
                logits=logits.detach().clone(),
                probs=probs.detach().clone(),
                training_loss=losses,
            )

    return PolicyEstimationResult(policies=policies)


# ----------------------------
# Target construction (Eq. 11)
# ----------------------------

def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float) -> torch.Tensor:
    """KL(p || q) with clamping for numerical stability."""
    p_clamped = torch.clamp(p, eps, 1.0)
    q_clamped = torch.clamp(q, eps, 1.0)
    return torch.sum(p_clamped * (torch.log(p_clamped) - torch.log(q_clamped)), dim=-1)


def compute_reward_targets(
    policy_estimation: PolicyEstimationResult,
    stage_datasets: Sequence[StageDataset],
    config: ExperimentConfig,
    device: torch.device,
) -> List[List[torch.Tensor | None]]:
    """
    Compute Y_h targets for each agent and stage according to Eq. (11):

      Y_h^i(s,a^i) = α * log π_h^i(a^i | s)
                      + α * γ * E_{a^{-i} ~ π_h^{-i}(·|s), s' ~ P(·|s,a)} [ KL( π_{h-1}^i(·|s') || π_h^i(·|s') ) ]

    Notes:
      * KL is evaluated at the NEXT state s'
      * Stages with no next state (last stage) have no targets (None)
    """
    env = GridWorld(
        size=config.environment.grid_size,
        start_positions=tuple(tuple(pos) for pos in config.environment.start_positions),
        goal_position=tuple(config.environment.goal_position),
        reward_family=RewardFamily(config.environment.reward_family),
    )

    num_agents = 2
    num_stages = len(stage_datasets)
    targets: List[List[torch.Tensor | None]] = [[None] * num_stages for _ in range(num_agents)]
    eps = config.malfl.prob_clip
    gamma = config.gamma
    alpha = config.alpha

    # stages: 1..H-1 (stage 0 has no prev; stage H-1 has no next)
    for stage_idx in range(1, num_stages):
        for agent_id in range(num_agents):
            logging.info(
                "[MA-LfL][Targets] Stage %d/%d - Agent %d",
                stage_idx,
                num_stages - 1,
                agent_id,
            )
            current_policy = policy_estimation.policies[agent_id][stage_idx]
            prev_policy = policy_estimation.policies[agent_id][stage_idx - 1]
            opponent_id = 1 - agent_id
            current_opponent_policy = policy_estimation.policies[opponent_id][stage_idx]

            # table over (state, own_action)
            y_table = torch.zeros(
                (config.num_states, config.num_actions), dtype=torch.float32, device=device
            )

            for state_index in range(config.num_states):
                joint_state = env.index_to_joint_state(state_index)
                # log π_h^i(a^i | s)
                log_probs = torch.log(torch.clamp(current_policy.probs[state_index], eps, 1.0))

                for own_action in range(config.num_actions):
                    # E_{a^{-i} ~ π_h^{-i}(·|s)} KL( π_{h-1}^i(·|s') || π_h^i(·|s') )
                    kl_expectation = 0.0
                    for opp_action in range(config.num_actions):
                        prob_opp = float(current_opponent_policy.probs[state_index, opp_action].clamp(min=eps))
                        if agent_id == 0:
                            joint_action = (own_action, opp_action)
                        else:
                            joint_action = (opp_action, own_action)

                        next_state = env.transition(joint_state, joint_action)
                        next_state_index = env.joint_state_to_index(next_state)

                        kl_value = float(
                            _kl_divergence(
                                prev_policy.probs[next_state_index].unsqueeze(0),
                                current_policy.probs[next_state_index].unsqueeze(0),
                                eps,
                            ).item()
                        )
                        kl_expectation += prob_opp * kl_value

                    # Eq. (11) 正号
                    y_value = alpha * log_probs[own_action] + alpha * gamma * kl_expectation
                    y_table[state_index, own_action] = y_value

            targets[agent_id][stage_idx] = y_table

    return targets


# ----------------------------
# Reward + shared shaping learning (Eq. 12 with shared g)
# ----------------------------

def learn_rewards(
    stage_datasets: Sequence[StageDataset],
    targets: List[List[torch.Tensor | None]],
    config: ExperimentConfig,
    device: torch.device,
) -> RewardLearningResult:
    num_agents = 2
    num_stages = len(stage_datasets)

    bundles: List[RewardModelBundle] = []
    predicted_tables: List[torch.Tensor] = []

    for agent_id in range(num_agents):
        reward_net = JointRewardNetwork(config.num_states, config.num_actions).to(device)
        # 共享势函数：跨所有阶段共用一个 g
        shared_potential = PotentialNetwork(config.num_states).to(device)

        optimizer = optim.Adam(
            [
                {"params": list(reward_net.parameters()), "lr": float(config.malfl.reward_lr), "weight_decay": WEIGHT_DECAY},
                {"params": list(shared_potential.parameters()), "lr": float(config.malfl.shaping_lr), "weight_decay": WEIGHT_DECAY},
            ]
        )
        losses: List[float] = []

        # aggregate training tensors across stages (only those with targets)
        all_states: List[torch.Tensor] = []
        all_next_states: List[torch.Tensor] = []
        all_joint_actions: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        all_stage_ids: List[torch.Tensor] = []

        for stage_idx, dataset in enumerate(stage_datasets):
            if targets[agent_id][stage_idx] is None:
                continue

            tensors = dataset.concatenated(device=torch.device("cpu"), pin_memory=False)

            current_targets_cuda = targets[agent_id][stage_idx]
            assert current_targets_cuda is not None
            current_targets = current_targets_cuda.detach().cpu()

            own_actions = tensors.joint_actions[:, agent_id]
            per_sample_targets = current_targets[tensors.states, own_actions]

            all_states.append(tensors.states.to(device, non_blocking=True))
            all_next_states.append(tensors.next_states.to(device, non_blocking=True))
            all_joint_actions.append(tensors.joint_actions.to(device, non_blocking=True))
            all_targets.append(per_sample_targets.to(device, non_blocking=True))
            all_stage_ids.append(torch.full_like(tensors.states, stage_idx, dtype=torch.long, device=device))

        if not all_states:
            raise ValueError("Insufficient stages with targets to learn rewards.")

        states_tensor = torch.cat(all_states, dim=0)
        next_states_tensor = torch.cat(all_next_states, dim=0)
        joint_actions_tensor = torch.cat(all_joint_actions, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        stage_ids_tensor = torch.cat(all_stage_ids, dim=0)

        # cap batch size to avoid full-batch rigidity
        batch_size = min(config.malfl.reward_batch_size, states_tensor.size(0), MAX_BATCH_SIZE)
        epochs = config.malfl.reward_epochs
        logging.info(
            "[MA-LfL][Reward] Agent %d - dataset size=%d (batch=%d, epochs=%d, reward_lr=%.2e, shaping_lr=%.2e)",
            agent_id,
            states_tensor.size(0),
            batch_size,
            epochs,
            float(config.malfl.reward_lr),
            float(config.malfl.shaping_lr),
        )

        for epoch in range(epochs):
            perm = torch.randperm(states_tensor.size(0), device=device)
            epoch_loss = 0.0
            batches = 0

            for start in range(0, states_tensor.size(0), batch_size):
                idx = perm[start:start + batch_size]
                batch_states = states_tensor[idx]
                batch_next_states = next_states_tensor[idx]
                batch_joint_actions = joint_actions_tensor[idx]
                batch_targets = targets_tensor[idx]
                batch_stage_ids = stage_ids_tensor[idx]

                # 预测裸奖励
                reward_values = reward_net(batch_states, batch_joint_actions)

                # 共享势函数的 shaping：g(s) - γ g(s')
                g_curr = shared_potential(batch_states)
                g_next = shared_potential(batch_next_states)

                # 数值稳定：按“同一小批的阶段”分别去均值（不过度约束 g 的零点）
                shaping_terms = torch.zeros_like(reward_values)
                for s_idx in batch_stage_ids.unique().tolist():
                    mask = (batch_stage_ids == s_idx)
                    if not torch.any(mask):
                        continue
                    g_c = g_curr[mask] - g_curr[mask].mean()
                    g_n = g_next[mask] - g_next[mask].mean()
                    shaping_terms[mask] = g_c - config.gamma * g_n

                prediction = reward_values + shaping_terms
                loss = F.mse_loss(prediction, batch_targets)

                # 势函数 L2 正则（有梯度）
                if config.malfl.potential_reg_weight > 0:
                    g_reg = (g_curr - g_curr.mean()).pow(2).mean()
                    loss = loss + config.malfl.potential_reg_weight * g_reg

                optimizer.zero_grad()
                loss.backward()

                # 梯度裁剪（两组参数一起裁剪）
                all_params = list(reward_net.parameters()) + list(shared_potential.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP_MAX_NORM)

                optimizer.step()

                epoch_loss += float(loss.item())
                batches += 1

            losses.append(epoch_loss / max(1, batches))
            if (epoch + 1) % max(1, epochs // 10) == 0 or (epoch + 1) == epochs:
                logging.info(
                    "[MA-LfL][Reward] Agent %d - %d/%d epochs completed (latest loss=%.4f)",
                    agent_id,
                    epoch + 1,
                    epochs,
                    losses[-1],
                )

        # 导出裸奖励表（不含整形）
        with torch.no_grad():
            state_grid = torch.arange(config.num_states, device=device, dtype=torch.long)
            joint_action_grid = torch.cartesian_prod(
                torch.arange(config.num_actions, device=device),
                torch.arange(config.num_actions, device=device),
            )
            repeated_states = state_grid.unsqueeze(1).repeat(1, joint_action_grid.size(0)).reshape(-1)
            repeated_actions = joint_action_grid.repeat(config.num_states, 1)
            rewards = reward_net(repeated_states, repeated_actions).reshape(
                config.num_states, config.num_actions, config.num_actions
            )

        bundles.append(
            RewardModelBundle(
                reward_network=reward_net,
                potential_networks=[shared_potential],  # 共享版本：仅一个
                losses=losses,
            )
        )
        predicted_tables.append(rewards.detach().clone())

    return RewardLearningResult(bundles=bundles, predicted_reward_tables=predicted_tables)


# ----------------------------
# Alignment evaluation (extra, without changing your existing evaluate_rewards)
# ----------------------------

def _pearson_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    # 处理常数向量情况
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan")

    # Pearson
    px = float(np.corrcoef(x, y)[0, 1])

    # Spearman（纯 numpy 实现）
    def ranks(a: np.ndarray) -> np.ndarray:
        order = a.argsort(kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a), dtype=np.float64)
        # 平均并列名次（处理 ties）
        vals = a[order]
        start = 0
        while start < len(vals):
            end = start + 1
            while end < len(vals) and vals[end] == vals[start]:
                end += 1
            if end - start > 1:
                avg = (start + end - 1) / 2.0
                ranks[order[start:end]] = avg
            start = end
        return ranks

    rx = ranks(x)
    ry = ranks(y)
    # 再做 Pearson 即 Spearman
    sx = float(np.corrcoef(rx, ry)[0, 1])
    return px, sx


def compute_alignment_metrics_and_save(
    config: ExperimentConfig,
    reward_learning: RewardLearningResult,
    targets: List[List[torch.Tensor | None]],
    stage_datasets: Sequence[StageDataset],
    output_dir: Path,
    device: torch.device,
) -> None:
    """
    计算并保存“对齐评估”：
      对每个智能体，按样本拼接 (s, a, s')，计算
        Z_hat = R_hat(s,a) + g(s) - gamma g(s')
      与训练目标 Y_h(s, a^i) 的 Pearson/Spearman（整体 + 分阶段）。
    保存到 <output_dir>/alignment_metrics.json
    """
    num_agents = 2
    num_stages = len(stage_datasets)
    payload = {"agents": []}

    for agent_id in range(num_agents):
        bundle = reward_learning.bundles[agent_id]
        reward_net = bundle.reward_network
        # 共享势函数：列表中只有一个
        g_net = bundle.potential_networks[0]
        g_net.eval()
        reward_net.eval()

        # 拼接所有可用阶段的样本
        all_pred: List[np.ndarray] = []
        all_targ: List[np.ndarray] = []
        per_stage_stats: List[Dict[str, float]] = []

        for stage_idx, dataset in enumerate(stage_datasets):
            if targets[agent_id][stage_idx] is None:
                per_stage_stats.append({"stage": stage_idx, "pearson": float("nan"), "spearman": float("nan")})
                continue

            tensors = dataset.concatenated(device=device)
            own_actions = tensors.joint_actions[:, agent_id]
            # 取目标 Y_h(s, a^i)
            stage_targets = targets[agent_id][stage_idx]
            assert stage_targets is not None
            y_vals = stage_targets[tensors.states, own_actions]  # [N]
            # 预测 Z_hat
            with torch.no_grad():
                r_hat = reward_net(tensors.states, tensors.joint_actions)  # [N]
                g_s = g_net(tensors.states)
                g_sp = g_net(tensors.next_states)
                # 与训练时一致：按当前阶段样本去均值，避免零点漂移影响
                g_s = g_s - g_s.mean()
                g_sp = g_sp - g_sp.mean()
                z_hat = r_hat + (g_s - config.gamma * g_sp)

            x = z_hat.detach().cpu().numpy().astype(np.float64)
            y = y_vals.detach().cpu().numpy().astype(np.float64)
            p, s = _pearson_spearman(x, y)
            per_stage_stats.append({"stage": stage_idx, "pearson": p, "spearman": s})
            all_pred.append(x)
            all_targ.append(y)

        if all_pred:
            X = np.concatenate(all_pred, axis=0)
            Y = np.concatenate(all_targ, axis=0)
            p_all, s_all = _pearson_spearman(X, Y)
        else:
            p_all, s_all = float("nan"), float("nan")

        payload["agents"].append(
            {
                "agent_id": agent_id,
                "overall": {"pearson": p_all, "spearman": s_all},
                "stages": per_stage_stats,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "alignment_metrics.json").write_text(json.dumps(payload, indent=2))


# ----------------------------
# Orchestration
# ----------------------------

def run_ma_lfl(
    config: ExperimentConfig,
    stage_datasets: Sequence[StageDataset],
    output_dir: Path,
    device: torch.device | None = None,
) -> MALFLArtifacts:
    device = device or torch.device(config.device)

    logging.info("[MA-LfL] Estimating policies...")
    policy_estimation = estimate_policies(stage_datasets, config, device)

    logging.info("[MA-LfL] Computing reward targets...")
    targets = compute_reward_targets(policy_estimation, stage_datasets, config, device)

    logging.info("[MA-LfL] Learning rewards...")
    reward_learning = learn_rewards(stage_datasets, targets, config, device)

    # === 准备势函数表 ===
    potential_tables = []
    for bundle in reward_learning.bundles:
        assert len(bundle.potential_networks) == 1
        g_net = bundle.potential_networks[0]
        g_net.eval()
        with torch.no_grad():
            g_table = g_net(torch.arange(config.num_states, device=device, dtype=torch.long)).detach().cpu()
        potential_tables.append(g_table)

    # === 评估阶段 ===
    evaluation_dir = output_dir / "evaluation"
    logging.info("[MA-LfL] Evaluating rewards (bare + shaped)...")
    evaluation_result = evaluate_rewards(
        config=config,
        predicted_tables=reward_learning.predicted_reward_tables,
        targets=targets,
        stage_datasets=stage_datasets,
        output_dir=evaluation_dir,
        potential_tables=potential_tables,
    )

    # 兼容处理：新版 evaluate_rewards 返回 dict，否则旧版返回单个对象
    if isinstance(evaluation_result, dict):
        evaluation_modes = evaluation_result
        evaluation_main = evaluation_result.get("bare") or list(evaluation_result.values())[0]
    else:
        evaluation_modes = {"bare": evaluation_result}
        evaluation_main = evaluation_result

    # === 额外的对齐评估 ===
    logging.info("[MA-LfL] Computing alignment metrics (R_hat + g - gamma*g' vs Y_h)...")
    compute_alignment_metrics_and_save(
        config=config,
        reward_learning=reward_learning,
        targets=targets,
        stage_datasets=stage_datasets,
        output_dir=evaluation_dir,
        device=device,
    )

    # === 保存所有结果 ===
    artifacts = MALFLArtifacts(
        policy_estimation=policy_estimation,
        reward_learning=reward_learning,
        targets=targets,
        evaluation=evaluation_main,
    )
    # 新增：保存多模式评估结果（bare / shaped）
    artifacts.evaluation_modes = evaluation_modes  # type: ignore

    artifacts.save(output_dir)
    logging.info("[MA-LfL] Pipeline complete.")
    return artifacts
