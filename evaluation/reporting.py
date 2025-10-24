from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from config import ExperimentConfig
from environments.gridworld import GridWorld, RewardFamily
from data.structures import StageDataset
from evaluation.metrics import correlation_pair
from evaluation.visualization import plot_reward_comparison, plot_trend


@dataclass
class AgentMetrics:
    pearson: float
    spearman: float


@dataclass
class RewardEvaluationResult:
    agent_metrics: List[AgentMetrics]
    mean_metrics: AgentMetrics
    trend_stages: List[int]
    trend_pearson: List[float]
    trend_spearman: List[float]
    heatmap_paths: Dict[str, Path]
    trend_path: Path


def _ground_truth_reward_table(
    env: GridWorld, agent_id: int, num_states: int, num_actions: int
) -> np.ndarray:
    table = np.zeros((num_states, num_actions, num_actions), dtype=np.float32)
    for state_index in range(num_states):
        joint_state = env.index_to_joint_state(state_index)
        reward = env.reward(agent_id, joint_state)
        table[state_index, :, :] = reward
    return table


def _state_heatmap_from_table(table: np.ndarray, grid_size: int) -> np.ndarray:
    flattened = table.mean(axis=(1, 2))
    return flattened.reshape(grid_size * grid_size, grid_size * grid_size)


def _compute_trend(
    targets: Sequence[Sequence[torch.Tensor | None]],
    stage_datasets: Sequence[StageDataset],
    true_reward_states: np.ndarray,
) -> Tuple[List[int], List[float], List[float]]:
    stages: List[int] = []
    pearson: List[float] = []
    spearman: List[float] = []

    num_agents = len(targets)
    max_stage = min(len(stage_datasets), len(targets[0]))

    for stage_idx in range(max_stage):
        if any(agent_targets[stage_idx] is None for agent_targets in targets):
            continue

        dataset = stage_datasets[stage_idx]
        tensors = dataset.concatenated(device=torch.device("cpu"), pin_memory=False)
        states_np = tensors.states.cpu().numpy()
        joint_actions_np = tensors.joint_actions.cpu().numpy()
        if states_np.size == 0:
            continue

        agent_vectors = []
        for agent_id in range(num_agents):
            tensor = targets[agent_id][stage_idx]
            assert tensor is not None
            target_np = tensor.detach().cpu().numpy()
            y_values = target_np[states_np, joint_actions_np[:, agent_id]]
            agent_vectors.append(y_values)

        if not agent_vectors:
            continue

        stage_matrix = np.stack(agent_vectors, axis=0)
        stage_mean = np.mean(stage_matrix, axis=0)
        true_values = true_reward_states[states_np]

        p, s = correlation_pair(stage_mean, true_values)
        stages.append(stage_idx)
        pearson.append(p)
        spearman.append(s)

    return stages, pearson, spearman


def evaluate_rewards(
    config: ExperimentConfig,
    predicted_tables: Sequence[torch.Tensor],
    targets: Sequence[Sequence[torch.Tensor | None]],
    stage_datasets: Sequence[StageDataset],
    output_dir: Path,
) -> RewardEvaluationResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = GridWorld(
        size=config.environment.grid_size,
        start_positions=tuple(tuple(pos) for pos in config.environment.start_positions),
        goal_position=tuple(config.environment.goal_position),
        reward_family=RewardFamily(config.environment.reward_family),
    )

    predicted_np = [table.detach().cpu().numpy() for table in predicted_tables]

    agent_metrics: List[AgentMetrics] = []
    heatmap_paths: Dict[str, Path] = {}
    correlations_pcc: List[float] = []
    correlations_scc: List[float] = []

    for agent_id, pred in enumerate(predicted_np):
        true_table = _ground_truth_reward_table(
            env, agent_id, config.num_states, config.num_actions
        )
        pcc, scc = correlation_pair(pred, true_table)
        agent_metrics.append(AgentMetrics(pearson=pcc, spearman=scc))
        correlations_pcc.append(pcc)
        correlations_scc.append(scc)

        heatmap_matrix_true = _state_heatmap_from_table(true_table, config.environment.grid_size)
        heatmap_matrix_pred = _state_heatmap_from_table(pred, config.environment.grid_size)
        agent_prefix = f"agent_{agent_id}"
        agent_dir = output_dir / agent_prefix
        plot_reward_comparison(
            heatmap_matrix_true,
            heatmap_matrix_pred,
            config.environment.grid_size,
            agent_dir,
            agent_prefix,
        )
        heatmap_paths[agent_prefix] = agent_dir

    mean_metrics = AgentMetrics(
        pearson=float(np.mean(correlations_pcc)),
        spearman=float(np.mean(correlations_scc)),
    )

    true_rewards_agent0 = np.array(
        [env.reward(0, env.index_to_joint_state(idx)) for idx in range(config.num_states)],
        dtype=np.float32,
    )
    true_rewards_agent1 = np.array(
        [env.reward(1, env.index_to_joint_state(idx)) for idx in range(config.num_states)],
        dtype=np.float32,
    )
    true_rewards_state = 0.5 * (true_rewards_agent0 + true_rewards_agent1)
    stages, trend_pcc, trend_scc = _compute_trend(targets, stage_datasets, true_rewards_state)
    trend_path = output_dir / "reward_trend.png"
    if stages:
        plot_trend(stages, trend_pcc, trend_scc, trend_path)
    else:
        trend_path.touch()

    return RewardEvaluationResult(
        agent_metrics=agent_metrics,
        mean_metrics=mean_metrics,
        trend_stages=stages,
        trend_pearson=trend_pcc,
        trend_spearman=trend_scc,
        heatmap_paths=heatmap_paths,
        trend_path=trend_path,
    )
