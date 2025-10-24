from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import itertools
import json
from typing import Dict, List, Sequence, Tuple

import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig
from data.structures import StageDataset
from environments.gridworld import GridWorld, RewardFamily
from evaluation.reporting import RewardEvaluationResult, evaluate_rewards
from models.policy_net import PolicyNetwork
from models.reward_net import JointRewardNetwork
from models.shaping_net import PotentialNetwork


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

        targets_dir = base_dir / "targets"
        targets_dir.mkdir(exist_ok=True)
        for agent_id, agent_targets in enumerate(self.targets):
            agent_dir = targets_dir / f"agent_{agent_id}"
            agent_dir.mkdir(exist_ok=True)
            for stage_idx, tensor in enumerate(agent_targets):
                if tensor is None:
                    continue
                torch.save(tensor.cpu(), agent_dir / f"stage_{stage_idx:02d}.pt")

        reward_dir = base_dir / "reward_learning"
        reward_dir.mkdir(exist_ok=True)
        for agent_id, bundle in enumerate(self.reward_learning.bundles):
            agent_dir = reward_dir / f"agent_{agent_id}"
            agent_dir.mkdir(exist_ok=True)
            torch.save(bundle.reward_network.state_dict(), agent_dir / "reward_model.pt")
            potentials_dir = agent_dir / "potentials"
            potentials_dir.mkdir(exist_ok=True)
            for stage_idx, net in enumerate(bundle.potential_networks):
                torch.save(net.state_dict(), potentials_dir / f"stage_{stage_idx:02d}.pt")
            np.save(agent_dir / "loss.npy", np.array(bundle.losses, dtype=np.float32))
            torch.save(self.reward_learning.predicted_reward_tables[agent_id].cpu(), agent_dir / "rewards.pt")

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
            losses: List[float] = []
            batch_size = min(config.malfl.policy_estimation_batch, states.size(0))
            for epoch in range(config.malfl.policy_estimation_epochs):
                perm = torch.randperm(states.size(0), device=device)
                epoch_loss = 0.0
                batches = 0
                for start in range(0, states.size(0), batch_size):
                    idx = perm[start : start + batch_size]
                    batch_states = states[idx]
                    batch_actions = actions[idx]
                    logits = model(batch_states)
                    dist = torch.distributions.Categorical(logits=logits)
                    log_prob = dist.log_prob(batch_actions)
                    entropy = dist.entropy()
                    loss = -log_prob.mean() - config.malfl.policy_entropy_coef * entropy.mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += float(loss.item())
                    batches += 1
            losses.append(epoch_loss / max(1, batches))
            if (epoch + 1) % max(1, config.malfl.policy_estimation_epochs // 5) == 0 or (epoch + 1) == config.malfl.policy_estimation_epochs:
                logging.info(
                    "[MA-LfL][Policy] Stage %d/%d - Agent %d: %d/%d epochs completed (loss=%.4f, baseline=%.4f)",
                    stage_idx + 1,
                    num_stages,
                    agent_id,
                    epoch + 1,
                    config.malfl.policy_estimation_epochs,
                    losses[-1],
                    baseline_loss,
                )

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


def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float) -> torch.Tensor:
    p_clamped = torch.clamp(p, eps, 1.0)
    q_clamped = torch.clamp(q, eps, 1.0)
    return torch.sum(p_clamped * (torch.log(p_clamped) - torch.log(q_clamped)), dim=-1)


def compute_reward_targets(
    policy_estimation: PolicyEstimationResult,
    stage_datasets: Sequence[StageDataset],
    config: ExperimentConfig,
    device: torch.device,
) -> List[List[torch.Tensor | None]]:
    """Compute Y_h targets for each agent and stage according to Eq (11)."""

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
            prev_opponent_policy = policy_estimation.policies[opponent_id][stage_idx - 1]
            current_opponent_policy = policy_estimation.policies[opponent_id][stage_idx]

            y_table = torch.zeros(
                (config.num_states, config.num_actions), dtype=torch.float32, device=device
            )

            for state_index in range(config.num_states):
                joint_state = env.index_to_joint_state(state_index)
                log_probs = torch.log(torch.clamp(current_policy.probs[state_index], eps, 1.0))

                for own_action in range(config.num_actions):
                    kl_expectation = 0.0
                    for opp_action in range(config.num_actions):
                        prob = float(current_opponent_policy.probs[state_index, opp_action].clamp(min=eps))
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
                        kl_expectation += prob * kl_value

                    y_value = -alpha * log_probs[own_action] - alpha * gamma * kl_expectation
                    y_table[state_index, own_action] = y_value

            targets[agent_id][stage_idx] = y_table

    return targets


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
        potentials = [
            PotentialNetwork(config.num_states).to(device) for _ in range(num_stages)
        ]
        reward_params = list(reward_net.parameters())
        potential_params = [param for net in potentials for param in net.parameters()]
        optimizer = optim.Adam(
            [
                {"params": reward_params, "lr": config.malfl.reward_lr},
                {"params": potential_params, "lr": config.malfl.shaping_lr},
            ]
        )
        losses: List[float] = []

        # Build dataset tensors
        all_states: List[torch.Tensor] = []
        all_next_states: List[torch.Tensor] = []
        all_joint_actions: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        all_stage_ids: List[torch.Tensor] = []

        pin_batches = device.type == "cuda"

        for stage_idx, dataset in enumerate(stage_datasets):
            if targets[agent_id][stage_idx] is None:
                continue
            tensors = dataset.concatenated(
                device=torch.device("cpu"),
                pin_memory=False,
            )
            current_targets_cuda = targets[agent_id][stage_idx]
            assert current_targets_cuda is not None
            current_targets = current_targets_cuda.detach().cpu()
            own_actions = tensors.joint_actions[:, agent_id]
            per_sample_targets = current_targets[tensors.states, own_actions]

            all_states.append(tensors.states.to(device, non_blocking=True))
            all_next_states.append(tensors.next_states.to(device, non_blocking=True))
            all_joint_actions.append(tensors.joint_actions.to(device, non_blocking=True))
            all_targets.append(per_sample_targets.to(device, non_blocking=True))
            all_stage_ids.append(
                torch.full_like(tensors.states, stage_idx, dtype=torch.long, device=device)
            )

        if not all_states:
            raise ValueError("Insufficient stages with targets to learn rewards.")

        states_tensor = torch.cat(all_states, dim=0)
        next_states_tensor = torch.cat(all_next_states, dim=0)
        joint_actions_tensor = torch.cat(all_joint_actions, dim=0)
        targets_tensor = torch.cat(all_targets, dim=0)
        stage_ids_tensor = torch.cat(all_stage_ids, dim=0)

        stage_indices_map: Dict[int, torch.Tensor] = {}
        for stage_idx in range(num_stages):
            mask = (stage_ids_tensor == stage_idx).nonzero(as_tuple=False).squeeze(-1)
            if mask.numel() > 0:
                stage_indices_map[stage_idx] = mask

        batch_size = min(config.malfl.reward_batch_size, states_tensor.size(0))
        epochs = config.malfl.reward_epochs
        logging.info(
            "[MA-LfL][Reward] Agent %d - dataset size=%d (batch=%d, epochs=%d)",
            agent_id,
            states_tensor.size(0),
            batch_size,
            epochs,
        )

        for epoch in range(epochs):
            perm = torch.randperm(states_tensor.size(0), device=device)
            epoch_loss = 0.0
            batches = 0
            for start in range(0, states_tensor.size(0), batch_size):
                idx = perm[start : start + batch_size]
                batch_states = states_tensor[idx]
                batch_next_states = next_states_tensor[idx]
                batch_joint_actions = joint_actions_tensor[idx]
                batch_targets = targets_tensor[idx]
                batch_stage_ids = stage_ids_tensor[idx]

                reward_values = reward_net(batch_states, batch_joint_actions)

                shaping_terms = torch.zeros_like(reward_values)
                unique_stages = batch_stage_ids.unique()
                for s_idx in unique_stages.tolist():
                    mask = batch_stage_ids == s_idx
                    if not torch.any(mask):
                        continue
                    g_curr = potentials[s_idx](batch_states[mask])
                    g_curr = g_curr - g_curr.mean()
                    g_next = potentials[s_idx](batch_next_states[mask])
                    g_next = g_next - g_next.mean()
                    shaping_terms[mask] = g_curr - config.gamma * g_next

                prediction = reward_values + shaping_terms
                loss = F.mse_loss(prediction, batch_targets)

                if config.malfl.potential_reg_weight > 0:
                    reg_loss = 0.0
                    for s_idx in unique_stages.tolist():
                        mask = batch_stage_ids == s_idx
                        if not torch.any(mask):
                            continue
                        ref_idx = stage_indices_map.get(s_idx)
                        if ref_idx is None:
                            continue
                        g_values = potentials[s_idx](states_tensor[ref_idx])
                        g_values = g_values - g_values.mean()
                        reg_loss = reg_loss + g_values.pow(2).mean()
                    loss = loss + config.malfl.potential_reg_weight * reg_loss

                optimizer.zero_grad()
                loss.backward()
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

        # Evaluate reward table
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
                potential_networks=potentials,
                losses=losses,
            )
        )
        predicted_tables.append(rewards.detach().clone())

    return RewardLearningResult(bundles=bundles, predicted_reward_tables=predicted_tables)


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
    evaluation_dir = output_dir / "evaluation"
    logging.info("[MA-LfL] Evaluating rewards...")
    evaluation = evaluate_rewards(
        config,
        reward_learning.predicted_reward_tables,
        targets,
        stage_datasets,
        evaluation_dir,
    )

    artifacts = MALFLArtifacts(
        policy_estimation=policy_estimation,
        reward_learning=reward_learning,
        targets=targets,
        evaluation=evaluation,
    )
    artifacts.save(output_dir)
    logging.info("[MA-LfL] Pipeline complete.")
    return artifacts
