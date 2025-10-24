from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import logging
import numpy as np
import torch

from agents.ma_spi_agent import MASPIAgent, MASPIAgentConfig
from config import ExperimentConfig
from data.structures import StageDataset, Trajectory
from environments.gridworld import GridWorld, RewardFamily


@dataclass
class StageArtifacts:
    dataset: StageDataset
    policy_snapshots: List[Dict[str, torch.Tensor]]
    policy_logits: List[List[List[float]]]


@dataclass
class MASPIArtifacts:
    stages: List[StageArtifacts] = field(default_factory=list)
    q_losses: Dict[int, List[float]] = field(default_factory=lambda: {0: [], 1: []})
    policy_losses: Dict[int, List[float]] = field(default_factory=lambda: {0: [], 1: []})
    random_seed: int | None = None
    final_policy_snapshots: List[Dict[str, torch.Tensor]] | None = None
    final_policy_logits: List[List[List[float]]] | None = None

    def save(self, base_dir: Path) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        metadata = {"seed": self.random_seed, "num_stages": len(self.stages)}
        (base_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

        for stage_idx, artifacts in enumerate(self.stages):
            stage_dir = base_dir / f"stage_{stage_idx:02d}"
            stage_dir.mkdir(exist_ok=True)

            # Save trajectory tensors
            trajectories = []
            for traj in artifacts.dataset.trajectories:
                trajectories.append(
                    {
                        "states": traj.states,
                        "joint_actions": traj.joint_actions,
                        "next_states": traj.next_states,
                        "rewards_agent_0": traj.rewards_agent_0,
                        "rewards_agent_1": traj.rewards_agent_1,
                        "log_probs_agent_0": traj.action_log_probs_agent_0,
                        "log_probs_agent_1": traj.action_log_probs_agent_1,
                    }
                )
            (stage_dir / "trajectories.json").write_text(json.dumps(trajectories, indent=2))

            # Save policy logits snapshot
            snapshot_dir = stage_dir / "policy_snapshots"
            snapshot_dir.mkdir(exist_ok=True)
            (stage_dir / "policy_logits.json").write_text(json.dumps(artifacts.policy_logits))
            for agent_id, snapshot in enumerate(artifacts.policy_snapshots):
                torch.save(snapshot, snapshot_dir / f"agent_{agent_id}.pt")

        torch.save(self.q_losses, base_dir / "q_losses.pt")
        torch.save(self.policy_losses, base_dir / "policy_losses.pt")

        if self.final_policy_logits is not None:
            final_dir = base_dir / "final_policy"
            final_dir.mkdir(exist_ok=True)
            (final_dir / "policy_logits.json").write_text(json.dumps(self.final_policy_logits))
            if self.final_policy_snapshots is not None:
                for agent_id, snapshot in enumerate(self.final_policy_snapshots):
                    torch.save(snapshot, final_dir / f"agent_{agent_id}.pt")


def clone_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def run_ma_spi(config: ExperimentConfig) -> MASPIArtifacts:
    torch.manual_seed(config.maspi.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.maspi.seed)
    np.random.seed(config.maspi.seed)
    if config.maspi.num_workers > 0:
        torch.set_num_threads(config.maspi.num_workers)

    env = GridWorld(
        size=config.environment.grid_size,
        start_positions=tuple(tuple(pos) for pos in config.environment.start_positions),
        goal_position=tuple(config.environment.goal_position),
        reward_family=RewardFamily(config.environment.reward_family),
        seed=config.maspi.seed,
    )

    device = torch.device(config.device)
    agents = [
        MASPIAgent(
            agent_id=0,
            config=MASPIAgentConfig(
                num_states=config.num_states,
                num_actions=config.num_actions,
                alpha=config.alpha,
                gamma=config.gamma,
                policy_lr=config.optimization.policy_lr,
                q_lr=config.optimization.q_lr,
                value_regularizer=config.optimization.value_regularizer,
                device=device,
                use_amp=config.maspi.use_amp,
            ),
        ),
        MASPIAgent(
            agent_id=1,
            config=MASPIAgentConfig(
                num_states=config.num_states,
                num_actions=config.num_actions,
                alpha=config.alpha,
                gamma=config.gamma,
                policy_lr=config.optimization.policy_lr,
                q_lr=config.optimization.q_lr,
                value_regularizer=config.optimization.value_regularizer,
                device=device,
                use_amp=config.maspi.use_amp,
            ),
        ),
    ]

    artifacts = MASPIArtifacts(random_seed=config.maspi.seed)

    num_stages = config.maspi.num_iterations
    logging.info(
        "[MA-SPI] Starting run with %d stages, %d episodes/stage, episode length %d",
        num_stages,
        config.maspi.evaluation_episodes_per_iteration,
        config.maspi.episode_length,
    )
    for stage_idx in range(num_stages):
        stage_label = f"[MA-SPI] Stage {stage_idx + 1}/{num_stages}"
        logging.info("%s - collecting trajectories...", stage_label)
        # Snapshot current policies before data collection
        policy_snapshots = [clone_state_dict(agent.policy_net) for agent in agents]
        state_grid = torch.arange(config.num_states, device=device, dtype=torch.long)
        policy_logits_values = []
        for agent in agents:
            with torch.no_grad():
                logits = agent.policy_net(state_grid).cpu().tolist()
            policy_logits_values.append(logits)

        stage_dataset = StageDataset()
        total_episodes = config.maspi.evaluation_episodes_per_iteration
        episode_interval = max(1, total_episodes // 5)
        for episode_idx in range(total_episodes):
            env.reset()
            trajectory = Trajectory()

            for step_idx in range(config.maspi.episode_length):
                state_index = env.state_index
                state_tensor = torch.tensor([state_index], device=device, dtype=torch.long)

                joint_action = []
                log_probs = []
                for agent in agents:
                    with torch.no_grad():
                        dist = agent.distribution(state_tensor)
                        action = dist.sample()
                        log_prob = dist.log_prob(action)
                    joint_action.append(int(action.item()))
                    log_probs.append(float(log_prob.item()))

                joint_action_tuple = (joint_action[0], joint_action[1])
                next_state, rewards = env.step(joint_action_tuple)
                next_state_index = env.joint_state_to_index(next_state)

                trajectory.append(
                    state_index=state_index,
                    joint_action=joint_action_tuple,
                    next_state_index=next_state_index,
                    reward_agent_0=rewards[0],
                    reward_agent_1=rewards[1],
                    log_prob_agent_0=log_probs[0],
                    log_prob_agent_1=log_probs[1],
                )

            stage_dataset.add(trajectory)
            if (episode_idx + 1) % episode_interval == 0 or (episode_idx + 1) == total_episodes:
                logging.info(
                    "%s - collected %d/%d episodes",
                    stage_label,
                    episode_idx + 1,
                    total_episodes,
                )

        artifacts.stages.append(
            StageArtifacts(
                dataset=stage_dataset,
                policy_snapshots=policy_snapshots,
                policy_logits=policy_logits_values,
            )
        )

        dataset_device = (
            device if (config.maspi.cache_stage_on_gpu and device.type == "cuda") else torch.device("cpu")
        )
        pin_memory = config.maspi.pin_memory and dataset_device.type == "cpu" and device.type == "cuda"
        tensors = stage_dataset.concatenated(device=dataset_device, pin_memory=pin_memory)
        dataset_size = tensors.states.size(0)
        # Episode-level reward statistics
        rewards_agent_0 = [float(np.sum(traj.rewards_agent_0)) for traj in stage_dataset.trajectories]
        rewards_agent_1 = [float(np.sum(traj.rewards_agent_1)) for traj in stage_dataset.trajectories]
        if rewards_agent_0:
            logging.info(
                "%s - Agent 0 avg episode reward: %.3f (std %.3f)",
                stage_label,
                float(np.mean(rewards_agent_0)),
                float(np.std(rewards_agent_0)),
            )
            logging.info(
                "%s - Agent 1 avg episode reward: %.3f (std %.3f)",
                stage_label,
                float(np.mean(rewards_agent_1)),
                float(np.std(rewards_agent_1)),
            )
        logging.info("%s - dataset size: %d transitions", stage_label, dataset_size)
        if dataset_size == 0:
            raise ValueError("Stage dataset is empty; cannot perform MA-SPI updates.")
        if dataset_size > 5_000_000:
            logging.warning(
                "%s - large dataset detected (%d samples). Consider reducing evaluation episodes if runtime is excessive.",
                stage_label,
                dataset_size,
            )

        max_batch_size = min(config.maspi.update_batch_size, dataset_size)
        min_batch_size = max(1, min(config.maspi.min_update_batch_size, max_batch_size))
        shrink_factor = float(config.maspi.batch_shrink_factor)
        growth_factor = float(config.maspi.batch_growth_factor)
        if shrink_factor <= 0 or shrink_factor >= 1:
            shrink_factor = 0.5
        if growth_factor <= 1:
            growth_factor = 1.1

        def _sample_indices(current_batch_size: int) -> torch.Tensor:
            if dataset_size <= current_batch_size:
                return torch.arange(dataset_size, dtype=torch.long)
            return torch.randint(0, dataset_size, (current_batch_size,), dtype=torch.long)

        def _shrink(size: int) -> int:
            new_size = max(int(size * shrink_factor), min_batch_size)
            if new_size >= size and size > min_batch_size:
                new_size = max(min_batch_size, size - 1)
            return max(new_size, min_batch_size)

        def _grow(size: int) -> int:
            if size >= max_batch_size:
                return size
            new_size = int(size * growth_factor)
            if new_size <= size:
                new_size = size + 1
            return min(new_size, max_batch_size)

        def _gather_rows(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
            if tensor.device.type == "cpu":
                return tensor[indices]
            idx_device = indices.to(tensor.device, non_blocking=True)
            return tensor.index_select(0, idx_device)

        q_batch_size = max_batch_size
        total_q_updates = config.maspi.q_updates_per_stage
        last_q_losses: Dict[int, float] = {}
        for update_idx in range(config.maspi.q_updates_per_stage):
            attempts = 0
            while True:
                batch_idx = _sample_indices(q_batch_size)
                try:
                    batch_states = _gather_rows(tensors.states, batch_idx)
                    batch_joint_actions = _gather_rows(tensors.joint_actions, batch_idx)
                    batch_next_states = _gather_rows(tensors.next_states, batch_idx)
                    agent_losses: List[tuple[int, float]] = []
                    for agent_id, agent in enumerate(agents):
                        rewards_tensor = tensors.rewards_agent_0 if agent_id == 0 else tensors.rewards_agent_1
                        batch_rewards = _gather_rows(rewards_tensor, batch_idx)
                        loss = agent.update_q(
                            states=batch_states,
                            joint_actions=batch_joint_actions,
                            rewards=batch_rewards,
                            next_states=batch_next_states,
                        )
                        agent_losses.append((agent_id, loss))
                    for agent_id, loss in agent_losses:
                        artifacts.q_losses[agent_id].append(loss)
                        last_q_losses[agent_id] = loss
                    break
                except torch.cuda.OutOfMemoryError:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    previous_size = q_batch_size
                    q_batch_size = _shrink(q_batch_size)
                    attempts += 1
                    logging.warning(
                        "%s - Q update %d: CUDA OOM, reducing batch %d -> %d",
                        stage_label,
                        update_idx + 1,
                        previous_size,
                        q_batch_size,
                    )
                    if q_batch_size == previous_size and q_batch_size == min_batch_size:
                        raise
                    continue
            if attempts == 0:
                q_batch_size = _grow(q_batch_size)
            if (update_idx + 1) % max(1, total_q_updates // 5) == 0 or (update_idx + 1) == total_q_updates:
                if not last_q_losses:
                    last_q_losses = {agent_id: artifacts.q_losses[agent_id][-1] for agent_id in artifacts.q_losses}
                loss_report = ", ".join(
                    f"A{agent_id}_Q_loss:{loss_value:.4f}"
                    for agent_id, loss_value in sorted(last_q_losses.items())
                )
                logging.info(
                    "%s - Q updates %d/%d (current batch %d) [%s]",
                    stage_label,
                    update_idx + 1,
                    total_q_updates,
                    q_batch_size,
                    loss_report,
                )

        policy_batch_size = q_batch_size
        total_policy_updates = config.maspi.policy_updates_per_stage
        last_policy_losses: Dict[int, float] = {}
        for update_idx in range(config.maspi.policy_updates_per_stage):
            attempts = 0
            while True:
                batch_idx = _sample_indices(policy_batch_size)
                try:
                    batch_states = _gather_rows(tensors.states, batch_idx)
                    for agent_id, agent in enumerate(agents):
                        loss = agent.update_policy(batch_states)
                        artifacts.policy_losses[agent_id].append(loss)
                        last_policy_losses[agent_id] = loss
                    break
                except torch.cuda.OutOfMemoryError:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    previous_size = policy_batch_size
                    policy_batch_size = _shrink(policy_batch_size)
                    attempts += 1
                    logging.warning(
                        "%s - policy update %d: CUDA OOM, reducing batch %d -> %d",
                        stage_label,
                        update_idx + 1,
                        previous_size,
                        policy_batch_size,
                    )
                    if policy_batch_size == previous_size and policy_batch_size == min_batch_size:
                        raise
                    continue
            if attempts == 0:
                policy_batch_size = _grow(policy_batch_size)
            if (update_idx + 1) % max(1, total_policy_updates // 5) == 0 or (update_idx + 1) == total_policy_updates:
                if not last_policy_losses:
                    last_policy_losses = {
                        agent_id: artifacts.policy_losses[agent_id][-1] for agent_id in artifacts.policy_losses
                    }
                loss_report = ", ".join(
                    f"A{agent_id}_Policy_loss:{loss_value:.4f}"
                    for agent_id, loss_value in sorted(last_policy_losses.items())
                )
                logging.info(
                    "%s - policy updates %d/%d (current batch %d) [%s]",
                    stage_label,
                    update_idx + 1,
                    total_policy_updates,
                    policy_batch_size,
                    loss_report,
                )

        logging.info("%s - completed", stage_label)

    final_policy_snapshots = [clone_state_dict(agent.policy_net) for agent in agents]
    state_grid = torch.arange(config.num_states, device=device, dtype=torch.long)
    final_policy_logits = []
    for agent in agents:
        with torch.no_grad():
            logits = agent.policy_net(state_grid).cpu().tolist()
        final_policy_logits.append(logits)

    artifacts.final_policy_snapshots = final_policy_snapshots
    artifacts.final_policy_logits = final_policy_logits

    return artifacts
