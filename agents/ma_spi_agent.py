from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.distributions import Categorical

try:  # PyTorch >= 2.3
    from torch.amp import GradScaler as _TorchGradScaler
    from torch.amp import autocast as _torch_autocast

    def _create_grad_scaler(enabled: bool) -> _TorchGradScaler:
        try:
            return _TorchGradScaler("cuda", enabled=enabled)
        except TypeError:
            return _TorchGradScaler(enabled=enabled)

    def _autocast(enabled: bool):
        return _torch_autocast("cuda", enabled=enabled)

except ImportError:  # Fallback for older versions
    from torch.cuda.amp import GradScaler as _TorchGradScaler
    from torch.cuda.amp import autocast as _torch_autocast

    def _create_grad_scaler(enabled: bool) -> _TorchGradScaler:
        return _TorchGradScaler(enabled=enabled)

    def _autocast(enabled: bool):
        return _torch_autocast(enabled=enabled)

from models.policy_net import PolicyNetwork
from models.q_net import SoftQNetwork


@dataclass
class MASPIAgentConfig:
    num_states: int
    num_actions: int
    alpha: float
    gamma: float
    policy_lr: float
    q_lr: float
    value_regularizer: float
    device: torch.device
    use_amp: bool = False


class MASPIAgent:
    """Implements the MA-SPI learner for a single agent."""

    def __init__(self, agent_id: int, config: MASPIAgentConfig) -> None:
        self.agent_id = agent_id
        self.config = config
        self.device = config.device
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.value_regularizer = config.value_regularizer
        self.use_amp = bool(config.use_amp and self.device.type == "cuda")

        self.policy_net = PolicyNetwork(config.num_states, config.num_actions).to(self.device)
        self.q_net = SoftQNetwork(config.num_states, config.num_actions).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=config.policy_lr)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=config.q_lr)
        self.q_scaler = _create_grad_scaler(self.use_amp)
        self.policy_scaler = _create_grad_scaler(self.use_amp)

    def distribution(self, state_indices: torch.Tensor) -> Categorical:
        logits = self.policy_net(state_indices)
        return Categorical(logits=logits)

    def get_action(self, state_index: int) -> int:
        state_tensor = torch.tensor([state_index], device=self.device, dtype=torch.long)
        dist = self.distribution(state_tensor)
        action = dist.sample()
        return int(action.item())

    def log_prob(self, state_indices: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.policy_net.log_prob(state_indices, actions)

    def entropy(self, state_indices: torch.Tensor) -> torch.Tensor:
        logits = self.policy_net(state_indices)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -torch.sum(probs * log_probs, dim=-1)
        return entropy

    def update_q(
        self,
        states: torch.Tensor,
        joint_actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
    ) -> float:
        """One-step soft Bellman update following Eq (5)."""
        non_blocking = self.device.type == "cuda"
        states = states.to(self.device, non_blocking=non_blocking)
        joint_actions = joint_actions.to(self.device, non_blocking=non_blocking)
        rewards = rewards.to(self.device, non_blocking=non_blocking)
        next_states = next_states.to(self.device, non_blocking=non_blocking)
        own_actions = joint_actions[:, self.agent_id]

        with torch.no_grad():
            next_logits = self.policy_net(next_states)
            next_log_probs = torch.log_softmax(next_logits, dim=-1)
            next_probs = next_log_probs.exp()
            next_q_values = self.q_net.all_action_values(next_states)
            entropy = -torch.sum(next_probs * next_log_probs, dim=-1)
            expected_q = torch.sum(next_probs * next_q_values, dim=-1) + self.alpha * entropy
            target = rewards + self.gamma * expected_q
            target = target.detach()

        self.q_optimizer.zero_grad(set_to_none=True)
        with _autocast(self.use_amp):
            q_values = self.q_net(states, own_actions)
            target_cast = target.to(dtype=q_values.dtype)
            loss = 0.5 * F.mse_loss(q_values, target_cast)
            if self.value_regularizer > 0:
                loss = loss + 0.5 * self.value_regularizer * (q_values ** 2).mean()

        self.q_scaler.scale(loss).backward()
        self.q_scaler.step(self.q_optimizer)
        self.q_scaler.update()
        loss_value = loss.detach().float().cpu().item()
        return float(loss_value)

    def update_policy(self, states: torch.Tensor) -> float:
        """Soft policy improvement step matching Eq (6)."""
        non_blocking = self.device.type == "cuda"
        states = states.to(self.device, non_blocking=non_blocking)
        with torch.no_grad():
            target_logits = self.q_net.all_action_values(states) / self.alpha
            target_probs = torch.softmax(target_logits, dim=-1)
            target_log_probs = torch.log(target_probs + 1e-8)

        self.policy_optimizer.zero_grad(set_to_none=True)
        with _autocast(self.use_amp):
            current_logits = self.policy_net(states)
            current_log_probs = torch.log_softmax(current_logits, dim=-1)
            target_probs_cast = target_probs.to(dtype=current_logits.dtype)
            target_log_probs_cast = target_log_probs.to(dtype=current_logits.dtype)
            loss = torch.sum(
                target_probs_cast * (target_log_probs_cast - current_log_probs), dim=-1
            ).mean()

        self.policy_scaler.scale(loss).backward()
        self.policy_scaler.step(self.policy_optimizer)
        self.policy_scaler.update()
        loss_value = loss.detach().float().cpu().item()
        return float(loss_value)

    def state_dict(self) -> dict:
        return {
            "policy": self.policy_net.state_dict(),
            "q_network": self.q_net.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.policy_net.load_state_dict(state_dict["policy"])
        self.q_net.load_state_dict(state_dict["q_network"])
