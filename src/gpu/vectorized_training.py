from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from src.mesoscopic.rl_common import RunningMeanStd
from src.mesoscopic.rl_layer import ActorCritic
from src.mesoscopic.sac_training import QNetwork, SACActor


def resolve_torch_device(requested: str | None = "auto") -> torch.device:
    preference = (requested or "auto").lower()
    if preference in {"auto", "cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    if preference in {"auto", "mps"} and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_episode_metrics(
    speed_trace: torch.Tensor,
    acc_trace: torch.Tensor,
    gap_trace: torch.Tensor,
    dt: float,
) -> dict[str, torch.Tensor]:
    """Match the scalar metrics used by run_training.py for a batched rollout."""
    speed_flat = speed_trace.permute(1, 0, 2).reshape(speed_trace.shape[1], -1)
    acc_flat = acc_trace.permute(1, 0, 2).reshape(acc_trace.shape[1], -1)
    gap_flat = gap_trace.permute(1, 0, 2).reshape(gap_trace.shape[1], -1)

    last_window = min(100, speed_trace.shape[0])
    jerk = torch.diff(acc_trace, dim=0) / max(dt, 1e-6)
    if jerk.shape[0] > 0:
        jerk_sq = jerk.pow(2).mean(dim=0)
        rms_jerk = torch.sqrt(jerk_sq.mean(dim=1))
    else:
        rms_jerk = torch.zeros(speed_trace.shape[1], device=speed_trace.device)

    return {
        "speed_var_global": speed_flat.var(dim=1, unbiased=False),
        "speed_std_time_mean": speed_trace.std(dim=2, unbiased=False).mean(dim=0),
        "oscillation_amplitude": speed_trace.amax(dim=(0, 2)) - speed_trace.amin(dim=(0, 2)),
        "min_gap": gap_flat.amin(dim=1),
        "rms_acc": torch.sqrt(acc_flat.pow(2).mean(dim=1)),
        "rms_jerk": rms_jerk,
        "mean_speed": speed_flat.mean(dim=1),
        "mean_speed_last100": speed_trace[-last_window:].mean(dim=(0, 2)),
    }


@dataclass
class VectorizedRolloutBuffer:
    rollout_steps: int
    num_envs: int
    num_vehicles: int
    state_dim: int
    device: torch.device

    def __post_init__(self) -> None:
        shape_bn = (self.rollout_steps, self.num_envs, self.num_vehicles)
        self.obs = torch.zeros(*shape_bn, self.state_dim, device=self.device)
        self.actions = torch.zeros(*shape_bn, device=self.device)
        self.log_probs = torch.zeros(*shape_bn, device=self.device)
        self.values = torch.zeros(*shape_bn, device=self.device)
        self.rewards = torch.zeros(*shape_bn, device=self.device)
        self.dones = torch.zeros(*shape_bn, device=self.device)
        self.masks = torch.zeros(*shape_bn, device=self.device, dtype=torch.bool)
        self.action_lows = torch.zeros(*shape_bn, device=self.device)
        self.action_highs = torch.zeros(*shape_bn, device=self.device)
        self.ptr = 0

    def add(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        masks: torch.Tensor,
        action_lows: torch.Tensor,
        action_highs: torch.Tensor,
    ) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.values[self.ptr] = values
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        self.masks[self.ptr] = masks
        self.action_lows[self.ptr] = action_lows
        self.action_highs[self.ptr] = action_highs
        self.ptr += 1

    def compute_gae(
        self,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(self.values[: self.ptr])
        gae = torch.zeros_like(last_values)

        for step in reversed(range(self.ptr)):
            next_values = last_values if step == self.ptr - 1 else self.values[step + 1]
            mask = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * mask - self.values[step]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[step] = gae

        returns = advantages + self.values[: self.ptr]
        return advantages, returns

    def batches(
        self,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        minibatch_size: int,
    ) -> Iterator[dict[str, torch.Tensor]]:
        valid_idx = self.masks[: self.ptr].reshape(-1).nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            return

        flat_obs = self.obs[: self.ptr].reshape(-1, self.state_dim)[valid_idx]
        flat_actions = self.actions[: self.ptr].reshape(-1)[valid_idx]
        flat_log_probs = self.log_probs[: self.ptr].reshape(-1)[valid_idx]
        flat_values = self.values[: self.ptr].reshape(-1)[valid_idx]
        flat_advantages = advantages.reshape(-1)[valid_idx]
        flat_returns = returns.reshape(-1)[valid_idx]
        flat_lows = self.action_lows[: self.ptr].reshape(-1)[valid_idx]
        flat_highs = self.action_highs[: self.ptr].reshape(-1)[valid_idx]

        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        permutation = torch.randperm(valid_idx.numel(), device=self.device)

        for start in range(0, valid_idx.numel(), minibatch_size):
            batch_idx = permutation[start : start + minibatch_size]
            yield {
                "obs": flat_obs[batch_idx],
                "actions": flat_actions[batch_idx],
                "old_log_probs": flat_log_probs[batch_idx],
                "old_values": flat_values[batch_idx],
                "advantages": flat_advantages[batch_idx],
                "returns": flat_returns[batch_idx],
                "action_lows": flat_lows[batch_idx],
                "action_highs": flat_highs[batch_idx],
            }


class VectorizedPPOTrainer:
    """Batched PPO using the existing mustykey actor-critic."""

    def __init__(
        self,
        *,
        state_dim: int = 8,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip: float = 0.2,
        train_epochs: int = 10,
        gae_lambda: float = 0.95,
        minibatch_size: int = 256,
        value_coef: float = 0.5,
        entropy_coef: float = 1e-3,
        max_grad_norm: float = 0.5,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.model = ActorCritic(state_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip = clip
        self.train_epochs = train_epochs
        self.gae_lambda = gae_lambda
        self.minibatch_size = minibatch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def select_actions(
        self,
        obs: torch.Tensor,
        action_lows: torch.Tensor,
        action_highs: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_shape = obs.shape[:2]
        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_lows = action_lows.reshape(-1)
        flat_highs = action_highs.reshape(-1)

        actions, log_probs, _, values = self.model.sample_action(
            flat_obs,
            action_low=flat_lows,
            action_high=flat_highs,
            deterministic=deterministic,
        )

        actions = actions.reshape(*batch_shape)
        log_probs = log_probs.reshape(*batch_shape)
        values = values.reshape(*batch_shape)
        actions = torch.where(masks, actions, torch.zeros_like(actions))
        log_probs = torch.where(masks, log_probs, torch.zeros_like(log_probs))
        values = torch.where(masks, values, torch.zeros_like(values))
        return actions, log_probs, values

    def update(
        self,
        buffer: VectorizedRolloutBuffer,
        next_obs: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            flat_next_obs = next_obs.reshape(-1, next_obs.shape[-1])
            _, _, next_values = self.model(flat_next_obs)
            next_values = next_values.reshape(next_obs.shape[0], next_obs.shape[1])

        advantages, returns = buffer.compute_gae(next_values, self.gamma, self.gae_lambda)

        updates = 0
        actor_loss_total = 0.0
        critic_loss_total = 0.0
        entropy_total = 0.0

        for _ in range(self.train_epochs):
            for batch in buffer.batches(advantages, returns, self.minibatch_size):
                log_probs, entropy, values = self.model.evaluate_actions(
                    batch["obs"],
                    batch["actions"],
                    batch["action_lows"],
                    batch["action_highs"],
                )

                ratios = torch.exp(log_probs - batch["old_log_probs"])
                surr1 = ratios * batch["advantages"]
                surr2 = torch.clamp(ratios, 1.0 - self.clip, 1.0 + self.clip) * batch["advantages"]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, batch["returns"])
                entropy_term = entropy.mean()

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_term
                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                actor_loss_total += float(actor_loss.item())
                critic_loss_total += float(critic_loss.item())
                entropy_total += float(entropy_term.item())
                updates += 1

        divisor = max(updates, 1)
        return {
            "updates": divisor,
            "actor_loss": actor_loss_total / divisor,
            "critic_loss": critic_loss_total / divisor,
            "entropy": entropy_total / divisor,
        }

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


class TensorReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, device: torch.device) -> None:
        self.capacity = max(int(capacity), 1)
        self.device = device
        self.ptr = 0
        self.size = 0
        self.states = torch.zeros(self.capacity, state_dim, device=device)
        self.actions = torch.zeros(self.capacity, device=device)
        self.rewards = torch.zeros(self.capacity, device=device)
        self.next_states = torch.zeros(self.capacity, state_dim, device=device)
        self.dones = torch.zeros(self.capacity, device=device)
        self.action_lows = torch.zeros(self.capacity, device=device)
        self.action_highs = torch.zeros(self.capacity, device=device)
        self.next_action_lows = torch.zeros(self.capacity, device=device)
        self.next_action_highs = torch.zeros(self.capacity, device=device)

    def add_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        action_lows: torch.Tensor,
        action_highs: torch.Tensor,
        next_action_lows: torch.Tensor,
        next_action_highs: torch.Tensor,
    ) -> None:
        batch_size = states.shape[0]
        indices = (torch.arange(batch_size, device=self.device) + self.ptr) % self.capacity
        self.states[indices] = states
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.next_states[indices] = next_states
        self.dones[indices] = dones
        self.action_lows[indices] = action_lows
        self.action_highs[indices] = action_highs
        self.next_action_lows[indices] = next_action_lows
        self.next_action_highs[indices] = next_action_highs
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
            "action_lows": self.action_lows[idx],
            "action_highs": self.action_highs[idx],
            "next_action_lows": self.next_action_lows[idx],
            "next_action_highs": self.next_action_highs[idx],
        }

    def __len__(self) -> int:
        return self.size


class VectorizedSACTrainer:
    """Batched SAC preserving the existing inference actor shape."""

    def __init__(
        self,
        *,
        state_dim: int = 8,
        lr: float = 3e-4,
        gamma: float = 0.99,
        train_epochs: int = 10,
        tau: float = 0.005,
        entropy_alpha: float = 0.2,
        batch_size: int = 256,
        replay_size: int = 100000,
        replay_warmup: int = 2048,
        auto_entropy_tuning: bool = True,
        target_entropy: float = -1.0,
        normalize_states: bool = True,
        normalize_rewards: bool = True,
        reward_scale_epsilon: float = 1e-6,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.device = device
        self.model = SACActor(state_dim=state_dim).to(device)
        self.q1 = QNetwork(state_dim).to(device)
        self.q2 = QNetwork(state_dim).to(device)
        self.target_q1 = QNetwork(state_dim).to(device)
        self.target_q2 = QNetwork(state_dim).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        self.gamma = gamma
        self.train_epochs = train_epochs
        self.tau = tau
        self.batch_size = batch_size
        self.replay_warmup = max(batch_size, int(replay_warmup))
        self.replay = TensorReplayBuffer(replay_size, state_dim, device)

        self.auto_entropy_tuning = auto_entropy_tuning
        self.target_entropy = float(target_entropy)
        self.fixed_entropy_alpha = float(entropy_alpha)
        if auto_entropy_tuning:
            init_alpha = max(float(entropy_alpha), 1e-6)
            self.log_entropy_alpha = torch.tensor(
                np.log(init_alpha),
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            self.alpha_optimizer = optim.Adam([self.log_entropy_alpha], lr=lr)
        else:
            self.log_entropy_alpha = None
            self.alpha_optimizer = None

        self.normalize_states = normalize_states
        self.normalize_rewards = normalize_rewards
        self.reward_scale_epsilon = reward_scale_epsilon
        self.state_rms = RunningMeanStd(shape=(state_dim,))
        self.reward_rms = RunningMeanStd(shape=())
        self._sync_actor_state_stats()

    def _sync_actor_state_stats(self) -> None:
        if self.normalize_states:
            self.model.set_state_stats(self.state_rms.mean, self.state_rms.var, self.state_rms.count)
        else:
            state_dim = self.model.state_mean.shape[0]
            self.model.set_state_stats(
                np.zeros(state_dim, dtype=np.float32),
                np.ones(state_dim, dtype=np.float32),
                1e-4,
            )

    def _normalize_states_for_q(self, states: torch.Tensor) -> torch.Tensor:
        if not self.normalize_states:
            return states
        mean = self.model.state_mean.to(states.device)
        var = self.model.state_var.to(states.device)
        return (states - mean) / torch.sqrt(var + 1e-6)

    def _normalize_rewards_for_training(self, rewards: torch.Tensor) -> torch.Tensor:
        if not self.normalize_rewards:
            return rewards
        reward_scale = max(float(np.sqrt(self.reward_rms.var + self.reward_scale_epsilon)), self.reward_scale_epsilon)
        return rewards / reward_scale

    def _current_entropy_alpha(self) -> torch.Tensor:
        if self.auto_entropy_tuning:
            return self.log_entropy_alpha.exp()
        return torch.tensor(self.fixed_entropy_alpha, device=self.device)

    def _set_critic_grad(self, enabled: bool) -> None:
        for parameter in self.q1.parameters():
            parameter.requires_grad_(enabled)
        for parameter in self.q2.parameters():
            parameter.requires_grad_(enabled)

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

    @torch.no_grad()
    def select_actions(
        self,
        obs: torch.Tensor,
        action_lows: torch.Tensor,
        action_highs: torch.Tensor,
        masks: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_obs = obs.reshape(-1, obs.shape[-1])
        flat_lows = action_lows.reshape(-1)
        flat_highs = action_highs.reshape(-1)
        actions, log_probs, _ = self.model.sample(
            flat_obs,
            action_low=flat_lows,
            action_high=flat_highs,
            deterministic=deterministic,
        )
        actions = actions.reshape(obs.shape[0], obs.shape[1])
        log_probs = log_probs.reshape(obs.shape[0], obs.shape[1])
        actions = torch.where(masks, actions, torch.zeros_like(actions))
        log_probs = torch.where(masks, log_probs, torch.zeros_like(log_probs))
        return actions, log_probs

    def add_transitions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        masks: torch.Tensor,
        action_lows: torch.Tensor,
        action_highs: torch.Tensor,
        next_action_lows: torch.Tensor,
        next_action_highs: torch.Tensor,
    ) -> int:
        valid_idx = masks.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        if valid_idx.numel() == 0:
            return 0

        flat_states = obs.reshape(-1, obs.shape[-1])[valid_idx]
        flat_actions = actions.reshape(-1)[valid_idx]
        flat_rewards = rewards.reshape(-1)[valid_idx]
        flat_next_states = next_obs.reshape(-1, next_obs.shape[-1])[valid_idx]
        flat_dones = dones.reshape(-1)[valid_idx]
        flat_lows = action_lows.reshape(-1)[valid_idx]
        flat_highs = action_highs.reshape(-1)[valid_idx]
        flat_next_lows = next_action_lows.reshape(-1)[valid_idx]
        flat_next_highs = next_action_highs.reshape(-1)[valid_idx]

        if self.normalize_states:
            state_batch = torch.cat([flat_states, flat_next_states], dim=0).detach().cpu().numpy()
            self.state_rms.update(state_batch)
            self._sync_actor_state_stats()
        if self.normalize_rewards:
            self.reward_rms.update(flat_rewards.detach().cpu().numpy())

        self.replay.add_batch(
            flat_states,
            flat_actions,
            flat_rewards,
            flat_next_states,
            flat_dones,
            flat_lows,
            flat_highs,
            flat_next_lows,
            flat_next_highs,
        )
        return int(valid_idx.numel())

    def update(self) -> dict[str, float]:
        if len(self.replay) < self.replay_warmup:
            return {
                "replay_size": len(self.replay),
                "updates": 0,
                "entropy_alpha": float(self._current_entropy_alpha().item()),
            }

        q1_loss_total = 0.0
        q2_loss_total = 0.0
        actor_loss_total = 0.0
        alpha_loss_total = 0.0

        for _ in range(self.train_epochs):
            batch = self.replay.sample(min(self.batch_size, len(self.replay)))
            states = batch["states"]
            actions = batch["actions"]
            rewards = self._normalize_rewards_for_training(batch["rewards"])
            next_states = batch["next_states"]
            dones = batch["dones"]
            action_lows = batch["action_lows"]
            action_highs = batch["action_highs"]
            next_action_lows = batch["next_action_lows"]
            next_action_highs = batch["next_action_highs"]

            q_states = self._normalize_states_for_q(states)
            q_next_states = self._normalize_states_for_q(next_states)

            with torch.no_grad():
                next_actions, next_log_probs, _ = self.model.sample(
                    next_states,
                    action_low=next_action_lows,
                    action_high=next_action_highs,
                    deterministic=False,
                )
                entropy_alpha = self._current_entropy_alpha().detach()
                target_q1 = self.target_q1(q_next_states, next_actions)
                target_q2 = self.target_q2(q_next_states, next_actions)
                target_q = torch.min(target_q1, target_q2) - entropy_alpha * next_log_probs
                q_target = rewards + self.gamma * (1.0 - dones) * target_q

            q1_pred = self.q1(q_states, actions)
            q2_pred = self.q2(q_states, actions)
            q1_loss = F.mse_loss(q1_pred, q_target)
            q2_loss = F.mse_loss(q2_pred, q_target)

            self.q1_optimizer.zero_grad()
            q1_loss.backward()
            self.q1_optimizer.step()

            self.q2_optimizer.zero_grad()
            q2_loss.backward()
            self.q2_optimizer.step()

            self._set_critic_grad(False)
            policy_actions, log_probs, _ = self.model.sample(
                states,
                action_low=action_lows,
                action_high=action_highs,
                deterministic=False,
            )
            q_policy = torch.min(self.q1(q_states, policy_actions), self.q2(q_states, policy_actions))
            entropy_alpha = self._current_entropy_alpha().detach()
            actor_loss = (entropy_alpha * log_probs - q_policy).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._set_critic_grad(True)

            if self.auto_entropy_tuning:
                alpha_loss = -(
                    self.log_entropy_alpha * (log_probs + self.target_entropy).detach()
                ).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                alpha_loss_total += float(alpha_loss.item())

            self._soft_update(self.q1, self.target_q1)
            self._soft_update(self.q2, self.target_q2)
            q1_loss_total += float(q1_loss.item())
            q2_loss_total += float(q2_loss.item())
            actor_loss_total += float(actor_loss.item())

        updates = max(self.train_epochs, 1)
        return {
            "replay_size": len(self.replay),
            "updates": self.train_epochs,
            "entropy_alpha": float(self._current_entropy_alpha().item()),
            "q1_loss": q1_loss_total / updates,
            "q2_loss": q2_loss_total / updates,
            "actor_loss": actor_loss_total / updates,
            "alpha_loss": alpha_loss_total / updates if self.auto_entropy_tuning else 0.0,
        }

    def save_model(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load_model(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
