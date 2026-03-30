# -------------------------------------------------------------
# File: src/mesoscopic/sac_training.py
# -------------------------------------------------------------

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.mesoscopic.rl_common import RunningMeanStd, sample_bounded_normal
from src.mesoscopic.rl_rewards import compute_residual_headway_reward


class SACActor(nn.Module):
    """Bounded stochastic actor used by SAC."""

    def __init__(
        self,
        state_dim=8,
        hidden=64,
        log_std_min=-5.0,
        log_std_max=2.0,
    ):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_std_head = nn.Linear(hidden, 1)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.eps = 1e-6

        self.register_buffer("state_mean", torch.zeros(state_dim, dtype=torch.float32))
        self.register_buffer("state_var", torch.ones(state_dim, dtype=torch.float32))
        self.register_buffer("state_count", torch.tensor(1e-4, dtype=torch.float32))

    def set_state_stats(self, mean, var, count):
        self.state_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.state_var.copy_(torch.as_tensor(var, dtype=torch.float32))
        self.state_count.fill_(float(count))

    def normalize_state(self, x):
        mean = self.state_mean.to(x.device)
        var = self.state_var.to(x.device)
        return (x - mean) / torch.sqrt(var + self.eps)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = self.shared(self.normalize_state(x))
        mean = self.mean_head(h)
        log_std = torch.clamp(self.log_std_head(h), self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, x, action_low, action_high, deterministic=False):
        mean, log_std = self.forward(x)
        action, log_prob, _ = sample_bounded_normal(
            mean,
            log_std,
            action_low,
            action_high,
            deterministic=deterministic,
        )
        return action.squeeze(-1), log_prob, action.squeeze(-1)


class QNetwork(nn.Module):
    """State-action critic for SAC."""

    def __init__(self, state_dim=8, hidden=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 0:
            action = action.unsqueeze(0).unsqueeze(-1)
        elif action.dim() == 1:
            action = action.unsqueeze(-1)

        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


class SACTrainer:
    """Off-policy SAC trainer for the bounded residual action."""

    def __init__(
        self,
        state_dim=8,
        lr=3e-4,
        gamma=0.99,
        train_epochs=10,
        reward_cfg=None,
        cth_cfg=None,
        acc_cfg=None,
        tau=0.005,
        entropy_alpha=0.2,
        batch_size=256,
        action_limit=0.1,
        replay_size=100000,
        replay_warmup=2048,
        auto_entropy_tuning=True,
        target_entropy=-1.0,
        normalize_states=True,
        normalize_rewards=True,
        reward_scale_epsilon=1e-6,
    ):
        self.model = SACActor(state_dim=state_dim)
        self.q1 = QNetwork(state_dim)
        self.q2 = QNetwork(state_dim)
        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)

        self.actor_optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        self.gamma = gamma
        self.train_epochs = train_epochs
        self.tau = tau
        self.batch_size = batch_size
        self.replay_warmup = max(batch_size, int(replay_warmup))
        self.replay_buffer = deque(maxlen=max(int(replay_size), self.replay_warmup))

        self.auto_entropy_tuning = auto_entropy_tuning
        self.target_entropy = float(target_entropy)
        self.fixed_entropy_alpha = float(entropy_alpha)
        if self.auto_entropy_tuning:
            init_alpha = max(float(entropy_alpha), 1e-6)
            self.log_entropy_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32, requires_grad=True)
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

        self.reward_cfg = reward_cfg or {}
        self.cth_cfg = cth_cfg or {}
        self.acc_cfg = acc_cfg or {}

        # Steps from the current episode are staged here, then committed to replay.
        self.memory = []

    def store_step(
        self,
        state,
        action,
        log_prob,
        value,
        reward,
        next_state=None,
        done=False,
        action_low=-1.0,
        action_high=1.0,
        next_action_low=None,
        next_action_high=None,
    ):
        self.memory.append({
            "state": np.asarray(state, dtype=np.float32),
            "action": float(action),
            "reward": float(reward),
            "next_state": np.asarray(next_state if next_state is not None else state, dtype=np.float32),
            "done": bool(done),
            "action_low": float(action_low),
            "action_high": float(action_high),
            "next_action_low": float(action_low if next_action_low is None else next_action_low),
            "next_action_high": float(action_high if next_action_high is None else next_action_high),
        })

    def compute_reward(
        self,
        v,
        s,
        mu_v,
        sigma_v2,
        alpha,
        a,
        a_prev,
        dt,
        e_leader=0.0,
    ):
        return compute_residual_headway_reward(
            reward_cfg=self.reward_cfg,
            cth_cfg=self.cth_cfg,
            acc_cfg=self.acc_cfg,
            v=v,
            s=s,
            mu_v=mu_v,
            sigma_v2=sigma_v2,
            alpha=alpha,
            a=a,
            a_prev=a_prev,
            dt=dt,
            e_leader=e_leader,
        )

    def _sync_actor_state_stats(self):
        if self.normalize_states:
            self.model.set_state_stats(self.state_rms.mean, self.state_rms.var, self.state_rms.count)
        else:
            state_dim = self.model.state_mean.shape[0]
            self.model.set_state_stats(
                np.zeros(state_dim, dtype=np.float32),
                np.ones(state_dim, dtype=np.float32),
                1e-4,
            )

    def _normalize_states_for_q(self, states):
        if not self.normalize_states:
            return states

        mean = self.model.state_mean.to(states.device)
        var = self.model.state_var.to(states.device)
        return (states - mean) / torch.sqrt(var + 1e-6)

    def _normalize_rewards_for_training(self, rewards):
        if not self.normalize_rewards:
            return rewards

        reward_scale = float(np.sqrt(self.reward_rms.var + self.reward_scale_epsilon))
        reward_scale = max(reward_scale, self.reward_scale_epsilon)
        return rewards / reward_scale

    def _current_entropy_alpha(self):
        if self.auto_entropy_tuning:
            return self.log_entropy_alpha.exp()
        return torch.tensor(self.fixed_entropy_alpha, dtype=torch.float32)

    def _set_critic_grad(self, enabled):
        for param in self.q1.parameters():
            param.requires_grad_(enabled)
        for param in self.q2.parameters():
            param.requires_grad_(enabled)

    def _soft_update(self, source, target):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def _commit_episode_memory(self):
        """Move the latest episode into replay and update normalization stats."""
        if not self.memory:
            return 0

        states = np.asarray([m["state"] for m in self.memory], dtype=np.float32)
        next_states = np.asarray([m["next_state"] for m in self.memory], dtype=np.float32)
        rewards = np.asarray([m["reward"] for m in self.memory], dtype=np.float32)

        if self.normalize_states:
            self.state_rms.update(np.concatenate((states, next_states), axis=0))
            self._sync_actor_state_stats()

        if self.normalize_rewards:
            self.reward_rms.update(rewards)

        self.replay_buffer.extend(self.memory)
        appended = len(self.memory)
        self.memory = []
        return appended

    def train(self):
        """Run several off-policy SAC updates from the replay buffer."""
        appended = self._commit_episode_memory()
        replay_size = len(self.replay_buffer)

        if replay_size < self.replay_warmup:
            return {
                "appended": appended,
                "replay_size": replay_size,
                "updates": 0,
                "entropy_alpha": float(self._current_entropy_alpha().item()),
            }

        batch_size = min(self.batch_size, replay_size)

        q1_loss_total = 0.0
        q2_loss_total = 0.0
        actor_loss_total = 0.0
        alpha_loss_total = 0.0

        for _ in range(self.train_epochs):
            batch_idx = np.random.randint(0, replay_size, size=batch_size)
            batch = [self.replay_buffer[idx] for idx in batch_idx]

            states = torch.tensor(np.asarray([b["state"] for b in batch]), dtype=torch.float32)
            actions = torch.tensor(np.asarray([b["action"] for b in batch]), dtype=torch.float32)
            rewards = torch.tensor(np.asarray([b["reward"] for b in batch]), dtype=torch.float32)
            next_states = torch.tensor(np.asarray([b["next_state"] for b in batch]), dtype=torch.float32)
            dones = torch.tensor(np.asarray([b["done"] for b in batch]), dtype=torch.float32)
            action_lows = torch.tensor(np.asarray([b["action_low"] for b in batch]), dtype=torch.float32)
            action_highs = torch.tensor(np.asarray([b["action_high"] for b in batch]), dtype=torch.float32)
            next_action_lows = torch.tensor(np.asarray([b["next_action_low"] for b in batch]), dtype=torch.float32)
            next_action_highs = torch.tensor(np.asarray([b["next_action_high"] for b in batch]), dtype=torch.float32)

            q_states = self._normalize_states_for_q(states)
            q_next_states = self._normalize_states_for_q(next_states)
            rewards = self._normalize_rewards_for_training(rewards)

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
            q_policy = torch.min(
                self.q1(q_states, policy_actions),
                self.q2(q_states, policy_actions)
            )
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
            "appended": appended,
            "replay_size": replay_size,
            "updates": self.train_epochs,
            "entropy_alpha": float(self._current_entropy_alpha().item()),
            "q1_loss": q1_loss_total / updates,
            "q2_loss": q2_loss_total / updates,
            "actor_loss": actor_loss_total / updates,
            "alpha_loss": alpha_loss_total / updates if self.auto_entropy_tuning else 0.0,
        }

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
