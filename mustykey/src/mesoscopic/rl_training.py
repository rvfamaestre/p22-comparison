# -------------------------------------------------------------
# File: src/mesoscopic/rl_training.py
# -------------------------------------------------------------

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from src.mesoscopic.rl_layer import ActorCritic
from src.mesoscopic.rl_rewards import compute_residual_headway_reward


# -------------------------------------------------------------
# PPO Trainer
# -------------------------------------------------------------

class PPOTrainer:
    """Standard on-policy PPO trainer for the bounded residual action."""

    def __init__(
        self,
        state_dim=8,
        lr=3e-4,
        gamma=0.99,
        clip=0.2,
        train_epochs=10,
        gae_lambda=0.95,
        minibatch_size=256,
        value_coef=0.5,
        entropy_coef=1e-3,
        max_grad_norm=0.5,
        reward_cfg=None,
        cth_cfg=None,
        acc_cfg=None,
    ):
        self.model = ActorCritic(state_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.clip = clip
        self.train_epochs = train_epochs
        self.gae_lambda = gae_lambda
        self.minibatch_size = minibatch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.reward_cfg = reward_cfg or {}
        self.cth_cfg = cth_cfg or {}
        self.acc_cfg = acc_cfg or {}

        # The rollout buffer stores one on-policy episode at a time.
        self.memory = []

    # ---------------------------------------------------------
    # store rollout step
    # ---------------------------------------------------------

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
            "state": state,
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "action_low": action_low,
            "action_high": action_high,
        })

    # ---------------------------------------------------------
    # compute reward
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # PPO update
    # ---------------------------------------------------------

    def _compute_gae(self, rewards, dones, values, next_values):
        """Generalized advantage estimation on the collected rollout."""
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, dtype=rewards.dtype)

        for idx in reversed(range(len(rewards))):
            mask = 1.0 - dones[idx]
            delta = rewards[idx] + self.gamma * next_values[idx] * mask - values[idx]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[idx] = gae

        returns = advantages + values
        return advantages, returns

    def train(self):
        """Run the PPO update over the latest collected on-policy rollout."""
        if not self.memory:
            return {"updates": 0, "samples": 0}

        states = torch.tensor(
            np.array([m["state"] for m in self.memory]),
            dtype=torch.float32
        )

        actions = torch.tensor(
            np.array([m["action"] for m in self.memory]),
            dtype=torch.float32
        )

        old_log_probs = torch.tensor(
            np.array([m["log_prob"] for m in self.memory]),
            dtype=torch.float32
        )

        old_values = torch.tensor(
            np.array([m["value"] for m in self.memory]),
            dtype=torch.float32
        )

        rewards = torch.tensor(
            np.array([m["reward"] for m in self.memory]),
            dtype=torch.float32
        )
        dones = torch.tensor(
            np.array([m.get("done", False) for m in self.memory]),
            dtype=torch.float32
        )
        next_states = torch.tensor(
            np.array([m["next_state"] for m in self.memory]),
            dtype=torch.float32
        )
        action_lows = torch.tensor(
            np.array([m.get("action_low", -1.0) for m in self.memory]),
            dtype=torch.float32
        )
        action_highs = torch.tensor(
            np.array([m.get("action_high", 1.0) for m in self.memory]),
            dtype=torch.float32
        )

        with torch.no_grad():
            _, _, next_values = self.model(next_states)
            next_values = next_values.squeeze(-1)

        advantages, returns = self._compute_gae(rewards, dones, old_values, next_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n_samples = states.shape[0]
        minibatch_size = min(self.minibatch_size, n_samples)
        updates = 0
        actor_loss_value = 0.0
        critic_loss_value = 0.0
        entropy_value = 0.0

        for _ in range(self.train_epochs):
            permutation = torch.randperm(n_samples)

            for start in range(0, n_samples, minibatch_size):
                batch_idx = permutation[start:start + minibatch_size]

                log_probs, entropy, values = self.model.evaluate_actions(
                    states[batch_idx],
                    actions[batch_idx],
                    action_lows[batch_idx],
                    action_highs[batch_idx],
                )

                ratios = torch.exp(log_probs - old_log_probs[batch_idx])

                s1 = ratios * advantages[batch_idx]
                s2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages[batch_idx]

                actor_loss = -torch.min(s1, s2).mean()
                critic_loss = F.mse_loss(values, returns[batch_idx])
                entropy_term = entropy.mean()

                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy_term
                )

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                actor_loss_value += float(actor_loss.item())
                critic_loss_value += float(critic_loss.item())
                entropy_value += float(entropy_term.item())
                updates += 1

        self.memory = []
        updates = max(updates, 1)
        return {
            "updates": updates,
            "samples": n_samples,
            "actor_loss": actor_loss_value / updates,
            "critic_loss": critic_loss_value / updates,
            "entropy": entropy_value / updates,
        }

    # ---------------------------------------------------------
    # evaluate log-prob and value for a chosen action
    # ---------------------------------------------------------

    def evaluate_action(self, state, action, action_low, action_high):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor([action], dtype=torch.float32)
        action_low_tensor = torch.tensor([action_low], dtype=torch.float32)
        action_high_tensor = torch.tensor([action_high], dtype=torch.float32)

        with torch.no_grad():
            log_prob, _, value = self.model.evaluate_actions(
                state_tensor,
                action_tensor,
                action_low_tensor,
                action_high_tensor,
            )

        return float(log_prob.squeeze().item()), float(value.squeeze().item())

    # ---------------------------------------------------------
    # save policy
    # ---------------------------------------------------------

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
