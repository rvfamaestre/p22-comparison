# -*- coding: utf-8 -*-
"""
GPU-native PPO trainer that works with VecRingRoadEnv.

Key differences from the CPU trainer:
* Rollouts from ``num_envs`` environments are collected in parallel.
* GAE and minibatch generation happen entirely on GPU.
* The observation normaliser runs on-device.
* A single training script orchestrates everything without Python-level
  vehicle loops.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.agents.rl_types import OBS_DIM


# --------------------------------------------------------------------------- #
# Running Normaliser (GPU, Welford)
# --------------------------------------------------------------------------- #


class GPURunningNormalizer:
    """Welford running mean/var on-device with freeze support."""

    def __init__(self, dim: int, device: torch.device, clip: float = 10.0):
        self.dim = dim
        self.clip = clip
        self.device = device
        # MPS does not support float64 tensors, so the stats dtype adapts.
        self.stats_dtype = torch.float32 if device.type == "mps" else torch.float64
        self.count = 0
        self.mean = torch.zeros(dim, device=device, dtype=self.stats_dtype)
        self.var = torch.ones(dim, device=device, dtype=self.stats_dtype)
        self._M2 = torch.zeros(dim, device=device, dtype=self.stats_dtype)
        self.frozen = False

    @torch.no_grad()
    def update(self, batch: torch.Tensor):
        """Update with a batch of shape (*, dim)."""
        if self.frozen:
            return
        flat = batch.reshape(-1, self.dim).to(self.stats_dtype)
        for row in flat:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self._M2 += delta * delta2
            self.var = self._M2 / max(self.count, 2) + 1e-8

    @torch.no_grad()
    def update_batch(self, batch: torch.Tensor):
        """Batch Welford update (much faster than row-by-row)."""
        if self.frozen:
            return
        flat = batch.reshape(-1, self.dim).to(self.stats_dtype)
        n = flat.shape[0]
        if n == 0:
            return
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        batch_count = n

        old_count = self.count
        new_count = old_count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / max(new_count, 1)
        m_a = self.var * max(old_count, 1)
        m_b = batch_var * batch_count
        self._M2 = (
            m_a + m_b + delta.pow(2) * old_count * batch_count / max(new_count, 1)
        )
        self.mean = new_mean
        self.count = new_count
        self.var = self._M2 / max(self.count, 2) + 1e-8

    def normalize(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalize obs to ~N(0,1) and clip."""
        mean = self.mean.float()
        std = self.var.float().sqrt()
        return ((obs - mean) / std).clamp(-self.clip, self.clip)

    def freeze(self):
        self.frozen = True

    def state_dict(self):
        return {
            "mean": self.mean.cpu(),
            "var": self.var.cpu(),
            "M2": self._M2.cpu(),
            "count": self.count,
            "stats_dtype": str(self.stats_dtype),
        }

    def load_state_dict(self, d):
        self.mean = d["mean"].to(self.device, dtype=self.stats_dtype)
        self.var = d["var"].to(self.device, dtype=self.stats_dtype)
        self._M2 = d["M2"].to(self.device, dtype=self.stats_dtype)
        self.count = int(d["count"])


# --------------------------------------------------------------------------- #
# Actor-Critic (identical architecture, but lives on GPU)
# --------------------------------------------------------------------------- #


class ActorCriticGPU(nn.Module):
    """Shared actor-critic with tanh-squashed Gaussian policy."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        hidden_dim: int = 128,
        num_hidden: int = 2,
        delta_alpha_max: float = 0.5,
    ):
        super().__init__()
        self.delta_alpha_max = delta_alpha_max

        dims = [obs_dim] + [hidden_dim] * num_hidden
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.value_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs):
        h = self.backbone(obs)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        value = self.value_head(h)
        return mean, std, value

    def get_action_and_value(self, obs, deterministic=False):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        u = mean if deterministic else dist.rsample()
        action = self.delta_alpha_max * torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def evaluate_actions(self, obs, raw_actions):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        clipped = (raw_actions / self.delta_alpha_max).clamp(-0.999, 0.999)
        u = torch.atanh(clipped)
        log_prob = dist.log_prob(u) - torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6)
        entropy = dist.entropy()
        return log_prob, entropy, value


# --------------------------------------------------------------------------- #
# GPU Rollout Buffer
# --------------------------------------------------------------------------- #


class GPURolloutBuffer:
    """
    Stores rollouts from ``num_envs`` environments on GPU.

    Shape convention: (rollout_steps, num_envs, N, *feature)
    but we only store CAV-relevant data using the CAV mask.
    For simplicity we store *all* vehicle slots and mask at loss time.
    """

    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        N: int,
        obs_dim: int,
        device: torch.device,
    ):
        self.T = rollout_steps
        self.B = num_envs
        self.N = N
        self.device = device

        self.obs = torch.zeros(self.T, self.B, N, obs_dim, device=device)
        self.actions = torch.zeros(self.T, self.B, N, 1, device=device)
        self.log_probs = torch.zeros(self.T, self.B, N, 1, device=device)
        self.rewards = torch.zeros(self.T, self.B, device=device)
        self.values = torch.zeros(self.T, self.B, N, 1, device=device)
        self.dones = torch.zeros(self.T, self.B, device=device)
        self.masks = torch.zeros(self.T, self.B, N, device=device, dtype=torch.bool)

        self.ptr = 0

    def add(self, obs, actions, log_probs, rewards, values, dones, masks):
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.values[t] = values
        self.dones[t] = dones.float()
        self.masks[t] = masks
        self.ptr += 1

    def reset(self):
        self.ptr = 0

    def compute_gae(self, last_values: torch.Tensor, gamma: float, lam: float):
        """
        Compute GAE advantages and returns on GPU.

        last_values : (B, N, 1)

        Returns advantages (T, B, N, 1), returns (T, B, N, 1)
        """
        T = self.ptr
        advantages = torch.zeros_like(self.values[:T])
        gae = torch.zeros_like(last_values)

        for t in reversed(range(T)):
            next_val = last_values if t == T - 1 else self.values[t + 1]
            d = self.dones[t].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
            r = self.rewards[t].unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
            delta = r + gamma * (1.0 - d) * next_val - self.values[t]
            gae = delta + gamma * lam * (1.0 - d) * gae
            advantages[t] = gae

        returns = advantages + self.values[:T]
        return advantages, returns

    def generate_batches(
        self, advantages: torch.Tensor, returns: torch.Tensor, minibatch_size: int
    ):
        """
        Yield flat minibatches, masking out non-CAV slots.

        Each batch is a dict of tensors on device.
        """
        T = self.ptr
        # Flatten (T, B, N) -> (T*B*N)
        flat_mask = self.masks[:T].reshape(-1)  # bool
        valid_idx = flat_mask.nonzero(as_tuple=False).squeeze(1)

        total = valid_idx.shape[0]
        if total == 0:
            return

        # Flatten all buffers
        flat_obs = self.obs[:T].reshape(-1, self.obs.shape[-1])
        flat_act = self.actions[:T].reshape(-1, 1)
        flat_lp = self.log_probs[:T].reshape(-1, 1)
        flat_adv = advantages.reshape(-1, 1)
        flat_ret = returns.reshape(-1, 1)
        flat_val = self.values[:T].reshape(-1, 1)

        # Select valid entries
        sel_obs = flat_obs[valid_idx]
        sel_act = flat_act[valid_idx]
        sel_lp = flat_lp[valid_idx]
        sel_adv = flat_adv[valid_idx]
        sel_ret = flat_ret[valid_idx]
        sel_val = flat_val[valid_idx]

        # Normalise advantages
        adv_mean = sel_adv.mean()
        adv_std = sel_adv.std() + 1e-8
        sel_adv = (sel_adv - adv_mean) / adv_std

        # Shuffle
        perm = torch.randperm(total, device=self.device)

        for start in range(0, total, minibatch_size):
            end = min(start + minibatch_size, total)
            idx = perm[start:end]
            yield {
                "obs": sel_obs[idx],
                "actions": sel_act[idx],
                "old_log_probs": sel_lp[idx],
                "advantages": sel_adv[idx],
                "returns": sel_ret[idx],
                "old_values": sel_val[idx],
            }


# --------------------------------------------------------------------------- #
# GPU PPO Trainer
# --------------------------------------------------------------------------- #


class GPUPPOTrainer:
    """Full PPO trainer operating on GPU tensors."""

    def __init__(
        self,
        *,
        obs_dim: int = OBS_DIM,
        hidden_dim: int = 128,
        num_hidden: int = 2,
        delta_alpha_max: float = 0.5,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        eps_clip: float = 0.2,
        clip_value_loss: bool = True,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 10,
        minibatch_size: int = 256,
        gamma: float = 0.99,
        lam_gae: float = 0.95,
        rollout_steps: int = 256,
        num_envs: int = 64,
        N: int = 22,
        device: torch.device = torch.device("cpu"),
        normalizer_warmup: int = 50_000,
    ):
        self.device = device
        self.eps_clip = eps_clip
        self.clip_value_loss = clip_value_loss
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.gamma = gamma
        self.lam_gae = lam_gae
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.N = N
        self.normalizer_warmup = normalizer_warmup

        self.policy = ActorCriticGPU(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            delta_alpha_max=delta_alpha_max,
        ).to(device)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.backbone.parameters(), "lr": lr_actor},
                {"params": self.policy.mean_head.parameters(), "lr": lr_actor},
                {"params": [self.policy.log_std], "lr": lr_actor},
                {"params": self.policy.value_head.parameters(), "lr": lr_critic},
            ]
        )

        self.buffer = GPURolloutBuffer(rollout_steps, num_envs, N, obs_dim, device)
        self.normalizer = GPURunningNormalizer(obs_dim, device)
        self.global_step = 0

        # Store initial LRs for linear schedule
        self._initial_lrs = [pg["lr"] for pg in self.optimizer.param_groups]

    def update_lr(self, progress: float):
        """Linear LR decay. progress = fraction of training completed."""
        frac = max(1.0 - progress, 0.0)
        for pg, base_lr in zip(self.optimizer.param_groups, self._initial_lrs):
            pg["lr"] = base_lr * frac

    @torch.no_grad()
    def select_actions(
        self, obs: torch.Tensor, mask: torch.Tensor, deterministic: bool = False
    ):
        """
        obs  : (B, N, 12)
        mask : (B, N) bool

        Returns actions (B, N, 1), log_probs (B, N, 1), values (B, N, 1)
        """
        B, N, D = obs.shape
        flat_obs = obs.reshape(B * N, D)
        actions, log_probs, _, values = self.policy.get_action_and_value(
            flat_obs, deterministic=deterministic
        )
        actions = actions.reshape(B, N, 1)
        log_probs = log_probs.reshape(B, N, 1)
        values = values.reshape(B, N, 1)

        # Zero out non-CAV slots
        mask3 = mask.unsqueeze(2)
        actions = actions * mask3.float()
        log_probs = log_probs * mask3.float()

        return actions, log_probs, values

    def update(self) -> Dict[str, float]:
        """Run PPO update on current buffer contents."""
        # Bootstrap last value
        last_obs = self.buffer.obs[self.buffer.ptr - 1]
        B, N, D = last_obs.shape
        flat = last_obs.reshape(B * N, D)
        with torch.no_grad():
            _, _, last_val = self.policy(flat)
        last_val = last_val.reshape(B, N, 1)

        advantages, returns = self.buffer.compute_gae(
            last_val, self.gamma, self.lam_gae
        )

        total_pg = 0.0
        total_vl = 0.0
        total_ent = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            for batch in self.buffer.generate_batches(
                advantages, returns, self.minibatch_size
            ):
                obs_b = batch["obs"]
                act_b = batch["actions"]
                old_lp = batch["old_log_probs"]
                adv_b = batch["advantages"]
                ret_b = batch["returns"]
                old_val = batch["old_values"]

                new_lp, entropy, values = self.policy.evaluate_actions(obs_b, act_b)

                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv_b
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * adv_b
                )
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss with optional clipping
                if self.clip_value_loss:
                    v_clipped = old_val + (values - old_val).clamp(
                        -self.eps_clip, self.eps_clip
                    )
                    v_loss_unclipped = (values - ret_b).pow(2)
                    v_loss_clipped = (v_clipped - ret_b).pow(2)
                    v_loss = torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = nn.functional.mse_loss(values, ret_b)

                ent_loss = -entropy.mean()

                loss = pg_loss + self.value_coef * v_loss + self.entropy_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_pg += pg_loss.item()
                total_vl += v_loss.item()
                total_ent += (-ent_loss).item()
                n_updates += 1

        n = max(n_updates, 1)
        self.buffer.reset()
        return {
            "pg_loss": total_pg / n,
            "v_loss": total_vl / n,
            "entropy": total_ent / n,
        }

    def save(self, path: str):
        import os

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "normalizer": self.normalizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        if "normalizer" in ckpt:
            self.normalizer.load_state_dict(ckpt["normalizer"])

