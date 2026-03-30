# -*- coding: utf-8 -*-
"""
GPU-native Soft Actor-Critic (SAC) trainer for the vectorized ring-road env.

Why SAC over PPO for this environment
--------------------------------------
The HDV dynamics are driven by Euler-Maruyama noise, so every episode is a
different realisation of the same stochastic process.  PPO is on-policy: it
collects a rollout, runs K PPO epochs, and discards all transitions.  SAC is
off-policy: every transition is stored in a **replay buffer** and can be
re-sampled many times, extracting far more gradient signal from the expensive
stochastic experience.

Architecture (mirrors PPO actor; new parts: twin-Q critics, target nets,
             replay buffer, auto-entropy tuning)
----------------------------------------------------------------
Actor  : shared backbone → tanh-squashed Gaussian  (same as PPO)
Critics: two independent Q(o, a) networks — clipped double-Q trick
Target : EMA copies of the two critics (soft target update, τ ≪ 1)
Entropy: learned temperature α via gradient on E[-log π(a|o) + H_target]

The actor can be initialised from a pre-trained PPO checkpoint so that the
replay buffer fills with useful experience from the first episode.

All operations are purely on-device (CUDA / MPS / CPU).  Shapes follow the
same (B, N, D) convention as the vectorized environment.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from src.agents.rl_types import OBS_DIM
from src.gpu.gpu_ppo import GPURunningNormalizer


# --------------------------------------------------------------------------- #
# Shared backbone utility
# --------------------------------------------------------------------------- #


def _build_mlp(in_dim: int, hidden_dim: int, num_hidden: int, out_dim: int) -> nn.Sequential:
    """Build a Tanh-MLP: in → [hidden]×num_hidden → out."""
    dims = [in_dim] + [hidden_dim] * num_hidden + [out_dim]
    layers: list = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# Actor (tanh-squashed Gaussian — identical semantics to PPO actor)
# --------------------------------------------------------------------------- #


class SACActorGPU(nn.Module):
    """Stochastic actor with state-dependent log_std (SAC convention).

    Unlike the PPO actor where log_std is a lone scalar parameter shared
    across all inputs, the SAC actor learns log_std as a function of the
    observation — this gives richer exploration and faster convergence on
    heterogeneous states.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

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
        layers: list = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        self.backbone = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)  # state-dependent log_std

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, log_std) before squashing."""
        h = self.backbone(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (or take mean) action and compute log-prob.

        Returns
        -------
        action   : (batch, 1)  in [-delta_alpha_max, delta_alpha_max]
        log_prob : (batch, 1)
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mean, std)

        u = mean if deterministic else dist.rsample()
        # tanh squashing
        action = self.delta_alpha_max * torch.tanh(u)
        # Log-prob with change-of-variables correction
        log_prob = dist.log_prob(u) - torch.log(
            1.0 - torch.tanh(u).pow(2) + 1e-6
        )
        return action, log_prob


# --------------------------------------------------------------------------- #
# Critic (Q-network)
# --------------------------------------------------------------------------- #


class SACCriticGPU(nn.Module):
    """Q(o, a) network: takes flattened (obs ∥ action) as input."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        action_dim: int = 1,
        hidden_dim: int = 128,
        num_hidden: int = 2,
    ):
        super().__init__()
        self.net = _build_mlp(obs_dim + action_dim, hidden_dim, num_hidden, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return Q-value (batch, 1)."""
        return self.net(torch.cat([obs, action], dim=-1))


# --------------------------------------------------------------------------- #
# GPU Replay Buffer
# --------------------------------------------------------------------------- #


class GPUReplayBuffer:
    """Circular replay buffer stored on-device.

    All tensors live on the accelerator throughout training; there are no
    host-device copies during sampling.

    Shapes
    ------
    obs / next_obs : (capacity, N, obs_dim)
    actions        : (capacity, N, 1)
    rewards        : (capacity,)
    dones          : (capacity,)
    masks          : (capacity, N)   — True = real CAV slot
    """

    def __init__(
        self,
        capacity: int,
        N: int,
        obs_dim: int,
        device: torch.device,
    ):
        self.capacity = capacity
        self.N = N
        self.obs_dim = obs_dim
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = torch.zeros(capacity, N, obs_dim, device=device)
        self.next_obs = torch.zeros(capacity, N, obs_dim, device=device)
        self.actions = torch.zeros(capacity, N, 1, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.dones = torch.zeros(capacity, device=device)
        self.masks = torch.zeros(capacity, N, device=device, dtype=torch.bool)

    def add(
        self,
        obs: torch.Tensor,        # (B, N, D)
        next_obs: torch.Tensor,   # (B, N, D)
        actions: torch.Tensor,    # (B, N, 1)
        rewards: torch.Tensor,    # (B,)
        dones: torch.Tensor,      # (B,) bool or float
        masks: torch.Tensor,      # (B, N) bool
    ):
        """Add a batch of B transitions; wraps around at capacity."""
        B = obs.shape[0]
        indices = torch.arange(B, device=self.device)
        idx = (self.ptr + indices) % self.capacity

        self.obs[idx] = obs
        self.next_obs[idx] = next_obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards.float()
        self.dones[idx] = dones.float()
        self.masks[idx] = masks

        self.ptr = (self.ptr + B) % self.capacity
        self.size = min(self.size + B, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample ``batch_size`` *individual CAV* transitions.

        We sample replay-buffer steps first, then select valid CAV slots
        from those steps.  This matches the PPO minibatch convention: only
        real CAV slots contribute to the gradient.

        Returns a dict of tensors of shape (batch_size, *).
        """
        # Sample random steps (with replacement)
        step_idx = torch.randint(0, self.size, (batch_size * 4,), device=self.device)

        # Flatten masks: (size, N) → select those steps
        flat_mask = self.masks[step_idx]        # (batch_size*4, N)
        # Build (cav, slot) index pairs
        pairs = flat_mask.nonzero(as_tuple=False)  # (K, 2): [step_local, vehicle]
        if pairs.shape[0] == 0:
            # Edge case: no valid CAV slots (should not happen in practice)
            pairs = torch.zeros(1, 2, dtype=torch.long, device=self.device)

        # Select up to batch_size pairs
        if pairs.shape[0] >= batch_size:
            sel = torch.randperm(pairs.shape[0], device=self.device)[:batch_size]
        else:
            sel = torch.randint(0, pairs.shape[0], (batch_size,), device=self.device)
        pairs = pairs[sel]  # (batch_size, 2)

        step_local = pairs[:, 0]    # index into step_idx
        veh_slot = pairs[:, 1]      # vehicle index within each step

        real_step = step_idx[step_local]  # actual replay-buffer row indices

        return {
            "obs":      self.obs[real_step, veh_slot],        # (B, D)
            "next_obs": self.next_obs[real_step, veh_slot],   # (B, D)
            "actions":  self.actions[real_step, veh_slot],    # (B, 1)
            "rewards":  self.rewards[real_step].unsqueeze(1), # (B, 1) — global reward
            "dones":    self.dones[real_step].unsqueeze(1),   # (B, 1)
        }

    def __len__(self) -> int:
        return self.size


# --------------------------------------------------------------------------- #
# GPU SAC Trainer
# --------------------------------------------------------------------------- #


class GPUSACTrainer:
    """Soft Actor-Critic trainer operating on GPU tensors.

    Key hyper-parameters (all have sensible defaults)
    ---------------------------------------------------
    lr_actor / lr_critic : Adam learning rates
    tau                  : soft target update coefficient (0.005 typical)
    gamma                : discount factor (0.99)
    alpha_init           : initial entropy temperature (auto-tuned if
                           auto_entropy=True, which is the default)
    target_entropy       : H_target = -action_dim * alpha_entropy_scale
                           (default: -1.0, one scalar action per CAV)
    replay_capacity      : replay buffer size (in env steps, i.e. B*step rows)
    warmup_steps         : number of env steps to collect before first update
    update_every         : update the networks every N env steps
    updates_per_step     : gradient updates per env step
    """

    def __init__(
        self,
        *,
        obs_dim: int = OBS_DIM,
        hidden_dim: int = 128,
        num_hidden: int = 2,
        delta_alpha_max: float = 0.5,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.2,
        auto_entropy: bool = True,
        target_entropy: float = -1.0,   # -action_dim
        replay_capacity: int = 500_000,
        N: int = 22,
        device: torch.device = torch.device("cpu"),
        normalizer_warmup: int = 10_000,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.auto_entropy = auto_entropy
        self.target_entropy = target_entropy
        self.N = N
        self.normalizer_warmup = normalizer_warmup

        # ---- Actor ----
        self.actor = SACActorGPU(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_hidden=num_hidden,
            delta_alpha_max=delta_alpha_max,
        ).to(device)

        # ---- Twin critics ----
        self.critic1 = SACCriticGPU(obs_dim, 1, hidden_dim, num_hidden).to(device)
        self.critic2 = SACCriticGPU(obs_dim, 1, hidden_dim, num_hidden).to(device)

        # ---- Target critics (frozen copies, updated via EMA) ----
        self.target_critic1 = deepcopy(self.critic1)
        self.target_critic2 = deepcopy(self.critic2)
        for p in self.target_critic1.parameters():
            p.requires_grad_(False)
        for p in self.target_critic2.parameters():
            p.requires_grad_(False)

        # ---- Entropy temperature ----
        self.log_alpha = torch.tensor(
            [float(torch.tensor(alpha_init).log())], device=device, requires_grad=True
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_actor)

        # ---- Optimisers ----
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr_critic,
        )

        # ---- Replay buffer ----
        self.replay = GPUReplayBuffer(
            capacity=replay_capacity, N=N, obs_dim=obs_dim, device=device
        )

        # ---- Observation normalizer ----
        self.normalizer = GPURunningNormalizer(obs_dim, device)

        # ---- Step counter ----
        self.global_step = 0

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy temperature (detached scalar)."""
        return self.log_alpha.exp().detach()

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def select_actions(
        self,
        obs: torch.Tensor,            # (B, N, D)
        mask: torch.Tensor,           # (B, N) bool
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Select actions for all vehicle slots; zero out non-CAV slots.

        Returns
        -------
        actions : (B, N, 1)
        """
        B, N, D = obs.shape
        flat_obs = obs.reshape(B * N, D)
        actions, _ = self.actor.get_action(flat_obs, deterministic=deterministic)
        actions = actions.reshape(B, N, 1)
        # Zero non-CAV slots
        actions = actions * mask.unsqueeze(2).float()
        return actions

    # ------------------------------------------------------------------ #
    # Soft target update
    # ------------------------------------------------------------------ #

    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
            p_tgt.data.copy_(self.tau * p_src.data + (1.0 - self.tau) * p_tgt.data)

    # ------------------------------------------------------------------ #
    # SAC update
    # ------------------------------------------------------------------ #

    def update(self, batch_size: int = 256) -> Dict[str, float]:
        """Run one gradient update step on a mini-batch from the replay buffer.

        Returns
        -------
        dict with keys: q1_loss, q2_loss, actor_loss, alpha_loss, alpha
        """
        if len(self.replay) < batch_size:
            return {}

        batch = self.replay.sample(batch_size)
        obs_b      = batch["obs"]       # (B, D)
        next_obs_b = batch["next_obs"]  # (B, D)
        act_b      = batch["actions"]   # (B, 1)
        rew_b      = batch["rewards"]   # (B, 1)
        done_b     = batch["dones"]     # (B, 1)

        # ---------- Critic update ----------
        with torch.no_grad():
            next_actions, next_log_pi = self.actor.get_action(next_obs_b)
            q1_next = self.target_critic1(next_obs_b, next_actions)
            q2_next = self.target_critic2(next_obs_b, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = rew_b + self.gamma * (1.0 - done_b) * q_next

        q1_pred = self.critic1(obs_b, act_b)
        q2_pred = self.critic2(obs_b, act_b)
        q1_loss = F.mse_loss(q1_pred, q_target)
        q2_loss = F.mse_loss(q2_pred, q_target)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------- Actor update ----------
        new_actions, log_pi = self.actor.get_action(obs_b)
        q1_new = self.critic1(obs_b, new_actions)
        q2_new = self.critic2(obs_b, new_actions)
        q_min_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_pi - q_min_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------- Entropy temperature update ----------
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.auto_entropy:
            alpha_loss = -(
                self.log_alpha.exp() * (log_pi.detach() + self.target_entropy)
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # ---------- Soft target update ----------
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

        return {
            "q1_loss":    q1_loss.item(),
            "q2_loss":    q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha":      self.alpha.item(),
        }

    # ------------------------------------------------------------------ #
    # Checkpoint
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "actor_state_dict":          self.actor.state_dict(),
                "critic1_state_dict":        self.critic1.state_dict(),
                "critic2_state_dict":        self.critic2.state_dict(),
                "target_critic1_state_dict": self.target_critic1.state_dict(),
                "target_critic2_state_dict": self.target_critic2.state_dict(),
                "actor_optimizer_state_dict":  self.actor_optimizer.state_dict(),
                "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
                "log_alpha":                 self.log_alpha.detach().cpu(),
                "alpha_optimizer_state_dict": self.alpha_optimizer.state_dict(),
                "global_step":               self.global_step,
                "normalizer":                self.normalizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic1.load_state_dict(ckpt["critic1_state_dict"])
        self.critic2.load_state_dict(ckpt["critic2_state_dict"])
        self.target_critic1.load_state_dict(ckpt["target_critic1_state_dict"])
        self.target_critic2.load_state_dict(ckpt["target_critic2_state_dict"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        if "normalizer" in ckpt:
            self.normalizer.load_state_dict(ckpt["normalizer"])

    def load_ppo_actor(self, ppo_ckpt_path: str):
        """Bootstrap the SAC actor from a pre-trained PPO checkpoint.

        Only the backbone and mean head weights are transferred (the log_std
        head is freshly initialised — SAC uses a state-dependent log_std head,
        which has no PPO counterpart).
        """
        ckpt = torch.load(ppo_ckpt_path, map_location=self.device, weights_only=False)
        ppo_sd = ckpt.get("policy_state_dict", ckpt)
        actor_sd = self.actor.state_dict()
        transferred = 0
        for k, v in ppo_sd.items():
            # Map PPO keys: "backbone.*" and "mean_head.*" exist in both
            if k in actor_sd and actor_sd[k].shape == v.shape:
                actor_sd[k] = v
                transferred += 1
        self.actor.load_state_dict(actor_sd)
        print(
            f"[GPUSACTrainer] Transferred {transferred} weight tensors "
            f"from PPO checkpoint ({ppo_ckpt_path})"
        )
