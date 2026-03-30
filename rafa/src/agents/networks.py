# -*- coding: utf-8 -*-
"""
Actor-critic networks for PPO with continuous residual action.

Architecture (formulation.tex):
    - Shared MLP backbone
    - Actor head: outputs mean and log_std for Gaussian -> tanh squashing
    - Critic head: outputs scalar V(o)
    - Parameter sharing across all CAVs
"""

import torch
import torch.nn as nn
from torch.distributions import Normal

from src.agents.rl_types import OBS_DIM, RLConfig


class ActorCritic(nn.Module):
    """Shared actor-critic with bounded continuous action."""

    def __init__(self, cfg: RLConfig, obs_dim: int = OBS_DIM):
        super().__init__()
        self.delta_alpha_max = cfg.delta_alpha_max

        dims = [obs_dim] + [cfg.hidden_dim] * cfg.num_hidden
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        self.backbone = nn.Sequential(*layers)

        # Actor head
        self.mean_head = nn.Linear(cfg.hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))  # learnable log_std

        # Critic head
        self.value_head = nn.Linear(cfg.hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        # Smaller init for policy head (start near zero residual)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: torch.Tensor):
        """
        Parameters
        ----------
        obs : Tensor, shape (..., OBS_DIM)

        Returns
        -------
        mean : Tensor, shape (..., 1)  — squashed action mean
        std  : Tensor, shape (..., 1)
        value: Tensor, shape (..., 1)
        """
        h = self.backbone(obs)
        mean = self.mean_head(h)
        std = self.log_std.exp().expand_as(mean)
        value = self.value_head(h)
        return mean, std, value

    # ------------------------------------------------------------------
    def get_action_and_value(self, obs: torch.Tensor, deterministic: bool = False):
        """
        Sample (or take mean) action, compute log-prob, entropy, value.

        Parameters
        ----------
        obs : Tensor, shape (B, OBS_DIM)
        deterministic : bool

        Returns
        -------
        action      : Tensor (B, 1)   in [-delta_alpha_max, delta_alpha_max]
        log_prob    : Tensor (B, 1)
        entropy     : Tensor (B, 1)
        value       : Tensor (B, 1)
        """
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)

        if deterministic:
            u = mean
        else:
            u = dist.rsample()

        # tanh squashing
        action = self.delta_alpha_max * torch.tanh(u)

        # log-prob correction for tanh squashing
        log_prob = dist.log_prob(u) - torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def evaluate_actions(self, obs: torch.Tensor, raw_actions: torch.Tensor):
        """
        Re-evaluate log-prob and value for stored actions (PPO update).

        Parameters
        ----------
        obs : Tensor (B, OBS_DIM)
        raw_actions : Tensor (B, 1)  — the *squashed* actions stored in buffer

        Returns
        -------
        log_prob : Tensor (B, 1)
        entropy  : Tensor (B, 1)
        value    : Tensor (B, 1)
        """
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)

        # invert tanh to recover u
        clipped = torch.clamp(raw_actions / self.delta_alpha_max, -0.999, 0.999)
        u = torch.atanh(clipped)

        log_prob = dist.log_prob(u) - torch.log(1.0 - torch.tanh(u).pow(2) + 1e-6)
        entropy = dist.entropy()
        return log_prob, entropy, value
