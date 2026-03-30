# -*- coding: utf-8 -*-
"""
PPO trainer for the residual-headway RL agent.

Implements the clipped PPO objective from formulation.tex:
    L_clip = E[ min(rho*A, clip(rho, 1-eps, 1+eps)*A) ]
"""

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.agents.rl_types import RLConfig
from src.agents.networks import ActorCritic
from src.agents.buffer import RolloutBuffer


class PPOTrainer:
    """Proximal Policy Optimisation trainer for continuous residual actions."""

    def __init__(self, cfg: RLConfig, num_cav: int, device: str = "cpu"):
        self.cfg = cfg
        self.num_cav = num_cav
        self.device = torch.device(device)

        self.policy = ActorCritic(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.backbone.parameters(), "lr": cfg.lr_actor},
                {"params": self.policy.mean_head.parameters(), "lr": cfg.lr_actor},
                {"params": [self.policy.log_std], "lr": cfg.lr_actor},
                {"params": self.policy.value_head.parameters(), "lr": cfg.lr_critic},
            ]
        )

        self.buffer = RolloutBuffer(cfg, num_cav)
        self.global_step = 0

        # Running stats for logging
        self._update_info: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_actions(self, obs: np.ndarray, deterministic: bool = False):
        """
        Select actions for all CAVs given observation matrix.

        Parameters
        ----------
        obs : np.ndarray (num_cav, OBS_DIM)
        deterministic : bool

        Returns
        -------
        actions   : np.ndarray (num_cav, 1)
        log_probs : np.ndarray (num_cav, 1)
        values    : np.ndarray (num_cav, 1)
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        actions, log_probs, _, values = self.policy.get_action_and_value(
            obs_t, deterministic=deterministic
        )
        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            values.cpu().numpy(),
        )

    @torch.no_grad()
    def get_value(self, obs: np.ndarray) -> np.ndarray:
        """Bootstrap V(o) for GAE."""
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        _, _, value = self.policy(obs_t)
        return value.cpu().numpy()

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------
    def update(self) -> Dict[str, float]:
        """
        Run PPO update using data in the rollout buffer.

        Returns
        -------
        dict of loss components for logging.
        """
        # Bootstrap last value
        # (caller must ensure buffer.obs[buffer.ptr-1] has the last obs)
        last_obs = self.buffer.obs[self.buffer.ptr - 1]
        last_value = self.get_value(last_obs)

        advantages, returns = self.buffer.compute_returns_and_advantages(last_value)

        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_ent = 0.0
        n_updates = 0

        for _ in range(self.cfg.ppo_epochs):
            for batch in self.buffer.get_batches(advantages, returns):
                obs_b = batch["obs"].to(self.device)
                act_b = batch["actions"].to(self.device)
                old_lp = batch["old_log_probs"].to(self.device)
                adv_b = batch["advantages"].to(self.device)
                ret_b = batch["returns"].to(self.device)

                new_lp, entropy, values = self.policy.evaluate_actions(obs_b, act_b)

                # PPO clipped objective
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * adv_b
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.cfg.eps_clip, 1.0 + self.cfg.eps_clip)
                    * adv_b
                )
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                v_loss = nn.functional.mse_loss(values, ret_b)

                # Entropy bonus
                ent_loss = -entropy.mean()

                loss = (
                    pg_loss
                    + self.cfg.value_coef * v_loss
                    + self.cfg.entropy_coef * ent_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_ent += (-ent_loss).item()
                n_updates += 1

        n = max(n_updates, 1)
        self._update_info = {
            "pg_loss": total_pg_loss / n,
            "v_loss": total_v_loss / n,
            "entropy": total_ent / n,
        }

        self.buffer.reset()
        return self._update_info

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "cfg": self.cfg,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
