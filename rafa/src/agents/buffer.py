# -*- coding: utf-8 -*-
"""
Rollout buffer for PPO.

Stores transitions collected during one rollout phase and computes
GAE advantages and discounted returns when the rollout is complete.
"""

from typing import List

import numpy as np
import torch

from src.agents.rl_types import RLConfig, OBS_DIM


class RolloutBuffer:
    """Fixed-length rollout storage with GAE computation."""

    def __init__(self, cfg: RLConfig, num_cav: int):
        self.cfg = cfg
        self.num_cav = num_cav
        self.max_steps = cfg.rollout_steps

        # Pre-allocate arrays
        self.obs = np.zeros((self.max_steps, num_cav, OBS_DIM), dtype=np.float32)
        self.actions = np.zeros((self.max_steps, num_cav, 1), dtype=np.float32)
        self.log_probs = np.zeros((self.max_steps, num_cav, 1), dtype=np.float32)
        self.rewards = np.zeros(self.max_steps, dtype=np.float32)
        self.values = np.zeros((self.max_steps, num_cav, 1), dtype=np.float32)
        self.dones = np.zeros(self.max_steps, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def reset(self):
        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,  # (num_cav, OBS_DIM)
        actions: np.ndarray,  # (num_cav, 1)
        log_probs: np.ndarray,  # (num_cav, 1)
        reward: float,
        value: np.ndarray,  # (num_cav, 1)
        done: bool,
    ):
        assert self.ptr < self.max_steps, "Buffer overflow"
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = actions
        self.log_probs[self.ptr] = log_probs
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = float(done)
        self.ptr += 1
        if self.ptr == self.max_steps:
            self.full = True

    # ------------------------------------------------------------------
    def compute_returns_and_advantages(self, last_value: np.ndarray):
        """
        Compute GAE-lambda advantages and discounted returns.

        Parameters
        ----------
        last_value : np.ndarray, shape (num_cav, 1)
            V(o_{T+1}) bootstrap value from the critic.

        Returns
        -------
        advantages : np.ndarray (T, num_cav, 1)
        returns    : np.ndarray (T, num_cav, 1)
        """
        T = self.ptr
        gamma = self.cfg.gamma
        lam = self.cfg.lam_gae

        advantages = np.zeros_like(self.values[:T])
        gae = np.zeros_like(last_value)

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]

            # Broadcast scalar reward to per-CAV shape
            r = self.rewards[t]
            d = self.dones[t]

            delta = r + gamma * (1.0 - d) * next_value - self.values[t]
            gae = delta + gamma * lam * (1.0 - d) * gae
            advantages[t] = gae

        returns = advantages + self.values[:T]
        return advantages, returns

    # ------------------------------------------------------------------
    def get_batches(self, advantages: np.ndarray, returns: np.ndarray):
        """
        Yield randomised minibatches of flattened (CAV, timestep) samples.

        Yields dicts of torch tensors with keys:
            obs, actions, old_log_probs, advantages, returns
        """
        T = self.ptr
        C = self.num_cav
        batch_size = T * C
        mb = self.cfg.minibatch_size

        # Flatten: (T, C, F) -> (T*C, F)
        flat_obs = self.obs[:T].reshape(batch_size, OBS_DIM)
        flat_actions = self.actions[:T].reshape(batch_size, 1)
        flat_log_probs = self.log_probs[:T].reshape(batch_size, 1)
        flat_adv = advantages.reshape(batch_size, 1)
        flat_ret = returns.reshape(batch_size, 1)

        # Normalise advantages
        adv_mean = flat_adv.mean()
        adv_std = flat_adv.std() + 1e-8
        flat_adv = (flat_adv - adv_mean) / adv_std

        indices = np.arange(batch_size)
        np.random.shuffle(indices)

        for start in range(0, batch_size, mb):
            end = min(start + mb, batch_size)
            idx = indices[start:end]
            yield {
                "obs": torch.as_tensor(flat_obs[idx], dtype=torch.float32),
                "actions": torch.as_tensor(flat_actions[idx], dtype=torch.float32),
                "old_log_probs": torch.as_tensor(
                    flat_log_probs[idx], dtype=torch.float32
                ),
                "advantages": torch.as_tensor(flat_adv[idx], dtype=torch.float32),
                "returns": torch.as_tensor(flat_ret[idx], dtype=torch.float32),
            }
