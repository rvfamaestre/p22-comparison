"""Shared utilities for the bounded continuous action space used by PPO and SAC."""

import numpy as np
import torch
from torch.distributions import Normal


class RunningMeanStd:
    """Online mean/variance tracker for state and reward normalization."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            return

        if np.shape(self.mean) == ():
            x = x.reshape(-1)
            batch_mean = float(np.mean(x))
            batch_var = float(np.var(x))
            batch_count = x.shape[0]
        else:
            x = x.reshape(-1, *np.shape(self.mean))
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        if batch_count <= 0:
            return

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count

        self.mean = np.asarray(new_mean, dtype=np.float32)
        self.var = np.asarray(m_2 / total_count, dtype=np.float32)
        self.count = float(total_count)


def prepare_action_bounds(action_low, action_high, reference):
    """Broadcast scalar or vector action bounds to the shape of a model output."""
    low = torch.as_tensor(action_low, dtype=reference.dtype, device=reference.device)
    high = torch.as_tensor(action_high, dtype=reference.dtype, device=reference.device)

    if low.dim() == 0:
        low = low.reshape(1, 1)
        high = high.reshape(1, 1)
    elif low.dim() == 1 and reference.dim() == 2:
        low = low.unsqueeze(-1)
        high = high.unsqueeze(-1)

    return low.expand_as(reference), high.expand_as(reference)


def sample_bounded_normal(mean, log_std, action_low, action_high, deterministic=False, eps=1e-6):
    """
    Sample from a tanh-squashed Gaussian and rescale it to the feasible action range.

    This is used by both PPO and SAC so that training always sees the bounded action
    that the simulator can actually execute.
    """
    low, high = prepare_action_bounds(action_low, action_high, mean)
    std = torch.exp(log_std)
    dist = Normal(mean, std)

    pre_tanh = mean if deterministic else dist.rsample()
    squashed = torch.tanh(pre_tanh)

    scale = 0.5 * (high - low)
    bias = 0.5 * (high + low)
    safe_scale = torch.clamp(scale, min=eps)

    action = bias + scale * squashed

    log_prob = (
        dist.log_prob(pre_tanh)
        - torch.log(safe_scale)
        - torch.log(1.0 - squashed.pow(2) + eps)
    )

    free_mask = (scale > eps).to(mean.dtype)
    log_prob = (log_prob * free_mask).sum(dim=-1)
    entropy = (dist.entropy() * free_mask).sum(dim=-1)

    return action, log_prob, entropy


def log_prob_bounded_normal(mean, log_std, actions, action_low, action_high, eps=1e-6):
    """Evaluate the corrected log-probability of an already bounded action."""
    if actions.dim() == 1 and mean.dim() == 2:
        actions = actions.unsqueeze(-1)

    low, high = prepare_action_bounds(action_low, action_high, mean)
    std = torch.exp(log_std)
    dist = Normal(mean, std)

    scale = 0.5 * (high - low)
    bias = 0.5 * (high + low)
    safe_scale = torch.clamp(scale, min=eps)

    normalized = ((actions - bias) / safe_scale).clamp(-1.0 + eps, 1.0 - eps)
    pre_tanh = 0.5 * (torch.log1p(normalized) - torch.log1p(-normalized))

    log_prob = (
        dist.log_prob(pre_tanh)
        - torch.log(safe_scale)
        - torch.log(1.0 - normalized.pow(2) + eps)
    )

    free_mask = (scale > eps).to(mean.dtype)
    return (log_prob * free_mask).sum(dim=-1)
