# -*- coding: utf-8 -*-
"""
Shared data structures for the RL layer.

Extended observation (15-dim):
    o_{i,t} = (mu_v, sigma_v^2, delta_i, v_i, s_i, dv_i, a_leader, alpha_{i,t-1},
               cav_share, mean_speed, speed_var, mean_alpha,
               is_leader_cav, is_follower_cav, local_cav_density)
    a_{i,t} = Delta_alpha_{i,t} in [-Delta_alpha_max, Delta_alpha_max]
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# Observation dimension (15 per CAV: 8 local + 4 global context + 3 neighborhood)
OBS_DIM = 15

# Observation feature names in order
OBS_KEYS = [
    "mu_v",            # upstream mean speed
    "sigma_v_sq",      # upstream speed variance
    "delta",           # speed mismatch v_i - mu_v
    "v",               # own speed
    "s",               # bumper-to-bumper gap
    "dv",              # relative speed (v_self - v_leader, IDM convention)
    "a_leader",        # leader acceleration (previous timestep)
    "alpha_prev",      # previous headway scaling factor
    "cav_share",       # fraction of fleet that is CAV
    "mean_speed",      # global mean speed across all vehicles
    "speed_var",       # global speed variance
    "mean_alpha",      # average alpha across all CAVs
    "is_leader_cav",   # 1.0 if leader is a CAV, 0.0 otherwise
    "is_follower_cav", # 1.0 if follower is a CAV, 0.0 otherwise
    "local_cav_density", # fraction of M upstream vehicles that are CAVs
]


@dataclass
class RLConfig:
    """RL-specific configuration (loaded from YAML).

    All fields mirror the VecEnvConfig defaults so that the CPU training
    path (train_rl.py + reward.py) is feature-identical to the GPU path.
    """

    # Action bounds (wider for exploration)
    delta_alpha_max: float = 0.5

    # Headway scaling limits
    alpha_min: float = 0.5
    alpha_max: float = 2.0

    # -------------------------------------------------------------------------
    # Reward weights (formulation.tex section "Reward")
    # All terms are dimensionless after normalisation in reward.py.
    # Values are kept in sync with config/rl_train.yaml (run-10 baseline).
    # -------------------------------------------------------------------------
    w_s: float = 5.0       # safety: gap violation
    w_tau: float = 3.0     # safety: time-gap violation
    w_v: float = 5.0       # efficiency: speed deficit (one-sided, run-5)
    w_j: float = 0.15      # comfort: jerk penalty (run-5)
    w_ss: float = 0.75     # string stability: error amplification (run-6)
    w_sigma: float = 0.5   # fleet speed variance – soft pacing regulariser (run-6)
    w_sigma_cav: float = 2.0  # CAV-local speed variance (fully controllable, run-6)
    w_alpha: float = 0.3   # residual regularisation — annealed during training
    w_alpha_final: float = 0.10  # final w_alpha after linear annealing (run-10)
    w_damp: float = 0.3    # damping bonus weight (run-9)
    sigma_ref_damp: float = 1.0  # normalisation scale for damping bonus

    # Collision floor: exponential penalty below critical gap threshold
    w_collision_floor: float = 10.0  # weight for exponential safety floor
    collision_gap_critical: float = 1.0  # threshold (m) below which penalty escalates

    # Reference / threshold values
    s_min: float = 2.0     # minimum safe gap (m)
    tau_min: float = 0.8   # minimum safe time-gap (s) — V2V CACC standard
    v_ref: float = 5.5     # reference speed (m/s) — raised in run-5

    # Adaptive v_ref: v_ref_effective = v_ref + v_ref_delta * cav_fraction
    adaptive_v_ref: bool = True
    v_ref_delta: float = 3.0  # run-9: 4.0 → 3.0 (achievable targets at all HRs)

    j_ref: float = 5.0    # jerk normalisation (m/s^3)
    eps_v: float = 0.1    # velocity floor for time-gap computation

    # Fleet-wide penalty scaling by CAV fraction (run-5)
    fleet_penalty_scaling: bool = True

    # PPO hyper-parameters
    gamma: float = 0.99
    lam_gae: float = 0.95
    eps_clip: float = 0.2
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    minibatch_size: int = 64

    # Training control
    rollout_steps: int = 1024     # steps per rollout before PPO update
    total_timesteps: int = 1_000_000
    eval_interval: int = 10_000
    log_interval: int = 1_000
    save_interval: int = 200_000

    # Network architecture
    hidden_dim: int = 128
    num_hidden: int = 2

    # Whether to use mesoscopic baseline as prior (residual mode)
    use_meso_baseline: bool = True

    # RL execution mode: "off" | "rule" | "residual"
    rl_mode: str = "off"

    # Domain randomization: human_rate options and sampling weights for training
    hr_options: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75])
    hr_weights: Optional[List[float]] = None  # sampling weights (None = uniform)

    # Randomise CAV positions each episode (prevents spatial overfitting, run-7)
    shuffle_cav_positions: bool = True

    # Normalizer freeze: stop updating after this many steps
    normalizer_warmup_steps: int = 50_000


@dataclass
class Transition:
    """A single (o, a, r, o', done, log_prob, value) tuple."""

    obs: np.ndarray        # (num_cav, OBS_DIM)
    actions: np.ndarray    # (num_cav,)
    reward: float          # scalar global reward
    next_obs: np.ndarray   # (num_cav, OBS_DIM)
    done: bool
    log_probs: np.ndarray  # (num_cav,)
    values: np.ndarray     # (num_cav,)
