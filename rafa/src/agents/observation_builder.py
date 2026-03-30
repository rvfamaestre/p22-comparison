# -*- coding: utf-8 -*-
"""
Observation builder for the RL agent.

Assembles per-CAV observation vectors from the simulator state.
Supported observation layouts:
- 8-dim legacy PPO checkpoints
- 12-dim intermediate checkpoints
- 15-dim extended checkpoints

Extended formulation (15-dim):
    o_{i,t} = (mu_v, sigma_v^2, delta_i, v_i, s_i, dv_i, a_leader, alpha_{i,t-1},
               cav_share, mean_speed, speed_var, mean_alpha,
               is_leader_cav, is_follower_cav, local_cav_density)

The last 7 features are context that enable the policy to condition on
traffic composition and local CAV topology (critical for domain
randomisation and shuffled CAV positions during training).

Includes a Welford running normalizer so all features are zero-mean,
unit-variance when fed to the policy network.
"""

from typing import Dict, List, Tuple

import numpy as np

from src.agents.rl_types import OBS_DIM, OBS_KEYS
from src.mesoscopic.meso_adapter import get_M_leaders_ring
from src.vehicles.cav_vehicle import CAVVehicle


# ---------------------------------------------------------------
# Running observation normalizer (Welford one-pass algorithm)
# ---------------------------------------------------------------


class RunningNormalizer:
    """Tracks running mean & variance and normalizes observations."""

    def __init__(self, dim: int, clip: float = 10.0):
        self.dim = dim
        self.clip = clip
        self.count = 0
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self._M2 = np.zeros(dim, dtype=np.float64)
        self.frozen = False

    def update(self, batch: np.ndarray):
        """Update stats with a batch of shape (B, dim)."""
        if self.frozen:
            return
        for row in batch:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self._M2 += delta * delta2
            self.var = self._M2 / max(self.count, 2) + 1e-8

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize obs (shape (B, dim)) to zero-mean, unit-variance."""
        return np.clip(
            (obs - self.mean.astype(np.float32)) / np.sqrt(self.var).astype(np.float32),
            -self.clip,
            self.clip,
        ).astype(np.float32)

    def save(self, path: str):
        np.savez(
            path, mean=self.mean, var=self.var, M2=self._M2, count=np.array(self.count)
        )

    def load(self, path: str):
        data = np.load(path)
        self.mean = data["mean"]
        self.var = data["var"]
        self._M2 = data["M2"]
        self.count = int(data["count"])

    def freeze(self):
        """Stop updating statistics (use current mean/var for normalization)."""
        self.frozen = True


# ---------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------


class ObservationBuilder:
    """Builds observation arrays for all CAVs from the current simulator state."""

    def __init__(
        self,
        M: int = 8,
        v_eps: float = 0.5,
        normalize: bool = True,
        obs_dim: int = OBS_DIM,
    ):
        """
        Parameters
        ----------
        M : int
            Number of upstream vehicles to sample for statistics.
        v_eps : float
            Velocity floor for upstream statistics (m/s).
        normalize : bool
            Whether to apply running normalization to observations.
        obs_dim : int
            Observation dimension. Supported values are:
            - 8 for legacy checkpoints
            - 12 for intermediate checkpoints
            - 15 for extended checkpoints with neighborhood features
        """
        if obs_dim not in (8, 12, 15):
            raise ValueError(
                f"Unsupported obs_dim={obs_dim}. Expected one of 8, 12, or 15."
            )
        self.M = M
        self.v_eps = v_eps
        self.normalize = normalize
        self.obs_dim = obs_dim
        self.normalizer = RunningNormalizer(obs_dim) if normalize else None

    def build(
        self,
        vehicles: list,
        L: float,
        alpha_prev: Dict[int, float],
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Build observation matrix for all CAVs.

        Parameters
        ----------
        vehicles : list[Vehicle]
            All vehicles, **already sorted by position** (the simulator does
            this at the start of each step).
        L : float
            Ring circumference.
        alpha_prev : dict[int, float]
            Previous alpha for each CAV (keyed by vehicle id).

        Returns
        -------
        obs : np.ndarray, shape (num_cav, OBS_DIM)
            Row *k* is the observation for the *k*-th CAV (in the order they
            appear in `cav_ids`).
        cav_ids : list[int]
            Vehicle ids of the CAVs, in the same row-order as `obs`.
        """
        cav_indices: List[Tuple[int, CAVVehicle]] = []
        for idx, v in enumerate(vehicles):
            if isinstance(v, CAVVehicle):
                cav_indices.append((idx, v))

        num_cav = len(cav_indices)
        if num_cav == 0:
            return np.empty((0, self.obs_dim), dtype=np.float32), []

        obs = np.zeros((num_cav, self.obs_dim), dtype=np.float32)
        cav_ids: List[int] = []

        # --- global / composition features (shared across all CAVs) ---
        all_speeds = np.array([v.v for v in vehicles])
        cav_share = float(num_cav) / max(len(vehicles), 1)
        mean_speed_global = float(np.mean(all_speeds))
        speed_var_global = float(np.var(all_speeds))
        alpha_vals = [alpha_prev.get(v.id, 1.0) for _, v in cav_indices]
        mean_alpha = float(np.mean(alpha_vals)) if alpha_vals else 1.0

        for k, (list_idx, cav) in enumerate(cav_indices):
            cav_ids.append(cav.id)

            # --- upstream statistics (mu_v, sigma_v^2) ---
            upstream_vels = get_M_leaders_ring(vehicles, list_idx, self.M, L)
            upstream_safe = [max(u, self.v_eps) for u in upstream_vels]
            mu_v = float(np.mean(upstream_safe)) if upstream_safe else 0.0
            sigma_v_sq = float(np.var(upstream_safe)) if upstream_safe else 0.0

            # --- local variables ---
            s = cav.compute_gap(L)
            dv = cav.v - cav.leader.v  # IDM convention (approaching = positive)
            a_leader = getattr(cav.leader, "acceleration", 0.0)
            alpha_p = alpha_prev.get(cav.id, 1.0)

            base_features = [
                mu_v,
                sigma_v_sq,
                cav.v - mu_v,  # delta_i
                cav.v,
                s,
                dv,
                a_leader,
                alpha_p,
            ]
            context_features = [
                cav_share,
                mean_speed_global,
                speed_var_global,
                mean_alpha,
            ]
            if self.obs_dim == 8:
                obs[k, :8] = base_features
            else:
                obs[k, :12] = base_features + context_features

            # Extended features (15-dim): local neighborhood topology
            if self.obs_dim >= 15:
                N = len(vehicles)
                # Is leader a CAV?
                leader_is_cav = 1.0 if isinstance(cav.leader, CAVVehicle) else 0.0
                # Is follower a CAV? (vehicle behind in ring)
                follower_idx = (list_idx - 1) % N
                follower_is_cav = (
                    1.0 if isinstance(vehicles[follower_idx], CAVVehicle) else 0.0
                )
                # Local CAV density among M upstream vehicles
                upstream_cav_count = sum(
                    1
                    for m in range(1, self.M + 1)
                    if isinstance(vehicles[(list_idx + m) % N], CAVVehicle)
                )
                local_cav_density = upstream_cav_count / max(self.M, 1)
                obs[k, 12] = leader_is_cav
                obs[k, 13] = follower_is_cav
                obs[k, 14] = local_cav_density

        # Running normalization (zero-mean, unit-variance)
        if self.normalizer is not None and num_cav > 0:
            self.normalizer.update(obs)
            obs = self.normalizer.normalize(obs)

        return obs, cav_ids
