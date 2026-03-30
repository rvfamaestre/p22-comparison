# -*- coding: utf-8 -*-
"""
Vectorized Ring-Road Environment — fully batched on GPU via PyTorch.

Implements the *entire* ring-road physics (IDM + CACC + mesoscopic adapter)
as tensor operations so that ``NUM_ENVS`` independent simulations run in
parallel on a single GPU kernel launch.

Key design choices
------------------
* **No Python loops over vehicles or environments.**  Every quantity is a
  (num_envs, N) tensor and uses only ``torch`` broadcasting / indexing.
* Vehicle type is encoded in a boolean mask ``is_cav`` of shape (num_envs, N).
* Mesoscopic adaptation, RL residual injection, observation building, and
  reward computation are all fused into the ``step`` call.
* Domain randomisation is vectorised: each env can have a different
  human_rate, resampled independently on reset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Configuration (mirrors RLConfig + simulator config, but for GPU env)
# --------------------------------------------------------------------------- #


@dataclass
class VecEnvConfig:
    """All parameters needed by the vectorized environment."""

    # Ring road
    N: int = 22
    L: float = 300.0
    dt: float = 0.1
    episode_steps: int = 1500  # T / dt

    # IDM (human) parameters
    idm_s0: float = 2.0
    idm_T: float = 1.5
    idm_a: float = 0.3
    idm_b: float = 3.0
    idm_v0: float = 14.0
    idm_delta: float = 4.0
    noise_Q: float = 0.45

    # CACC (CAV) parameters
    cacc_ks: float = 0.4
    cacc_kv: float = 1.3
    cacc_kf: float = 0.95
    cacc_kv0: float = 0.05
    cacc_b_max: float = 3.0
    cacc_a_max: float = 1.5
    cacc_v_max: float = 20.0
    cacc_v_des: float = 14.0
    cth_d0: float = 3.5
    cth_hc: float = 1.2

    # Mesoscopic adaptation
    meso_enabled: bool = True
    meso_M: int = 8
    meso_lambda_rho: float = 0.8
    meso_gamma: float = 0.4
    meso_alpha_min: float = 0.7
    meso_alpha_max: float = 2.0
    meso_max_alpha_rate: float = 0.2
    meso_sigma_v_ema_lambda: float = 0.9
    meso_sigma_v_min_threshold: float = 0.2
    meso_v_eps_sigma: float = 0.5
    meso_psi_deadband: float = 0.5

    # RL residual action
    rl_delta_alpha_max: float = 0.5
    rl_alpha_min: float = 0.5
    rl_alpha_max: float = 2.0

    # Reward weights (dimensionless)
    w_s: float = 5.0
    w_tau: float = 3.0
    w_v: float = 5.0
    w_j: float = 0.15
    w_ss: float = 0.75
    w_sigma: float = 0.5  # Fleet variance — soft regulariser for pacing effects
    w_sigma_cav: float = 2.0  # CAV-local speed variance (controllable)
    w_alpha: float = 0.3
    s_min_reward: float = 2.0  # Gap safety threshold
    tau_min_reward: float = 0.8  # Time-gap threshold (V2V cooperative ACC standard)
    v_ref: float = 5.5
    j_ref: float = 5.0
    eps_v: float = 0.1
    fleet_penalty_scaling: bool = True  # scale fleet-wide terms by CAV fraction

    # Perturbation (curriculum: randomise magnitude, timing, target)
    perturbation_enabled: bool = True
    perturbation_time: float = 3.0  # legacy default (used if curriculum disabled)
    perturbation_delta_v: float = -3.0  # legacy default
    noise_warmup_time: float = 3.0
    perturb_curriculum: bool = True  # enable perturbation randomisation
    perturb_dv_min: float = -4.0  # strongest brake (m/s)
    perturb_dv_max: float = -1.0  # lightest brake (m/s)
    perturb_time_min: float = 2.0  # earliest perturbation (s)
    perturb_time_max: float = 8.0  # latest perturbation (s)
    perturb_random_target: bool = True  # random vehicle or always vehicle 0

    # Warm-up
    warmup_duration: float = 10.0
    warmup_accel_limit: float = 1.0

    # Domain randomisation
    hr_options: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)
    hr_weights: Optional[Tuple[float, ...]] = None  # sampling weights (None = uniform)
    shuffle_cav_positions: bool = False

    # Damping reward (bonus for reducing fleet variance)
    w_damp: float = 0.3  # damping reward weight (run-9: 1.0→0.3)
    sigma_ref_damp: float = 1.0  # reference sigma for normalisation

    # Collision floor: exponential penalty when any gap falls below critical threshold
    w_collision_floor: float = 10.0  # weight for exponential safety floor
    collision_gap_critical: float = 1.0  # threshold (m) below which penalty escalates

    # Adaptive v_ref: scales reference speed with CAV fraction
    adaptive_v_ref: bool = True
    v_ref_delta: float = (
        3.0  # v_ref = v_ref + v_ref_delta * cav_fraction (run-9: 4.0→3.0)
    )

    # w_alpha annealing: final value after linear decay
    w_alpha_final: float = (
        0.10  # run-10: 0.0→0.10 (keep gradient guide above critical ~0.12)
    )


# --------------------------------------------------------------------------- #
# Vectorized environment
# --------------------------------------------------------------------------- #


class VecRingRoadEnv:
    """
    Batched ring-road environment running entirely on GPU.

    Shapes (B = num_envs, N = num vehicles per ring):
        x       : (B, N)     positions
        v       : (B, N)     velocities
        a_prev  : (B, N)     accelerations from previous step
        is_cav  : (B, N)     bool mask
        alpha   : (B, N)     current headway scaling (CAVs only, 1.0 for HDVs)
    """

    def __init__(self, num_envs: int, cfg: VecEnvConfig, device: torch.device):
        self.B = num_envs
        self.cfg = cfg
        self.N = cfg.N
        self.device = device

        # Pre-compute constants
        self.veh_length = 5.0
        self.min_gap = 0.3

        # hr_options as tensor for fast sampling
        self._hr_opts = torch.tensor(cfg.hr_options, device=device, dtype=torch.float32)
        # hr_weights for importance-weighted domain randomisation
        if cfg.hr_weights is not None:
            w = torch.tensor(cfg.hr_weights, device=device, dtype=torch.float32)
            self._hr_weights = w / w.sum()  # normalise to probabilities
        else:
            self._hr_weights = None  # uniform sampling

        # Allocate persistent state tensors
        self.x = torch.zeros(num_envs, self.N, device=device)
        self.v = torch.zeros(num_envs, self.N, device=device)
        self.a_prev = torch.zeros(num_envs, self.N, device=device)
        self.is_cav = torch.zeros(num_envs, self.N, device=device, dtype=torch.bool)
        self.alpha = torch.ones(num_envs, self.N, device=device)
        self.alpha_rule = torch.ones(num_envs, self.N, device=device)

        # Mesoscopic adapter state (per CAV per env)
        self.meso_rho = torch.zeros(num_envs, self.N, device=device)
        self.meso_alpha_prev = torch.ones(num_envs, self.N, device=device)
        self.meso_sigma_smooth = torch.zeros(num_envs, self.N, device=device)

        # Episode bookkeeping
        self.step_count = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.human_rate = torch.zeros(num_envs, device=device)
        self.num_cav = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.perturbation_applied = torch.zeros(
            num_envs, device=device, dtype=torch.bool
        )
        self.collision_clamp_count = torch.zeros(
            num_envs, device=device, dtype=torch.long
        )

        # Per-env perturbation parameters (curriculum)
        self.perturb_target = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.perturb_time = torch.full(
            (num_envs,), cfg.perturbation_time, device=device
        )
        self.perturb_dv = torch.full(
            (num_envs,), cfg.perturbation_delta_v, device=device
        )

        # Damping reward state
        self.prev_fleet_var = torch.zeros(num_envs, device=device)

        # w_alpha annealing (mutable, updated by training loop)
        self.current_w_alpha = cfg.w_alpha

        # For reward jerk computation
        self.a_prev_reward = torch.zeros(num_envs, self.N, device=device)

        # Sort indices (reused)
        self._sort_idx = torch.zeros(num_envs, self.N, device=device, dtype=torch.long)
        self._leader_idx = torch.zeros(
            num_envs, self.N, device=device, dtype=torch.long
        )
        self._follower_idx = torch.zeros(
            num_envs, self.N, device=device, dtype=torch.long
        )

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #
    def reset(self, env_mask: Optional[torch.Tensor] = None) -> None:
        """
        Reset environments indicated by ``env_mask`` (default: all).

        After reset every env has uniform spacing, equal speed, and a
        freshly sampled human_rate from ``hr_options``.
        """
        if env_mask is None:
            env_mask = torch.ones(self.B, device=self.device, dtype=torch.bool)

        B_reset = int(env_mask.sum().item())
        if B_reset == 0:
            return

        cfg = self.cfg
        N = self.N

        # Domain randomisation: sample new human_rate per env
        if self._hr_weights is not None:
            idx = torch.multinomial(self._hr_weights, B_reset, replacement=True)
        else:
            idx = torch.randint(len(self._hr_opts), (B_reset,), device=self.device)
        hr = self._hr_opts[idx]
        self.human_rate[env_mask] = hr

        # Number of humans & CAVs
        num_human = (hr * N).long()
        num_cav = N - num_human
        self.num_cav[env_mask] = num_cav

        # Build is_cav mask: last num_cav vehicles in each env are CAVs
        # Then optionally shuffle to intersperse CAVs among HDVs
        is_cav_new = torch.zeros(B_reset, N, device=self.device, dtype=torch.bool)
        for i in range(B_reset):
            nc = int(num_cav[i].item())
            if nc > 0:
                is_cav_new[i, N - nc :] = True
                if getattr(cfg, "shuffle_cav_positions", False):
                    # Shuffle CAV positions so they're not always clustered
                    perm = torch.randperm(N, device=self.device)
                    is_cav_new[i] = is_cav_new[i, perm]
        self.is_cav[env_mask] = is_cav_new

        # Uniform spacing
        spacing = cfg.L / N
        positions = torch.arange(N, device=self.device, dtype=torch.float32) * spacing
        self.x[env_mask] = positions.unsqueeze(0).expand(B_reset, -1)

        # Uniform initial speed
        init_v = cfg.L / N / cfg.cth_hc  # ~equilibrium speed
        init_v = min(init_v, 5.5)
        self.v[env_mask] = init_v

        # Zero accelerations
        self.a_prev[env_mask] = 0.0
        self.a_prev_reward[env_mask] = 0.0

        # Reset mesoscopic state
        self.alpha[env_mask] = 1.0
        self.alpha_rule[env_mask] = 1.0
        self.meso_rho[env_mask] = 0.0
        self.meso_alpha_prev[env_mask] = 1.0
        self.meso_sigma_smooth[env_mask] = 0.0

        # Reset step counter and perturbation flag
        self.step_count[env_mask] = 0
        self.perturbation_applied[env_mask] = False
        self.collision_clamp_count[env_mask] = 0

        # Perturbation curriculum: sample per-env perturbation parameters
        if not cfg.perturbation_enabled:
            self.perturb_time[env_mask] = float("inf")
            self.perturb_dv[env_mask] = 0.0
            self.perturb_target[env_mask] = 0
        elif cfg.perturb_curriculum:
            self.perturb_time[env_mask] = torch.empty(
                B_reset, device=self.device
            ).uniform_(cfg.perturb_time_min, cfg.perturb_time_max)
            self.perturb_dv[env_mask] = torch.empty(
                B_reset, device=self.device
            ).uniform_(cfg.perturb_dv_min, cfg.perturb_dv_max)
            if cfg.perturb_random_target:
                self.perturb_target[env_mask] = torch.randint(
                    0, N, (B_reset,), device=self.device
                )
            else:
                self.perturb_target[env_mask] = 0
        else:
            self.perturb_time[env_mask] = cfg.perturbation_time
            self.perturb_dv[env_mask] = cfg.perturbation_delta_v
            self.perturb_target[env_mask] = 0

        # Reset damping state
        self.prev_fleet_var[env_mask] = 0.0

    # ------------------------------------------------------------------ #
    # Sorted-order helpers
    # ------------------------------------------------------------------ #
    def _sort_and_gaps(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sort vehicles by position, compute leader indices and gaps.

        Returns
        -------
        sort_idx   : (B, N)  indices that sort x ascending
        gaps       : (B, N)  bumper-to-bumper gap to leader (original vehicle order)
        dv_leader  : (B, N)  v_leader - v_self (CACC convention, original vehicle order)
        """
        sort_idx = self.x.argsort(
            dim=1
        )  # sort_idx[b, j] = orig index of j-th sorted vehicle

        # Inverse permutation: inv_sort[b, v] = sort position of vehicle v
        arange_N = (
            torch.arange(self.N, device=self.device).unsqueeze(0).expand(self.B, -1)
        )
        inv_sort = torch.zeros_like(sort_idx)
        inv_sort.scatter_(1, sort_idx, arange_N)

        # Leader of each original vehicle: the next vehicle in ascending-position order
        leader_sort_pos = (inv_sort + 1) % self.N  # (B, N)
        leader_idx = sort_idx.gather(
            1, leader_sort_pos
        )  # (B, N) in original vehicle order

        # Follower of each original vehicle: the previous vehicle in ascending-position order
        follower_sort_pos = (inv_sort - 1) % self.N  # (B, N)
        follower_idx = sort_idx.gather(
            1, follower_sort_pos
        )  # (B, N) in original vehicle order

        x_leader = self.x.gather(1, leader_idx)
        v_leader = self.v.gather(1, leader_idx)

        # Gap: handle ring wraparound
        raw_gap = x_leader - self.x
        gaps = torch.where(raw_gap < 0, raw_gap + self.cfg.L, raw_gap) - self.veh_length

        # Relative speed (CACC convention: positive = opening)
        dv_leader = v_leader - self.v

        self._sort_idx = sort_idx
        self._leader_idx = leader_idx
        self._follower_idx = follower_idx
        return sort_idx, gaps, dv_leader

    # ------------------------------------------------------------------ #
    # Upstream statistics for mesoscopic adapter (vectorised)
    # ------------------------------------------------------------------ #
    def _upstream_stats(
        self, sort_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each vehicle compute mean and variance of M upstream vehicles'
        speeds. Fully vectorised using gather over a window.

        Returns (B, N) tensors: mu_v, sigma_v_sq.
        """
        M = self.cfg.meso_M
        v_s = self.v.gather(1, sort_idx)  # sorted speeds

        # Build M-column index matrix: for vehicle at sorted position j,
        # upstream vehicles are at sorted positions j+1 .. j+M (mod N)
        arange_N = torch.arange(self.N, device=self.device)
        # offsets (1..M)
        offsets = torch.arange(1, M + 1, device=self.device)
        # (N, M) upstream sorted indices
        up_sorted = (arange_N.unsqueeze(1) + offsets.unsqueeze(0)) % self.N
        # expand to (B, N, M)
        up_sorted = up_sorted.unsqueeze(0).expand(self.B, -1, -1)

        # Gather upstream speeds: v_s is (B, N), up_sorted is (B, N, M)
        # Index dim=1 of v_s for each of the M upstream positions
        v_ups = torch.zeros(self.B, self.N, M, device=self.device)
        for m in range(M):
            v_ups[:, :, m] = v_s.gather(1, up_sorted[:, :, m])

        # Clamp to v_eps
        v_ups = v_ups.clamp(min=self.cfg.meso_v_eps_sigma)

        mu_v = v_ups.mean(dim=2)  # (B, N)
        sigma_v_sq = v_ups.var(dim=2)  # (B, N)

        # Un-sort: put back in original vehicle order
        # sort_idx maps sorted→original, we need inverse
        inv_sort = torch.zeros_like(sort_idx)
        inv_sort.scatter_(
            1,
            sort_idx,
            torch.arange(self.N, device=self.device).unsqueeze(0).expand(self.B, -1),
        )
        mu_v = mu_v.gather(1, inv_sort)
        sigma_v_sq = sigma_v_sq.gather(1, inv_sort)

        return mu_v, sigma_v_sq

    # ------------------------------------------------------------------ #
    # Mesoscopic alpha computation (vectorised)
    # ------------------------------------------------------------------ #
    def _compute_meso_alpha(
        self, mu_v: torch.Tensor, sigma_v_sq: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorised mesoscopic adaptation for all CAVs in all envs.

        Returns alpha (B, N) — only meaningful where is_cav is True.
        """
        cfg = self.cfg
        sigma_v = sigma_v_sq.sqrt()

        # EMA smoothing of sigma_v
        lam = cfg.meso_sigma_v_ema_lambda
        self.meso_sigma_smooth = lam * self.meso_sigma_smooth + (1 - lam) * sigma_v

        # Thresholding
        sig = self.meso_sigma_smooth.clone()
        sig = torch.where(
            sig < cfg.meso_sigma_v_min_threshold, torch.zeros_like(sig), sig
        )

        # Velocity mismatch (psi) with deadband
        psi = self.v - mu_v
        psi = torch.where(psi.abs() < cfg.meso_psi_deadband, torch.zeros_like(psi), psi)

        # Beta = sig * sign(psi) — direction-aware stress input
        beta = sig * psi.sign()

        # EMA filter: rho_{t+1} = lambda * rho_t + (1-lambda) * gamma * beta
        self.meso_rho = (
            cfg.meso_lambda_rho * self.meso_rho
            + (1 - cfg.meso_lambda_rho) * cfg.meso_gamma * beta
        )

        # Alpha = 1 + rho, clamped
        alpha_raw = 1.0 + self.meso_rho

        # Rate limiting: |alpha - alpha_prev| <= max_rate * dt
        max_delta = cfg.meso_max_alpha_rate * cfg.dt
        alpha_clamped = self.meso_alpha_prev + (alpha_raw - self.meso_alpha_prev).clamp(
            -max_delta, max_delta
        )

        alpha_out = alpha_clamped.clamp(cfg.meso_alpha_min, cfg.meso_alpha_max)

        # Update state (only for CAVs, but tensor-wide is fine; HDV values ignored)
        self.meso_alpha_prev = alpha_out.clone()

        return alpha_out

    # ------------------------------------------------------------------ #
    # IDM acceleration (vectorised, applied to HDVs)
    # ------------------------------------------------------------------ #
    def _idm_acc(
        self, gaps: torch.Tensor, v: torch.Tensor, dv_closing: torch.Tensor
    ) -> torch.Tensor:
        """
        IDM acceleration for all vehicles.  dv_closing = v_self - v_leader.

        Returns (B, N) tensor.
        """
        cfg = self.cfg
        s = gaps.clamp(min=cfg.idm_s0)
        s_star = (
            cfg.idm_s0
            + v * cfg.idm_T
            + (v * dv_closing) / (2.0 * math.sqrt(cfg.idm_a * cfg.idm_b))
        )
        s_star = s_star.clamp(min=cfg.idm_s0)

        acc = cfg.idm_a * (
            1.0 - (v / cfg.idm_v0).pow(cfg.idm_delta) - (s_star / s).pow(2)
        )
        acc = acc.clamp(min=-cfg.idm_b)
        return acc

    # ------------------------------------------------------------------ #
    # CACC acceleration (vectorised, applied to CAVs)
    # ------------------------------------------------------------------ #
    def _cacc_acc(
        self,
        gaps: torch.Tensor,
        v: torch.Tensor,
        dv_open: torch.Tensor,
        a_leader: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        CACC acceleration with adapted gains from mesoscopic alpha.

        dv_open = v_leader - v_self (positive = gap opening).

        Returns (B, N) tensor.
        """
        cfg = self.cfg

        # Adapted gains from alpha
        h_c = alpha * cfg.cth_hc
        k_f = (cfg.cacc_kf * alpha.sqrt()).clamp(max=1.0)
        k_v0 = cfg.cacc_kv0 / alpha.clamp(min=0.01)

        s_des = cfg.cth_d0 + h_c * v
        e_s = gaps - s_des
        e_v = dv_open
        e_vdes = cfg.cacc_v_des - v

        acc = cfg.cacc_ks * e_s + cfg.cacc_kv * e_v + k_f * a_leader + k_v0 * e_vdes

        # Emergency braking for very small gaps
        emergency = gaps < 0.1
        acc = torch.where(emergency, torch.full_like(acc, -cfg.cacc_b_max), acc)

        # Saturation
        acc = acc.clamp(-cfg.cacc_b_max, cfg.cacc_a_max)
        return acc

    # ------------------------------------------------------------------ #
    # Observation builder (vectorised, 15-dim per CAV)
    # ------------------------------------------------------------------ #
    def build_obs(
        self,
        mu_v: torch.Tensor,
        sigma_v_sq: torch.Tensor,
        gaps: torch.Tensor,
        dv_leader: torch.Tensor,
        a_leader: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build (B, N, 15) observation tensor and a validity mask.

        We pad to max_cav = N (worst case all CAVs).  The mask indicates
        which entries are real CAVs.

        Returns
        -------
        obs   : (B, N, 15)
        mask  : (B, N) bool — True where a real CAV exists
        """
        cfg = self.cfg
        B, N = self.B, self.N

        # Per-vehicle features (all B, N)
        delta = self.v - mu_v
        cav_share = (self.num_cav.float() / N).unsqueeze(1).expand(B, N)
        mean_speed = self.v.mean(dim=1, keepdim=True).expand(B, N)
        speed_var = self.v.var(dim=1, keepdim=True).expand(B, N)

        # mean_alpha across CAVs per env
        alpha_masked = self.alpha.clone()
        alpha_masked[~self.is_cav] = 0.0
        cav_count = self.num_cav.float().clamp(min=1).unsqueeze(1)
        mean_alpha = (alpha_masked.sum(dim=1, keepdim=True) / cav_count).expand(B, N)

        # Local neighborhood features
        leader_is_cav = self.is_cav.gather(1, self._leader_idx).float()  # (B, N)
        follower_is_cav = self.is_cav.gather(1, self._follower_idx).float()  # (B, N)

        # Local CAV density: fraction of M upstream vehicles that are CAVs
        M = cfg.meso_M
        sort_idx = self._sort_idx
        is_cav_sorted = self.is_cav.float().gather(1, sort_idx)  # (B, N) sorted
        arange_sorted = torch.arange(N, device=self.device).unsqueeze(0).expand(B, -1)
        local_cav_sum = torch.zeros(B, N, device=self.device)
        for m in range(1, M + 1):
            up_pos = (arange_sorted + m) % N
            local_cav_sum += is_cav_sorted.gather(1, up_pos)
        local_cav_density_sorted = local_cav_sum / M
        # Un-sort to original vehicle order
        local_cav_density = torch.zeros_like(local_cav_density_sorted)
        local_cav_density.scatter_(1, sort_idx, local_cav_density_sorted)

        obs = torch.stack(
            [
                mu_v,  # 0
                sigma_v_sq,  # 1
                delta,  # 2
                self.v,  # 3
                gaps,  # 4
                -dv_leader,  # 5  IDM convention: v_self - v_leader
                a_leader,  # 6
                self.alpha,  # 7  alpha_prev
                cav_share,  # 8
                mean_speed,  # 9
                speed_var,  # 10
                mean_alpha,  # 11
                leader_is_cav,  # 12
                follower_is_cav,  # 13
                local_cav_density,  # 14
            ],
            dim=2,
        )  # (B, N, 15)

        return obs, self.is_cav  # mask

    # ------------------------------------------------------------------ #
    # Reward (vectorised, global scalar per env)
    # ------------------------------------------------------------------ #
    def compute_reward(
        self, gaps: torch.Tensor, dv_leader: torch.Tensor, delta_alphas: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the global scalar reward for each env.

        delta_alphas : (B, N)  — RL residual (0 for HDVs)

        Returns (B,) reward tensor.

        Fleet-wide penalty scaling
        --------------------------
        When ``fleet_penalty_scaling`` is enabled, reward components that
        average over *all* vehicles (efficiency, fleet speed variance)
        are multiplied by the CAV fraction so that uncontrollable HDV
        behaviour does not dominate the learning signal at high human
        rates.  Safety penalties (gap + time-gap) are **not** scaled by
        eta so the agent cannot exploit reduced penalties at high human
        ratios.  A separate CAV-only speed-variance term
        (``w_sigma_cav``) provides a clean, controllable signal.
        """
        cfg = self.cfg
        B, N = self.B, self.N
        v = self.v
        eps_v = cfg.eps_v

        # CAV fraction per env (B,) — used to scale fleet-wide terms
        cav_frac = (self.num_cav.float() / N).clamp(min=1.0 / N)
        # eta = 1 when all CAVs, shrinks linearly with fewer CAVs
        eta = (
            cav_frac if cfg.fleet_penalty_scaling else torch.ones(B, device=self.device)
        )

        # Adaptive v_ref: scale reference speed with CAV fraction
        if cfg.adaptive_v_ref:
            v_ref = cfg.v_ref + cfg.v_ref_delta * cav_frac  # (B,)
            v_ref = v_ref.unsqueeze(1)  # (B, 1) for broadcasting
            v_ref_scalar = v_ref.squeeze(1)  # (B,) for scalar terms
        else:
            v_ref = max(cfg.v_ref, 1e-3)
            v_ref_scalar = torch.full((B,), v_ref, device=self.device)

        # -- r_sf: safety (fleet-wide, NOT scaled by eta) --
        # Safety penalties always apply at full weight regardless of CAV fraction
        # so the agent cannot exploit reduced penalties at high human ratios.
        gap_viol = (1.0 - gaps / max(cfg.s_min_reward, 1e-3)).clamp(min=0).pow(2)
        time_gap = gaps / v.clamp(min=eps_v)
        tau_viol = (1.0 - time_gap / max(cfg.tau_min_reward, 1e-3)).clamp(min=0).pow(2)
        r_sf = -(
            cfg.w_s * gap_viol.mean(dim=1) + cfg.w_tau * tau_viol.mean(dim=1)
        )

        # -- r_collision_floor: exponential penalty below critical gap --
        # Makes near-collision states non-negotiable (unbounded as gap → 0).
        s_crit = max(cfg.collision_gap_critical, 1e-3)
        crit_viol = (s_crit - gaps).clamp(min=0) / s_crit  # (B, N) in [0, ~1]
        # exp(x)-1 gives smooth escalation: 0 at boundary, steep inside
        r_collision_floor = -cfg.w_collision_floor * (
            crit_viol.exp() - 1.0
        ).mean(dim=1)

        # -- r_ef: efficiency (fleet-wide, scaled by eta) --
        # One-sided: only penalise speeds BELOW v_ref (don't punish high throughput)
        speed_deficit = ((v_ref - v) / v_ref.clamp(min=1e-3)).clamp(min=0)
        r_ef = -eta * cfg.w_v * speed_deficit.pow(2).mean(dim=1)

        # -- r_cf: comfort (jerk), only CAVs --
        jerk = (self.a_prev - self.a_prev_reward) / max(cfg.dt, 1e-6)
        jerk_norm = (jerk / max(cfg.j_ref, 1e-3)).pow(2)
        jerk_norm_cav = jerk_norm * self.is_cav.float()
        r_cf = -cfg.w_j * jerk_norm_cav.sum(dim=1) / self.num_cav.float().clamp(min=1)

        # -- r_ss: string stability (CAV-only) --
        alpha_eff = torch.where(self.is_cav, self.alpha, torch.ones_like(self.alpha))
        h_c_eff = alpha_eff * cfg.cth_hc
        s_des = cfg.cth_d0 + h_c_eff * v
        e = gaps - s_des
        # leader error: gather the spacing error of each vehicle's actual leader
        e_leader = e.gather(1, self._leader_idx)
        ss_term = (e.abs() - e_leader.abs()).clamp(min=0) / max(cfg.L / N, 1.0)
        ss_cav = ss_term * self.is_cav.float()
        r_ss = -cfg.w_ss * ss_cav.sum(dim=1) / self.num_cav.float().clamp(min=1)

        # -- r_sigma: fleet speed variance (scaled by eta) --
        fleet_var = v.var(dim=1)
        r_sigma = -eta * cfg.w_sigma * fleet_var / v_ref_scalar.pow(2).clamp(min=1e-3)

        # -- r_sigma_cav: CAV-local speed variance (fully controllable) --
        v_cav = v * self.is_cav.float()  # zero out HDV speeds
        cav_mean_v = v_cav.sum(dim=1) / self.num_cav.float().clamp(min=1)  # (B,)
        cav_dev = (v - cav_mean_v.unsqueeze(1)).pow(2) * self.is_cav.float()
        cav_var = cav_dev.sum(dim=1) / self.num_cav.float().clamp(min=1)
        r_sigma_cav = -cfg.w_sigma_cav * cav_var / v_ref_scalar.pow(2).clamp(min=1e-3)

        # -- r_alpha: residual regularisation (annealable) --
        da_max = max(cfg.rl_delta_alpha_max, 1e-6)
        da_norm = (delta_alphas / da_max).pow(2) * self.is_cav.float()
        r_alpha = (
            -self.current_w_alpha
            * da_norm.sum(dim=1)
            / self.num_cav.float().clamp(min=1)
        )

        # -- r_damp: damping bonus (positive reward for reducing fleet variance) --
        damping = (self.prev_fleet_var - fleet_var).clamp(min=0)
        r_damp = cfg.w_damp * damping / max(cfg.sigma_ref_damp, 1e-6)

        # Update damping state for next step
        self.prev_fleet_var = fleet_var.detach()

        return r_sf + r_collision_floor + r_ef + r_cf + r_ss + r_sigma + r_sigma_cav + r_alpha + r_damp

    # ------------------------------------------------------------------ #
    # Main step
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(
        self, delta_alphas: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Advance all environments by one dt.

        Parameters
        ----------
        delta_alphas : (B, N)
            RL residual for every vehicle slot.  Should be 0 for HDV slots.

        Returns
        -------
        obs       : (B, N, 15)
        reward    : (B,)
        done      : (B,) bool
        mask      : (B, N) bool  — which slots are real CAVs
        info      : dict of (B,) tensors for logging
        """
        cfg = self.cfg
        B, N = self.B, self.N
        t = self.step_count.float() * cfg.dt  # current time per env

        # ===== Perturbation injection (curriculum: per-env timing, magnitude, target) =====
        should_perturb = (
            cfg.perturbation_enabled
            & (~self.perturbation_applied)
            & (t >= self.perturb_time)
        )
        if should_perturb.any():
            dv = torch.zeros(B, N, device=self.device)
            # Scatter each env's perturbation magnitude to its target vehicle
            targets = self.perturb_target.unsqueeze(1)  # (B, 1)
            magnitudes = self.perturb_dv.unsqueeze(1)  # (B, 1)
            dv.scatter_(1, targets, magnitudes)
            self.v = self.v + dv * should_perturb.unsqueeze(1).float()
            self.v = self.v.clamp(min=0.0)
            self.perturbation_applied = self.perturbation_applied | should_perturb

        # ===== Sort and compute gaps =====
        sort_idx, gaps, dv_leader = self._sort_and_gaps()

        # ===== Upstream stats =====
        mu_v, sigma_v_sq = self._upstream_stats(sort_idx)

        # ===== Mesoscopic alpha =====
        if cfg.meso_enabled:
            alpha_meso = self._compute_meso_alpha(mu_v, sigma_v_sq)
        else:
            alpha_meso = torch.ones_like(self.alpha)
            self.meso_rho.zero_()
            self.meso_alpha_prev.fill_(1.0)
            self.meso_sigma_smooth.zero_()

        # Store rule-based alpha
        self.alpha_rule = alpha_meso.clone()

        # ===== RL residual injection (ratchet-free) =====
        alpha_rl = (alpha_meso + delta_alphas).clamp(cfg.rl_alpha_min, cfg.rl_alpha_max)
        # Only apply to CAVs
        self.alpha = torch.where(self.is_cav, alpha_rl, torch.ones_like(alpha_rl))

        # ===== Leader acceleration for feedforward =====
        a_leader = self.a_prev.gather(1, self._leader_idx)

        # ===== Compute accelerations =====
        # IDM for humans (dv_closing = -dv_leader = v_self - v_leader)
        acc_idm = self._idm_acc(gaps, self.v, -dv_leader)
        # CACC for CAVs
        acc_cacc = self._cacc_acc(gaps, self.v, dv_leader, a_leader, self.alpha)

        acc = torch.where(self.is_cav, acc_cacc, acc_idm)

        # Warm-up limiting
        in_warmup = t.unsqueeze(1) < cfg.warmup_duration  # (B, 1)
        acc = torch.where(
            in_warmup, acc.clamp(-cfg.warmup_accel_limit, cfg.warmup_accel_limit), acc
        )

        # Store for jerk reward (before updating a_prev)
        self.a_prev_reward = self.a_prev.clone()

        # ===== Update velocities =====
        # Noise for humans (suppressed during warmup)
        noise_on = (t >= cfg.noise_warmup_time).unsqueeze(1).float()  # (B, 1)
        noise = (
            torch.randn(B, N, device=self.device)
            * math.sqrt(cfg.noise_Q * cfg.dt)
            * noise_on
        )
        noise = noise * (~self.is_cav).float()  # only HDVs get noise

        self.v = (self.v + (acc + noise) * cfg.dt).clamp(min=0.0)
        # HDV speed cap at idm_v0
        hdv_cap = (
            ~self.is_cav
        ).float() * cfg.idm_v0 + self.is_cav.float() * cfg.cacc_v_max
        self.v = self.v.clamp(max=hdv_cap)

        # Store acceleration
        self.a_prev = acc

        # ===== Update positions =====
        self.x = self.x + self.v * cfg.dt

        # Collision prevention
        _, gaps_new, _ = self._sort_and_gaps()
        too_close = gaps_new < self.min_gap
        if too_close.any():
            self.collision_clamp_count += too_close.sum(dim=1).to(torch.long)
            self.v = torch.where(too_close, torch.zeros_like(self.v), self.v)

        # Wraparound
        self.x = self.x % cfg.L

        # ===== Reward =====
        _, gaps_final, dv_final = self._sort_and_gaps()
        reward = self.compute_reward(gaps_final, dv_final, delta_alphas)

        # ===== Build observations =====
        mu_v_new, sigma_v_sq_new = self._upstream_stats(self._sort_idx)
        a_leader_new = self.a_prev.gather(1, self._leader_idx)
        obs, mask = self.build_obs(
            mu_v_new, sigma_v_sq_new, gaps_final, dv_final, a_leader_new
        )

        # ===== Step counter and done =====
        self.step_count += 1
        done = self.step_count >= cfg.episode_steps

        return (
            obs,
            reward,
            done,
            mask,
            {
                "alpha_mean": self.alpha[self.is_cav].mean()
                if self.is_cav.any()
                else torch.tensor(0.0),
                "collision_clamp_count": self.collision_clamp_count.clone(),
            },
        )

    # ------------------------------------------------------------------ #
    # Auto-reset done environments
    # ------------------------------------------------------------------ #
    def auto_reset(self, done: torch.Tensor) -> None:
        """Reset any environments that are done."""
        if done.any():
            self.reset(env_mask=done)
