from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class VecEnvConfig:
    """Batched torch implementation of the mustykey training environment."""

    N: int
    L: float
    dt: float
    episode_steps: int
    initial_speed: float
    initial_conditions: str
    human_ratio: float
    perturbation_enabled: bool
    perturbation_vehicle: int
    perturbation_time: float
    perturbation_delta_v: float
    noise_warmup_time: float
    noise_Q: float
    warmup_duration: float
    warmup_accel_limit: float
    idm_s0: float
    idm_T: float
    idm_a: float
    idm_b: float
    idm_v0: float
    idm_delta: float
    cacc_ks: float
    cacc_kv: float
    cacc_kf: float
    cacc_kv0: float
    cacc_b_max: float
    cacc_b_comfort: float
    cacc_a_max: float
    cacc_v_max: float
    cacc_v_des: float
    cth_d0: float
    cth_hc: float
    meso_enabled: bool
    meso_M: int
    meso_lambda_rho: float
    meso_gamma: float
    meso_alpha_min: float
    meso_alpha_max: float
    meso_sigma_v_min_threshold: float
    meso_v_eps_sigma: float
    meso_max_alpha_rate: float
    meso_sigma_v_ema_lambda: float
    meso_psi_deadband: float
    rl_delta_alpha_max: float
    rl_alpha_min: float
    rl_alpha_max: float
    reward_s_min: float
    reward_tau_min: float
    reward_j_ref: float
    reward_epsilon_v: float
    reward_epsilon_s: float
    reward_w_s: float
    reward_w_tau: float
    reward_w_e: float
    reward_w_v: float
    reward_w_j: float
    reward_w_ss: float
    reward_w_sigma2: float

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "VecEnvConfig":
        idm = config.get("idm_params", {})
        acc = config.get("acc_params", {})
        cth = config.get("cth_params", {})
        meso = config.get("mesoscopic", {})
        rl = config.get("rl_layer", {})
        reward = config.get("rl_training", {}).get("reward", {})
        return cls(
            N=int(config["N"]),
            L=float(config["L"]),
            dt=float(config["dt"]),
            episode_steps=int(round(float(config["T"]) / float(config["dt"]))),
            initial_speed=float(config.get("initial_speed", 5.5)),
            initial_conditions=str(config.get("initial_conditions", "random")).lower(),
            human_ratio=float(config.get("human_ratio", 0.5)),
            perturbation_enabled=bool(config.get("perturbation_enabled", False)),
            perturbation_vehicle=int(config.get("perturbation_vehicle", 0)),
            perturbation_time=float(config.get("perturbation_time", 3.0)),
            perturbation_delta_v=float(config.get("perturbation_delta_v", -3.0)),
            noise_warmup_time=float(config.get("noise_warmup_time", 0.0)),
            noise_Q=float(config.get("noise_Q", 0.0)),
            warmup_duration=float(config.get("warmup_duration", 10.0)),
            warmup_accel_limit=float(config.get("warmup_accel_limit", 1.0)),
            idm_s0=float(idm.get("s0", 2.0)),
            idm_T=float(idm.get("T", 1.5)),
            idm_a=float(idm.get("a", 0.3)),
            idm_b=float(idm.get("b", 3.0)),
            idm_v0=float(idm.get("v0", 14.0)),
            idm_delta=float(idm.get("delta", 4.0)),
            cacc_ks=float(acc.get("ks", 0.4)),
            cacc_kv=float(acc.get("kv", 1.3)),
            cacc_kf=float(acc.get("kf", 0.95)),
            cacc_kv0=float(acc.get("kv0", 0.05)),
            cacc_b_max=float(acc.get("b_max", 3.0)),
            cacc_b_comfort=float(acc.get("b_comfort", 1.5)),
            cacc_a_max=float(acc.get("a_max", 1.5)),
            cacc_v_max=float(acc.get("v_max", 20.0)),
            cacc_v_des=float(acc.get("v_des", 14.0)),
            cth_d0=float(cth.get("d0", 2.0)),
            cth_hc=float(cth.get("hc", 1.2)),
            meso_enabled=bool(meso.get("enabled", False)),
            meso_M=int(meso.get("M", 8)),
            meso_lambda_rho=float(meso.get("lambda_rho", 0.8)),
            meso_gamma=float(meso.get("gamma", 0.4)),
            meso_alpha_min=float(meso.get("alpha_min", 0.9)),
            meso_alpha_max=float(meso.get("alpha_max", 1.6)),
            meso_sigma_v_min_threshold=float(meso.get("sigma_v_min_threshold", 0.03)),
            meso_v_eps_sigma=float(meso.get("v_eps_sigma", 0.5)),
            meso_max_alpha_rate=float(meso.get("max_alpha_rate", 0.3)),
            meso_sigma_v_ema_lambda=float(meso.get("sigma_v_ema_lambda", 0.5)),
            meso_psi_deadband=float(meso.get("psi_deadband", 0.2)),
            rl_delta_alpha_max=float(rl.get("delta_alpha_max", 0.1)),
            rl_alpha_min=float(rl.get("alpha_min", 0.9)),
            rl_alpha_max=float(rl.get("alpha_max", 1.6)),
            reward_s_min=float(reward.get("s_min", 2.0)),
            reward_tau_min=float(reward.get("tau_min", 0.6)),
            reward_j_ref=float(reward.get("j_ref", 3.0)),
            reward_epsilon_v=float(reward.get("epsilon_v", 1e-3)),
            reward_epsilon_s=float(reward.get("epsilon_s", 1e-3)),
            reward_w_s=float(reward.get("w_s", 2.0)),
            reward_w_tau=float(reward.get("w_tau", 1.0)),
            reward_w_e=float(reward.get("w_e", 1.0)),
            reward_w_v=float(reward.get("w_v", 1.0)),
            reward_w_j=float(reward.get("w_j", 0.2)),
            reward_w_ss=float(reward.get("w_ss", 0.5)),
            reward_w_sigma2=float(reward.get("w_sigma2", 0.1)),
        )


class VecRingRoadEnv:
    """Vectorized training-only environment compatible with mustykey policies."""

    def __init__(self, num_envs: int, cfg: VecEnvConfig, device: torch.device):
        self.B = int(num_envs)
        self.cfg = cfg
        self.N = cfg.N
        self.device = device
        self.veh_length = 5.0
        self.min_gap = 0.3

        shape = (self.B, self.N)
        self.x = torch.zeros(shape, device=device)
        self.v = torch.zeros(shape, device=device)
        self.a_prev = torch.zeros(shape, device=device)
        self.is_cav = torch.zeros(shape, device=device, dtype=torch.bool)
        self.alpha_applied = torch.ones(shape, device=device)

        self.meso_rho = torch.zeros(shape, device=device)
        self.meso_alpha_prev = torch.ones(shape, device=device)
        self.meso_sigma_smooth = torch.zeros(shape, device=device)

        self.step_count = torch.zeros(self.B, device=device, dtype=torch.long)
        self.collision_clamp_count = torch.zeros(self.B, device=device, dtype=torch.long)
        self.perturbation_applied = torch.zeros(self.B, device=device, dtype=torch.bool)
        self.perturb_target = torch.zeros(self.B, device=device, dtype=torch.long)

        self._sort_idx = torch.zeros(shape, device=device, dtype=torch.long)
        self._leader_idx = torch.zeros(shape, device=device, dtype=torch.long)
        self._follower_idx = torch.zeros(shape, device=device, dtype=torch.long)
        self._cache: dict[str, torch.Tensor] | None = None
        self.last_gaps = torch.zeros(shape, device=device)

    def reset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        num_human = int(cfg.N * cfg.human_ratio)
        num_cav = cfg.N - num_human

        self.is_cav.zero_()
        if num_cav > 0:
            self.is_cav[:, cfg.N - num_cav :] = True

        if cfg.initial_conditions == "uniform":
            spacing = cfg.L / cfg.N
            positions = torch.arange(cfg.N, device=self.device, dtype=torch.float32) * spacing
            self.x = positions.unsqueeze(0).expand(self.B, -1).clone()
            self.v.fill_(cfg.initial_speed)
        else:
            self._reset_random_initial_conditions()

        self.a_prev.zero_()
        self.alpha_applied.fill_(1.0)
        self.meso_rho.zero_()
        self.meso_alpha_prev.fill_(1.0)
        self.meso_sigma_smooth.zero_()
        self.step_count.zero_()
        self.collision_clamp_count.zero_()
        self.perturbation_applied.zero_()

        if cfg.perturbation_vehicle == -1:
            self.perturb_target = torch.randint(0, cfg.N, (self.B,), device=self.device)
        else:
            self.perturb_target.fill_(cfg.perturbation_vehicle)

        return self._prepare_observation()

    def _reset_random_initial_conditions(self) -> None:
        cfg = self.cfg
        nominal_spacing = cfg.L / cfg.N
        sigma_spacing = 0.25 * nominal_spacing
        min_spacing = 2.5 * max(cfg.idm_s0, cfg.cth_d0)
        max_spacing = 1.8 * nominal_spacing

        positions = torch.zeros(self.B, cfg.N, device=self.device)
        for env_idx in range(self.B):
            first_spacings = torch.randn(cfg.N - 1, device=self.device) * sigma_spacing + nominal_spacing
            first_spacings = first_spacings.clamp(min=min_spacing, max=max_spacing)
            residual = cfg.L - float(first_spacings.sum().item())
            if residual < min_spacing or residual > max_spacing:
                available = cfg.L - min_spacing
                scale = available / max(float(first_spacings.sum().item()), 1e-6)
                first_spacings = first_spacings * scale
            positions[env_idx, 1:] = torch.cumsum(first_spacings, dim=0)

        velocity_noise = torch.randn(self.B, cfg.N, device=self.device) * 3.0
        self.x = positions
        self.v = (cfg.initial_speed + velocity_noise).clamp(min=0.5, max=cfg.cacc_v_max)

    def _apply_pending_perturbation(self) -> None:
        cfg = self.cfg
        if not cfg.perturbation_enabled:
            return

        current_time = self.step_count.float() * cfg.dt
        should_perturb = (~self.perturbation_applied) & (current_time >= cfg.perturbation_time)
        if not bool(should_perturb.any().item()):
            return

        delta_v = torch.zeros_like(self.v)
        delta_v.scatter_(1, self.perturb_target.unsqueeze(1), cfg.perturbation_delta_v)
        self.v = torch.where(should_perturb.unsqueeze(1), self.v + delta_v, self.v).clamp(min=0.0)
        self.perturbation_applied |= should_perturb

    def _sort_and_gaps(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sort_idx = self.x.argsort(dim=1)
        arange_n = torch.arange(self.N, device=self.device).unsqueeze(0).expand(self.B, -1)
        inv_sort = torch.zeros_like(sort_idx)
        inv_sort.scatter_(1, sort_idx, arange_n)

        leader_sort_pos = (inv_sort + 1) % self.N
        follower_sort_pos = (inv_sort - 1) % self.N
        leader_idx = sort_idx.gather(1, leader_sort_pos)
        follower_idx = sort_idx.gather(1, follower_sort_pos)

        x_leader = self.x.gather(1, leader_idx)
        raw_gap = x_leader - self.x
        gaps = torch.where(raw_gap < 0.0, raw_gap + self.cfg.L, raw_gap) - self.veh_length
        dv_open = self.v.gather(1, leader_idx) - self.v

        self._sort_idx = sort_idx
        self._leader_idx = leader_idx
        self._follower_idx = follower_idx
        self.last_gaps = gaps
        return sort_idx, gaps, dv_open

    def _upstream_stats(self, sort_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg
        sorted_v = self.v.gather(1, sort_idx)
        upstream = torch.zeros(self.B, self.N, cfg.meso_M, device=self.device)
        arange_sorted = torch.arange(self.N, device=self.device).unsqueeze(0).expand(self.B, -1)
        for offset in range(1, cfg.meso_M + 1):
            upstream_pos = (arange_sorted + offset) % self.N
            upstream[:, :, offset - 1] = sorted_v.gather(1, upstream_pos)

        upstream = upstream.clamp(min=cfg.meso_v_eps_sigma)
        mu_v_sorted = upstream.mean(dim=2)
        sigma_v_sq_sorted = upstream.var(dim=2, unbiased=False)

        inv_sort = torch.zeros_like(sort_idx)
        inv_sort.scatter_(
            1,
            sort_idx,
            torch.arange(self.N, device=self.device).unsqueeze(0).expand(self.B, -1),
        )
        mu_v = mu_v_sorted.gather(1, inv_sort)
        sigma_v_sq = sigma_v_sq_sorted.gather(1, inv_sort)
        return mu_v, sigma_v_sq

    def _compute_meso_alpha(self, mu_v: torch.Tensor, sigma_v_sq: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        sigma_v = sigma_v_sq.sqrt()
        lam = cfg.meso_sigma_v_ema_lambda
        self.meso_sigma_smooth = lam * self.meso_sigma_smooth + (1.0 - lam) * sigma_v
        sigma_filtered = torch.where(
            self.meso_sigma_smooth < cfg.meso_sigma_v_min_threshold,
            torch.zeros_like(self.meso_sigma_smooth),
            self.meso_sigma_smooth,
        )

        beta = (2.0 * sigma_filtered / max(cfg.cacc_v_max, 1e-3)).clamp(max=1.0)
        velocity_difference = self.v - mu_v
        psi = torch.where(
            velocity_difference.abs() > cfg.meso_psi_deadband,
            beta * velocity_difference.sign(),
            torch.zeros_like(beta),
        )

        self.meso_rho = cfg.meso_lambda_rho * self.meso_rho + cfg.meso_gamma * psi
        alpha_raw = (1.0 + self.meso_rho).clamp(cfg.meso_alpha_min, cfg.meso_alpha_max)

        max_delta = cfg.meso_max_alpha_rate * cfg.dt
        alpha = self.meso_alpha_prev + (alpha_raw - self.meso_alpha_prev).clamp(-max_delta, max_delta)
        alpha = alpha.clamp(cfg.meso_alpha_min, cfg.meso_alpha_max)
        self.meso_alpha_prev = alpha
        return alpha

    def _prepare_observation(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._apply_pending_perturbation()

        sort_idx, gaps, dv_open = self._sort_and_gaps()
        mu_v, sigma_v_sq = self._upstream_stats(sort_idx)

        if self.cfg.meso_enabled:
            alpha_rule = self._compute_meso_alpha(mu_v, sigma_v_sq)
        else:
            alpha_rule = torch.ones_like(self.v)

        a_leader = self.a_prev.gather(1, self._leader_idx)
        alpha_prev_feature = torch.where(
            self.step_count.unsqueeze(1) == 0,
            alpha_rule,
            self.alpha_applied,
        )

        obs = torch.stack(
            [
                mu_v,
                sigma_v_sq,
                self.v - mu_v,
                self.v,
                gaps,
                dv_open,
                a_leader,
                alpha_prev_feature,
            ],
            dim=2,
        )

        action_low = torch.maximum(
            torch.full_like(alpha_rule, -self.cfg.rl_delta_alpha_max),
            self.cfg.rl_alpha_min - alpha_rule,
        )
        action_high = torch.minimum(
            torch.full_like(alpha_rule, self.cfg.rl_delta_alpha_max),
            self.cfg.rl_alpha_max - alpha_rule,
        )
        action_low = torch.where(self.is_cav, action_low, torch.zeros_like(action_low))
        action_high = torch.where(self.is_cav, action_high, torch.zeros_like(action_high))

        self._cache = {
            "obs": obs,
            "mask": self.is_cav.clone(),
            "action_low": action_low,
            "action_high": action_high,
            "alpha_rule": alpha_rule,
            "mu_v": mu_v,
            "sigma_v_sq": sigma_v_sq,
            "gaps": gaps,
            "dv_open": dv_open,
            "a_leader": a_leader,
        }
        return obs, self.is_cav.clone(), action_low, action_high

    def _idm_acc(self, gaps: torch.Tensor, dv_closing: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        s = gaps.clamp(min=cfg.idm_s0)
        s_star = cfg.idm_s0 + self.v * cfg.idm_T + (self.v * dv_closing) / (
            2.0 * math.sqrt(cfg.idm_a * cfg.idm_b)
        )
        s_star = s_star.clamp(min=cfg.idm_s0)
        acc = cfg.idm_a * (1.0 - (self.v / cfg.idm_v0).pow(cfg.idm_delta) - (s_star / s).pow(2))
        return acc.clamp(min=-cfg.idm_b)

    def _cacc_acc(
        self,
        gaps: torch.Tensor,
        dv_open: torch.Tensor,
        a_leader: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        h_c = alpha * cfg.cth_hc
        k_f = (cfg.cacc_kf * alpha.sqrt()).clamp(max=1.0)
        k_v0 = cfg.cacc_kv0 / alpha.clamp(min=1e-3)

        s_des = cfg.cth_d0 + h_c * self.v
        spacing_ratio = gaps / s_des.clamp(min=0.1)
        acc = (
            cfg.cacc_ks * (gaps - s_des)
            + cfg.cacc_kv * dv_open
            + k_f * a_leader
        )

        blend = torch.zeros_like(spacing_ratio)
        blend = torch.where(spacing_ratio > 2.0, torch.full_like(blend, 0.3), blend)
        blend = torch.where(
            (spacing_ratio > 1.5) & (spacing_ratio <= 2.0),
            torch.full_like(blend, 0.1),
            blend,
        )
        blend = torch.where(
            (spacing_ratio > 1.3) & (spacing_ratio <= 1.5),
            torch.full_like(blend, 0.05),
            blend,
        )
        acc = acc + blend * k_v0 * (cfg.cacc_v_des - self.v)

        gap_deficit = (cfg.cth_d0 - gaps).clamp(min=0.0)
        emergency_brake = -cfg.cacc_b_max * (gap_deficit / max(cfg.cth_d0, 1e-3))
        acc = torch.where(gaps < cfg.cth_d0, torch.minimum(acc, emergency_brake), acc)
        acc = torch.where(
            (spacing_ratio > 2.0) & (acc < -cfg.cacc_b_comfort),
            torch.full_like(acc, -cfg.cacc_b_comfort),
            acc,
        )

        closing_rate = -dv_open
        ttc = gaps / closing_rate.abs().clamp(min=0.1)
        collision_avoidance = -closing_rate.abs() / 2.0
        acc = torch.where(
            (closing_rate > 1.0) & (ttc < 3.0),
            torch.minimum(acc, collision_avoidance),
            acc,
        )
        acc = torch.where(gaps < 0.1, torch.full_like(acc, -cfg.cacc_b_max), acc)
        return acc.clamp(min=-cfg.cacc_b_max, max=cfg.cacc_a_max)

    def _compute_local_reward(self, alpha: torch.Tensor, acc: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        cache = self._cache
        assert cache is not None

        s = cache["gaps"]
        mu_v = cache["mu_v"]
        sigma_v_sq = cache["sigma_v_sq"]

        s_des = cfg.cth_d0 + alpha * self.v
        spacing_error = s - s_des
        tau = s / self.v.clamp(min=cfg.reward_epsilon_v)
        jerk = (acc - self.a_prev) / max(cfg.dt, 1e-6)

        reward = -(
            cfg.reward_w_s * (cfg.reward_s_min - s).clamp(min=0.0).pow(2)
            + cfg.reward_w_tau * (cfg.reward_tau_min - tau).clamp(min=0.0).pow(2)
        )
        reward = reward - cfg.reward_w_e * (
            spacing_error / s_des.clamp(min=cfg.reward_epsilon_s)
        ).pow(2)
        reward = reward - cfg.reward_w_v * (
            (self.v - mu_v) / max(cfg.cacc_v_max, cfg.reward_epsilon_v)
        ).pow(2)
        reward = reward - cfg.reward_w_j * (
            jerk / max(cfg.reward_j_ref, 1e-3)
        ).pow(2)
        reward = reward - cfg.reward_w_ss * spacing_error.abs()
        reward = reward - cfg.reward_w_sigma2 * sigma_v_sq
        return torch.where(self.is_cav, reward, torch.zeros_like(reward))

    @torch.no_grad()
    def step(
        self,
        delta_alpha: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self._cache is None:
            raise RuntimeError("VecRingRoadEnv.step() called before reset().")

        cfg = self.cfg
        cache = self._cache

        if delta_alpha.dim() == 3:
            delta_alpha = delta_alpha.squeeze(-1)

        bounded_delta = torch.maximum(
            torch.minimum(delta_alpha, cache["action_high"]),
            cache["action_low"],
        )
        bounded_delta = torch.where(self.is_cav, bounded_delta, torch.zeros_like(bounded_delta))
        alpha = cache["alpha_rule"] + bounded_delta
        alpha = torch.where(
            self.is_cav,
            alpha.clamp(cfg.rl_alpha_min, cfg.rl_alpha_max),
            torch.ones_like(alpha),
        )

        acc_idm = self._idm_acc(cache["gaps"], -cache["dv_open"])
        acc_cacc = self._cacc_acc(cache["gaps"], cache["dv_open"], cache["a_leader"], alpha)
        acc_det = torch.where(self.is_cav, acc_cacc, acc_idm)

        current_time = self.step_count.float() * cfg.dt
        in_warmup = current_time.unsqueeze(1) < cfg.warmup_duration
        acc_det = torch.where(
            in_warmup,
            acc_det.clamp(min=-cfg.warmup_accel_limit, max=cfg.warmup_accel_limit),
            acc_det,
        )

        reward = self._compute_local_reward(alpha, acc_det)

        noise_on = (current_time >= cfg.noise_warmup_time).unsqueeze(1).float()
        noise = torch.randn_like(self.v) * math.sqrt(cfg.noise_Q * cfg.dt)
        noise = noise * (~self.is_cav).float() * noise_on
        executed_acc = torch.where(self.is_cav, acc_det, acc_det + noise)

        old_v = self.v.clone()
        hdv_v = (old_v + executed_acc * cfg.dt).clamp(min=0.0, max=cfg.idm_v0)

        cav_v_candidate = (old_v + executed_acc * cfg.dt).clamp(min=0.0, max=cfg.cacc_v_max)
        cav_delta_v = (cav_v_candidate - old_v).clamp(min=-2.0 * cfg.dt, max=2.0 * cfg.dt)
        cav_v = (old_v + cav_delta_v).clamp(min=0.0, max=cfg.cacc_v_max)

        self.v = torch.where(self.is_cav, cav_v, hdv_v)
        self.a_prev = executed_acc
        self.alpha_applied = alpha
        self.x = self.x + self.v * cfg.dt

        _, gaps_after, _ = self._sort_and_gaps()
        too_close = gaps_after < self.min_gap
        if bool(too_close.any().item()):
            self.collision_clamp_count += too_close.sum(dim=1).to(torch.long)
            x_leader = self.x.gather(1, self._leader_idx)
            leader_wrapped = torch.where(x_leader < self.x, x_leader + cfg.L, x_leader)
            corrected_x = leader_wrapped - self.veh_length - self.min_gap
            corrected_x = torch.where(corrected_x >= cfg.L, corrected_x - cfg.L, corrected_x)
            self.x = torch.where(too_close, corrected_x, self.x)
            self.v = torch.where(too_close, torch.zeros_like(self.v), self.v)
            self.a_prev = torch.where(too_close, torch.zeros_like(self.a_prev), self.a_prev)

        self.x = torch.remainder(self.x, cfg.L)
        _, gaps_final, _ = self._sort_and_gaps()
        self.last_gaps = gaps_final

        self.step_count += 1
        done = self.step_count >= cfg.episode_steps
        info = {"collision_clamp_count": self.collision_clamp_count.clone()}

        if bool(done.all().item()):
            next_obs = cache["obs"]
            next_mask = cache["mask"]
            next_action_low = cache["action_low"]
            next_action_high = cache["action_high"]
            self._cache = None
        else:
            next_obs, next_mask, next_action_low, next_action_high = self._prepare_observation()

        return (
            next_obs,
            reward,
            done,
            next_mask,
            next_action_low,
            next_action_high,
            info,
        )
