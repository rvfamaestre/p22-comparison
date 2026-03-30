# -*- coding: utf-8 -*-
"""
Reward function for the RL agent (CPU path).

Implements the global reward from formulation.tex:
    r_t = r_sf + r_collision_floor + r_ef + r_cf + r_ss + r_sigma + r_sigma_cav + r_alpha + r_damp

All terms are made **dimensionless** before weighting so that the
weight vector directly controls relative importance regardless of
physical units.  This implementation is kept in exact feature-parity
with the GPU vectorized environment (src/gpu/vec_env.py).

Feature history
---------------
run-5  : one-sided efficiency penalty (penalise only v < v_ref)
run-6  : w_sigma_cav (CAV-local variance), fleet_penalty_scaling
run-8  : r_damp (damping bonus), adaptive_v_ref
run-10 : current defaults (see RLConfig)

Normalisation conventions
-------------------------
- Gap safety:      (max(0, 1 - s/s_min))^2              in [0, 1]
- Time-gap:        (max(0, 1 - tau/tau_min))^2           in [0, 1]
- Efficiency:      (max(0, (v_ref-v)/v_ref))^2           in [0, 1]  [one-sided]
- Comfort:         (jerk / j_ref)^2                      in [0, ~4]
- String stab:     max(0, |e_i| - |e_l|) / s_ref         in [0, ~2]
- Fleet variance:  Var(v) / v_ref^2                      in [0, ~1]
- CAV variance:    Var_cav(v) / v_ref^2                  in [0, ~1]
- Alpha reg:       (delta_alpha / delta_alpha_max)^2      in [0, 1]
- Damping bonus:   max(0, sigma^2_{t-1} - sigma^2_t) / sigma_ref_damp
"""

from typing import Dict, List

import numpy as np

from src.agents.rl_types import RLConfig
from src.vehicles.cav_vehicle import CAVVehicle


class RewardFunction:
    """Computes the global scalar reward at each timestep.

    Parameters
    ----------
    cfg : RLConfig
        Full RL configuration including all weight/threshold fields.
    """

    def __init__(self, cfg: RLConfig):
        self.cfg = cfg
        # Store previous CAV accelerations for jerk computation
        self._prev_acc: Dict[int, float] = {}
        # Damping bonus: previous fleet speed variance
        self._prev_fleet_var: float = 0.0

    def reset(self):
        """Call at the start of each episode."""
        self._prev_acc.clear()
        self._prev_fleet_var = 0.0

    # ------------------------------------------------------------------
    def compute(
        self,
        vehicles: list,
        L: float,
        dt: float,
        delta_alphas: Dict[int, float],
        *,
        current_w_alpha: float = None,
    ) -> Dict[str, float]:
        """Compute reward components and the total scalar reward.

        Parameters
        ----------
        vehicles : list[Vehicle]
            All vehicles (sorted by position, leaders already assigned).
        L : float
            Ring circumference.
        dt : float
            Simulation timestep.
        delta_alphas : dict[int, float]
            RL residual actions keyed by CAV id.  Empty when RL is off.
        current_w_alpha : float, optional
            Annealed regularisation weight.  Falls back to cfg.w_alpha
            when not provided (compatible with non-annealing callers).

        Returns
        -------
        dict with keys "total", "safety", "efficiency", "comfort",
             "string_stability", "speed_var", "speed_var_cav",
             "alpha_reg", "damping".
        """
        c = self.cfg
        N = len(vehicles)
        if N == 0:
            return {k: 0.0 for k in ("total", "safety", "collision_floor",
                                      "efficiency", "comfort",
                                      "string_stability", "speed_var", "speed_var_cav",
                                      "alpha_reg", "damping")}

        cavs: List[CAVVehicle] = [v for v in vehicles if isinstance(v, CAVVehicle)]
        num_cav = max(len(cavs), 1)               # avoid /0
        cav_frac = len(cavs) / max(N, 1)          # in [0, 1]
        # η scales fleet-wide terms by CAV fraction so uncontrollable HDV
        # behaviour does not swamp the learning signal at high human rates.
        eta = cav_frac if c.fleet_penalty_scaling else 1.0

        # Effective regularisation weight (supports w_alpha annealing)
        w_alpha_eff = current_w_alpha if current_w_alpha is not None else c.w_alpha

        # ---- Adaptive v_ref ----
        if c.adaptive_v_ref:
            v_ref = c.v_ref + c.v_ref_delta * cav_frac
        else:
            v_ref = c.v_ref
        v_ref = max(v_ref, 1e-3)

        # ---- Gather per-vehicle quantities ----
        speeds = np.array([v.v for v in vehicles])
        gaps = np.array([v.compute_gap(L) for v in vehicles])
        time_gaps = gaps / np.clip(speeds, c.eps_v, None)

        # ---- r_sf (safety) — fleet-wide, NOT scaled by eta ----
        # Safety penalties always apply at full weight regardless of CAV fraction
        # so the agent cannot exploit reduced penalties at high human ratios.
        gap_viol = np.maximum(0.0, 1.0 - gaps / max(c.s_min, 1e-3)) ** 2
        tau_viol = np.maximum(0.0, 1.0 - time_gaps / max(c.tau_min, 1e-3)) ** 2
        r_sf = -(c.w_s * gap_viol.mean() + c.w_tau * tau_viol.mean())

        # ---- r_collision_floor: exponential penalty below critical gap ----
        # Makes near-collision states non-negotiable (unbounded as gap → 0).
        s_crit = max(c.collision_gap_critical, 1e-3)
        crit_viol = np.maximum(0.0, (s_crit - gaps) / s_crit)
        r_collision_floor = -c.w_collision_floor * float((np.exp(crit_viol) - 1.0).mean())

        # ---- r_ef (efficiency) — one-sided, scaled by eta ----
        # Only penalise speeds BELOW v_ref; do not punish high throughput.
        speed_deficit = np.maximum(0.0, (v_ref - speeds) / v_ref) ** 2
        r_ef = -eta * c.w_v * float(speed_deficit.mean())

        # ---- r_cf (comfort / jerk) — CAVs only ----
        jerk_sum = 0.0
        for cav in cavs:
            a_prev = self._prev_acc.get(cav.id, cav.acceleration)
            jerk = (cav.acceleration - a_prev) / max(dt, 1e-6)
            jerk_sum += (jerk / max(c.j_ref, 1e-3)) ** 2
        r_cf = -c.w_j * jerk_sum / num_cav

        # ---- r_ss (string stability) — CAVs only ----
        s_ref = max(L / N, 1.0)
        ss_sum = 0.0
        for cav in cavs:
            s_des = cav.d0 + getattr(cav, "_meso_h_c", cav.hc) * cav.v
            e_i = cav.compute_gap(L) - s_des
            leader = cav.leader
            if isinstance(leader, CAVVehicle):
                s_des_l = leader.d0 + getattr(leader, "_meso_h_c", leader.hc) * leader.v
                e_l = leader.compute_gap(L) - s_des_l
            else:
                e_l = 0.0  # HDV leader: assume reference error = 0
            ss_sum += max(0.0, abs(e_i) - abs(e_l)) / s_ref
        r_ss = -c.w_ss * ss_sum / num_cav

        # ---- r_sigma (fleet speed variance) — scaled by eta ----
        fleet_var = float(np.var(speeds))
        r_sigma = -eta * c.w_sigma * fleet_var / max(v_ref ** 2, 1e-3)

        # ---- r_sigma_cav (CAV-local speed variance) ----
        if len(cavs) > 1:
            cav_speeds = np.array([v.v for v in cavs])
            cav_var = float(np.var(cav_speeds))
        else:
            cav_var = 0.0
        r_sigma_cav = -c.w_sigma_cav * cav_var / max(v_ref ** 2, 1e-3)

        # ---- r_alpha (residual regularisation) ----
        da_max = max(c.delta_alpha_max, 1e-6)
        alpha_sq_sum = sum(
            (delta_alphas.get(cav.id, 0.0) / da_max) ** 2 for cav in cavs
        )
        r_alpha = -w_alpha_eff * alpha_sq_sum / num_cav

        # ---- r_damp (damping bonus: positive reward for reducing fleet var) ----
        damping = max(0.0, self._prev_fleet_var - fleet_var)
        r_damp = c.w_damp * damping / max(c.sigma_ref_damp, 1e-6)

        # ---- Update state for next step ----
        self._prev_fleet_var = fleet_var
        for cav in cavs:
            self._prev_acc[cav.id] = cav.acceleration

        total = r_sf + r_collision_floor + r_ef + r_cf + r_ss + r_sigma + r_sigma_cav + r_alpha + r_damp
        return {
            "total": float(total),
            "safety": float(r_sf),
            "collision_floor": float(r_collision_floor),
            "efficiency": float(r_ef),
            "comfort": float(r_cf),
            "string_stability": float(r_ss),
            "speed_var": float(r_sigma),
            "speed_var_cav": float(r_sigma_cav),
            "alpha_reg": float(r_alpha),
            "damping": float(r_damp),
        }
