# -*- coding: utf-8 -*-
"""Tests for RewardFunction."""

import numpy as np
import pytest

from src.agents.reward import RewardFunction
from src.agents.rl_types import RLConfig
from src.vehicles.cav_vehicle import CAVVehicle
from src.vehicles.human_vehicle import HumanVehicle


def _make_ring(num_human=3, num_cav=2, L=200.0, v=10.0):
    """Build a minimal ring-road for testing."""
    N = num_human + num_cav
    spacing = L / N
    vehicles = []
    idm_p = {"s0": 2.0, "T": 1.2, "a": 1.0, "b": 1.5, "v0": 15.0, "delta": 4}
    acc_p = {
        "ks": 0.4,
        "kv": 1.3,
        "kf": 0.7,
        "kv0": 0.05,
        "b_max": 3.0,
        "v_max": 20.0,
        "v_des": 15.0,
    }
    cth_p = {"d0": 5.0, "hc": 1.0}

    for i in range(num_human):
        vh = HumanVehicle(i, i * spacing, v, idm_p, 0.1)
        vehicles.append(vh)
    for j in range(num_cav):
        vid = num_human + j
        vh = CAVVehicle(vid, vid * spacing, v, acc_p, cth_p)
        vehicles.append(vh)

    vehicles.sort(key=lambda vh: vh.x)
    for idx, vh in enumerate(vehicles):
        vh.leader = vehicles[(idx + 1) % N]
        vh.L = L
    return vehicles, L


class TestRewardFunction:
    def test_keys(self):
        cfg = RLConfig()
        rf = RewardFunction(cfg)
        vehicles, L = _make_ring()
        out = rf.compute(vehicles, L, 0.1, {})
        expected_keys = {
            "total",
            "safety",
            "efficiency",
            "collision_floor",
            "comfort",
            "string_stability",
            "speed_var",
            "speed_var_cav",
            "alpha_reg",
            "damping",
        }
        assert set(out.keys()) == expected_keys

    def test_all_finite(self):
        cfg = RLConfig()
        rf = RewardFunction(cfg)
        vehicles, L = _make_ring()
        out = rf.compute(vehicles, L, 0.1, {})
        for k, v in out.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"

    def test_safety_penalty_direction(self):
        """Smaller gaps should produce more negative safety reward."""
        cfg = RLConfig(s_min=10.0, fleet_penalty_scaling=False)  # raise threshold to trigger
        rf = RewardFunction(cfg)
        vehicles, L = _make_ring(3, 2, L=50.0)  # tight ring
        out = rf.compute(vehicles, L, 0.1, {})
        assert out["safety"] < 0.0, "Expected negative safety penalty on tight ring"

    def test_alpha_reg_zero_residual(self):
        """Zero residual should produce zero regularisation penalty."""
        cfg = RLConfig()
        rf = RewardFunction(cfg)
        vehicles, L = _make_ring()
        out = rf.compute(vehicles, L, 0.1, {})
        assert out["alpha_reg"] == pytest.approx(0.0)

    def test_alpha_reg_nonzero(self):
        """Non-zero residual should produce negative regularisation penalty."""
        cfg = RLConfig()
        rf = RewardFunction(cfg)
        vehicles, L = _make_ring(2, 2)
        cav_ids = [v.id for v in vehicles if isinstance(v, CAVVehicle)]
        deltas = {cid: 0.2 for cid in cav_ids}
        out = rf.compute(vehicles, L, 0.1, deltas)
        assert out["alpha_reg"] < 0.0

    def test_efficiency_one_sided(self):
        """Efficiency penalty is zero when all vehicles are AT or ABOVE v_ref."""
        v_ref = 5.5
        cfg = RLConfig(v_ref=v_ref, adaptive_v_ref=False, fleet_penalty_scaling=False)
        rf = RewardFunction(cfg)
        # Vehicles at exactly v_ref
        vehicles, L = _make_ring(3, 2, v=v_ref)
        out = rf.compute(vehicles, L, 0.1, {})
        assert out["efficiency"] == pytest.approx(0.0, abs=1e-6), (
            "Should be zero when all vehicles are at v_ref (one-sided penalty)"
        )
        # Vehicles ABOVE v_ref: still zero (one-sided)
        vehicles_fast, L = _make_ring(3, 2, v=v_ref + 2.0)
        rf2 = RewardFunction(cfg)
        out2 = rf2.compute(vehicles_fast, L, 0.1, {})
        assert out2["efficiency"] == pytest.approx(0.0, abs=1e-6), (
            "One-sided: should not penalise speeds above v_ref"
        )

    def test_efficiency_below_vref_is_negative(self):
        """Efficiency penalty is negative when all vehicles are below v_ref."""
        v_ref = 10.0
        cfg = RLConfig(v_ref=v_ref, adaptive_v_ref=False, fleet_penalty_scaling=False)
        rf = RewardFunction(cfg)
        vehicles, L = _make_ring(3, 2, v=4.0)  # well below v_ref
        out = rf.compute(vehicles, L, 0.1, {})
        assert out["efficiency"] < 0.0

    def test_damping_bonus_positive_on_variance_reduction(self):
        """Damping bonus should be positive when fleet variance goes down."""
        cfg = RLConfig(w_damp=1.0, sigma_ref_damp=1.0,
                       fleet_penalty_scaling=False, adaptive_v_ref=False)
        rf = RewardFunction(cfg)
        vehicles, L = _make_ring(3, 2)
        # First call: initialize prev_fleet_var
        rf.compute(vehicles, L, 0.1, {})
        # Artificially set prev_fleet_var high so next call shows a reduction
        rf._prev_fleet_var = 100.0
        out = rf.compute(vehicles, L, 0.1, {})
        assert out["damping"] > 0.0, "Damping bonus should be positive when variance decreased"

    def test_w_alpha_annealing(self):
        """current_w_alpha=0 should produce zero alpha penalty even with large residuals."""
        cfg = RLConfig()
        rf = RewardFunction(cfg)
        vehicles, L = _make_ring(2, 2)
        cav_ids = [v.id for v in vehicles if isinstance(v, CAVVehicle)]
        deltas = {cid: 0.4 for cid in cav_ids}
        out = rf.compute(vehicles, L, 0.1, deltas, current_w_alpha=0.0)
        assert out["alpha_reg"] == pytest.approx(0.0, abs=1e-9)
