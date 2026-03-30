# -*- coding: utf-8 -*-
"""
Zero-residual regression test.

Verifies that when RL outputs zero residual everywhere, the simulator
behaviour matches the pure rule-based mesoscopic baseline.
"""

import numpy as np
import pytest
import yaml
from copy import deepcopy

from src.simulation.scenario_manager import ScenarioManager
from src.vehicles.cav_vehicle import CAVVehicle
from src.agents.observation_builder import ObservationBuilder
from src.agents.action_adapter import ActionAdapter
from src.agents.rl_types import RLConfig


# Minimal config for deterministic comparison
_BASE_CFG = {
    "N": 10,
    "L": 200.0,
    "dt": 0.1,
    "T": 5.0,  # 50 steps
    "initial_speed": 10.0,
    "seed": 123,
    "output_path": "output/_test_zero_residual",
    "initial_conditions": "uniform",
    "perturbation_enabled": False,
    "noise_warmup_time": 0.0,
    "enable_live_viz": False,
    "play_recording": False,
    "human_ratio": 0.5,
    "idm_params": {"s0": 2.0, "T": 1.2, "a": 1.0, "b": 1.5, "v0": 15.0, "delta": 4},
    "noise_Q": 0.0,  # deterministic human model
    "acc_params": {
        "ks": 0.4,
        "kv": 1.3,
        "kf": 0.7,
        "kv0": 0.05,
        "b_max": 3.0,
        "b_comfort": 1.5,
        "a_max": 1.5,
        "v_max": 20.0,
        "v_des": 15.0,
    },
    "cth_params": {"d0": 5.0, "hc": 1.0},
    "mesoscopic": {
        "enabled": True,
        "M": 4,
        "lambda_rho": 0.8,
        "gamma": 0.4,
        "alpha_min": 0.7,
        "alpha_max": 2.0,
        "enable_gain_scheduling": False,
        "enable_danger_mode": False,
        "sigma_v_min_threshold": 0.2,
        "v_eps_sigma": 0.5,
        "max_alpha_rate": 0.2,
        "sigma_v_ema_lambda": 0.9,
        "enable_k_f_adaptation": False,
        "psi_deadband": 0.5,
        "adaptation_mode": "highway",
    },
    "macro_teacher": "none",
    "save_macro_dataset": False,
    "dx": 1.0,
    "kernel_h": 3.0,
}


def _run_sim(cfg):
    """Run a short simulation and return final vehicle states."""
    mgr = ScenarioManager(cfg)
    sim = mgr.build(live_viz=None)
    sim.run()
    return {v.id: (v.x, v.v) for v in sim.env.vehicles}


class TestZeroResidualRegression:
    def test_zero_residual_matches_baseline(self):
        """Zero RL residual reproduces the rule-based mesoscopic baseline."""
        # --- run 1: rule-only (no RL) ---
        cfg_rule = deepcopy(_BASE_CFG)
        cfg_rule["rl"] = {"rl_mode": "off"}
        states_rule = _run_sim(cfg_rule)

        # --- run 2: RL residual mode with zero residual ---
        cfg_rl = deepcopy(_BASE_CFG)
        cfg_rl["rl"] = {
            "rl_mode": "residual",
            "delta_alpha_max": 0.3,
            "alpha_min": 0.7,
            "alpha_max": 2.0,
        }
        mgr = ScenarioManager(cfg_rl)
        sim = mgr.build(live_viz=None)

        # Inject zero residual every step
        for _ in range(sim.steps):
            cav_ids = [v.id for v in sim.env.vehicles if isinstance(v, CAVVehicle)]
            sim.set_rl_actions({cid: 0.0 for cid in cav_ids})
            sim.step()
        sim.logger.save()

        states_rl = {v.id: (v.x, v.v) for v in sim.env.vehicles}

        # Compare
        for vid in states_rule:
            x_rule, v_rule = states_rule[vid]
            x_rl, v_rl = states_rl[vid]
            assert x_rl == pytest.approx(x_rule, abs=1e-6), (
                f"Vehicle {vid} position mismatch: rule={x_rule:.6f}, rl_zero={x_rl:.6f}"
            )
            assert v_rl == pytest.approx(v_rule, abs=1e-6), (
                f"Vehicle {vid} velocity mismatch: rule={v_rule:.6f}, rl_zero={v_rl:.6f}"
            )

    def test_action_clipping(self):
        """RL actions are clipped so alpha stays in [alpha_min, alpha_max]."""
        rl_cfg = RLConfig(delta_alpha_max=0.3, alpha_min=0.7, alpha_max=2.0)
        adapter = ActionAdapter(rl_cfg)

        # Extreme residual that would push alpha out of bounds
        result = adapter.apply(
            {0: 5.0, 1: -5.0},
            {0: 1.0, 1: 1.0},
        )
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(0.7)
