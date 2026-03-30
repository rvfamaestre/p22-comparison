# -*- coding: utf-8 -*-
"""Tests for ObservationBuilder."""

import numpy as np
import pytest

from src.agents.observation_builder import ObservationBuilder
from src.agents.rl_types import OBS_DIM
from src.vehicles.cav_vehicle import CAVVehicle
from src.vehicles.human_vehicle import HumanVehicle


def _make_ring(num_human=3, num_cav=2, L=200.0, spacing=None):
    """Create a minimal set of vehicles on a ring."""
    N = num_human + num_cav
    if spacing is None:
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
        v = HumanVehicle(i, i * spacing, 10.0, idm_p, 0.1)
        vehicles.append(v)
    for j in range(num_cav):
        vid = num_human + j
        v = CAVVehicle(vid, vid * spacing, 10.0, acc_p, cth_p)
        vehicles.append(v)

    # Sort and assign leaders
    vehicles.sort(key=lambda v: v.x)
    for idx, v in enumerate(vehicles):
        v.leader = vehicles[(idx + 1) % N]
        v.L = L
    return vehicles, L


class TestObservationBuilder:
    def test_shape(self):
        vehicles, L = _make_ring(3, 2)
        builder = ObservationBuilder(M=2, normalize=False)
        obs, cav_ids = builder.build(vehicles, L, {})
        assert obs.shape == (2, OBS_DIM)
        assert len(cav_ids) == 2

    def test_no_cavs(self):
        vehicles, L = _make_ring(5, 0)
        builder = ObservationBuilder(M=2, normalize=False)
        obs, cav_ids = builder.build(vehicles, L, {})
        assert obs.shape == (0, OBS_DIM)
        assert cav_ids == []

    def test_alpha_prev_default(self):
        vehicles, L = _make_ring(2, 1)
        builder = ObservationBuilder(M=2, normalize=False)
        obs, cav_ids = builder.build(vehicles, L, {})
        # alpha_prev should default to 1.0 (index 7)
        assert obs[0, 7] == pytest.approx(1.0)

    def test_alpha_prev_custom(self):
        vehicles, L = _make_ring(2, 1)
        cav_id = [v.id for v in vehicles if isinstance(v, CAVVehicle)][0]
        builder = ObservationBuilder(M=2, normalize=False)
        obs, _ = builder.build(vehicles, L, {cav_id: 1.5})
        assert obs[0, 7] == pytest.approx(1.5)

    def test_finite_values(self):
        vehicles, L = _make_ring(3, 3)
        builder = ObservationBuilder(M=3, normalize=False)
        obs, _ = builder.build(vehicles, L, {})
        assert np.all(np.isfinite(obs))

    def test_normalization_produces_finite(self):
        """Normalized output should still be finite."""
        vehicles, L = _make_ring(3, 3)
        builder = ObservationBuilder(M=3, normalize=True)
        # Call twice so normalizer has some stats
        builder.build(vehicles, L, {})
        obs, _ = builder.build(vehicles, L, {})
        assert np.all(np.isfinite(obs))

    def test_global_features(self):
        """Global context features (cav_share, mean_speed, speed_var, mean_alpha)."""
        vehicles, L = _make_ring(3, 2)
        builder = ObservationBuilder(M=2, normalize=False)
        obs, cav_ids = builder.build(vehicles, L, {})
        # cav_share = 2/5 = 0.4
        assert obs[0, 8] == pytest.approx(0.4, abs=1e-5)
        # mean_speed should be > 0 (all vehicles at 10.0 m/s)
        assert obs[0, 9] > 0
        # speed_var >= 0
        assert obs[0, 10] >= 0
        # mean_alpha defaults to 1.0 when no alpha_prev given
        assert obs[0, 11] == pytest.approx(1.0)
