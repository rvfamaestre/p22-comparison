# -*- coding: utf-8 -*-

# -------------------------------------------------------------
# File: src/vehicles/human_vehicle.py
# Stochastic IDM (Intelligent Driver Model) for human drivers.
# Implements Treiber & Helbing (2000) with Euler-Maruyama noise.
# -------------------------------------------------------------

import numpy as np
from .vehicle import Vehicle
from ..utils.random_utils import random_normal


class HumanVehicle(Vehicle):
    def __init__(self, vid, x0, v0, idm_params, noise_Q):
        super().__init__(vid, x0, v0)
        self.idm = idm_params
        self.Q = noise_Q

    # ---------------------------------------------------------
    # IDM acceleration
    # ---------------------------------------------------------
    def compute_idm_acc(self, s, v, dv):
        """
        Intelligent Driver Model (IDM) acceleration.
        
        Mathematical Formulation (Treiber & Helbing 2000):
        a = a_max * [1 - (v/v0)^δ - (s*/s)^2]
        
        where desired dynamic gap:
        s*(v, Δv) = s0 + v*T + (v*Δv)/(2*√(a*b))
        
        Parameters:
        -----------
        s : float
            Current spacing to leader (meters)
        v : float
            Current velocity (m/s)
        dv : float
            Closing rate: v_self - v_leader (m/s)
            Positive = approaching, Negative = separating
        """

        # IDM parameters
        s0 = self.idm["s0"]  
        T  = self.idm["T"]
        a  = self.idm["a"]
        b  = self.idm["b"]
        v0 = self.idm["v0"]
        delta = self.idm["delta"]

        # 1. enforce minimum physical spacing
        s = max(s, s0)

        # 2. closing rate already correct from compute_closing_rate()
        # dv = v_self - v_leader (no flip needed)

        # 3. desired dynamic spacing
        s_star = s0 + v * T + (v * dv) / (2 * np.sqrt(a * b))
        s_star = max(s0, s_star)

        # 4. IDM core formula
        acc = a * (1 - (v / v0)**delta - (s_star / s)**2)

        # clamp braking
        acc = max(acc, -b)
        return acc

    # ---------------------------------------------------------
    # Noise for human drivers (with warmup suppression)
    # ---------------------------------------------------------
    def sample_noise(self, dt, current_time=None, warmup_time=0.0):
        """
        Sample stochastic noise for human driver.
        
        If current_time < warmup_time, returns 0 (no noise).
        This allows perfect uniformity in early simulation phase.
        """
        if current_time is not None and current_time < warmup_time:
            return 0.0  # Suppress noise during warmup
        return np.sqrt(self.Q * dt) * random_normal()

    # ---------------------------------------------------------
    # Unified velocity update using base class
    # ---------------------------------------------------------
    def update_velocity(self, acc_det, dt, current_time=None, warmup_time=0.0):
        noise = self.sample_noise(dt, current_time, warmup_time) if self.leader is not None else 0.0
        a = acc_det + noise

        # use unified Vehicle base update
        super().base_update_velocity(a, dt)

        # clamp max speed separately
        self.v = min(self.v, self.idm["v0"])
