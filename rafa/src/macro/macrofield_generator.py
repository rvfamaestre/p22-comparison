# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 18:05:38 2025

@author: shnoz
"""

# -------------------------------------------------------------
# File: src/macro/macrofield_generator.py
# SPH-based macrofield computation for density and velocity.
# -------------------------------------------------------------

import numpy as np
from .sph import gaussian_kernel


def ring_distance(xj, x_veh, L):
    """
    Compute minimal signed distance on a ring.

    Works for scalars or numpy arrays.
    r = (xj - x_i) wrapped into [-L/2, L/2].
    """
    r = xj - x_veh
    r = (r + 0.5 * L) % L - 0.5 * L
    return r


class MacrofieldGenerator:
    def __init__(self, L, dx, h):
        """
        L: ring length
        dx: grid spacing
        h: kernel smoothing length
        """
        self.L = L
        self.x_grid = np.arange(0, L, dx)
        self.h = h

    def compute_density(self, vehicles):
        """Compute density field ρ(x) using SPH."""
        rho = np.zeros_like(self.x_grid, dtype=float)
        veh_x = np.array([v.x for v in vehicles])

        for j, xj in enumerate(self.x_grid):
            r = ring_distance(xj, veh_x, self.L)
            rho[j] = gaussian_kernel(r, self.h).sum()

        return rho

    def compute_velocity(self, vehicles, rho):
        """Compute velocity field u(x) using SPH."""
        u = np.zeros_like(rho)
        veh_x = np.array([v.x for v in vehicles])
        veh_v = np.array([v.v for v in vehicles])

        for j, xj in enumerate(self.x_grid):
            r = ring_distance(xj, veh_x, self.L)
            w = gaussian_kernel(r, self.h)
            num = (veh_v * w).sum()

            rho_safe = max(rho[j], 1e-6)
            u[j] = num / rho_safe

        return u

    def compute_macrofields(self, vehicles):
        rho = self.compute_density(vehicles)
        u = self.compute_velocity(vehicles, rho)
        return rho, u
