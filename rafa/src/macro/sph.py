# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 18:05:00 2025

@author: shnoz
"""

# -------------------------------------------------------------
# File: src/macro/sph.py
# Gaussian kernel for SPH smoothing (compact support).
# -------------------------------------------------------------

import numpy as np


def gaussian_kernel(r, h):
    """
    Gaussian SPH kernel with compact support (3h).
    
    r: distance (can be scalar or numpy array)
    h: smoothing length
    """

    # 1. Ensure positive smoothing length
    h = max(h, 1e-6)

    r = np.asarray(r)

    # 2. Compact support: ignore points beyond 3h
    mask = np.abs(r) <= 3.0 * h

    # Output array
    W = np.zeros_like(r, dtype=float)

    # 3. Evaluate kernel only inside support
    coeff = 1.0 / (np.sqrt(2 * np.pi) * h)
    W[mask] = coeff * np.exp(-(r[mask] ** 2) / (2.0 * h * h))

    return W
