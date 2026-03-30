# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 18:04:05 2025

@author: shnoz
"""

# -------------------------------------------------------------
# File: src/utils/random_utils.py
# Random utility functions.
# -------------------------------------------------------------

import numpy as np

# Create a module-level RNG
rng = np.random.default_rng()


def set_random_seed(seed):
    """Allow deterministic simulation runs."""
    global rng
    rng = np.random.default_rng(seed)


def random_normal():
    """Return a sample from N(0,1)."""
    return rng.normal(0.0, 1.0)
