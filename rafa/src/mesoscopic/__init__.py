"""
Mesoscopic adaptation layer for human-inspired CAV control.

Based on:
- Iovine et al. (2015): Safe Human-Inspired Mesoscopic Hybrid Automaton
- Mirabilio et al. (2023): Mesoscopic Human-Inspired Adaptive Cruise Control for Eco-Driving
"""

from .meso_adapter import (
    MesoConfig,
    CavGains,
    MesoAdapter,
    get_M_leaders_ring,
    create_passthrough_adapter
)

__all__ = [
    'MesoConfig',
    'CavGains',
    'MesoAdapter',
    'get_M_leaders_ring',
    'create_passthrough_adapter'
]
