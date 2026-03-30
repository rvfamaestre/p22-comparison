# -*- coding: utf-8 -*-

# -------------------------------------------------------------
# File: src/environment/ring_road.py
# -------------------------------------------------------------


class RingRoadEnv:
    def __init__(self, L, vehicles):
        """
        Ring road environment container.
        
        CRITICAL: Vehicle order is established at initialization
        by Simulator and NEVER changes during simulation.
        
        Parameters:
        -----------
        L : float
            Ring road length (meters)
        vehicles : list[Vehicle]
            List of vehicle objects
        """
        self.L = L
        self.vehicles = vehicles

    # ---------------------------------------------------------
    # Wraparound only (no sorting!)
    # ---------------------------------------------------------
    def apply_wraparound_all(self):
        """Apply periodic boundary to all vehicles."""
        for v in self.vehicles:
            v.apply_wraparound(self.L)

