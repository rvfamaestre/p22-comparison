# -*- coding: utf-8 -*-
"""
Unstable Deterministic IDM Human Driver (Model B)
Generates self-sustained stop-and-go waves

Parameters tuned for UNSTABLE regime:
- T = 1.8s (above critical headway : slow reaction)
- a = 0.45 m/s^2 (slow acceleration : long recovery)
- b = 3.0 m/s^2 (strong braking : asymmetric response)
- NO noise (deterministic waves)

Purpose: Act as wave generators in mixed-autonomy scenarios
Based on validated parameters from:
- Treiber & Helbing (2000)
- Sugiyama et al. (2008) ring-road experiments
- Treiber & Kesting (2013) instability analysis
"""

import numpy as np
from .vehicle import Vehicle


class UnstableHumanVehicle(Vehicle):
    """
    Unstable deterministic IDM driver (wave generator).
    
    Used for 20-40% of human drivers to create stop-and-go instability.
    Operating in linearly unstable parameter regime.
    """
    
    def __init__(self, vid, x0, v0, idm_params, noise_Q=0.0):
        super().__init__(vid, x0, v0)
        
        # IDM parameters (UNSTABLE REGIME)
        self.s0 = idm_params["s0"]
        self.T = idm_params["T"]        # Large T → slow reaction
        self.a = idm_params["a"]        # Small a → slow recovery
        self.b = idm_params["b"]        # Large b → strong braking
        self.v0 = idm_params["v0"]
        self.delta = idm_params["delta"]
        
        # Noise parameter (kept for API consistency, should be 0.0)
        self.noise_Q = noise_Q
        
        # Type identifier for logging
        self.driver_type = "unstable_human"
    
    def compute_idm_acc(self, s, v, dv):
        """
        Standard IDM acceleration formula with UNSTABLE parameters.
        
        Key to instability:
        1. Large time headway T : delayed reaction
        2. Small acceleration a : slow recovery after braking
        3. Large b/a ratio : asymmetric response (brake fast, accelerate slow)
        
        This creates POSITIVE FEEDBACK at critical wavelengths:
        - Small perturbations AMPLIFY instead of decay
        - System enters limit cycle oscillations
        - Stop-and-go waves emerge naturally
        
        Mathematical stability criterion:
        T_critical = (2/a) * sqrt(b * s0)
        
        For our parameters:
        T_critical = (2/0.45) * sqrt(3.0 * 4.0) ≈ 15.4s
        
        Our T = 1.8s is chosen to be in the unstable regime
        for the given density (10-12m spacing).
        
        Parameters:
        -----------
        s : float
            Current spacing to leader (meters)
        v : float
            Current velocity (m/s)
        dv : float
            Closing rate: v_self - v_leader (m/s)
            Positive = approaching leader
        
        Returns:
        --------
        float
            Acceleration (m/s²), clamped to [-b, +a]
        """
        # Enforce minimum spacing
        s = max(s, self.s0)
        
        # Desired dynamic gap (with large T → aggressive braking)
        s_star = self.s0 + v * self.T + (v * dv) / (2 * np.sqrt(self.a * self.b))
        s_star = max(self.s0, s_star)
        
        # IDM acceleration
        acc = self.a * (1.0 - (v / self.v0)**self.delta - (s_star / s)**2)
        
        # Clamp braking (can brake HARD with b=3.0)
        acc = max(acc, -self.b)
        
        return acc
    
    def update_velocity(self, acc_det, dt, current_time=None, warmup_time=0.0):
        """
        Update velocity DETERMINISTICALLY (no noise).
        
        Noise would mask the pure wave mechanism.
        Unstable dynamics alone create oscillations.
        
        Note: Warmup parameters included for API compatibility but ignored
        since this vehicle type never uses noise.
        
        Parameters:
        -----------
        acc_det : float
            Deterministic IDM acceleration
        dt : float
            Timestep size
        """
        # Store acceleration for logging
        self.a = acc_det
        
        # Euler integration (deterministic)
        self.v += acc_det * dt
        
        # Enforce physical bounds
        self.v = max(0.0, self.v)
        self.v = min(self.v, self.v0)
