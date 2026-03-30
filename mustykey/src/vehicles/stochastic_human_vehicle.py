# -*- coding: utf-8 -*-

import numpy as np
from .vehicle import Vehicle


class StochasticHumanVehicle(Vehicle):
    """
    Stable stochastic IDM driver.
    
    Used for 60-80% of human drivers to add realism without instability.
    Based on standard IDM with moderate parameters and stochastic noise.
    """
    
    def __init__(self, vid, x0, v0, idm_params, noise_Q):
        super().__init__(vid, x0, v0)
        
        # IDM parameters
        self.s0 = idm_params["s0"]
        self.T = idm_params["T"]
        self.a = idm_params["a"]
        self.b = idm_params["b"]
        self.v0 = idm_params["v0"]
        self.delta = idm_params["delta"]
        
        # Stochastic noise intensity
        self.Q = noise_Q
        
        # Type identifier for logging
        self.driver_type = "stochastic_human"
    
    def compute_idm_acc(self, s, v, dv):
        """
        Standard IDM acceleration formula (Treiber & Helbing 2000).
        
        a = a_max * [1 - (v/v0)^delta - (s*/s)^2]
        
        where s* = s0 + v*T + (v*dv)/(2*sqrt(a*b))
        
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
        # Enforce minimum spacing (prevent singularity)
        s = max(s, self.s0)
        
        # Desired dynamic gap
        s_star = self.s0 + v * self.T + (v * dv) / (2 * np.sqrt(self.a * self.b))
        s_star = max(self.s0, s_star)
        
        # IDM acceleration
        acc = self.a * (1.0 - (v / self.v0)**self.delta - (s_star / s)**2)
        
        # Clamp braking to comfortable deceleration
        acc = max(acc, -self.b)
        
        return acc
    
    def sample_noise(self, dt):
        """
        Sample acceleration noise using Euler-Maruyama method.
        
        Noise term: alpha * sqrt(dt) * N(0,1)
        where alpha = sqrt(Q)
        
        Returns:
        --------
        float
            Stochastic acceleration perturbation (m/s^2)
        """
        if self.leader is None:
            return 0.0
        
        sigma = np.sqrt(self.Q * dt)
        return np.random.normal(0.0, sigma)
    
    def update_velocity(self, acc_det, dt, current_time=None, warmup_time=0.0):
        
        # Add stochastic noise
        noise = self.sample_noise(dt)
        a_total = acc_det + noise
        
        # Store acceleration for logging
        self.a = a_total
        
        # Euler integration
        self.v += a_total * dt
        
        # Enforce physical bounds
        self.v = max(0.0, self.v)
        self.v = min(self.v, self.v0)
