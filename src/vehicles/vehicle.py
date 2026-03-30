# -*- coding: utf-8 -*-
# -------------------------------------------------------------
# File: src/vehicles/vehicle.py
# -------------------------------------------------------------

import numpy as np

class Vehicle:
    def __init__(self, vid, x0, v0, length=5.0):
        """
        Base class for all vehicles on the ring road.
        
        DYNAMIC TOPOLOGY: Leader reassigned every timestep based on position.
        
        Parameters:
        -----------
        vid : int
            Unique vehicle identifier
        x0 : float
            Initial position on ring (meters)
        v0 : float
            Initial velocity (m/s)
        length : float, optional
            Vehicle length (meters). Default: 5.0
        """
        self.id = vid
        self.x = float(x0)
        self.v = float(v0)
        self.length = length
        self.leader = None
        self.acceleration = 0.0  # Current acceleration (for CACC feedforward and logging)
        self.a = 0.0   # Alias for backward compatibility

    def set_leader(self, leader_vehicle):
        """Assign leader (called ONCE at initialization)."""
        self.leader = leader_vehicle

    # ---------------------------------------------------------
    # Gap Computation - NET BUMPER-TO-BUMPER DISTANCE
    # ---------------------------------------------------------
    def compute_gap(self, L):
        """
        Compute net gap (bumper-to-bumper) to leader.
        
        Convention A (Treiber, Ploeg, Bando standard):
            s = x_leader,rear - x_follower,front
              = (x_leader - x_follower) - L_follower
        
        This is the standard used by:
        - Treiber IDM
        - Ploeg CACC  
        - Bando OV model
        - All ring-road stability papers
        
        Makes s_des = d0 + h*v where d0 is bumper-to-bumper minimum.
        
        Parameters:
        -----------
        L : float
            Ring road length (meters)
            
        Returns:
        --------
        float
            Net spacing to leader (meters)
        """
        leader_x = self.leader.x
        
        # Handle wraparound
        if leader_x < self.x:
            leader_x += L
        
        # Convention A: subtract follower's length
        gap = leader_x - self.x - self.length
        
        return gap

    # ---------------------------------------------------------
    # Closing rate for IDM convention (Treiber & Helbing 2000)
    # ---------------------------------------------------------
    def compute_closing_rate(self):
        """
        Closing rate for IDM: v_self - v_leader
        
        Convention:
        - Positive: Self faster than leader → approaching (need more gap)
        - Negative: Leader faster than self → separating (can accept smaller gap)
        
        This matches Treiber & Helbing (2000) IDM formulation.
        """
        return self.v - self.leader.v

    # ---------------------------------------------------------
    # Common velocity update for all vehicles
    # ---------------------------------------------------------
    def base_update_velocity(self, a, dt):
        """Euler integration for velocity."""
        self.v += a * dt
        self.v = max(0.0, self.v)  # No negative speeds
        self.acceleration = a
        self.a = a  # Alias for backward compatibility

    # ---------------------------------------------------------
    # Position update with PROPER collision prevention
    # ---------------------------------------------------------
    def update_position(self, dt):
        """
        Euler integration for position with HARD collision prevention.
        
        Properly handles ring-road wraparound.
        Prevents overtaking by enforcing minimum gap constraint.
        
        Returns:
        --------
        collision_clamp_triggered : bool
            True if collision prevention forced emergency braking (v=0)
        """
        if self.leader is None or self.L is None:
            # No leader assigned yet or ring length not set
            self.x += self.v * dt
            return False
        
        # Compute new position
        x_new = self.x + self.v * dt
        
        # ===== COLLISION PREVENTION (with wraparound handling) =====
        leader_x = self.leader.x
        
        # Handle wraparound for gap computation
        if leader_x < x_new:
            leader_x_wrapped = leader_x + self.L
        else:
            leader_x_wrapped = leader_x
        
        # Compute gap with new position
        gap = leader_x_wrapped - x_new - self.length
        
        # Enforce minimum gap (0.3m safety margin)
        MIN_GAP = 0.3
        collision_clamp_triggered = False
        
        if gap < MIN_GAP:
            # Position would violate minimum gap - clamp it
            x_new = leader_x_wrapped - self.length - MIN_GAP
            
            # Emergency brake (IMPULSE DISTURBANCE - invalidates string stability analysis!)
            self.v = 0.0
            collision_clamp_triggered = True
            
            # Handle reverse wraparound (if clamped position > L)
            if x_new >= self.L:
                x_new -= self.L
        
        self.x = x_new
        
        return collision_clamp_triggered

    # ---------------------------------------------------------
    # Wrap-around on ring road
    # ---------------------------------------------------------
    def apply_wraparound(self, L):
        """Periodic boundary condition."""
        self.x = self.x % L
        self.L = L  # Store for collision prevention

