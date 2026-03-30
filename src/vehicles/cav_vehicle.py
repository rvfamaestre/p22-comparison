# -*- coding: utf-8 -*-

# -------------------------------------------------------------
# File: src/vehicles/cav_vehicle.py
# String-Stable ACC/CTH Controller
# Follows Ploeg et al. (2014) "String-Stable CACC Design"
# -------------------------------------------------------------

from .vehicle import Vehicle


class CAVVehicle(Vehicle):
    def __init__(self, vid, x0, v0, acc_params, cth_params):
        super().__init__(vid, x0, v0)

        # CACC Controller gains
        self.kp   = acc_params["ks"]       # spacing gain (position error)
        self.kd   = acc_params["kv"]       # velocity gain (derivative)
        self.kf   = acc_params.get("kf", 0.7)  # feedforward gain (leader accel)
        self.kv0  = acc_params.get("kv0", 0.05)  # free-flow gain (very small for platoon matching)

        # Desired free-flow speed
        self.v_des = acc_params.get("v_des", 15.0)

        # Physical limits
        self.b_max = acc_params["b_max"]   # max braking (e.g., -3.0 m/s²)
        self.b_comfort = acc_params.get("b_comfort", 1.5)  # comfort braking
        self.v_max = acc_params["v_max"]   # speed cap
        self.a_max = acc_params.get("a_max", 1.5)  # acceleration limit

        # CTH parameters
        self.d0 = cth_params["d0"]         # standstill distance
        self.hc = cth_params["hc"]         # time headway
        
        # Platoon matching state (for Option B free-flow)
        self.spacing_above_threshold_time = 0.0  # Track sustained large spacing

    # ---------------------------------------------------------
    # Desired spacing under the CTH policy
    # ---------------------------------------------------------
    def compute_desired_spacing(self, v):
        """
        Constant Time Headway (CTH) spacing policy.
        
        s_des(v) = d0 + hc * v
        
        Parameters:
        -----------
        v : float
            Current velocity of the vehicle (m/s)
            
        Returns:
        --------
        float
            Desired spacing (meters)
        """
        return self.d0 + self.hc * v

    # ---------------------------------------------------------
    # String-Stable CACC Controller (Ploeg 2014, Naus 2010)
    # ---------------------------------------------------------
    def compute_cacc_acc(self, s, v, dv, a_leader):
        """
        Cooperative Adaptive Cruise Control (CACC) with feedforward.
        
        Control Law:
            a = Kp*(s - s_des) + Kd*(v_leader - v_self) + Kf*a_leader + Kv0*(v_des - v)
        
        Components:
        1. Spacing error: Kp*(s - s_des)  [proportional]
        2. Velocity error: Kd*dv  [derivative/damping]
        3. Feedforward: Kf*a_leader  [anticipatory, CRITICAL for string stability]
        4. Free-flow: Kv0*(v_des - v)  [gentle speed regulation when gap is large]
        
        String Stability Conditions (Ploeg 2014):
            - Kd > hc*Kp  (basic stability)
            - Kf ≈ 1.0 for perfect feedforward (0.5-0.8 typical)
            - Kv0 << Kp (free-flow doesn't dominate)
        
        Sign Convention:
            dv = v_leader - v_self  (positive = opening gap)
        
        Parameters:
        -----------
        s : float
            Current spacing to leader (meters)
        v : float
            Current velocity (m/s)
        dv : float
            Velocity difference: v_leader - v_self (m/s)
            Positive = leader faster (gap opening)
            Negative = self faster (gap closing)
        a_leader : float
            Leader's current acceleration (m/s²)
        
        Returns:
        --------
        float
            Commanded acceleration (m/s²), saturated
        """
        
        # ===== CTH Spacing Policy =====
        s_des = self.d0 + self.hc * v
        
        # ===== Handle Small/Negative Gaps =====
        # Don't clamp gap, but handle division carefully
        if s < 0.1:
            # Very close - emergency braking mode
            # Don't use standard control law, just brake hard
            return -self.b_max
        
        # ===== Error Terms =====
        e_s = s - s_des           # Spacing error (positive = too far)
        e_v = dv                  # Velocity error (v_leader - v_self)
        e_vdes = self.v_des - v   # Free-flow error
        
        # ===== Core CACC Control Law =====
        # 1. Proportional (spacing)
        acc_p = self.kp * e_s
        
        # 2. Derivative (velocity matching)
        acc_d = self.kd * e_v
        
        # 3. Feedforward (leader acceleration) - CRITICAL for string stability
        acc_f = self.kf * a_leader
        
        # Combine PD + Feedforward
        acc = acc_p + acc_d + acc_f
        
        # ===== Platoon-Matching Free-Flow (Option B) =====
        # Prevents CAVs from pulling away from human platoon
        # Only activate free-flow if spacing is large AND sustained
        spacing_ratio = s / max(s_des, 0.1)
        
        # Hysteresis to prevent oscillations
        if spacing_ratio > 1.5:
            # Large spacing - consider free-flow activation
            # But only if sustained (prevents transient activation)
            # Note: time tracking would need dt passed in, simplified here
            # Use aggressive threshold: only activate if VERY far
            if spacing_ratio > 2.0:
                blend = 0.3  # Gentle free-flow even when very far
            else:
                blend = 0.1  # Minimal free-flow in transition
        elif spacing_ratio > 1.3:
            # Moderate spacing - very gentle free-flow
            blend = 0.05
        else:
            # Close to leader → no free-flow (gap control dominates)
            # This keeps CAVs matched to platoon speed
            blend = 0.0
        
        acc_ff = blend * self.kv0 * e_vdes
        acc += acc_ff
        
        # ===== Safety Constraints =====
        # Emergency braking if too close
        if s < self.d0:
            gap_deficit = self.d0 - s
            acc_emergency = -self.b_max * (gap_deficit / self.d0)
            acc = min(acc, acc_emergency)
        
        # Prevent hard braking when spacing is large (physical sense check)
        if spacing_ratio > 2.0 and acc < -self.b_comfort:
            acc = max(acc, -self.b_comfort)
        
        # Collision avoidance: if closing too fast, brake proportionally
        if e_v < -1.0:  # Self much faster than leader
            time_to_collision = s / abs(e_v) if abs(e_v) > 0.1 else 999
            if time_to_collision < 3.0:  # Less than 3 seconds
                acc_ca = -abs(e_v) / 2.0
                acc = min(acc, acc_ca)
        
        # ===== Actuator Saturation =====
        acc = max(-self.b_max, min(acc, self.a_max))
        
        return acc

    # ---------------------------------------------------------
    # Velocity update with comprehensive saturation
    # ---------------------------------------------------------
    def update_velocity(self, acc, dt):
        
        # Euler integration
        v_new = self.v + acc * dt
        
        # Enforce strict velocity bounds
        v_new = max(0.0, min(v_new, self.v_max))
        
        # Additional safety: limit velocity change per timestep
        max_dv = 2.0 * dt  # Max 2 m/s² effective change
        if abs(v_new - self.v) > max_dv:
            if v_new > self.v:
                v_new = self.v + max_dv
            else:
                v_new = self.v - max_dv
        
        self.v = v_new
        self.acceleration = acc
        self.a = acc  # Alias for backward compatibility

