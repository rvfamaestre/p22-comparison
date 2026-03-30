# -*- coding: utf-8 -*-
"""
Human-Inspired Mesoscopic Adaptation Layer for CAV Control

Based on:
- Iovine et al. (2015): Variance-Driven Time Headway (VDT)
- Mirabilio et al. (2023): Mesoscopic Human-Inspired ACC with macroscopic filtering

Core concept: CAVs adapt their time headway and controller gains based on 
macroscopic speed statistics (mean/variance) of vehicles AHEAD, mimicking human 
psycho-physical responses to approaching traffic conditions.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np


@dataclass
class MesoConfig:
    """Configuration for mesoscopic adaptation layer."""
    M: int = 8                          # Number of vehicles AHEAD to sample (forward-looking window)
    lambda_rho: float = 0.8             # Filter coefficient for stress indicator
    gamma: float = 0.1                  # Scaling factor for psycho-physical response (REDUCED: gentler adaptation)
    alpha_min: float = 0.7              # Minimum headway scaling (aggressive)
    alpha_max: float = 2.0              # Maximum headway scaling (cautious)
    enable_gain_scheduling: bool = True # Enable gain adaptation
    enable_danger_mode: bool = False    # Enable emergency mode logic
    rho_crit: float = 0.6              # Critical stress threshold for danger mode
    sigma_v_min_threshold: float = 0.2  # Minimum σ_v threshold (m/s) - below this, treat as zero variance
    v_eps_sigma: float = 0.5            # Epsilon for velocity protection when computing stats (m/s)
    max_alpha_rate: float = 0.2         # Maximum α rate of change (1/s) - gives 0.02 per step (gentler)
    sigma_v_ema_lambda: float = 0.9     # EMA smoothing for σ_v (0.9 = very smooth)
    enable_k_f_adaptation: bool = False # Enable adaptive feedforward (disable for testing)
    psi_deadband: float = 0.5           # Deadband for velocity_difference (m/s) - reduce to make more responsive
    adaptation_mode: str = "highway"    # "highway" (increase h with variance) or "ringroad" (decrease h with variance)
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.M > 0, "M must be positive"
        assert 0 <= abs(self.lambda_rho) < 1, "lambda_rho must be in (-1, 1)"
        assert self.gamma >= 0, "gamma must be non-negative"
        assert 0 < self.alpha_min <= 1, "alpha_min must be in (0, 1]"
        assert self.alpha_max >= 1, "alpha_max must be >= 1"
        assert self.alpha_min <= self.alpha_max, "alpha_min must be <= alpha_max"
        assert self.sigma_v_min_threshold >= 0, "sigma_v_min_threshold must be non-negative"
        assert self.v_eps_sigma > 0, "v_eps_sigma must be positive"
        assert self.max_alpha_rate > 0, "max_alpha_rate must be positive"
        assert 0 <= self.sigma_v_ema_lambda < 1, "sigma_v_ema_lambda must be in [0, 1)"
        assert self.psi_deadband >= 0, "psi_deadband must be non-negative"
        assert self.adaptation_mode in ["highway", "ringroad", "variance", "spacing_error"], \
            "adaptation_mode must be 'highway', 'ringroad', 'variance', or 'spacing_error'"


@dataclass
class CavGains:
    """Baseline CAV controller gains."""
    k_s: float      # Spacing gain
    k_v: float      # Velocity gain
    k_v0: float     # Free-flow gain
    h_c: float      # Time headway
    
    def validate_string_stability(self):
        """
        Verify string stability condition: k_v > h_c * k_s
        
        This is the continuous-time stability criterion from Ploeg et al. (2014).
        """
        margin = self.k_v - self.h_c * self.k_s
        return margin > 0, margin


class MesoAdapter:
    
    def __init__(self, cfg: MesoConfig, v_max: float):
        """
        Initialize mesoscopic adapter.
        
        Parameters:
        -----------
        cfg : MesoConfig
            Configuration parameters
        v_max : float
            Maximum velocity for normalization (m/s)
        """
        cfg.validate()
        self.cfg = cfg
        self.v_max = v_max
        
        # Per-CAV filter state: rho_i (psycho-physical stress indicator)
        self.rho: Dict[int, float] = {}
        
        # Per-CAV previous alpha for rate limiting
        self.alpha_prev: Dict[int, float] = {}
        
        # Per-CAV smoothed sigma_v for EMA filtering
        self.sigma_v_smooth: Dict[int, float] = {}
        
        # Tracking for diagnostics
        self.beta_history: Dict[int, List[float]] = {}
        self.psi_history: Dict[int, List[float]] = {}
    
    def reset(self):
        """Clear all internal state (for new simulation runs)."""
        self.rho.clear()
        self.alpha_prev.clear()
        self.sigma_v_smooth.clear()
        self.beta_history.clear()
        self.psi_history.clear()
    
    def compute_alpha(self, 
                     cav_id: int, 
                     cav_velocity: float,
                     upstream_velocities: List[float]) -> Tuple[float, Dict[str, float]]:
      
        M_actual = len(upstream_velocities)
        
        if M_actual == 0:
            # No data from vehicles ahead - use neutral scaling
            return 1.0, {'mu_v': 0.0, 'sigma_v': 0.0, 'beta': 0.0, 'psi': 0.0, 'rho': 0.0, 'M_actual': 0}
        
        # Step 1 & 2: Compute mean and standard deviation
       
        upstream_velocities_safe = [max(v, self.cfg.v_eps_sigma) for v in upstream_velocities]
        mu_v = np.mean(upstream_velocities_safe)
        sigma_v_raw = np.std(upstream_velocities_safe)
        
        # EMA smoothing on sigma_v (prevents reacting to instantaneous noise)
        sigma_v_prev = self.sigma_v_smooth.get(cav_id, sigma_v_raw)
        sigma_v = self.cfg.sigma_v_ema_lambda * sigma_v_prev + (1 - self.cfg.sigma_v_ema_lambda) * sigma_v_raw
        self.sigma_v_smooth[cav_id] = sigma_v
        
        # Apply minimum threshold: small variance is treated as zero (homogeneous flow)
        if sigma_v < self.cfg.sigma_v_min_threshold:
            sigma_v = 0.0
        
        # Step 3: Normalized volatility (Mirabilio Eq. 10)
       
        beta = 2.0 * sigma_v / max(self.v_max, 0.1)
        beta = min(beta, 1.0)  # Clamp to [0, 1]
        
        # Step 4: Directional psycho-physical indicator (Mirabilio et al. 2023, Eq. perceptionsignalEq)
        
        velocity_difference = cav_velocity - mu_v  # CORRECTED to match paper (was: mu_v - cav_velocity)
        epsilon = self.cfg.psi_deadband  # Use configurable deadband (was hard-coded at 0.5)
        
        if abs(velocity_difference) > epsilon:
            psi = beta * np.sign(velocity_difference)
        else:
            psi = 0.0  # Neutral zone
        
        # Step 5: Filter to get psycho-physical stress (Mirabilio Eq. 12)
        # First-order low-pass filter with memory
        rho_prev = self.rho.get(cav_id, 0.0)
        rho = self.cfg.lambda_rho * rho_prev + self.cfg.gamma * psi
        self.rho[cav_id] = rho
        
        # Step 6: Map stress to headway scaling (Iovine VDT concept)
                
        if self.cfg.adaptation_mode == "spacing_error":
            # Spacing-error-driven adaptation (experimental)
            # Requires additional state tracking - not fully implemented
            # For now, fall back to variance mode
            alpha_raw = 1.0 + rho
        else:
            # Variance-driven mode ("variance", "highway", or "ringroad" for backward compatibility)
            # High variance → Larger headway → Wave damping
            # THIS IS CORRECT FOR BOTH HIGHWAY AND RING-ROAD!
            alpha_raw = 1.0 + rho
        
        alpha_raw = np.clip(alpha_raw, self.cfg.alpha_min, self.cfg.alpha_max)
        
         # Limit maximum rate of change to prevent rapid oscillations
        if cav_id in self.alpha_prev:
            alpha_prev = self.alpha_prev[cav_id]
            # Assume dt = 0.1 (typical timestep)
            dt_assumed = 0.1
            max_delta = self.cfg.max_alpha_rate * dt_assumed
            delta_alpha = alpha_raw - alpha_prev
            
            if abs(delta_alpha) > max_delta:
                # Clamp the change
                alpha = alpha_prev + np.sign(delta_alpha) * max_delta
            else:
                alpha = alpha_raw
        else:
            # First timestep - no limiting
            alpha = alpha_raw
        
        # Store for next timestep
        self.alpha_prev[cav_id] = alpha
        
        # Diagnostic data for logging
        diagnostics = {
            'mu_v': mu_v,
            'sigma_v': sigma_v,
            'beta': beta,
            'psi': psi,
            'rho': rho,
            'alpha': alpha,
            'alpha_raw': alpha_raw,  # Before rate limiting
            'M_actual': M_actual
        }
        
        return alpha, diagnostics
    
    def adapt_cav_policy(self, 
                        cav_id: int,
                        gains: CavGains,
                        alpha: float,
                        danger_override: bool = False,
                        k_f_baseline: float = 0.95) -> Tuple[float, float, float, float, float, Dict[str, float]]:
          
        # Danger mode override (Iovine et al. 2015 hybrid automaton concept)
        if danger_override and self.cfg.enable_danger_mode:
            # Force maximum caution
            alpha = self.cfg.alpha_max
            k_v0_prime = 0.0  # Disable free-flow in emergency
            h_c_prime = alpha * gains.h_c
            k_s_prime = gains.k_s  # Keep constant for stability
            k_v_prime = gains.k_v  # Keep constant for stability
            k_f_prime = k_f_baseline * np.sqrt(alpha)  # Adaptive feedforward
            
            # Ensure string stability
            margin = k_v_prime - h_c_prime * k_s_prime
            if margin <= 0:
                print(f"DANGER MODE: CAV {cav_id} stability violation, reducing alpha")
                alpha = gains.k_v / (gains.h_c * gains.k_s * 1.2)
                h_c_prime = alpha * gains.h_c
                k_f_prime = k_f_baseline * np.sqrt(alpha)
                margin = k_v_prime - h_c_prime * k_s_prime
        
        else:
            # Standard mesoscopic adaptation
            
            # 1. Headway scaling (REQUIRED)
            h_c_prime = alpha * gains.h_c
            
            # 2.  GAIN SCHEDULING FOR STRING STABILITY
            # Keep k_s and k_v CONSTANT to preserve stability margin
            # Only adapt feedforward k_f to match new dynamics
            k_s_prime = gains.k_s
            k_v_prime = gains.k_v
            
            # 3. Adaptive Feedforward 
            k_f_prime = k_f_baseline * np.sqrt(alpha)
            k_f_prime = min(k_f_prime, 1.0)  # Never exceed perfect feedforward
            
            # 4. Free-flow inverse scaling
            # Higher alpha(cautious) → reduce free-flow influence
            k_v0_prime = gains.k_v0 / alpha
            
            # String stability enforcement
            # With constant k_s and k_v, margin is ALWAYS preserved
            margin = k_v_prime - h_c_prime * k_s_prime
            # Margin = k_v - alpha*h_c*k_s (decreases with alpha, but slowly)
            
            if margin <= 0:
                # Should NEVER happen with this design, but safety check
                print(f"CRITICAL WARNING: CAV {cav_id} stability violation! α={alpha:.2f}, margin={margin:.3f}")
                # Emergency: reduce alpha to restore margin
                alpha_safe = k_v_prime / (gains.h_c * k_s_prime * 1.2)
                h_c_prime = alpha_safe * gains.h_c
                k_f_prime = k_f_baseline * np.sqrt(alpha_safe)
                margin = k_v_prime - h_c_prime * k_s_prime
        
        # Final validation
        is_stable, final_margin = CavGains(k_s_prime, k_v_prime, k_v0_prime, h_c_prime).validate_string_stability()
        
        diagnostics = {
            'alpha_applied': alpha,
            'k_f_adapted': k_f_prime,
            'string_stable': is_stable,
            'stability_margin': final_margin,
            'gain_scheduling_active': False,  # Now always False (gains fixed)
            'danger_mode': danger_override
        }
        
        return k_s_prime, k_v_prime, k_v0_prime, h_c_prime, k_f_prime, diagnostics
    
    def check_danger_condition(self, 
                              spacing: float,
                              velocity: float,
                              closing_rate: float,
                              s_emergency_threshold: float) -> bool:
        """
        Check if danger mode should activate (optional feature).
        
        Danger conditions (Iovine et al. 2015):
        1. Spacing below emergency threshold
        2. High closing rate (TTC < critical value)
        3. Psycho-physical stress exceeds threshold
        
        Parameters:
        -----------
        spacing : float
            Current spacing to leader (m)
        velocity : float
            CAV velocity (m/s)
        closing_rate : float
            Relative velocity: v_self - v_leader (m/s)
        s_emergency_threshold : float
            Critical spacing threshold (m)
        
        Returns:
        --------
        bool
            True if danger mode should activate
        """
        if not self.cfg.enable_danger_mode:
            return False
        
        # Condition 1: Very close spacing
        if spacing < s_emergency_threshold:
            return True
        
        # Condition 2: Time-to-collision check
        if closing_rate > 0.5:  # Approaching leader
            ttc = spacing / closing_rate if closing_rate > 0 else 999
            if ttc < 2.0:  # Less than 2 seconds
                return True
        
        # Condition 3: High psycho-physical stress (not directly accessible here)
        # This would need to be checked by caller using self.rho
        
        return False


def get_M_leaders_ring(vehicles: List, cav_index: int, M: int, L: float) -> List[float]:

    N = len(vehicles)
    upstream_velocities = []
    
    for offset in range(1, M + 1):
        upstream_idx = (cav_index + offset) % N  # LOOK AHEAD (forward in position)
        upstream_velocities.append(vehicles[upstream_idx].v)
    
    return upstream_velocities


# Convenience function for disabling mesoscopic layer
def create_passthrough_adapter(v_max: float) -> MesoAdapter:
    """Create adapter with alpha = 1 (no adaptation)."""
    cfg = MesoConfig(
        M=1,
        lambda_rho=0.0,
        gamma=0.0,
        alpha_min=1.0,
        alpha_max=1.0,
        enable_gain_scheduling=False,
        enable_danger_mode=False
    )
    return MesoAdapter(cfg, v_max)
