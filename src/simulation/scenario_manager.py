# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 09:19:26 2025

@author: shnoz
"""

# -------------------------------------------------------------
# File: src/simulation/scenario_manager.py
# Creates vehicle populations and builds the simulator.
# -------------------------------------------------------------

import numpy as np
from src.vehicles.human_vehicle import HumanVehicle
from src.vehicles.stochastic_human_vehicle import StochasticHumanVehicle
from src.vehicles.unstable_human_vehicle import UnstableHumanVehicle
from src.vehicles.cav_vehicle import CAVVehicle
from src.environment.ring_road import RingRoadEnv
from src.macro.macrofield_generator import MacrofieldGenerator
from src.utils.logger import Logger
from src.utils.random_utils import set_random_seed


class ScenarioManager:
    def __init__(self, config):
        self.config = config

        # Optional reproducibility
        if "seed" in config:
            set_random_seed(config["seed"])
            np.random.seed(config["seed"])

    # ---------------------------------------------------------
    # CREATE VEHICLES WITH NON-UNIFORM INITIAL CONDITIONS
    # ---------------------------------------------------------
    def create_vehicles(self):
        """
        Non-uniform initialization following Treiber/Stern/Ploeg standards.
        
        Key principles:
        1. Generate N-1 random spacings, compute Nth from periodic constraint
        2. Velocity perturbations for ALL vehicles (humans + CAVs)
        3. Maximum spacing bounded at ~2x nominal (prevents extreme gaps)
        4. CAVs start away from v_des to demonstrate regulation capability
        
        This approach:
        - Avoids equilibrium bias (controllers produce rich transient dynamics)
        - Preserves N-1 degrees of freedom (maximum randomness under constraint)
        - Satisfies sum(spacings) = L exactly (ring closure)
        - Prevents collisions (min_spacing enforcement)
        """
        N = self.config["N"]
        human_ratio = self.config["human_ratio"]
        num_human = int(N * human_ratio)
        num_cav = N - num_human
        L = self.config["L"]
        
        nominal_spacing = L / N
        
        # Safety check: warn if initial conditions too tight
        d0 = self.config["cth_params"]["d0"]
        vehicle_length = 5.0  # Standard vehicle length
        min_safe_spacing = 2.0 * d0  # 2x minimum bumper distance
        
        if nominal_spacing < (min_safe_spacing + vehicle_length):
            print(f"\n⚠ WARNING: Tight initial conditions!")
            print(f"  Current spacing: {nominal_spacing:.1f}m (L={L}m / N={N})")
            print(f"  Minimum safe: {min_safe_spacing + vehicle_length:.1f}m")
            print(f"  Recommendation: N <= {int(L / (min_safe_spacing + vehicle_length))} vehicles")
            print(f"                  OR L >= {int(N * (min_safe_spacing + vehicle_length))}m")
            print(f"  Expect collision warnings in first few seconds.\n")
        
        # =========================================================================
        # STEP 1: Generate N-1 random spacings + 1 residual (ring closure)
        # =========================================================================
        
        # 25% spacing variability (Treiber standard)
        sigma_spacing = 0.25 * nominal_spacing
        
        # =========================================================================
        # INITIALIZATION MODE: Uniform vs Random (CRITICAL EXPERIMENTAL DESIGN)
        # =========================================================================
        init_mode = self.config.get("initial_conditions", "random")
        
        if init_mode == "uniform":
            # =====================================================================
            # UNIFORM MODE: Identical spacing and velocity (Sugiyama protocol)
            # =====================================================================
            uniform_spacing = L / N
            spacings = [uniform_spacing] * N
            
            # Verify periodic constraint
            assert abs(sum(spacings) - L) < 1e-6, \
                f"Spacing sum error: {sum(spacings)} != {L}"
            
            # Positions from uniform spacing
            x_positions = np.array([i * uniform_spacing for i in range(N)])
            
            # UNIFORM: All vehicles at same initial speed
            v_init = self.config["initial_speed"]
            v_perturb = np.full(N, v_init, dtype=float)
            
            print(f"[ScenarioManager] UNIFORM initialization (Sugiyama protocol):")
            print(f"  - Spacing: {uniform_spacing:.2f}m (identical for all {N} vehicles)")
            print(f"  - Velocity: {v_init:.2f}m/s (identical)")
            print(f"  - Waves will emerge from IDM instability + stochastic noise")
            if self.config.get("perturbation_enabled", False):
                print(f"  - Perturbation: Vehicle {self.config.get('perturbation_vehicle', 0)} at t={self.config.get('perturbation_time', 10.0):.1f}s")
        
        else:
            # =====================================================================
            # RANDOM MODE: Stochastic initial conditions (old default)
            # =====================================================================
            # Conservative bounds - use 2.5x minimum to prevent early collisions
            min_spacing = 2.5 * max(
                self.config["idm_params"]["s0"],
                self.config["cth_params"]["d0"]
            )
            max_spacing = 1.8 * nominal_spacing  # Prevents extreme gaps
            
            spacings = []
            remaining_length = L
            
            # Generate N-1 independent random spacings
            for i in range(N - 1):
                s = np.random.normal(nominal_spacing, sigma_spacing)
                s = np.clip(s, min_spacing, max_spacing)
                spacings.append(s)
                remaining_length -= s
            
            # Compute last spacing as residual (enforces sum = L)
            last_spacing = remaining_length
            
            # Safety: if residual out of bounds, redistribute proportionally
            if last_spacing < min_spacing or last_spacing > max_spacing:
                available = L - min_spacing
                scale = available / sum(spacings)
                spacings = [s * scale for s in spacings]
                last_spacing = L - sum(spacings)
            
            spacings.append(last_spacing)
            
            # Verify periodic constraint
            assert abs(sum(spacings) - L) < 1e-6, \
                f"Spacing sum error: {sum(spacings)} != {L}"
            
            # Positions from random spacings
            x_positions = np.zeros(N)
            x_positions[0] = 0.0  # First vehicle at origin
            for i in range(1, N):
                x_positions[i] = x_positions[i-1] + spacings[i-1]
            
            # Random velocities
            v_nominal = self.config["initial_speed"]
            sigma_v = 3.0  # m/s (20% variability)
            v_perturb = np.random.normal(v_nominal, sigma_v, N)
            
            # Clamp to physical bounds
            v_max = self.config["acc_params"]["v_max"]
            v_perturb = np.clip(v_perturb, 0.5, v_max)
            
            print(f"[ScenarioManager] RANDOM initialization:")
            print(f"  - Spacing: {nominal_spacing:.2f}m ± {sigma_spacing:.2f}m")
            print(f"  - Velocity: {v_nominal:.2f}m/s ± {sigma_v:.2f}m/s")
        
        # =========================================================================
        # STEP 4: Create vehicle objects with perturbed initial conditions
        # =========================================================================
        
        vehicles = []
        idx = 0
        
        # ---------------------------------------------------------------------
        # DUAL HUMAN MODEL: Check if new parameters exist
        # ---------------------------------------------------------------------
        use_dual_model = ("stochastic_idm_params" in self.config and 
                         "unstable_idm_params" in self.config and
                         "unstable_human_ratio" in self.config)
        
        # Create humans
        if use_dual_model:
            # NEW: Dual human-driver model for stop-and-go wave generation
            
            unstable_ratio = self.config["unstable_human_ratio"]
            num_unstable = int(num_human * unstable_ratio)
            num_stochastic = num_human - num_unstable
            
            print(f"[ScenarioManager] Dual Human Model Active:")
            print(f"  - {num_stochastic} Stochastic IDM (stable + noise)")
            print(f"  - {num_unstable} Unstable IDM (wave generators)")
            print(f"  - {num_cav} CAVs (ACC+CTH)")
            
            # Create mixed vehicle type array and shuffle
            vehicle_types = (['stochastic'] * num_stochastic + 
                           ['unstable'] * num_unstable)
            np.random.shuffle(vehicle_types)
            
            # Instantiate humans based on shuffled types
            for vtype in vehicle_types:
                if vtype == 'stochastic':
                    vehicles.append(StochasticHumanVehicle(
                        vid=idx,
                        x0=float(x_positions[idx]),
                        v0=float(v_perturb[idx]),
                        idm_params=self.config["stochastic_idm_params"],
                        noise_Q=self.config["noise_Q_stochastic"]
                    ))
                else:  # 'unstable'
                    vehicles.append(UnstableHumanVehicle(
                        vid=idx,
                        x0=float(x_positions[idx]),
                        v0=float(v_perturb[idx]),
                        idm_params=self.config["unstable_idm_params"],
                        noise_Q=self.config["noise_Q_unstable"]
                    ))
                idx += 1
        else:
            # LEGACY: Single human model (backward compatibility)
            print(f"[ScenarioManager] Single Human Model (Legacy)")
            for _ in range(num_human):
                vehicles.append(HumanVehicle(
                    vid=idx,
                    x0=float(x_positions[idx]),
                    v0=float(v_perturb[idx]),
                    idm_params=self.config["idm_params"],
                    noise_Q=self.config["noise_Q"]
                ))
                idx += 1
        
        # Create CAVs
        for _ in range(num_cav):
            vehicles.append(CAVVehicle(
                vid=idx,
                x0=float(x_positions[idx]),
                v0=float(v_perturb[idx]),  # Perturbed velocity (NOT v_des)
                acc_params=self.config["acc_params"],
                cth_params=self.config["cth_params"]
            ))
            idx += 1
        
        return vehicles

    # ---------------------------------------------------------
    # BUILD SIMULATOR WITH FIXED TOPOLOGY
    # ---------------------------------------------------------
    def build(self, live_viz=None):
        """
        Creates environment, macro generator, logger, and simulator.
        
        Args:
            live_viz: Optional LiveVisualizer instance for real-time display
            
        Returns:
            Simulator configured with all components
        """
        vehicles = self.create_vehicles()

        # Environment
        env = RingRoadEnv(self.config["L"], vehicles)

        # Macrofield generator
        macro_gen = MacrofieldGenerator(
            L=self.config["L"],
            dx=self.config["dx"],
            h=self.config["kernel_h"]
        )

        # --------------------------------------------
        # BUILD METADATA FOR LOGGER
        # --------------------------------------------
        metadata = {
            "N": self.config["N"],
            "L": self.config["L"],
            "dt": self.config["dt"],
            "T": self.config["T"],
            "scenario": self.config.get("scenario", "default"),
            "seed": self.config.get("seed", None),
            "initial_conditions": self.config.get("initial_conditions", "random"),
            "human_ratio": self.config["human_ratio"],
            "idm_params": self.config.get("idm_params", {}),
            "stochastic_idm_params": self.config.get("stochastic_idm_params", {}),
            "unstable_idm_params": self.config.get("unstable_idm_params", {}),
            "noise_Q": self.config.get("noise_Q", None),
            "noise_Q_stochastic": self.config.get("noise_Q_stochastic", None),
            "noise_Q_unstable": self.config.get("noise_Q_unstable", None),
            "acc_params": self.config.get("acc_params", {}),
            "cth_params": self.config.get("cth_params", {}),
            "mesoscopic": self.config.get("mesoscopic", {}),
            "rl_layer": self.config.get("rl_layer", {}),
            "perturbation_enabled": self.config.get("perturbation_enabled", False),
            "perturbation_vehicle": self.config.get("perturbation_vehicle", 0),
            "perturbation_time": self.config.get("perturbation_time", None),
            "perturbation_delta_v": self.config.get("perturbation_delta_v", None),
            "macro_teacher": self.config.get("macro_teacher", "none"),
            "save_macro_dataset": self.config.get("save_macro_dataset", False),
            "macro_predictor": self.config.get("macro_predictor", "pde"),
            "arz_params": self.config.get("arz_params", {})
        }

        # --------------------------------------------
        # CREATE LOGGER WITH METADATA
        # --------------------------------------------
        logger = Logger(
            output_dir=self.config["output_path"],
            metadata=metadata
        )

        # Simulator
        from .simulator import Simulator
        sim = Simulator(
            env=env,
            macro_gen=macro_gen,
            logger=logger,
            dt=self.config["dt"],
            T=self.config["T"],
            config=self.config,  # Pass full config for ARZ solver
            live_viz=live_viz    # Pass live visualizer
        )

        return sim

