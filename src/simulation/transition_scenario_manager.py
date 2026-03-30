# -*- coding: utf-8 -*-
"""
Transition Scenario Manager
Builds the mixed-autonomy transition scenario with 100% human start.
"""

import numpy as np
from src.vehicles.human_vehicle import HumanVehicle
from src.vehicles.stochastic_human_vehicle import StochasticHumanVehicle
from src.vehicles.cav_vehicle import CAVVehicle
from src.environment.ring_road_transition import RingRoadTransitionEnv
from src.macro.macrofield_generator import MacrofieldGenerator
from src.utils.logger import Logger
from src.utils.random_utils import set_random_seed
from src.simulation.transition_simulator import TransitionSimulator


class TransitionScenarioManager:
    """
    Creates initial 100% human population and builds transition simulator.
    """
    
    def __init__(self, config):
        self.config = config
        
        # Set random seed if provided
        if "seed" in config:
            set_random_seed(config["seed"])
            np.random.seed(config["seed"])
    
    def create_initial_vehicles(self):
        """
        Create 100% human vehicle population with non-uniform spacing.
        
        Returns list of HumanVehicle instances.
        """
        N = self.config["N"]
        L = self.config["L"]
        idm_params = self.config["idm_params"]
        noise_Q = self.config.get("noise_Q", 0.0)
        
        # Use stochastic human vehicles
        use_stochastic = self.config.get("use_stochastic_human", True)
        
        nominal_spacing = L / N
        
        # Spacing variability
        sigma_spacing = 0.25 * nominal_spacing
        min_spacing = 2.5 * idm_params["s0"]
        max_spacing = 1.8 * nominal_spacing
        
        # Generate N-1 random spacings
        spacings = []
        remaining_length = L
        
        for i in range(N - 1):
            s = np.random.normal(nominal_spacing, sigma_spacing)
            s = np.clip(s, min_spacing, max_spacing)
            spacings.append(s)
            remaining_length -= s
        
        # Last spacing as residual
        last_spacing = remaining_length
        
        # Safety check
        if last_spacing < min_spacing or last_spacing > max_spacing:
            print(f"[ScenarioManager] Adjusting residual spacing: {last_spacing:.1f}m")
            # Redistribute
            ratio = remaining_length / sum(spacings)
            spacings = [s * ratio for s in spacings]
            last_spacing = L - sum(spacings)
        
        spacings.append(last_spacing)
        
        # Cumulative positions
        positions = [0.0]
        for s in spacings[:-1]:
            positions.append(positions[-1] + s)
        
        # Initial velocities (perturbed around mean)
        initial_speed = self.config.get("initial_speed", 10.0)
        sigma_v = 0.15 * initial_speed
        
        velocities = []
        for _ in range(N):
            v = np.random.normal(initial_speed, sigma_v)
            v = np.clip(v, 0.5 * initial_speed, 1.5 * initial_speed)
            velocities.append(v)
        
        # Create vehicles
        vehicles = []
        for i in range(N):
            if use_stochastic:
                v = StochasticHumanVehicle(
                    vid=i,
                    x0=positions[i],
                    v0=velocities[i],
                    idm_params=idm_params,
                    noise_Q=noise_Q
                )
            else:
                v = HumanVehicle(
                    vid=i,
                    x0=positions[i],
                    v0=velocities[i],
                    idm_params=idm_params,
                    noise_Q=noise_Q
                )
            vehicles.append(v)
        
        print(f"[ScenarioManager] Created {N} human vehicles")
        print(f"  Mean spacing: {np.mean(spacings):.2f}m (std={np.std(spacings):.2f}m)")
        print(f"  Mean velocity: {np.mean(velocities):.2f}m/s (std={np.std(velocities):.2f}m/s)")
        
        return vehicles
    
    def build(self):
        """
        Build complete simulation with transition environment.
        
        Returns TransitionSimulator instance.
        """
        # Create initial 100% human population
        vehicles = self.create_initial_vehicles()
        
        # Create transition environment with zones
        env = RingRoadTransitionEnv(
            L=self.config["L"],
            vehicles=vehicles,
            arrival_zone_config=self.config["arrival_zone"],
            departure_zone_config=self.config["departure_zone"]
        )
        
        print(f"[ScenarioManager] Arrival zone: center={self.config['arrival_zone']['center_x']:.1f}m, "
              f"width={self.config['arrival_zone']['width']:.1f}m")
        print(f"[ScenarioManager] Departure zone: center={self.config['departure_zone']['center_x']:.1f}m, "
              f"width={self.config['departure_zone']['width']:.1f}m")
        
        # Create macrofield generator (optional)
        macro_gen = MacrofieldGenerator(
            L=self.config["L"],
            dx=self.config.get("dx", 1.0),
            h=self.config.get("sph_smoothing_length", 10.0)
        )
        
        # Create logger
        metadata = {
            "scenario": "mixed_autonomy_transition",
            "N_initial": self.config["N"],
            "L": self.config["L"],
            "dt": self.config["dt"],
            "T": self.config["T"],
            "swap_interval": self.config.get("swap_interval", 20.0),
            "swap_ratio": self.config.get("swap_ratio", 0.25),
            "arrival_zone": self.config["arrival_zone"],
            "departure_zone": self.config["departure_zone"],
            "idm_params": self.config["idm_params"],
            "acc_params": self.config["acc_params"],
            "cth_params": self.config["cth_params"],
            "seed": self.config.get("seed", None)
        }
        
        logger = Logger(self.config["output_path"], metadata)
        
        # Create transition simulator
        sim = TransitionSimulator(
            env=env,
            macro_gen=macro_gen,
            logger=logger,
            dt=self.config["dt"],
            T=self.config["T"],
            swap_interval=self.config.get("swap_interval", 20.0),
            swap_ratio=self.config.get("swap_ratio", 0.25),
            idm_params=self.config["idm_params"],
            acc_params=self.config["acc_params"],
            cth_params=self.config["cth_params"],
            noise_Q=self.config.get("noise_Q", 0.0)
        )
        
        return sim
