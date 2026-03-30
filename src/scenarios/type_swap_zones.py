import numpy as np
from src.vehicles.human_vehicle import HumanVehicle
from src.vehicles.stochastic_human_vehicle import StochasticHumanVehicle
from src.vehicles.cav_vehicle import CAVVehicle
from src.environment.ring_road import RingRoadEnv
from src.simulation.simulator import Simulator
from src.utils.config_manager import ConfigManager
from src.utils.logger import Logger
from src.macro.macrofield_generator import MacrofieldGenerator
from src.visualization.visualizer import Visualizer


class TypeSwapSimulator(Simulator):
    """
    Simulator that performs in-place type swaps without disturbing traffic flow.
    
    Swap mechanism:
    - Every swap_interval seconds, convert k humans to CAVs
    - k = round(swap_ratio * N_total)
    - Conversion preserves exact position, velocity, acceleration
    - NO insertion/removal logic, NO gap creation
    """
    
    def __init__(self, env, macro_gen, logger, dt, T, config,
                 swap_interval=20.0, swap_ratio=0.25,
                 arrival_zone_center=225.0, arrival_zone_width=30.0,
                 departure_zone_center=75.0, departure_zone_width=30.0):
        super().__init__(env, macro_gen, logger, dt, T)
        
        self.config = config
        self.t = 0.0  # Track simulation time
        
        self.swap_interval = swap_interval
        self.swap_ratio = swap_ratio
        self.k_per_swap = max(1, round(swap_ratio * len(env.vehicles)))
        
        # Visual zones (for rendering only)
        self.arrival_zone = {
            'center': arrival_zone_center,
            'width': arrival_zone_width,
            'start': (arrival_zone_center - arrival_zone_width/2) % config['L'],
            'end': (arrival_zone_center + arrival_zone_width/2) % config['L']
        }
        self.departure_zone = {
            'center': departure_zone_center,
            'width': departure_zone_width,
            'start': (departure_zone_center - departure_zone_width/2) % config['L'],
            'end': (departure_zone_center + departure_zone_width/2) % config['L']
        }
        
        self.next_swap_time = swap_interval
        self.swap_count = 0
        self.total_swaps = 0
        
        # Lap tracking for vehicle 1 (index 0 after sorting)
        self.track_vehicle_id = 0  # Track vehicle with ID 0
        self.lap_count = 0
        self.last_x_tracked = None  # Previous position of tracked vehicle
        self.lap_completed = False
        
        # Perturbation parameters (optional)
        perturbation = config.get('perturbation', {})
        self.perturbation_enabled = perturbation.get('enabled', False)
        self.perturbation_vid = perturbation.get('vehicle_id', 0)
        self.perturbation_start = perturbation.get('t_start', 5.0)
        self.perturbation_duration = perturbation.get('duration', 2.0)
        self.perturbation_delta_v = perturbation.get('delta_v', -3.0)
        self.perturbation_applied = False
        self.perturbation_end = self.perturbation_start + self.perturbation_duration
        
        # Initialize tracked vehicle position
        for v in env.vehicles:
            if v.id == self.track_vehicle_id:
                self.last_x_tracked = v.x
                break
        
        print(f"\n{'='*70}")
        print(f"TYPE-SWAP TRANSITION SCENARIO (LAP-BASED)")
        print(f"{'='*70}")
        print(f"Initial: 100% Human ({len(env.vehicles)} vehicles)")
        print(f"Swap trigger: when vehicle {self.track_vehicle_id} completes a lap")
        print(f"Per swap: convert {self.k_per_swap} humans ({int(swap_ratio*100)}%) to CAVs")
        print(f"Ring length: {config['L']}m")
        print(f"Arrival zone (visual): center={arrival_zone_center}m, width={arrival_zone_width}m")
        print(f"Departure zone (visual): center={departure_zone_center}m, width={departure_zone_width}m")
        if self.perturbation_enabled:
            print(f"Perturbation: vehicle {self.perturbation_vid} at t={self.perturbation_start}s "
                  f"for {self.perturbation_duration}s (delta_v={self.perturbation_delta_v} m/s)")
        print(f"{'='*70}\n")
    
    def check_lap_completion(self):
        """
        Check if tracked vehicle completed a lap (crossed x=0 from high position).
        Returns True if lap completed this step.
        """
        if self.last_x_tracked is None:
            return False
        
        L = self.config['L']
        
        # Find tracked vehicle
        tracked_vehicle = None
        for v in self.env.vehicles:
            if v.id == self.track_vehicle_id:
                tracked_vehicle = v
                break
        
        if tracked_vehicle is None:
            return False
        
        x_current = tracked_vehicle.x
        x_prev = self.last_x_tracked
        
        # Detect wraparound: if vehicle went from near L to near 0
        lap_completed = False
        if x_prev > 0.75 * L and x_current < 0.25 * L:
            lap_completed = True
            self.lap_count += 1
            print(f"\n[LAP COMPLETED] Vehicle {self.track_vehicle_id} completed lap {self.lap_count} at t={self.t:.1f}s\n")
        
        # Update last position
        self.last_x_tracked = x_current
        
        return lap_completed
    
    def step(self):
        """Single simulation step with type-swap events."""
        # Apply perturbation if enabled and in time window
        if self.perturbation_enabled:
            if self.perturbation_start <= self.t < self.perturbation_end:
                if not self.perturbation_applied:
                    # Apply perturbation to leader vehicle
                    for v in self.env.vehicles:
                        if v.id == self.perturbation_vid:
                            v.v = max(0, v.v + self.perturbation_delta_v)
                            print(f"\n[PERTURBATION] t={self.t:.1f}s: Vehicle {v.id} brakes by {abs(self.perturbation_delta_v)} m/s\n")
                            self.perturbation_applied = True
                            break
        
        # Standard simulation step (updates positions)
        super().step()
        
        # Check if tracked vehicle completed a lap
        if self.check_lap_completion():
            self.perform_type_swap()
        
        # Increment time
        self.t += self.dt
    
    def perform_type_swap(self):
        """
        Perform in-place type swap: Human -> CAV
        
        CRITICAL: This is NOT insertion/removal!
        - Select k humans to convert
        - Create CAV at EXACT same state (x, v, a)
        - Replace in vehicle list
        - NO gaps, NO waiting, NO physics changes
        """
        vehicles = self.env.vehicles
        
        # Count current composition
        n_human = sum(1 for v in vehicles if isinstance(v, (HumanVehicle, StochasticHumanVehicle)))
        n_cav = sum(1 for v in vehicles if isinstance(v, CAVVehicle))
        
        self.swap_count += 1
        
        print(f"\n[SWAP EVENT {self.swap_count}] t={self.t:.1f}s (Lap {self.lap_count})")
        print(f"  Current: {n_human} humans, {n_cav} CAVs")
        
        # Check if all humans converted
        if n_human == 0:
            print(f"  All vehicles are CAVs - no more swaps needed")
            return
        
        # Find humans to convert (select k closest to departure zone for visual consistency)
        humans = [(i, v) for i, v in enumerate(vehicles) 
                  if isinstance(v, (HumanVehicle, StochasticHumanVehicle))]
        
        if not humans:
            return
        
        # Sort by distance to departure zone center (for visual realism)
        dep_center = self.departure_zone['center']
        L = self.config['L']
        
        def distance_to_zone(vehicle):
            """Circular distance to departure zone center."""
            dx = (vehicle.x - dep_center + L/2) % L - L/2
            return abs(dx)
        
        humans_sorted = sorted(humans, key=lambda iv: distance_to_zone(iv[1]))
        
        # Select k humans to convert
        k = min(self.k_per_swap, len(humans_sorted))
        selected = humans_sorted[:k]
        
        print(f"  Target: Convert {k} humans to CAVs")
        
        # Perform in-place type swap
        converted = 0
        for idx, human in selected:
            # Create CAV with EXACT same state
            new_cav = CAVVehicle(
                vid=human.id,
                x0=human.x,  # EXACT position
                v0=human.v,  # EXACT velocity
                acc_params=self.config['acc_params'],
                cth_params=self.config['cth_params']
            )
            
            # Inherit ALL attributes from human
            new_cav.leader = human.leader
            new_cav.L = self.config['L']  # Ring length (needed for collision prevention)
            new_cav.acceleration = human.acceleration
            
            # Replace in vehicle list (in-place)
            self.env.vehicles[idx] = new_cav
            converted += 1
        
        self.total_swaps += converted
        
        # CRITICAL: Re-sort vehicles by position and reassign leaders
        # This ensures proper ordering after type swaps
        vehicles.sort(key=lambda v: v.x)
        
        # Reassign leaders (circular: last vehicle's leader is first vehicle)
        for i in range(len(vehicles)):
            vehicles[i].leader = vehicles[(i + 1) % len(vehicles)]
        
        # Recount after swap
        n_human_after = sum(1 for v in self.env.vehicles if isinstance(v, (HumanVehicle, StochasticHumanVehicle)))
        n_cav_after = sum(1 for v in self.env.vehicles if isinstance(v, CAVVehicle))
        
        print(f"  Converted {converted} vehicles")
        print(f"  New total: {n_human_after} humans, {n_cav_after} CAVs")
        print(f"  Leaders reassigned after swap")
        k = min(self.k_per_swap, len(humans_sorted))
        selected = humans_sorted[:k]
        
        print(f"  Target: Convert {k} humans to CAVs")
        
        # Perform in-place type swap
        converted = 0
        for idx, human in selected:
            # Create CAV with EXACT same state
            new_cav = CAVVehicle(
                vid=human.id,
                x0=human.x,  # EXACT position
                v0=human.v,  # EXACT velocity
                acc_params=self.config['acc_params'],
                cth_params=self.config['cth_params']
            )
            
            # Inherit ALL attributes from human
            new_cav.leader = human.leader
            new_cav.L = self.config['L']  # Ring length (needed for collision prevention)
            new_cav.acceleration = human.acceleration
            
            # Replace in vehicle list (in-place)
            self.env.vehicles[idx] = new_cav
            converted += 1
        
        self.total_swaps += converted
        
        # Recount after swap
        n_human_after = sum(1 for v in self.env.vehicles if isinstance(v, (HumanVehicle, StochasticHumanVehicle)))
        n_cav_after = sum(1 for v in self.env.vehicles if isinstance(v, CAVVehicle))
        
        print(f"  Converted {converted} vehicles")
        print(f"  New total: {n_human_after} humans, {n_cav_after} CAVs")
    
    def run(self):
        """Run simulation with type-swap events."""
        T = self.config['T']
        dt = self.config['dt']
        print(f"[Simulator] Starting type-swap simulation (t_max={T}s, dt={dt}s)")
        
        n_steps = self.steps
        
        for step_num in range(n_steps):
            self.step()
            
            # Check if all vehicles are CAVs
            n_human = sum(1 for v in self.env.vehicles if isinstance(v, (HumanVehicle, StochasticHumanVehicle)))
            
            # If all CAVs, run for 20 more seconds then stop
            if n_human == 0 and self.t >= self.next_swap_time - self.swap_interval + 20.0:
                print(f"\n[Simulator] All vehicles converted to CAVs - stopping after grace period")
                break
        
        # Save data
        self.logger.save()
        
        print(f"\n{'='*70}")
        print(f"SIMULATION COMPLETE")
        print(f"{'='*70}")
        print(f"Total simulation time: {self.t:.1f}s")
        print(f"Swap events: {self.swap_count}")
        print(f"Total humans converted: {self.total_swaps}")
        
        n_human_final = sum(1 for v in self.env.vehicles if isinstance(v, (HumanVehicle, StochasticHumanVehicle)))
        n_cav_final = sum(1 for v in self.env.vehicles if isinstance(v, CAVVehicle))
        print(f"Final composition: {n_human_final} humans, {n_cav_final} CAVs")
        print(f"{'='*70}\n")


def run_type_swap_zones(config_path, seed=None, visualize=True):
    """
    Run type-swap transition scenario with visual zones.
    
    Args:
        config_path: Path to YAML configuration file
        seed: Random seed (overrides config if provided)
        visualize: Whether to show visualization after simulation
        
    Returns:
        Simulator instance with results
    """
    # Load configuration
    config = ConfigManager.load(config_path)
    
    if seed is not None:
        config['seed'] = seed
    
    # Set random seed
    np.random.seed(config['seed'])
    
    # Extract zone parameters
    arrival_zone = config.get('arrival_zone', {'center_x': 225.0, 'width': 30.0})
    departure_zone = config.get('departure_zone', {'center_x': 75.0, 'width': 30.0})
    swap_interval = config.get('swap_interval', 20.0)
    swap_ratio = config.get('swap_ratio', 0.25)
    
    # Create initial vehicles (100% human)
    N = config['N']
    L = config['L']
    initial_speed = config.get('initial_speed', 10.0)
    vehicle_length = config.get('vehicle_length', 5.0)
    
    spacing = L / N
    vehicles = []
    
    for i in range(N):
        x = i * spacing
        v = initial_speed + np.random.normal(0, 0.5)
        v = max(0, v)  # Non-negative
        
        # Use StochasticHumanVehicle if noise_Q specified
        noise_Q = config.get('noise_Q', 0.0)
        if noise_Q > 0:
            vehicle = StochasticHumanVehicle(
                vid=i,
                x0=x,
                v0=v,
                idm_params=config['idm_params'],
                noise_Q=noise_Q
            )
        else:
            vehicle = HumanVehicle(
                vehicle_id=i,
                x=x,
                v=v,
                idm_params=config['idm_params'],
                length=vehicle_length
            )
        
        vehicles.append(vehicle)
    
    print(f"[ScenarioManager] Created {N} human vehicles (100% human)")
    
    # Sort vehicles and assign leaders (done once at initialization)
    vehicles.sort(key=lambda v: v.x)
    for i, vehicle in enumerate(vehicles):
        leader_idx = (i + 1) % N
        vehicle.leader = vehicles[leader_idx]
    
    # Create environment
    env = RingRoadEnv(L=L, vehicles=vehicles)
    
    # Create macrofield generator
    dx = config.get('dx', 1.0)
    sph_h = config.get('sph_h', 5.0)
    macrofield_gen = MacrofieldGenerator(L=L, dx=dx, h=sph_h)
    
    # Create logger
    metadata = {
        'N': N,
        'L': L,
        'dt': config['dt'],
        'T': config['T'],
        'seed': config['seed'],
        'scenario_type': 'type_swap_zones',
        'swap_interval': swap_interval,
        'swap_ratio': swap_ratio,
        'arrival_zone': arrival_zone,
        'departure_zone': departure_zone
    }
    logger = Logger(config['output_path'], metadata)
    
    # Create type-swap simulator
    sim = TypeSwapSimulator(
        env=env,
        macro_gen=macrofield_gen,
        logger=logger,
        dt=config['dt'],
        T=config['T'],
        config=config,
        swap_interval=swap_interval,
        swap_ratio=swap_ratio,
        arrival_zone_center=arrival_zone['center_x'],
        arrival_zone_width=arrival_zone['width'],
        departure_zone_center=departure_zone['center_x'],
        departure_zone_width=departure_zone['width']
    )
    
    # Run simulation
    sim.run()
    
    # Visualize if requested
    if visualize:
        print(f"[Visualizer] Creating visualization with zone overlays...")
        
        # Import zone overlay helpers
        from src.visualization.zone_overlays import draw_zone_arcs, draw_attached_roads
        
        # Get CAV IDs for color coding (will update during animation)
        from src.utils.logger import Logger as LoggerUtil
        import torch
        
        # Load micro data to get final CAV IDs
        micro_data = torch.load(f"{config['output_path']}/micro.pt")
        
        # Create custom visualizer with zone overlays
        viz = Visualizer(
            output_path=config['output_path'],
            L=L,
            cav_ids=[]  # Will update per frame
        )
        
        # Store zone info for rendering
        arrival_zone_info = {
            'center': arrival_zone['center_x'],
            'width': arrival_zone['width'],
            'start': (arrival_zone['center_x'] - arrival_zone['width']/2) % L,
            'end': (arrival_zone['center_x'] + arrival_zone['width']/2) % L
        }
        departure_zone_info = {
            'center': departure_zone['center_x'],
            'width': departure_zone['width'],
            'start': (departure_zone['center_x'] - departure_zone['width']/2) % L,
            'end': (departure_zone['center_x'] + departure_zone['width']/2) % L
        }
        
        # Monkey-patch the animate function to draw zones
        original_animate = viz.animate
        
        def animate_with_zones(frame):
            """Enhanced animation with zone overlays."""
            # Call original animation
            artists = original_animate(frame)
            
            # Add zone overlays on first frame
            if frame == 0:
                draw_zone_arcs(viz.ax, viz.R, arrival_zone_info, departure_zone_info, L)
                draw_attached_roads(viz.ax, viz.R, arrival_zone_info, departure_zone_info, L)
            
            return artists
        
        viz.animate = animate_with_zones
        
        viz.play(fps=30)
    
    return sim
