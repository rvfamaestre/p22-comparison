# -*- coding: utf-8 -*-
"""
Mixed-Autonomy Transition Environment
Extended ring road with arrival and departure zones for vehicle swapping.
"""

import numpy as np


class TransitionZone:
    """Represents a fixed zone on the ring road."""
    
    def __init__(self, center_x, width, L):
        """
        Parameters:
        -----------
        center_x : float
            Center position of zone on ring [0, L)
        width : float
            Width of zone (meters)
        L : float
            Ring length for wraparound handling
        """
        self.center_x = center_x
        self.width = width
        self.L = L
        self.half_width = width / 2.0
    
    def contains_vehicle(self, vehicle):
        """
        Check if vehicle is inside this zone.
        
        Uses circular distance to handle wraparound correctly.
        Vehicle is inside if circular distance to center <= half_width.
        """
        # Circular distance (shortest arc on ring)
        dx = vehicle.x - self.center_x
        
        # Wrap to [-L/2, L/2]
        if dx > self.L / 2:
            dx -= self.L
        elif dx < -self.L / 2:
            dx += self.L
        
        return abs(dx) <= self.half_width


class RingRoadTransitionEnv:
    """
    Ring road environment with arrival and departure zones.
    
    Manages vehicle swapping from 100% human to 100% CAV over time.
    """
    
    def __init__(self, L, vehicles, arrival_zone_config, departure_zone_config):
        """
        Parameters:
        -----------
        L : float
            Ring road length (meters)
        vehicles : list[Vehicle]
            Initial vehicle list (100% human at start)
        arrival_zone_config : dict
            {'center_x': float, 'width': float}
        departure_zone_config : dict
            {'center_x': float, 'width': float}
        """
        self.L = L
        self.vehicles = vehicles
        
        # Create zones
        self.arrival_zone = TransitionZone(
            arrival_zone_config['center_x'],
            arrival_zone_config['width'],
            L
        )
        self.departure_zone = TransitionZone(
            departure_zone_config['center_x'],
            departure_zone_config['width'],
            L
        )
        
        # Swap tracking
        self.pending_arrivals = 0
        self.pending_departures = 0
        
        # Statistics
        self.total_swapped = 0
        self.total_humans_removed = 0
        self.total_cavs_added = 0
    
    def apply_wraparound_all(self):
        """Apply periodic boundary to all vehicles."""
        for v in self.vehicles:
            v.apply_wraparound(self.L)
    
    def get_humans_in_departure_zone(self):
        """Return list of human vehicles currently in departure zone."""
        from src.vehicles.human_vehicle import HumanVehicle
        from src.vehicles.stochastic_human_vehicle import StochasticHumanVehicle
        from src.vehicles.unstable_human_vehicle import UnstableHumanVehicle
        
        humans = []
        for v in self.vehicles:
            if isinstance(v, (HumanVehicle, StochasticHumanVehicle, UnstableHumanVehicle)):
                if self.departure_zone.contains_vehicle(v):
                    humans.append(v)
        return humans
    
    def count_humans(self):
        """Count total human vehicles."""
        from src.vehicles.human_vehicle import HumanVehicle
        from src.vehicles.stochastic_human_vehicle import StochasticHumanVehicle
        from src.vehicles.unstable_human_vehicle import UnstableHumanVehicle
        
        count = 0
        for v in self.vehicles:
            if isinstance(v, (HumanVehicle, StochasticHumanVehicle, UnstableHumanVehicle)):
                count += 1
        return count
    
    def count_cavs(self):
        """Count total CAV vehicles."""
        from src.vehicles.cav_vehicle import CAVVehicle
        
        count = 0
        for v in self.vehicles:
            if isinstance(v, CAVVehicle):
                count += 1
        return count
    
    def is_safe_to_insert(self, x_insertion, min_gap=5.0):
        """
        Check if insertion at x_insertion is safe.
        
        Returns True if gaps before and after are >= min_gap.
        """
        if len(self.vehicles) == 0:
            return True
        
        # Find nearest vehicles before and after insertion point
        positions = [v.x for v in self.vehicles]
        positions.sort()
        
        # Find insertion index
        insert_idx = 0
        for i, x in enumerate(positions):
            if x <= x_insertion:
                insert_idx = i + 1
        
        # Check gap before
        if insert_idx > 0:
            x_before = positions[insert_idx - 1]
        else:
            x_before = positions[-1] - self.L  # wraparound
        
        gap_before = x_insertion - x_before
        if gap_before < 0:
            gap_before += self.L
        
        # Check gap after
        if insert_idx < len(positions):
            x_after = positions[insert_idx]
        else:
            x_after = positions[0] + self.L  # wraparound
        
        gap_after = x_after - x_insertion
        if gap_after < 0:
            gap_after += self.L
        
        return gap_before >= min_gap and gap_after >= min_gap
    
    def compute_local_mean_velocity(self, x_position, search_radius=20.0):
        """
        Compute mean velocity of vehicles near x_position.
        
        Used to initialize inserted CAV velocity.
        """
        if len(self.vehicles) == 0:
            return 10.0  # default fallback
        
        velocities = []
        for v in self.vehicles:
            # Circular distance
            dx = abs(v.x - x_position)
            if dx > self.L / 2:
                dx = self.L - dx
            
            if dx <= search_radius:
                velocities.append(v.v)
        
        if len(velocities) == 0:
            # No vehicles nearby, use global mean
            return np.mean([v.v for v in self.vehicles])
        
        return np.mean(velocities)
