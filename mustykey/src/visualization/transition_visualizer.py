# -*- coding: utf-8 -*-
"""
Transition Scenario Visualizer
Ring road with arrival and departure zone visualization (as attached roads).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class TransitionVisualizer:
    """
    Visualizer for mixed-autonomy transition scenario.
    
    Displays:
    - Ring road with vehicles
    - Arrival zone as attached road (right side)
    - Departure zone as attached road (left side)
    - Semi-transparent zone indicators on ring
    """
    
    def __init__(self, folder, L, arrival_zone_config, departure_zone_config, R=20.0):
        """
        Parameters:
        -----------
        folder : str
            Directory containing micro.pt
        L : float
            Ring length (meters)
        arrival_zone_config : dict
            {'center_x': float, 'width': float}
        departure_zone_config : dict
            {'center_x': float, 'width': float}
        R : float
            Visualization radius (ring)
        """
        # Load data
        raw = torch.load(f"{folder}/micro.pt", map_location="cpu")
        
        self.L = float(L)
        self.R = float(R)
        self.arrival_config = arrival_zone_config
        self.departure_config = departure_zone_config
        
        # Reconstruct frames
        self.frames = []
        self.vehicle_types = []  # Track types for each frame
        
        if len(raw) == 0:
            print("WARNING: No data in micro.pt")
            return
        
        # Group by timestep
        timesteps = {}
        for record in raw:
            step = record['step']
            if step not in timesteps:
                timesteps[step] = []
            timesteps[step].append(record)
        
        # Sort by step number
        for step in sorted(timesteps.keys()):
            records = timesteps[step]
            # Sort by vehicle ID
            records_sorted = sorted(records, key=lambda r: r['id'])
            
            x_list = [r['x'] for r in records_sorted]
            v_list = [r['v'] for r in records_sorted]
            types = [r['type'] for r in records_sorted]
            
            self.frames.append((x_list, v_list))
            self.vehicle_types.append(types)
        
        print(f"[TransitionVisualizer] Loaded {len(self.frames)} frames")
        
        # ===== FIGURE SETUP =====
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.set_aspect('equal', adjustable='box')
        
        # Expand limits to accommodate attached roads
        self.ax.set_xlim(-R - 15, R + 15)
        self.ax.set_ylim(-R - 5, R + 5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Mixed-Autonomy Transition on Ring Road")
        
        # ===== DRAW RING ROAD =====
        road_width = 3.0
        
        # Outer boundary
        outer_circle = patches.Circle((0, 0), R + road_width/2, fill=False,
                                     linewidth=2, color='black')
        self.ax.add_patch(outer_circle)
        
        # Inner boundary
        inner_circle = patches.Circle((0, 0), R - road_width/2, fill=False,
                                     linewidth=2, color='black')
        self.ax.add_patch(inner_circle)
        
        # Road surface
        road_surface = patches.Annulus((0, 0), R - road_width/2, road_width,
                                      fill=True, facecolor='lightgray',
                                      edgecolor='none', alpha=0.3)
        self.ax.add_patch(road_surface)
        
        # ===== DRAW ARRIVAL ZONE (RIGHT SIDE - as attached road) =====
        # Arrival road extends from ring to the right
        arrival_theta = self._x_to_theta(arrival_zone_config['center_x'])
        arrival_x = R * np.cos(arrival_theta)
        arrival_y = R * np.sin(arrival_theta)
        
        road_length = 8  # Length of attached road
        road_w = 2.5  # Width of attached road
        
        # Arrival road (pointing outward from ring)
        arrival_road_x = arrival_x + road_length * np.cos(arrival_theta)
        arrival_road_y = arrival_y + road_length * np.sin(arrival_theta)
        
        # Draw arrival road as rectangle
        arrival_rect = patches.FancyBboxPatch(
            (arrival_x, arrival_y - road_w/2),
            road_length, road_w,
            boxstyle="round,pad=0.05",
            edgecolor='green', facecolor='lightgreen',
            linewidth=2, alpha=0.4,
            transform=self._get_rotation_transform(arrival_theta, arrival_x, arrival_y)
        )
        self.ax.add_patch(arrival_rect)
        
        # Label
        self.ax.text(arrival_road_x + 2, arrival_road_y, "ARRIVAL",
                    fontsize=9, color='green', weight='bold',
                    ha='left', va='center')
        
        # Zone indicator on ring (semi-transparent arc)
        self._draw_zone_arc(arrival_zone_config, 'green', 'ARRIVAL')
        
        # ===== DRAW DEPARTURE ZONE (LEFT SIDE - as attached road) =====
        departure_theta = self._x_to_theta(departure_zone_config['center_x'])
        departure_x = R * np.cos(departure_theta)
        departure_y = R * np.sin(departure_theta)
        
        # Departure road (pointing outward from ring)
        departure_road_x = departure_x + road_length * np.cos(departure_theta)
        departure_road_y = departure_y + road_length * np.sin(departure_theta)
        
        # Draw departure road
        departure_rect = patches.FancyBboxPatch(
            (departure_x, departure_y - road_w/2),
            road_length, road_w,
            boxstyle="round,pad=0.05",
            edgecolor='red', facecolor='lightcoral',
            linewidth=2, alpha=0.4,
            transform=self._get_rotation_transform(departure_theta, departure_x, departure_y)
        )
        self.ax.add_patch(departure_rect)
        
        # Label
        self.ax.text(departure_road_x + 2, departure_road_y, "DEPARTURE",
                    fontsize=9, color='red', weight='bold',
                    ha='left', va='center')
        
        # Zone indicator on ring
        self._draw_zone_arc(departure_zone_config, 'red', 'DEPARTURE')
        
        # ===== ORIENTATION MARKERS =====
        tick_positions = [0, np.pi/2, np.pi, 3*np.pi/2]
        for th in tick_positions:
            x0 = (R + road_width/2) * np.cos(th)
            y0 = (R + road_width/2) * np.sin(th)
            x1 = (R + road_width/2 + 1.5) * np.cos(th)
            y1 = (R + road_width/2 + 1.5) * np.sin(th)
            self.ax.plot([x0, x1], [y0, y1], color='black', linewidth=2)
        
        # Legend
        self.ax.scatter([], [], c='blue', s=50, label="Human", marker='o')
        self.ax.scatter([], [], c='red', s=80, label="CAV", marker='s')
        self.ax.legend(loc='upper left', fontsize=10)
        
        # Scatter handles
        self.human_scatter = None
        self.cav_scatter = None
        
        plt.ion()
        plt.show()
    
    def _x_to_theta(self, x):
        """Convert ring position to angle (radians)."""
        return (2 * np.pi * x) / self.L
    
    def _get_rotation_transform(self, angle, x, y):
        """Get matplotlib transform for rotating patches."""
        import matplotlib.transforms as transforms
        trans = transforms.Affine2D().rotate_around(x, y, angle) + self.ax.transData
        return trans
    
    def _draw_zone_arc(self, zone_config, color, label):
        """
        Draw semi-transparent arc on ring to indicate zone.
        
        Handles wraparound at 0/L boundary correctly.
        """
        center_x = zone_config['center_x']
        width = zone_config['width']
        
        # Convert to angles
        theta_center = self._x_to_theta(center_x)
        d_theta = (2 * np.pi * width) / self.L
        
        theta_start = theta_center - d_theta / 2
        theta_end = theta_center + d_theta / 2
        
        # Handle wraparound: if zone crosses 0, draw two arcs
        if theta_start < 0:
            # Draw from [theta_start + 2π, 2π]
            theta1_start = theta_start + 2 * np.pi
            theta1_end = 2 * np.pi
            arc1 = patches.Wedge((0, 0), self.R + 1.5, 
                                np.degrees(theta1_start), np.degrees(theta1_end),
                                width=3.0, facecolor=color, edgecolor=color,
                                alpha=0.25, linewidth=0)
            self.ax.add_patch(arc1)
            
            # Draw from [0, theta_end]
            theta2_start = 0
            theta2_end = theta_end
            arc2 = patches.Wedge((0, 0), self.R + 1.5,
                                np.degrees(theta2_start), np.degrees(theta2_end),
                                width=3.0, facecolor=color, edgecolor=color,
                                alpha=0.25, linewidth=0)
            self.ax.add_patch(arc2)
        
        elif theta_end > 2 * np.pi:
            # Draw from [theta_start, 2π]
            theta1_start = theta_start
            theta1_end = 2 * np.pi
            arc1 = patches.Wedge((0, 0), self.R + 1.5,
                                np.degrees(theta1_start), np.degrees(theta1_end),
                                width=3.0, facecolor=color, edgecolor=color,
                                alpha=0.25, linewidth=0)
            self.ax.add_patch(arc1)
            
            # Draw from [0, theta_end - 2π]
            theta2_start = 0
            theta2_end = theta_end - 2 * np.pi
            arc2 = patches.Wedge((0, 0), self.R + 1.5,
                                np.degrees(theta2_start), np.degrees(theta2_end),
                                width=3.0, facecolor=color, edgecolor=color,
                                alpha=0.25, linewidth=0)
            self.ax.add_patch(arc2)
        
        else:
            # Normal case: no wraparound
            arc = patches.Wedge((0, 0), self.R + 1.5,
                               np.degrees(theta_start), np.degrees(theta_end),
                               width=3.0, facecolor=color, edgecolor=color,
                               alpha=0.25, linewidth=0)
            self.ax.add_patch(arc)
    
    def x_to_xy(self, x):
        """Convert ring position(s) to 2D coordinates."""
        theta = np.array(x) * (2 * np.pi / self.L)
        X = self.R * np.cos(theta)
        Y = self.R * np.sin(theta)
        return X, Y
    
    def play(self, pause=0.01):
        """Animate simulation with vehicle type tracking."""
        if len(self.frames) == 0:
            print("No frames to display")
            return
        
        for step, ((x_list, v_list), types) in enumerate(zip(self.frames, self.vehicle_types)):
            if len(x_list) == 0:
                continue
            
            X, Y = self.x_to_xy(x_list)
            
            # Separate by type
            human_idx = [i for i, t in enumerate(types) if 'Human' in t]
            cav_idx = [i for i, t in enumerate(types) if 'CAV' in t]
            
            # Update human scatter
            if len(human_idx) > 0:
                if self.human_scatter is None:
                    self.human_scatter = self.ax.scatter(
                        X[human_idx], Y[human_idx],
                        s=50, c='blue', marker='o', zorder=10
                    )
                else:
                    self.human_scatter.set_offsets(np.c_[X[human_idx], Y[human_idx]])
            elif self.human_scatter is not None:
                self.human_scatter.set_offsets(np.empty((0, 2)))
            
            # Update CAV scatter
            if len(cav_idx) > 0:
                if self.cav_scatter is None:
                    self.cav_scatter = self.ax.scatter(
                        X[cav_idx], Y[cav_idx],
                        s=80, c='red', marker='s', zorder=10
                    )
                else:
                    self.cav_scatter.set_offsets(np.c_[X[cav_idx], Y[cav_idx]])
            elif self.cav_scatter is not None:
                self.cav_scatter.set_offsets(np.empty((0, 2)))
            
            # Update title with composition
            n_human = len(human_idx)
            n_cav = len(cav_idx)
            total = n_human + n_cav
            if total > 0:
                pct_cav = (n_cav / total) * 100
                self.ax.set_title(
                    f"Mixed-Autonomy Transition | Step {step} | "
                    f"Humans: {n_human} | CAVs: {n_cav} ({pct_cav:.0f}%)"
                )
            
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(pause)
        
        plt.ioff()
        plt.show()
