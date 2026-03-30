# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 09:25:35 2025

@author: shnoz
"""

# -------------------------------------------------------------
# File: src/visualization/visualizer.py
# -------------------------------------------------------------

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Visualizer:
    def __init__(self, folder, L, N, cav_ids=None, R=20.0):
        """
        folder  : directory containing micro.pt
        L       : ring length
        N       : number of vehicles
        cav_ids : list of IDs for CAVs
        """
        # Load flat list of dicts
        raw = torch.load(f"{folder}/micro.pt", map_location="cpu")

        # Reconstruct frames:
        # raw = [dict for vehicle0_step0, dict for vehicle1_step0, ..., dict for vehicle19_step0,
        #        dict for vehicle0_step1, dict for vehicle1_step1, ..., ]
        assert len(raw) % N == 0, "Micro data length not divisible by N!"

        self.frames = []
        total_steps = len(raw) // N

        for step in range(total_steps):
            block = raw[step * N : (step + 1) * N]
           
            # But visualizer needs them indexed by vehicle ID
            block_sorted = sorted(block, key=lambda b: b["id"])
            
            x_list = [b["x"] for b in block_sorted]
            v_list = [b["v"] for b in block_sorted]
            self.frames.append((x_list, v_list))

        self.L = float(L)
        self.N = N
        self.R = float(R)
        self.cav_ids = set(cav_ids) if cav_ids is not None else set()

        # ---------------------------------------------------------
        # Figure setup
        # ---------------------------------------------------------
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-R - 5, R + 5)
        self.ax.set_ylim(-R - 5, R + 5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Title will update every frame
        self.ax.set_title("Ring-Road Simulation")

        # ---------------------------------------------------------
        # Two-circle road visualization (inner and outer boundaries)
        # ---------------------------------------------------------
        road_width = 3.0  # Width of the road in visualization units
        
        # Outer boundary circle
        outer_circle = patches.Circle((0, 0), R + road_width/2, fill=False, 
                                     linewidth=2, color='black')
        self.ax.add_patch(outer_circle)
        
        # Inner boundary circle
        inner_circle = patches.Circle((0, 0), R - road_width/2, fill=False, 
                                     linewidth=2, color='black')
        self.ax.add_patch(inner_circle)
        
        # Road surface (gray area between circles)
        road_surface = patches.Annulus((0, 0), R - road_width/2, road_width, 
                                      fill=True, facecolor='lightgray', 
                                      edgecolor='none', alpha=0.3)
        self.ax.add_patch(road_surface)

        # Orientation markers on outer circle (to detect movement)
        tick_positions = [0, np.pi/2, np.pi, 3*np.pi/2]
        for th in tick_positions:
            x0 = (R + road_width/2) * np.cos(th)
            y0 = (R + road_width/2) * np.sin(th)
            x1 = (R + road_width/2 + 1.5) * np.cos(th)
            y1 = (R + road_width/2 + 1.5) * np.sin(th)
            self.ax.plot([x0, x1], [y0, y1], color='black', linewidth=3)

        # Scatter handles for dynamic agents
        self.human_scatter = None
        self.cav_scatter = None

        plt.ion()
        plt.show()

    # ---------------------------------------------------------
    def x_to_xy(self, x):
        """Convert longitudinal x into 2D ring coordinates."""
        theta = (2 * np.pi / self.L) * np.array(x)
        X = self.R * np.cos(theta)
        Y = self.R * np.sin(theta)
        return X, Y

    # ---------------------------------------------------------
    def play(self, pause=0.01):
        """Animate reconstructed micro frames."""

        # Create legend once
        self.ax.scatter([], [], c='blue', s=50, label="Human")
        self.ax.scatter([], [], c='red', s=80, label="CAV")
        self.ax.legend(loc="upper right")

        # Animate each reconstructed step
        for step, (x_list, v_list) in enumerate(self.frames):

            X, Y = self.x_to_xy(x_list)

            human_idx = [i for i in range(self.N) if i not in self.cav_ids]
            cav_idx   = [i for i in range(self.N) if i in self.cav_ids]

            # Draw humans
            if self.human_scatter is None:
                self.human_scatter = self.ax.scatter(X[human_idx], Y[human_idx], s=50, c='blue')
            else:
                self.human_scatter.set_offsets(np.c_[X[human_idx], Y[human_idx]])

            # Draw CAVs
            if self.cav_scatter is None:
                self.cav_scatter = self.ax.scatter(X[cav_idx], Y[cav_idx], s=80, c='red')
            else:
                self.cav_scatter.set_offsets(np.c_[X[cav_idx], Y[cav_idx]])

            # Title update
            self.ax.set_title(f"Ring-Road Simulation | Step {step}")

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(pause)

        plt.ioff()
        plt.show()


# =============================================================================
# LIVE VISUALIZER - Real-time display during simulation
# =============================================================================

class LiveVisualizer:
    """Real-time visualization of ring-road simulation during execution."""
    
    def __init__(self, L, N, cav_ids=None, R=20.0, update_interval=10):
        """
        Args:
            L: Ring road length (m)
            N: Number of vehicles
            cav_ids: Set/list of CAV vehicle IDs
            R: Visualization radius
            update_interval: Update display every N simulation steps
        """
        self.L = float(L)
        self.N = N
        self.R = float(R)
        self.cav_ids = set(cav_ids) if cav_ids is not None else set()
        self.update_interval = update_interval
        self.step_counter = 0
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(-R - 5, R + 5)
        self.ax.set_ylim(-R - 5, R + 5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_title("Ring-Road Simulation (LIVE)")
        
        # Road visualization
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
        
        # Orientation markers
        tick_positions = [0, np.pi/2, np.pi, 3*np.pi/2]
        for th in tick_positions:
            x0 = (R + road_width/2) * np.cos(th)
            y0 = (R + road_width/2) * np.sin(th)
            x1 = (R + road_width/2 + 1.5) * np.cos(th)
            y1 = (R + road_width/2 + 1.5) * np.sin(th)
            self.ax.plot([x0, x1], [y0, y1], color='black', linewidth=3)
        
        # Legend
        self.ax.scatter([], [], c='blue', s=50, label="Human")
        self.ax.scatter([], [], c='red', s=80, label="CAV")
        self.ax.legend(loc="upper right")
        
        # Scatter plot handles (will be created on first update)
        self.human_scatter = None
        self.cav_scatter = None
        
        # Enable interactive mode
        plt.ion()
        plt.show()
    
    def x_to_xy(self, x_array):
        """Convert longitudinal positions to 2D ring coordinates."""
        theta = (2 * np.pi / self.L) * np.array(x_array)
        X = self.R * np.cos(theta)
        Y = self.R * np.sin(theta)
        return X, Y
    
    def update(self, vehicles, current_time, step):
        """
        Update visualization with current vehicle positions.
        
        Args:
            vehicles: List of vehicle objects with .id and .x attributes
            current_time: Current simulation time (s)
            step: Current simulation step number
        """
        self.step_counter += 1
        
        # Only update display every N steps (for performance)
        if self.step_counter % self.update_interval != 0:
            return
        
        # Extract positions and sort by vehicle ID
        vehicle_data = sorted([(v.id, v.x) for v in vehicles], key=lambda x: x[0])
        ids = [d[0] for d in vehicle_data]
        positions = [d[1] for d in vehicle_data]
        
        # Convert to ring coordinates
        X, Y = self.x_to_xy(positions)
        
        # Split into humans and CAVs
        human_idx = [i for i, vid in enumerate(ids) if vid not in self.cav_ids]
        cav_idx = [i for i, vid in enumerate(ids) if vid in self.cav_ids]
        
        # Update human scatter
        if len(human_idx) > 0:
            if self.human_scatter is None:
                self.human_scatter = self.ax.scatter(
                    [X[i] for i in human_idx],
                    [Y[i] for i in human_idx],
                    s=50, c='blue'
                )
            else:
                self.human_scatter.set_offsets(
                    np.c_[[X[i] for i in human_idx], [Y[i] for i in human_idx]]
                )
        
        # Update CAV scatter
        if len(cav_idx) > 0:
            if self.cav_scatter is None:
                self.cav_scatter = self.ax.scatter(
                    [X[i] for i in cav_idx],
                    [Y[i] for i in cav_idx],
                    s=80, c='red'
                )
            else:
                self.cav_scatter.set_offsets(
                    np.c_[[X[i] for i in cav_idx], [Y[i] for i in cav_idx]]
                )
        
        # Update title with time info
        self.ax.set_title(f"Ring-Road Simulation (LIVE) | t={current_time:.1f}s | Step {step}")
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        """Close the visualization window."""
        plt.ioff()
        plt.close(self.fig)

