"""
Visualization overlay helpers for zone-based scenarios.

These functions draw visual zones on ring-road animations WITHOUT
modifying the core Visualizer class.
"""

import numpy as np
import matplotlib.patches as mpatches


def draw_zone_arcs(ax, R, arrival_zone, departure_zone, L):
    """
    Draw arrival and departure zone arcs on ring visualization.
    
    Args:
        ax: Matplotlib axis
        R: Ring radius (visualization only)
        arrival_zone: Dict with 'center', 'width', 'start', 'end'
        departure_zone: Dict with 'center', 'width', 'start', 'end'
        L: Ring length
        
    CRITICAL: These are visual decorations only - NO physics impact
    """
    # Draw departure zone (RED)
    _draw_single_zone_arc(
        ax, R, departure_zone, L,
        color='red', alpha=0.15, label='DEPARTURE'
    )
    
    # Draw arrival zone (GREEN)
    _draw_single_zone_arc(
        ax, R, arrival_zone, L,
        color='green', alpha=0.15, label='ARRIVAL'
    )


def _draw_single_zone_arc(ax, R, zone, L, color, alpha, label):
    """Draw a single zone arc (handles wraparound)."""
    start = zone['start']
    end = zone['end']
    
    # Check for wraparound
    if start < end:
        # No wraparound
        theta_start = (start / L) * 2 * np.pi
        theta_end = (end / L) * 2 * np.pi
        _add_arc_patch(ax, R, theta_start, theta_end, color, alpha, label)
    else:
        # Wraparound: draw two arcs
        # Arc 1: from start to L
        theta_start1 = (start / L) * 2 * np.pi
        theta_end1 = 2 * np.pi
        _add_arc_patch(ax, R, theta_start1, theta_end1, color, alpha, label)
        
        # Arc 2: from 0 to end
        theta_start2 = 0
        theta_end2 = (end / L) * 2 * np.pi
        _add_arc_patch(ax, R, theta_start2, theta_end2, color, alpha, None)  # No label for second part


def _add_arc_patch(ax, R, theta_start, theta_end, color, alpha, label):
    """Add wedge patch to axis."""
    wedge = mpatches.Wedge(
        center=(0, 0),
        r=R + 3,  # Slightly outside vehicle ring
        theta1=np.degrees(theta_start),
        theta2=np.degrees(theta_end),
        facecolor=color,
        alpha=alpha,
        edgecolor=color,
        linewidth=2,
        label=label
    )
    ax.add_patch(wedge)


def draw_attached_roads(ax, R, arrival_zone, departure_zone, L, road_length=15):
    """
    Draw visual "attached roads" extending from zones.
    
    CRITICAL: These are VISUAL ONLY - not actual road geometry.
    They help viewers understand where swaps conceptually "happen".
    """
    # Departure road (RED, pointing outward left)
    dep_center = departure_zone['center']
    dep_angle = (dep_center / L) * 2 * np.pi
    
    dep_x = R * np.cos(dep_angle)
    dep_y = R * np.sin(dep_angle)
    
    # Extend outward
    dep_outer_x = (R + road_length) * np.cos(dep_angle)
    dep_outer_y = (R + road_length) * np.sin(dep_angle)
    
    ax.plot(
        [dep_x, dep_outer_x],
        [dep_y, dep_outer_y],
        color='red', linewidth=3, alpha=0.6,
        linestyle='--', label='Departure Road (visual)'
    )
    
    # Arrival road (GREEN, pointing outward right)
    arr_center = arrival_zone['center']
    arr_angle = (arr_center / L) * 2 * np.pi
    
    arr_x = R * np.cos(arr_angle)
    arr_y = R * np.sin(arr_angle)
    
    # Extend outward
    arr_outer_x = (R + road_length) * np.cos(arr_angle)
    arr_outer_y = (R + road_length) * np.sin(arr_angle)
    
    ax.plot(
        [arr_x, arr_outer_x],
        [arr_y, arr_outer_y],
        color='green', linewidth=3, alpha=0.6,
        linestyle='--', label='Arrival Road (visual)'
    )
    
    # Add labels
    ax.text(
        dep_outer_x * 1.1, dep_outer_y * 1.1,
        'DEPARTURE\n(visual only)',
        color='red', fontsize=8, ha='center',
        weight='bold', alpha=0.8
    )
    
    ax.text(
        arr_outer_x * 1.1, arr_outer_y * 1.1,
        'ARRIVAL\n(visual only)',
        color='green', fontsize=8, ha='center',
        weight='bold', alpha=0.8
    )
