# -*- coding: utf-8 -*-
"""
String stability analysis module.

Provides functions to inject velocity perturbations and analyze
their propagation through the vehicle platoon.
"""

import numpy as np
import torch


def inject_perturbation(vehicles, leader_id, delta_v, t_current, t_perturb, dt):
    """
    Inject a velocity perturbation into a specific vehicle at a given time.
    
    Parameters:
    -----------
    vehicles : list
        List of vehicle objects
    leader_id : int
        ID of the vehicle to perturb
    delta_v : float
        Magnitude of velocity perturbation (m/s)
    t_current : float
        Current simulation time (s)
    t_perturb : float
        Time at which to apply perturbation (s)
    dt : float
        Simulation timestep (s)
    
    Returns:
    --------
    bool
        True if perturbation was applied, False otherwise
    """
    # Check if we're at the perturbation time (within one timestep)
    if abs(t_current - t_perturb) < dt / 2:
        for v in vehicles:
            if v.id == leader_id:
                v.v += delta_v
                # Ensure velocity remains physical
                v.v = max(0.0, v.v)
                return True
    return False


def compute_velocity_amplification(micro_data, perturb_vehicle_id, t_perturb, dt, N_vehicles, L=300.0, vehicle_length=5.0):
    """
    Compute velocity fluctuation amplification downstream of perturbation.
    
    DEPRECATED WARNING: This function assumes micro_data is ordered consistently
    by vehicle ID at each timestep, which may NOT be true if vehicles are sorted
    by position before logging. Use CSV-based analysis with explicit ID tracking instead
    (see plot_string_stability_analysis.py).
    
    Analyzes how a velocity perturbation propagates through the platoon
    by measuring peak velocity deviations for each vehicle.
    
    Parameters:
    -----------
    micro_data : list of tuples
        Simulation micro data: [(id_list, x_list, v_list), ...]
        CRITICAL: Must include id_list to track vehicles correctly!
    perturb_vehicle_id : int
        ID of vehicle that was perturbed
    t_perturb : float
        Time when perturbation was applied (s)
    dt : float
        Simulation timestep (s)
    N_vehicles : int
        Total number of vehicles
    L : float
        Ring length (m) for wraparound correction
    vehicle_length : float
        Vehicle length (m) for bumper-to-bumper spacing
    
    Returns:
    --------
    amplification_ratios : numpy.ndarray
        Ratio of peak velocity deviation for each follower relative to leader
    spacing_variance : numpy.ndarray
        Time-series variance of inter-vehicle spacing
    velocity_std : numpy.ndarray
        Standard deviation of velocities over time for each vehicle
    """
    # Find perturbation timestep index
    perturb_step = int(t_perturb / dt)
    
    # Determine data format (old vs new)
    if len(micro_data[0]) == 2:
        print("\n[WARNING] String stability analysis using OLD format without ID tracking!")
        print("          Results may be INVALID if vehicles are sorted by position during logging.")
        print("          Use CSV-based analysis (plot_string_stability_analysis.py) instead.")
        # Old format: (x_list, v_list) - assume indices match IDs
        n_steps = len(micro_data)
        v_matrix = np.zeros((n_steps, N_vehicles))
        x_matrix = np.zeros((n_steps, N_vehicles))
        
        for step, (x_list, v_list) in enumerate(micro_data):
            v_matrix[step, :] = v_list
            x_matrix[step, :] = x_list
            
    elif len(micro_data[0]) == 3:
        # New format: (id_list, x_list, v_list) - track by ID
        print("\n[INFO] String stability analysis using ID-tracked format.")
        n_steps = len(micro_data)
        
        # Build ID-indexed matrices
        id_list_ref = micro_data[0][0]  # Get ID list from first timestep
        id_to_idx = {vid: i for i, vid in enumerate(id_list_ref)}
        
        v_matrix = np.zeros((n_steps, N_vehicles))
        x_matrix = np.zeros((n_steps, N_vehicles))
        
        for step, (id_list, x_list, v_list) in enumerate(micro_data):
            for i, vid in enumerate(id_list):
                idx = id_to_idx[vid]
                v_matrix[step, idx] = v_list[i]
                x_matrix[step, idx] = x_list[i]
    else:
        raise ValueError(f"Unexpected micro_data format: {len(micro_data[0])} elements per timestep")
    
    # Compute baseline (pre-perturbation) mean velocities
    if perturb_step > 10:
        baseline_v = np.mean(v_matrix[:perturb_step-5, :], axis=0)
    else:
        baseline_v = np.mean(v_matrix[:max(1, perturb_step), :], axis=0)
    
    # Compute velocity deviations from baseline
    v_deviations = v_matrix - baseline_v[np.newaxis, :]
    
    # Find peak absolute deviation after perturbation for each vehicle
    post_perturb_deviations = v_deviations[perturb_step:, :]
    peak_deviations = np.max(np.abs(post_perturb_deviations), axis=0)
    
    # Compute amplification ratio relative to perturbed vehicle
    leader_peak = peak_deviations[perturb_vehicle_id]
    if leader_peak > 0.01:  # Avoid division by very small numbers
        amplification_ratios = peak_deviations / leader_peak
    else:
        amplification_ratios = np.ones(N_vehicles)
    
    # Compute spacing variance over time (after perturbation)
    # Handle ring wraparound and vehicle length
    spacing_variance = []
    for step in range(perturb_step, n_steps):
        if len(micro_data[step]) == 2:
            x_list, _ = micro_data[step]
        else:
            _, x_list, _ = micro_data[step]  # id_list, x_list, v_list
        
        spacings = []
        for i in range(N_vehicles):
            next_i = (i + 1) % N_vehicles
            # Handle ring wraparound
            dx = x_list[next_i] - x_list[i]
            if dx < 0:  # Leader wrapped around
                dx += L
            # Bumper-to-bumper spacing (net gap)
            spacing = dx - vehicle_length
            spacings.append(max(spacing, 0.0))  # Avoid negative spacings
        spacing_variance.append(np.var(spacings))
    spacing_variance = np.array(spacing_variance)
    
    # Compute velocity standard deviation over time for each vehicle
    velocity_std = np.std(v_matrix[perturb_step:, :], axis=0)
    
    return amplification_ratios, spacing_variance, velocity_std


def analyze_string_stability(micro_data, perturb_vehicle_id, t_perturb, dt, N_vehicles):
    """
    Comprehensive string stability analysis.
    
    Determines whether the system is string-stable by checking if velocity
    fluctuations decay or amplify downstream.
    
    Parameters:
    -----------
    micro_data : list of tuples
        Simulation micro data
    perturb_vehicle_id : int
        ID of perturbed vehicle
    t_perturb : float
        Perturbation time (s)
    dt : float
        Timestep (s)
    N_vehicles : int
        Number of vehicles
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'is_stable': bool, True if string-stable
        - 'amplification_ratios': array of amplification ratios
        - 'max_amplification': float, maximum amplification
        - 'spacing_variance': array of spacing variance over time
        - 'velocity_std': array of velocity std for each vehicle
        - 'stability_metric': float, mean amplification downstream
    """
    amp_ratios, spacing_var, vel_std = compute_velocity_amplification(
        micro_data, perturb_vehicle_id, t_perturb, dt, N_vehicles
    )
    
    # Determine string stability: amplification should decay downstream
    # Focus on vehicles downstream of the perturbed one
    downstream_start = (perturb_vehicle_id + 1) % N_vehicles
    downstream_amplifications = []
    
    for offset in range(1, min(5, N_vehicles)):
        idx = (perturb_vehicle_id + offset) % N_vehicles
        downstream_amplifications.append(amp_ratios[idx])
    
    if len(downstream_amplifications) > 0:
        max_amplification = np.max(downstream_amplifications)
        mean_amplification = np.mean(downstream_amplifications)
    else:
        max_amplification = amp_ratios[perturb_vehicle_id]
        mean_amplification = amp_ratios[perturb_vehicle_id]
    
    # String-stable if amplification <= 1.0 (perturbations decay)
    is_stable = max_amplification <= 1.1  # Small tolerance for numerical error
    
    return {
        'is_stable': is_stable,
        'amplification_ratios': amp_ratios,
        'max_amplification': max_amplification,
        'spacing_variance': spacing_var,
        'velocity_std': vel_std,
        'stability_metric': mean_amplification
    }


def save_string_stability_results(results, output_path):
    """
    Save string stability analysis results to file.
    
    Parameters:
    -----------
    results : dict
        Results from analyze_string_stability()
    output_path : str
        Path to save results (PyTorch format)
    """
    torch.save(results, output_path)


def load_string_stability_results(input_path):
    """
    Load string stability results from file.
    
    Parameters:
    -----------
    input_path : str
        Path to load results from
    
    Returns:
    --------
    dict
        String stability analysis results
    """
    return torch.load(input_path, map_location='cpu')
