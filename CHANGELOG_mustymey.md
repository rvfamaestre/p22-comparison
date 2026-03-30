# Changelog — Mustafa Kamal (mustymey)

All changes from the initial repository state, attributable to Mustafa's work.
Rafael's contributions (GPU vectorized backend, HPC pipeline, GUI, independent test suite) are excluded.

---

## Core Simulation Framework

### `src/simulation/simulator.py`
- Full CPU-based ring-road traffic simulator with microscopic stepping
- Implements IDM dynamics for HDVs and CACC/CTH dynamics for CAVs
- Warm-up phase logic: suppresses noise and limits acceleration during the initial transient period
- Perturbation injection system: applies velocity drops to designated vehicles at configurable times
- Collision detection and collision-clamp counting for admissibility gating
- Macro/micro data logging at each timestep (positions, velocities, accelerations, gaps, rewards)

### `src/simulation/scenario_manager.py`
- Scenario configuration and vehicle initialization
- Random initialization mode: stochastic placement of vehicles with configurable spacing/velocity distributions
- HDV/CAV role assignment at configurable penetration rates (25%, 50%, 75%)
- Integration with mesoscopic adapter for CAV headway adaptation

### `src/simulation/transition_scenario_manager.py`
- Extended scenario manager supporting dynamic type-swap zones (HDV↔CAV transitions during simulation)

---

## Vehicle Models

### `src/vehicles/vehicle.py`
- Base vehicle class: position, velocity, acceleration, and vehicle-length properties
- Ring-road wrapping logic (modular arithmetic on positions)

### `src/vehicles/human_vehicle.py`
- Deterministic IDM implementation for human-driven vehicles

### `src/vehicles/stochastic_human_vehicle.py`
- Stochastic IDM (SIDM): adds Gaussian acceleration noise to the deterministic IDM, enabling realistic traffic wave generation near the string-stability threshold

### `src/vehicles/unstable_human_vehicle.py`
- IDM variant with deliberately aggressive/unstable parameters for stress-testing the control architecture

### `src/vehicles/cav_vehicle.py`
- CAV with CACC/CTH longitudinal controller
- Accepts adaptive time-headway parameter `α` from the mesoscopic layer
- Implements leader-acceleration feedforward term

---

## Mesoscopic Adaptation Layer

### `src/mesoscopic/meso_adapter.py`
- Core adapter: computes upstream macroscopic traffic state (mean speed, speed variance) per CAV
- Rule-based headway scaling `α_rule = sat(1 + ρ)` with first-order filtering
- Interfaces with the RL layer to obtain residual correction `Δα`

### `src/mesoscopic/rl_layer.py`
- RL policy wrapper: loads trained PPO/SAC policy networks, runs inference at each CAV control step
- Constructs the 8-dimensional observation vector from micro + macro state

### `src/mesoscopic/rl_common.py`
- Shared RL utilities: policy network architecture (MLP), action clipping, observation normalization

### `src/mesoscopic/rl_training.py`
- PPO training loop: rollout collection, advantage estimation (GAE), clipped policy gradient updates
- Episode-based training against the CPU simulator

### `src/mesoscopic/sac_training.py`
- SAC training loop: off-policy training with replay buffer, entropy-regularized actor-critic updates
- Soft Q-function with double-critic architecture

### `src/mesoscopic/rl_rewards.py`
- Multi-objective reward function: safety, spacing regulation, efficiency, comfort (jerk), string-stability, and traffic-smoothness terms
- Configurable weighting via YAML config

---

## Macroscopic Observer

### `src/macro/macrofield_generator.py`
- Generates continuous macroscopic traffic fields (density, velocity) from discrete vehicle positions using SPH interpolation
- Provides upstream perception window for each CAV

### `src/macro/sph.py`
- Smoothed Particle Hydrodynamics kernel for ring-road density/velocity field estimation

---

## String Stability Analysis

### `src/control/string_stability.py`
- Post-simulation string-stability metric computation
- Amplification-ratio calculation: measures maximum downstream-to-source speed-error ratio after perturbation
- Pairwise speed-gain analysis across all vehicle pairs
- Admissibility gating: validates that string-stability measurement is scientifically meaningful (zero pre-perturbation clamps, correctly applied perturbation, sufficient baseline window)

### `src/utils/string_stability_metrics.py`
- Helper utilities for string-stability extraction from simulation logs

---

## Experiment Orchestration

### `run_experiments.py`
- Multi-method comparison runner: orchestrates training + evaluation for No-RL / PPO / SAC across configurable penetration rates
- Automatic output directory management, config merging, and log aggregation
- Calls training, simulation, and report-building stages in sequence

### `run_simulation.py`
- Standalone simulation entry point: loads config, initializes scenario, runs simulator, saves results

### `scripts/build_rl_comparison_report.py`
- Post-experiment analysis: reads per-method simulation outputs, computes comparison metrics
- Generates:
  - `comparison_metrics.csv` / `.xlsx` — full evaluation table
  - `training_summary.csv` — training convergence summary
  - `experiment_report.md` — human-readable report
  - Bar plots and training-curve overlays for all metrics (speed variance, min gap, collisions, amplification ratio, etc.)

### `scripts/run_method_compare_ep800.sh`
- Shell launcher for the full 800-episode comparison campaign on the DCE cluster

---

## Configuration System

### `config/` directory
- Comprehensive YAML-based configuration for all simulation parameters:
  - IDM/CTH/CACC controller gains and physical constants
  - Penetration rates, vehicle counts, road length
  - Training hyperparameters (learning rate, batch size, entropy coefficient, etc.)
  - Perturbation protocol settings (timing, magnitude, target vehicle)
  - Early-stopping configuration (patience, EMA smoothing)
- Comparison-specific configs for each method × penetration-rate combination
- Rerun configs with updated episode counts and random-seed settings

---

## Utilities

### `src/utils/config.py`
- YAML config loader with defaults and deep-merge support

### `src/utils/config_manager.py`
- Config path resolution and validation

### `src/utils/logger.py`
- Structured logging for simulation runs (macro/micro CSV + `.pt` tensor dumps, metadata JSON, summary text)

### `src/utils/random_utils.py`
- Seed management for reproducibility across NumPy, PyTorch, and Python's random module

---

## Mid-Report Contributions

### `mid-report/`
- Co-authored the mid-report LaTeX document
- Sections contributed: Traffic Environment (IDM, SIDM, CAV/CTH, heterogeneous traffic), Reinforcement Learning (MDP formulation, reward design, PPO, SAC), System Architecture (layered control framework)

---

## Summary

Mustafa built the complete CPU simulation and training pipeline, including the microsopic vehicle models, the mesoscopic adaptation architecture, the RL training loops (both PPO and SAC), the reward function design, the string-stability analysis tools, and the experiment orchestration and reporting system. This constitutes the core scientific codebase of the project.
