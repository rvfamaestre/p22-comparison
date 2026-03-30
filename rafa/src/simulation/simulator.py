# -*- coding: utf-8 -*-
# -------------------------------------------------------------
# File: src/simulation/simulator.py

# -------------------------------------------------------------

from src.vehicles.human_vehicle import HumanVehicle
from src.vehicles.stochastic_human_vehicle import StochasticHumanVehicle
from src.vehicles.unstable_human_vehicle import UnstableHumanVehicle
from src.vehicles.cav_vehicle import CAVVehicle
from src.mesoscopic.meso_adapter import (
    MesoAdapter,
    MesoConfig,
    CavGains,
    get_M_leaders_ring,
)
from src.agents.rl_types import RLConfig

# ARZ macro teacher is optional in some experiment modes.
try:
    from src.macro.arz_solver import step_arz, validate_arz_params

    _ARZ_SOLVER_AVAILABLE = True
except ModuleNotFoundError:
    step_arz = None
    validate_arz_params = None
    _ARZ_SOLVER_AVAILABLE = False

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Simulator:
    def __init__(self, env, macro_gen, logger, dt, T, config=None, live_viz=None):
        assert dt > 0, "dt must be positive"
        self.env = env
        self.macro_gen = macro_gen
        self.logger = logger
        self.dt = dt
        self.config = config if config is not None else {}
        self.run_forever = bool(self.config.get("run_forever", False))
        self.steps = None if self.run_forever else int(T / dt)
        self.collision_count = 0
        self.live_viz = live_viz
        self.viz_update_interval = self.config.get("viz_update_interval", 10)
        self.step_log_interval = int(self.config.get("step_log_interval", 10))

        # Macro teacher (ARZ PDE solver) configuration
        self.macro_teacher_enabled = self.config.get("macro_teacher", "none") == "arz"
        self.save_macro_dataset = self.config.get("save_macro_dataset", False)
        self.macro_predictor = self.config.get(
            "macro_predictor", "pde"
        )  # 'pde' or 'surrogate'

        if self.macro_teacher_enabled or self.save_macro_dataset:
            if not _ARZ_SOLVER_AVAILABLE:
                raise ModuleNotFoundError(
                    "ARZ mode requested but src.macro.arz_solver is missing. "
                    "Set 'macro_teacher: none' and 'save_macro_dataset: false', "
                    "or restore src/macro/arz_solver.py."
                )

            # Validate and store ARZ parameters
            self.arz_params = self.config.get(
                "arz_params", {"v_max": 30.0, "rho_jam": 0.2, "gamma": 1.0}
            )
            validate_arz_params(self.arz_params)

            # Get spatial resolution from macro generator
            if self.macro_gen is not None and hasattr(self.macro_gen, "x_grid"):
                self.dx_macro = (
                    self.macro_gen.x_grid[1] - self.macro_gen.x_grid[0]
                    if len(self.macro_gen.x_grid) > 1
                    else 1.0
                )
            else:
                self.dx_macro = self.config.get("dx", 1.0)

            print(f"[Simulator] Macro teacher: ARZ PDE solver enabled")
            print(
                f"            ARZ params: v_max={self.arz_params['v_max']}, rho_jam={self.arz_params['rho_jam']}, gamma={self.arz_params['gamma']}"
            )
            print(
                f"            dx={self.dx_macro:.2f}m, save_dataset={self.save_macro_dataset}"
            )
            print(f"            Predictor mode: {self.macro_predictor}")

        # Step counter and time tracking
        self.current_step = 0
        self.current_time = 0.0

        # Collision clamp tracking (for string stability validation)
        self.collision_clamp_count = 0
        self.collision_clamp_events = []  # List of (time, vehicle_id)
        self.perturbation_target_vehicle_actual = None
        self.perturbation_applied = False

        # Mesoscopic adaptation layer (human-inspired CAV control)
        self.meso_enabled = self.config.get("mesoscopic", {}).get("enabled", False)

        if self.meso_enabled:
            meso_cfg_dict = self.config.get("mesoscopic", {})
            self.meso_config = MesoConfig(
                M=meso_cfg_dict.get("M", 8),
                lambda_rho=meso_cfg_dict.get("lambda_rho", 0.8),
                gamma=meso_cfg_dict.get("gamma", 0.5),
                alpha_min=meso_cfg_dict.get("alpha_min", 0.7),
                alpha_max=meso_cfg_dict.get("alpha_max", 2.0),
                enable_gain_scheduling=meso_cfg_dict.get(
                    "enable_gain_scheduling", True
                ),
                enable_danger_mode=meso_cfg_dict.get("enable_danger_mode", False),
                rho_crit=meso_cfg_dict.get("rho_crit", 0.6),
                sigma_v_min_threshold=meso_cfg_dict.get("sigma_v_min_threshold", 0.2),
                v_eps_sigma=meso_cfg_dict.get("v_eps_sigma", 0.5),
                max_alpha_rate=meso_cfg_dict.get("max_alpha_rate", 0.2),
                sigma_v_ema_lambda=meso_cfg_dict.get("sigma_v_ema_lambda", 0.9),
                enable_k_f_adaptation=meso_cfg_dict.get("enable_k_f_adaptation", False),
                psi_deadband=meso_cfg_dict.get("psi_deadband", 0.5),
                adaptation_mode=meso_cfg_dict.get("adaptation_mode", "highway"),
            )

            v_max = self.config.get("acc_params", {}).get("v_max", 20.0)
            self.meso_adapter = MesoAdapter(self.meso_config, v_max)

            print(f"[Simulator] Mesoscopic adaptation ENABLED")
            print(
                f"            M={self.meso_config.M}, lambda={self.meso_config.lambda_rho}, gamma={self.meso_config.gamma}"
            )
            print(
                f"            alpha in [{self.meso_config.alpha_min}, {self.meso_config.alpha_max}]"
            )
            print(
                f"            Gain scheduling: {self.meso_config.enable_gain_scheduling}"
            )
        else:
            self.meso_adapter = None
            print(f"[Simulator] Mesoscopic adaptation DISABLED (baseline PD-CTH only)")

        # RL interface: per-CAV residual alpha overrides (set externally before step())
        self._rl_delta_alphas = {}  # {cav_id: float}
        self._rl_alpha_rules = {}  # stores alpha_rule for each CAV (for reward/logging)
        rl_cfg_dict = self.config.get("rl", {})
        self.rl_mode = rl_cfg_dict.get("rl_mode", "off")  # "off" | "rule" | "residual"
        self.rl_config = RLConfig(
            delta_alpha_max=rl_cfg_dict.get("delta_alpha_max", 0.3),
            alpha_min=rl_cfg_dict.get("alpha_min", 0.7),
            alpha_max=rl_cfg_dict.get("alpha_max", 2.0),
        )

        # Warm-up phase to prevent violent transients from tight initial conditions
        self.warmup_duration = 10.0  # seconds (extended for recovery)
        self.warmup_accel_limit = (
            1.0  # m/s^2 (moderate - allows recovery while preventing violence)
        )

        # CRITICAL: Initialize ring length in all vehicles for collision prevention
        for v in self.env.vehicles:
            v.L = self.env.L

        # Keep state observation-ready even before the first integration step.
        self.refresh_topology()

        print(
            f"[Simulator] Dynamic topology enabled: Leaders reassigned every timestep"
        )
        print(
            f"[Simulator] Warm-up phase: {self.warmup_duration}s with |a| <= {self.warmup_accel_limit} m/s²"
        )
        print(f"[Simulator] Collision prevention active: MIN_GAP = 0.3m")

    def refresh_topology(self):
        """Sort vehicles by current position and refresh leader links."""
        vehicles = self.env.vehicles
        vehicles.sort(key=lambda v: v.x)
        num_vehicles = len(vehicles)
        for idx, vehicle in enumerate(vehicles):
            vehicle.leader = vehicles[(idx + 1) % num_vehicles]

    def update_logger_metadata(self):
        """Persist run-level diagnostics alongside the trajectory files."""
        if not hasattr(self.logger, "metadata") or self.logger.metadata is None:
            return

        self.logger.metadata.update(
            {
                "steps_completed": int(self.current_step),
                "elapsed_time_s": float(self.current_time),
                "collision_count": int(self.collision_count),
                "collision_clamp_count": int(self.collision_clamp_count),
                "string_stability_valid": bool(self.collision_clamp_count == 0),
                "perturbation_applied": bool(self.perturbation_applied),
                "perturbation_target_vehicle_actual": (
                    int(self.perturbation_target_vehicle_actual)
                    if self.perturbation_target_vehicle_actual is not None
                    else None
                ),
            }
        )

    # ---------------------------------------------------------
    # ONE SIMULATION STEP
    # ---------------------------------------------------------
    def step(self):
        """
        ONE SIMULATION TIMESTEP

        Order of operations (CRITICAL - DO NOT REORDER):
        0. SORT vehicles by position and REASSIGN leaders (DYNAMIC)
        1. Compute state information (gaps, velocities, leader accelerations)
        2. Compute accelerations (IDM or CACC with feedforward)
        3. Update velocities (Euler integration)
        4. Update positions (Euler integration)
        5. Apply wraparound (periodic boundary)
        6. Collision detection (warning only)
        7. Log data
        """

        if self.step_log_interval > 0 and self.current_step % self.step_log_interval == 0:
            print(f"[Step {self.current_step}] t={self.current_time:.2f}s", flush=True)

        L = self.env.L
        vehicles = self.env.vehicles
        N = len(vehicles)

        # ===== PERTURBATION INJECTION (Uniform Init Mode) =====
        # Apply one-time velocity perturbation to trigger wave formation
        # FIXED (Feb 12, 2026): Use vehicle ID lookup, not list index after sorting
        if (
            self.config.get("initial_conditions") == "uniform"
            and self.config.get("perturbation_enabled", False)
            and self.current_time >= self.config.get("perturbation_time", 10.0)
            and not hasattr(self, "_perturbation_applied")
        ):
            target_id = self.config.get("perturbation_vehicle", 0)
            if target_id == -1:  # Random vehicle
                target_id = np.random.randint(0, N)

            delta_v = self.config.get("perturbation_delta_v", -2.0)

            # Find vehicle by ID (critical: don't use list index after sorting!)
            target_vehicle = next((v for v in vehicles if v.id == target_id), None)

            if target_vehicle is not None:
                old_v = target_vehicle.v
                target_vehicle.v += delta_v
                target_vehicle.v = max(target_vehicle.v, 0.0)
                self.perturbation_target_vehicle_actual = int(target_id)
                self.perturbation_applied = True
                print(
                    f"[Perturbation] VEHICLE ID {target_id}: {old_v:.2f} -> {target_vehicle.v:.2f} m/s (dv={delta_v:.1f}) at t={self.current_time:.1f}s"
                )
                self._perturbation_applied = True
            else:
                self.perturbation_target_vehicle_actual = int(target_id)
                self.perturbation_applied = False
                print(f"[Perturbation] ERROR: Vehicle ID {target_id} not found!")
                self._perturbation_applied = True

        # ===== STEP 0: SORT AND REASSIGN LEADERS (CRITICAL FIX) =====
        self.refresh_topology()

        # ===== STEP 1: Compute State Information =====
        state_info = []
        for v in vehicles:
            # Gap computation (handles wraparound correctly)
            s = v.compute_gap(L)
            # Don't clamp gap - let models handle small gaps properly

            # Closing rate: CORRECTED SIGN CONVENTION
            # For CACC: dv = v_leader - v_self (positive = opening gap)
            dv_cacc = v.leader.v - v.v

            state_info.append((v, s, dv_cacc))

        # ===== STEP 2: Compute Accelerations (ALL BEFORE UPDATE) =====
        # Two-pass acceleration computation for true zero-delay feedforward
        # Pass 1: Use previous timestep's leader accelerations
        # Pass 2: Update with current timestep's leader accelerations (iterative convergence)
        # This is essential for string stability in CACC platoons (Ploeg et al. 2014)

        # Get noise warmup time from config (suppress noise initially for uniform mode)
        noise_warmup = self.config.get("noise_warmup_time", 0.0)

        # Initialize with previous accelerations (for feedforward)
        accelerations = [
            v.acceleration if hasattr(v, "acceleration") else 0.0 for v in vehicles
        ]

        # SINGLE-PASS with initialization from previous timestep
        # Ring topology creates circular dependency - perfect zero-delay impossible
        # But with good initialization, single pass gives ~1-step delay which is acceptable
        new_accelerations = []

        for i, (v, s, dv) in enumerate(state_info):
            if isinstance(
                v, (HumanVehicle, StochasticHumanVehicle, UnstableHumanVehicle)
            ):
                # IDM uses v_self - v_leader (positive = approaching)
                dv_idm = v.v - v.leader.v
                acc = v.compute_idm_acc(s, v.v, dv_idm)

            elif isinstance(v, CAVVehicle):
                # ===== LEADER ACCELERATION (FEEDFORWARD) =====
                # Use leader's acceleration from previous timestep
                leader_idx = (i + 1) % len(vehicles)
                a_leader = accelerations[leader_idx]  # From previous timestep

                # ===== MESOSCOPIC ADAPTATION (if enabled) =====
                if self.meso_adapter is not None:
                    # Get velocities of M vehicles AHEAD (leaders)
                    # CORRECTED (Feb 2026): Forward-looking statistics to match physics
                    upstream_vels = get_M_leaders_ring(
                        vehicles, i, self.meso_config.M, L
                    )

                    # Compute alpha and diagnostics
                    alpha, meso_diag = self.meso_adapter.compute_alpha(
                        v.id, v.v, upstream_vels
                    )

                    # Baseline gains
                    baseline_gains = CavGains(k_s=v.kp, k_v=v.kd, k_v0=v.kv0, h_c=v.hc)

                    # Check danger condition (optional)
                    danger = False
                    if self.meso_config.enable_danger_mode:
                        s_emg = v.d0 * 1.5  # 1.5x standstill = emergency threshold
                        closing_rate = v.v - v.leader.v
                        danger = self.meso_adapter.check_danger_condition(
                            s, v.v, closing_rate, s_emg
                        )

                        # Also check stress level
                        if v.id in self.meso_adapter.rho:
                            danger = danger or (
                                abs(self.meso_adapter.rho[v.id])
                                > self.meso_config.rho_crit
                            )

                    # Adapt gains (NOW INCLUDES ADAPTIVE FEEDFORWARD)
                    k_s_p, k_v_p, k_v0_p, h_c_p, k_f_p, adapt_diag = (
                        self.meso_adapter.adapt_cav_policy(
                            v.id, baseline_gains, alpha, danger, k_f_baseline=v.kf
                        )
                    )

                    # RL residual injection (if active)
                    alpha_rule = alpha  # preserve for logging
                    self._rl_alpha_rules[v.id] = alpha_rule
                    if self.rl_mode == "residual" and v.id in self._rl_delta_alphas:
                        da = self._rl_delta_alphas[v.id]
                        alpha = float(
                            np.clip(
                                alpha_rule + da,
                                self.rl_config.alpha_min,
                                self.rl_config.alpha_max,
                            )
                        )
                        # Re-adapt gains with corrected alpha
                        k_s_p, k_v_p, k_v0_p, h_c_p, k_f_p, adapt_diag = (
                            self.meso_adapter.adapt_cav_policy(
                                v.id, baseline_gains, alpha, danger, k_f_baseline=v.kf
                            )
                        )
                        # NOTE: Do NOT update meso_adapter.alpha_prev here.
                        # The rate-limiter must track the rule-based alpha only;
                        # otherwise the RL residual accumulates through the
                        # rate limiter creating a positive-feedback ratchet
                        # that drives alpha to saturation.

                    # Store adapted parameters for logging
                    v._meso_alpha = alpha
                    v._meso_h_c = h_c_p
                    v._meso_k_s = k_s_p
                    v._meso_k_v = k_v_p
                    v._meso_k_v0 = k_v0_p
                    v._meso_k_f = k_f_p  # NEW: store adapted feedforward
                    v._meso_diagnostics = {**meso_diag, **adapt_diag}

                else:
                    # No mesoscopic adaptation - use baseline
                    k_s_p = v.kp
                    k_v_p = v.kd
                    k_v0_p = v.kv0
                    h_c_p = v.hc
                    k_f_p = v.kf  # Use baseline feedforward
                    v._meso_alpha = 1.0
                    self._rl_alpha_rules[v.id] = 1.0

                # Compute CACC acceleration with adapted parameters
                # Temporarily override vehicle parameters (INCLUDING FEEDFORWARD)
                kp_orig, kd_orig, kv0_orig, hc_orig, kf_orig = (
                    v.kp,
                    v.kd,
                    v.kv0,
                    v.hc,
                    v.kf,
                )
                v.kp, v.kd, v.kv0, v.hc, v.kf = k_s_p, k_v_p, k_v0_p, h_c_p, k_f_p

                acc = v.compute_cacc_acc(s, v.v, dv, a_leader)

                # Restore original parameters (adapter is stateless per-timestep)
                v.kp, v.kd, v.kv0, v.hc, v.kf = (
                    kp_orig,
                    kd_orig,
                    kv0_orig,
                    hc_orig,
                    kf_orig,
                )

            # Warm-up phase: limit accelerations to prevent violent transients
            if self.current_time < self.warmup_duration:
                acc = max(-self.warmup_accel_limit, min(acc, self.warmup_accel_limit))

            new_accelerations.append(acc)

        # Store final accelerations
        accelerations = new_accelerations

        # ===== STEP 2B: Store ALL Accelerations =====
        # Store AFTER all computed - enables zero-delay feedforward for CAV platoons
        for v, acc in zip(vehicles, accelerations):
            v.acceleration = acc

        # ===== STEP 3: Update Velocities =====
        for v in vehicles:
            if isinstance(
                v, (HumanVehicle, StochasticHumanVehicle, UnstableHumanVehicle)
            ):
                # Pass current_time and warmup_time for noise suppression
                v.update_velocity(
                    v.acceleration, self.dt, self.current_time, noise_warmup
                )
            else:
                v.update_velocity(v.acceleration, self.dt)

        # ===== STEP 4: Update Positions =====
        # Track collision clamp events (invalidate string stability if any occur)
        for v in vehicles:
            clamp_triggered = v.update_position(self.dt)
            if clamp_triggered:
                self.collision_clamp_count += 1
                self.collision_clamp_events.append((self.current_time, v.id))
                if len(self.collision_clamp_events) <= 3:  # Only print first 3
                    print(
                        f"[COLLISION CLAMP] Vehicle {v.id} emergency brake at t={self.current_time:.2f}s (gap < MIN_GAP)"
                    )

        # ===== STEP 5: Apply Wraparound =====
        for v in vehicles:
            v.apply_wraparound(L)

        # ===== STEP 6: Collision Detection =====
        for v in vehicles:
            s = v.compute_gap(L)
            if s < 0.5:
                print(
                    f"WARNING: Vehicle {v.id} collision risk (gap={s:.2f}m) at t={self.current_time:.1f}s"
                )
                self.collision_count += 1

        # ===== STEP 7: Log Microscopic Data =====
        self.logger.log_micro(vehicles, self.current_time, self.current_step)

        # ===== STEP 8: Compute & Log Macroscopic Fields =====
        if self.macro_gen is not None:
            # Compute macro state at time t from microscopic state via SPH
            rho_t, u_t = self.macro_gen.compute_macrofields(vehicles)

            if rho_t is not None and u_t is not None:
                # Always log to macro.csv (standard logging)
                self.logger.log_macro(rho_t, u_t, self.current_time, self.current_step)

                # Additionally: Generate macro teacher dataset for GNN training
                if self.save_macro_dataset:
                    try:
                        # Use ARZ PDE solver to predict macro state at t+dt (teacher signal)
                        rho_pde, u_pde = step_arz(
                            rho_t, u_t, self.dt, self.dx_macro, self.arz_params
                        )

                        # Log the input-target pair for GNN training
                        self.logger.log_macro_teacher(
                            rho_t,
                            u_t,
                            rho_pde,
                            u_pde,
                            self.current_time,
                            self.current_step,
                        )
                    except Exception as e:
                        print(
                            f"WARNING: ARZ solver failed at t={self.current_time:.1f}s: {e}"
                        )

        # Preserve an observation-ready state for live controllers / HUDs.
        self.refresh_topology()

        self.current_step += 1
        self.current_time += self.dt

    # ---------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------
    def run(self):
        """Main simulation loop."""
        if self.run_forever:
            print("[Simulator] Running in continuous mode until externally stopped...")
        else:
            print(
                f"[Simulator] Running {self.steps} timesteps (T={self.steps * self.dt:.1f}s)..."
            )
        if self.live_viz is not None:
            print(
                f"[Simulator] Live visualization ENABLED (updating every {self.viz_update_interval} steps)"
            )

        if self.run_forever:
            step = 0
            while True:
                self.step()
                if self.live_viz is not None and step % self.viz_update_interval == 0:
                    self.live_viz.update(self.env.vehicles, self.current_time, step)
                step += 1
        else:
            for step in range(self.steps):
                self.step()

                # Update live visualization if enabled
                if self.live_viz is not None and step % self.viz_update_interval == 0:
                    self.live_viz.update(self.env.vehicles, self.current_time, step)

        # Close live visualization
        if self.live_viz is not None:
            self.live_viz.close()

        self.update_logger_metadata()
        self.logger.save()

        # Report collisions
        if self.collision_count > 0:
            print(
                f"\nWARNING: {self.collision_count} collision warnings during simulation"
            )
        else:
            print("\nSimulation completed with ZERO collisions")

        # Report collision clamps (CRITICAL for string stability validation!)
        if self.collision_clamp_count > 0:
            print(
                f"\n CRITICAL: {self.collision_clamp_count} COLLISION CLAMP events (emergency v=0 braking)"
            )
            print(
                f"     These are IMPULSE DISTURBANCES that INVALIDATE string stability analysis!"
            )
            print(f"     First 3 events:")
            for i, (t, vid) in enumerate(self.collision_clamp_events[:3]):
                print(f"       {i + 1}. Vehicle {vid} at t={t:.2f}s")
            if self.collision_clamp_count > 3:
                print(f"       ... and {self.collision_clamp_count - 3} more")
            print(f"     RECOMMENDATION: Discard this run or tune to prevent clamps.")
        else:
            print(f" ZERO collision clamps - string stability analysis is VALID")

    # ---------------------------------------------------------
    # RL INTERFACE HELPERS
    # ---------------------------------------------------------
    def set_rl_actions(self, delta_alphas):
        """
        Set per-CAV residual headway corrections for the next step().

        Parameters
        ----------
        delta_alphas : dict[int, float]
            Keyed by CAV vehicle id.
        """
        self._rl_delta_alphas = dict(delta_alphas)

    def get_rl_alpha_rules(self):
        """Return the last computed rule-based alpha for each CAV."""
        return dict(self._rl_alpha_rules)

    @property
    def done(self):
        """True when the simulation has finished all timesteps."""
        if self.run_forever:
            return False
        return self.current_step >= self.steps
