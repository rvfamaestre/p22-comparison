# -*- coding: utf-8 -*-
# -------------------------------------------------------------
# File: src/simulation/simulator.py
# -------------------------------------------------------------

from src.vehicles.human_vehicle import HumanVehicle
from src.vehicles.stochastic_human_vehicle import StochasticHumanVehicle
from src.vehicles.unstable_human_vehicle import UnstableHumanVehicle
from src.vehicles.cav_vehicle import CAVVehicle
from src.mesoscopic.meso_adapter import MesoAdapter, MesoConfig, CavGains, get_M_leaders_ring
from src.mesoscopic.rl_layer import ResidualHeadwayRLLayer, RLConfig
from src.utils.string_stability_metrics import STRING_STABILITY_BASELINE_WINDOW_STEPS

import numpy as np


class Simulator:
    def __init__(self, env, macro_gen, logger, dt, T, config=None, live_viz=None):
        assert dt > 0, "dt must be positive"
        self.env = env
        self.macro_gen = macro_gen
        self.logger = logger
        self.dt = dt
        self.steps = int(T / dt)
        self.collision_count = 0
        self.config = config if config is not None else {}
        self.live_viz = live_viz
        self.viz_update_interval = config.get('viz_update_interval', 10) if config else 10
        self.quiet = self.config.get("quiet", False)

        # Step counter and time tracking
        self.current_step = 0
        self.current_time = 0.0

        # Collision clamp tracking (for string stability validation)
        self.collision_clamp_count = 0
        self.collision_clamp_events = []  # List of (time, vehicle_id)
        self.perturbation_applied = False
        self.perturbation_applied_time = None
        self.perturbation_target_id = None

        # Mesoscopic adaptation layer (human-inspired CAV control)
        self.meso_enabled = self.config.get('mesoscopic', {}).get('enabled', False)

        if self.meso_enabled:
            meso_cfg_dict = self.config.get('mesoscopic', {})
            self.meso_config = MesoConfig(
                M=meso_cfg_dict.get('M', 8),
                lambda_rho=meso_cfg_dict.get('lambda_rho', 0.8),
                gamma=meso_cfg_dict.get('gamma', 0.5),
                alpha_min=meso_cfg_dict.get('alpha_min', 0.7),
                alpha_max=meso_cfg_dict.get('alpha_max', 2.0),
                enable_gain_scheduling=meso_cfg_dict.get('enable_gain_scheduling', True),
                enable_danger_mode=meso_cfg_dict.get('enable_danger_mode', False),
                rho_crit=meso_cfg_dict.get('rho_crit', 0.6),
                sigma_v_min_threshold=meso_cfg_dict.get('sigma_v_min_threshold', 0.2),
                v_eps_sigma=meso_cfg_dict.get('v_eps_sigma', 0.5),
                max_alpha_rate=meso_cfg_dict.get('max_alpha_rate', 0.2),
                sigma_v_ema_lambda=meso_cfg_dict.get('sigma_v_ema_lambda', 0.9),
                enable_k_f_adaptation=meso_cfg_dict.get('enable_k_f_adaptation', False),
                psi_deadband=meso_cfg_dict.get('psi_deadband', 0.5),
                adaptation_mode=meso_cfg_dict.get('adaptation_mode', 'highway')
            )

            v_max = self.config.get('acc_params', {}).get('v_max', 20.0)
            self.meso_adapter = MesoAdapter(self.meso_config, v_max)

            print(f"[Simulator] Mesoscopic adaptation ENABLED")
            print(
                f"            M={self.meso_config.M}, "
                f"lambda={self.meso_config.lambda_rho}, gamma={self.meso_config.gamma}"
            )
            print(
                f"            alpha in [{self.meso_config.alpha_min}, {self.meso_config.alpha_max}]"
            )
            print(f"            Gain scheduling: {self.meso_config.enable_gain_scheduling}")
        else:
            self.meso_adapter = None
            print(f"[Simulator] Mesoscopic adaptation DISABLED (baseline PD-CTH only)")

        # RL residual layer
        self.rl_enabled = self.config.get('rl_layer', {}).get('enabled', False)

        if self.rl_enabled:
            rl_cfg_dict = self.config.get('rl_layer', {})
            self.rl_config = RLConfig(
                enabled=rl_cfg_dict.get('enabled', False),
                mode=rl_cfg_dict.get('mode', 'inference'),
                algorithm=rl_cfg_dict.get('algorithm', 'ppo'),
                model_path=rl_cfg_dict.get('model_path', 'models/rl_policy.pt'),
                delta_alpha_max=rl_cfg_dict.get('delta_alpha_max', 0.1),
                alpha_min=rl_cfg_dict.get('alpha_min', 0.9),
                alpha_max=rl_cfg_dict.get('alpha_max', 1.6),
            )
            self.rl_layer = ResidualHeadwayRLLayer(self.rl_config)
            print(f"[Simulator] RL residual layer ENABLED")
            print(f"            algorithm={self.rl_config.algorithm}, mode={self.rl_config.mode}, model={self.rl_config.model_path}")
            print(
                f"            delta_alpha_max={self.rl_config.delta_alpha_max}, "
                f"alpha in [{self.rl_config.alpha_min}, {self.rl_config.alpha_max}]"
            )
        else:
            self.rl_layer = None
            print(f"[Simulator] RL residual layer DISABLED")

        # Warm-up phase to prevent violent transients from tight initial conditions.
        self.warmup_duration = float(self.config.get("warmup_duration", 10.0))
        self.warmup_accel_limit = float(self.config.get("warmup_accel_limit", 1.0))
        self.noise_warmup_time = float(self.config.get("noise_warmup_time", 0.0))
        self.perturbation_enabled = bool(self.config.get("perturbation_enabled", False))
        self.perturbation_time = (
            float(self.config.get("perturbation_time", 10.0))
            if self.perturbation_enabled
            else None
        )

        # CRITICAL: Initialize ring length in all vehicles for collision prevention
        for v in self.env.vehicles:
            v.L = self.env.L

        print(f"[Simulator] Dynamic topology enabled: Leaders reassigned every timestep")
        print(f"[Simulator] Warm-up phase: {self.warmup_duration}s with |a| <= {self.warmup_accel_limit} m/s²")
        print(f"[Simulator] Collision prevention active: MIN_GAP = 0.3m")
        if self.perturbation_enabled and self.perturbation_time is not None:
            protocol_ready_time = max(self.warmup_duration, self.noise_warmup_time)
            recommended_time = (
                protocol_ready_time
                + STRING_STABILITY_BASELINE_WINDOW_STEPS * self.dt
            )
            if self.perturbation_time <= protocol_ready_time:
                print(
                    "[Simulator] WARNING: perturbation_time is not after warm-up/noise "
                    f"release ({self.perturbation_time:.2f}s <= {protocol_ready_time:.2f}s)."
                )
            elif self.perturbation_time < recommended_time:
                print(
                    "[Simulator] WARNING: perturbation_time leaves less than one full "
                    "string-stability baseline window after warm-up "
                    f"({self.perturbation_time:.2f}s < {recommended_time:.2f}s)."
                )

    def _set_default_rl_diagnostics(self, vehicle, alpha_value):
        """Populate RL diagnostics when no learned residual is applied."""
        vehicle._rl_diagnostics = {
            "alpha_rule": alpha_value,
            "delta_alpha": 0.0,
            "alpha": alpha_value
        }
        vehicle._rl_alpha_rule = float(alpha_value)
        vehicle._rl_delta_alpha = 0.0
        vehicle._rl_alpha = float(alpha_value)

    def _append_rl_training_step(self, vehicle, rl_out, mu_v, sigma_v2, gap, a_leader):
        """Store one simulator step in the rollout history used by the trainer."""
        if not hasattr(vehicle, "_rl_history"):
            vehicle._rl_history = []

        vehicle._rl_history.append({
            "state": rl_out.get("state_vector", None),
            "action_raw": rl_out.get("delta_alpha_raw", 0.0),
            "action_applied": rl_out.get("delta_alpha", 0.0),
            "action_low": rl_out.get("action_low", -self.rl_layer.config.delta_alpha_max),
            "action_high": rl_out.get("action_high", self.rl_layer.config.delta_alpha_max),
            "log_prob": rl_out.get("log_prob", None),
            "value": rl_out.get("value", None),
            "v": float(vehicle.v),
            "s": float(gap),
            "mu_v": float(mu_v),
            "sigma_v2": float(sigma_v2),
            "alpha": float(rl_out.get("alpha", rl_out.get("alpha_rule", 1.0))),
            "a_prev": float(getattr(vehicle, "acceleration", 0.0)),
            "a_lead": float(a_leader),
        })

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

        if self.current_step % 10 == 0 and not self.quiet:
            print(f"[Step {self.current_step}] t={self.current_time:.2f}s", flush=True)

        L = self.env.L
        vehicles = self.env.vehicles
        N = len(vehicles)

        # ===== PERTURBATION INJECTION =====
        if (
            self.perturbation_enabled
            and self.perturbation_time is not None
            and self.current_time >= self.perturbation_time
            and not self.perturbation_applied
        ):
            target_id = self.config.get("perturbation_vehicle", 0)
            if target_id == -1:
                target_id = np.random.randint(0, N)

            delta_v = self.config.get("perturbation_delta_v", -2.0)
            target_vehicle = next((v for v in vehicles if v.id == target_id), None)

            if target_vehicle is not None:
                old_v = target_vehicle.v
                target_vehicle.v += delta_v
                target_vehicle.v = max(target_vehicle.v, 0.0)
                if not self.quiet:
                    print(
                        f"[Perturbation] VEHICLE ID {target_id}: "
                        f"{old_v:.2f} -> {target_vehicle.v:.2f} m/s "
                        f"(delta_v={delta_v:.1f}) at t={self.current_time:.1f}s"
                    )
                self.perturbation_applied = True
                self.perturbation_applied_time = float(self.current_time)
                self.perturbation_target_id = int(target_id)
            else:
                if not self.quiet:
                    print(f"[Perturbation] ERROR: Vehicle ID {target_id} not found!")
                self.perturbation_applied = True
                self.perturbation_applied_time = float(self.current_time)
                self.perturbation_target_id = int(target_id)

        # ===== STEP 0: SORT AND REASSIGN LEADERS =====
        vehicles.sort(key=lambda v: v.x)

        for i, vehicle in enumerate(vehicles):
            leader_idx = (i + 1) % N
            vehicle.leader = vehicles[leader_idx]

        # ===== STEP 1: Compute State Information =====
        state_info = []
        for v in vehicles:
            s = v.compute_gap(L)
            dv_cacc = v.leader.v - v.v
            state_info.append((v, s, dv_cacc))

        # ===== STEP 2: Compute Accelerations =====
        noise_warmup = self.noise_warmup_time

        # Use previous timestep accelerations for feedforward initialization
        accelerations = [v.acceleration if hasattr(v, 'acceleration') else 0.0 for v in vehicles]
        new_accelerations = []

        for i, (v, s, dv) in enumerate(state_info):
            if isinstance(v, (HumanVehicle, StochasticHumanVehicle, UnstableHumanVehicle)):
                # IDM uses v_self - v_leader (positive = approaching)
                dv_idm = v.v - v.leader.v
                acc = v.compute_idm_acc(s, v.v, dv_idm)

            elif isinstance(v, CAVVehicle):
                # ===== LEADER ACCELERATION (FEEDFORWARD) =====
                leader_idx = (i + 1) % len(vehicles)
                a_leader = accelerations[leader_idx]

                # ===== MESOSCOPIC ADAPTATION (if enabled) =====
                if self.meso_adapter is not None:
                    upstream_vels = get_M_leaders_ring(vehicles, i, self.meso_config.M, L)

                    # Rule-based alpha and diagnostics
                    alpha_rule, meso_diag = self.meso_adapter.compute_alpha(
                        v.id, v.v, upstream_vels
                    )

                    # RL residual on top of the rule-based mesoscopic alpha.
                    if self.rl_layer is not None and self.rl_layer.config.enabled:
                        mu_v = meso_diag.get("mu_v", 0.0)
                        sigma_v = meso_diag.get("sigma_v", 0.0)
                        sigma_v2 = sigma_v ** 2
                        speed_mismatch = v.v - mu_v

                        rl_out = self.rl_layer.compute_alpha(
                            cav_id=v.id,
                            alpha_rule=alpha_rule,
                            mu_v=mu_v,
                            sigma_v2=sigma_v2,
                            speed_mismatch=speed_mismatch,
                            v=v.v,
                            s=s,
                            delta_v=dv,
                            a_lead=a_leader
                        )

                        alpha = rl_out["alpha"]

                        v._rl_diagnostics = rl_out
                        v._rl_alpha_rule = float(rl_out.get("alpha_rule", alpha_rule))
                        v._rl_delta_alpha = float(rl_out.get("delta_alpha", 0.0))
                        v._rl_alpha = float(rl_out.get("alpha", alpha))

                        # In training mode the simulator acts as the rollout collector.
                        if self.rl_layer.config.mode == "train":
                            self._append_rl_training_step(v, rl_out, mu_v, sigma_v2, s, a_leader)
                    else:
                        alpha = alpha_rule
                        self._set_default_rl_diagnostics(v, alpha_rule)

                    baseline_gains = CavGains(
                        k_s=v.kp,
                        k_v=v.kd,
                        k_v0=v.kv0,
                        h_c=v.hc
                    )

                    danger = False
                    if self.meso_config.enable_danger_mode:
                        s_emg = v.d0 * 1.5
                        closing_rate = v.v - v.leader.v
                        danger = self.meso_adapter.check_danger_condition(
                            s, v.v, closing_rate, s_emg
                        )

                        if v.id in self.meso_adapter.rho:
                            danger = danger or (abs(self.meso_adapter.rho[v.id]) > self.meso_config.rho_crit)

                    k_s_p, k_v_p, k_v0_p, h_c_p, k_f_p, adapt_diag = self.meso_adapter.adapt_cav_policy(
                        v.id, baseline_gains, alpha, danger, k_f_baseline=v.kf
                    )

                    v._meso_alpha = alpha
                    v._meso_h_c = h_c_p
                    v._meso_k_s = k_s_p
                    v._meso_k_v = k_v_p
                    v._meso_k_v0 = k_v0_p
                    v._meso_k_f = k_f_p
                    v._meso_diagnostics = {**meso_diag, **adapt_diag}

                else:
                    k_s_p = v.kp
                    k_v_p = v.kd
                    k_v0_p = v.kv0
                    h_c_p = v.hc
                    k_f_p = v.kf

                    v._meso_alpha = 1.0
                    self._set_default_rl_diagnostics(v, 1.0)

                # Compute CACC acceleration with adapted parameters
                kp_orig, kd_orig, kv0_orig, hc_orig, kf_orig = v.kp, v.kd, v.kv0, v.hc, v.kf
                v.kp, v.kd, v.kv0, v.hc, v.kf = k_s_p, k_v_p, k_v0_p, h_c_p, k_f_p

                acc = v.compute_cacc_acc(s, v.v, dv, a_leader)

                # Restore original parameters
                v.kp, v.kd, v.kv0, v.hc, v.kf = kp_orig, kd_orig, kv0_orig, hc_orig, kf_orig

            else:
                raise TypeError(f"Unsupported vehicle type: {type(v).__name__}")

            # IMPORTANT: this applies to ALL vehicles
            if self.current_time < self.warmup_duration:
                acc = max(-self.warmup_accel_limit, min(acc, self.warmup_accel_limit))

            # Save EXECUTED acceleration for RL training (CAVs only)
            if (
                isinstance(v, CAVVehicle)
                and self.rl_layer is not None
                and self.rl_layer.config.enabled
                and self.rl_layer.config.mode == "train"
                and hasattr(v, "_rl_history")
                and len(v._rl_history) > 0
            ):
                v._rl_history[-1]["a"] = float(acc)

            # IMPORTANT: append for EVERY vehicle
            new_accelerations.append(acc)

        # Store final accelerations
        accelerations = new_accelerations

        # ===== STEP 2B: Store ALL Accelerations =====
        for v, acc in zip(vehicles, accelerations):
            v.acceleration = acc

        # ===== STEP 3: Update Velocities =====
        for v in vehicles:
            if isinstance(v, (HumanVehicle, StochasticHumanVehicle, UnstableHumanVehicle)):
                v.update_velocity(v.acceleration, self.dt, self.current_time, noise_warmup)
            else:
                v.update_velocity(v.acceleration, self.dt)

        # ===== STEP 4: Update Positions =====
        for v in vehicles:
            clamp_triggered = v.update_position(self.dt)
            if clamp_triggered:
                self.collision_clamp_count += 1
                self.collision_clamp_events.append((self.current_time, v.id))
                if len(self.collision_clamp_events) <= 3 and not self.quiet:
                    print(f"[COLLISION CLAMP] Vehicle {v.id} emergency brake at t={self.current_time:.2f}s (gap < MIN_GAP)")

        # ===== STEP 5: Apply Wraparound =====
        for v in vehicles:
            v.apply_wraparound(L)

        # ===== STEP 6: Collision Detection =====
        for v in vehicles:
            s = v.compute_gap(L)
            if s < 0.5:
                if not self.quiet:
                    print(f"WARNING: Vehicle {v.id} collision risk (gap={s:.2f}m) at t={self.current_time:.1f}s")
                self.collision_count += 1

        # ===== STEP 7: Log Microscopic Data =====
        self.logger.log_micro(vehicles, self.current_time, self.current_step)

        # ===== STEP 8: Compute & Log Macroscopic Fields =====
        if self.macro_gen is not None:
            rho_t, u_t = self.macro_gen.compute_macrofields(vehicles)

            if rho_t is not None and u_t is not None:
                self.logger.log_macro(rho_t, u_t, self.current_time, self.current_step)

        self.current_step += 1
        self.current_time += self.dt

    # ---------------------------------------------------------
    # MAIN LOOP
    # ---------------------------------------------------------
    def run(self):
        """Main simulation loop."""
        if not self.quiet:
            print(f"[Simulator] Running {self.steps} timesteps (T={self.steps*self.dt:.1f}s)...")
            if self.live_viz is not None:
                print(f"[Simulator] Live visualization ENABLED (updating every {self.viz_update_interval} steps)")

        for step in range(self.steps):
            self.step()

            if self.live_viz is not None and step % self.viz_update_interval == 0:
                self.live_viz.update(self.env.vehicles, self.current_time, step)

        if self.live_viz is not None:
            self.live_viz.close()

        # Save collision counts into metadata before logger.save()
        self.logger.metadata["collision_count"] = self.collision_count
        self.logger.metadata["collision_clamp_count"] = self.collision_clamp_count
        self.logger.metadata["noise_warmup_time"] = self.noise_warmup_time
        self.logger.metadata["warmup_duration"] = self.warmup_duration
        self.logger.metadata["warmup_accel_limit"] = self.warmup_accel_limit
        self.logger.metadata["perturbation_enabled"] = self.perturbation_enabled
        self.logger.metadata["perturbation_time"] = self.perturbation_time
        self.logger.metadata["perturbation_applied"] = bool(self.perturbation_applied)
        self.logger.metadata["perturbation_applied_time"] = self.perturbation_applied_time
        self.logger.metadata["perturbation_target_id"] = self.perturbation_target_id

        analysis_perturbation_time = (
            self.perturbation_applied_time
            if self.perturbation_applied_time is not None
            else self.perturbation_time
        )
        clamp_events = [
            {"time": float(t), "vehicle_id": int(vid)}
            for t, vid in self.collision_clamp_events
        ]
        if analysis_perturbation_time is None:
            pre_events = clamp_events
            post_events = []
        else:
            pre_events = [
                event for event in clamp_events
                if float(event["time"]) < float(analysis_perturbation_time)
            ]
            post_events = [
                event for event in clamp_events
                if float(event["time"]) >= float(analysis_perturbation_time)
            ]

        protocol_ready_time = max(self.warmup_duration, self.noise_warmup_time)
        recommended_perturbation_time = (
            protocol_ready_time
            + STRING_STABILITY_BASELINE_WINDOW_STEPS * self.dt
        )
        self.logger.metadata["collision_clamp_events"] = clamp_events
        self.logger.metadata["pre_perturbation_collision_clamp_count"] = len(pre_events)
        self.logger.metadata["post_perturbation_collision_clamp_count"] = len(post_events)
        self.logger.metadata["pre_perturbation_collision_clamp_events"] = pre_events
        self.logger.metadata["post_perturbation_collision_clamp_events"] = post_events
        self.logger.metadata["protocol_ready_time"] = protocol_ready_time
        self.logger.metadata["protocol_recommended_perturbation_time"] = recommended_perturbation_time
        self.logger.metadata["protocol_perturbation_after_warmup"] = bool(
            analysis_perturbation_time is not None
            and analysis_perturbation_time > protocol_ready_time
        )
        self.logger.metadata["protocol_baseline_window_after_warmup"] = bool(
            analysis_perturbation_time is not None
            and analysis_perturbation_time >= recommended_perturbation_time
        )

        self.logger.save()

        if not self.quiet:
            if self.collision_count > 0:
                print(f"\nWARNING: {self.collision_count} collision warnings during simulation")
            else:
                print("\nSimulation completed with ZERO collisions")

            if self.collision_clamp_count > 0:
                print(f"\n CRITICAL: {self.collision_clamp_count} COLLISION CLAMP events (emergency v=0 braking)")
                print(f"     These are IMPULSE DISTURBANCES that INVALIDATE string stability analysis!")
                print(f"     First 3 events:")
                for i, (t, vid) in enumerate(self.collision_clamp_events[:3]):
                    print(f"       {i+1}. Vehicle {vid} at t={t:.2f}s")
                if self.collision_clamp_count > 3:
                    print(f"       ... and {self.collision_clamp_count - 3} more")
                print(f"     RECOMMENDATION: Discard this run or tune to prevent clamps.")
            else:
                print(" ZERO collision clamps - string stability analysis is VALID")
