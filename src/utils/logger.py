# -*- coding: utf-8 -*-
"""
Saves:
    - micro.csv  : microscopic trajectories (t, step, id, x, v, a, type)
    - macro.csv  : macroscopic fields (t, step, cell, rho, u)
    - metadata.json : simulation constants and parameters
    - micro.pt / macro.pt : PyTorch archives
    - summary.txt : compact run summary
"""

import os
import csv
import json
import torch
import numpy as np


class Logger:
    """Collect and save the simulator outputs used for training and evaluation."""

    def __init__(self, output_dir, metadata):
        """
        Args:
            output_dir  : where files are saved
            metadata    : dictionary containing N, L, dt, scenario, params, etc.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Buffers
        self.micro_records = []
        self.macro_records = []
        
        # Macro teacher dataset (for GNN training)
        self.macro_teacher_enabled = metadata.get('save_macro_dataset', False)
        if self.macro_teacher_enabled:
            self.macro_input_rho = []   # List of rho arrays at time t
            self.macro_input_u = []     # List of u arrays at time t
            self.macro_target_rho = []  # List of rho arrays at time t+dt (PDE prediction)
            self.macro_target_u = []    # List of u arrays at time t+dt (PDE prediction)
            self.macro_teacher_times = []  # Timestamps for each pair

        # Store metadata now
        self.metadata = metadata

        # Paths
        self.micro_csv_path = os.path.join(output_dir, "micro.csv")
        self.macro_csv_path = os.path.join(output_dir, "macro.csv")
        self.meta_json_path = os.path.join(output_dir, "metadata.json")
        self.micro_pt_path  = os.path.join(output_dir, "micro.pt")
        self.macro_pt_path  = os.path.join(output_dir, "macro.pt")
        self.summary_txt_path = os.path.join(output_dir, "summary.txt")
        
        # Macro teacher dataset paths
        if self.macro_teacher_enabled:
            self.macro_teacher_input_path = os.path.join(output_dir, "macro_teacher_input.npz")
            self.macro_teacher_target_path = os.path.join(output_dir, "macro_teacher_target.npz")

    # ---------------------------------------------------------
    # MICROSCOPIC LOGGER
    # ---------------------------------------------------------
    def log_micro(self, vehicles, t, step):
        """Store microscopic data for one timestep."""
        L = self.metadata.get("L", None)

        for v in vehicles:
            # Prefer simulator-updated acceleration if available
            a_val = getattr(v, "acceleration", getattr(v, "a", 0.0))

            # Log gap for downstream metrics (e.g. min_gap during training)
            if L is not None and hasattr(v, "compute_gap"):
                try:
                    gap_s = float(v.compute_gap(L))
                except Exception:
                    gap_s = float("nan")
            else:
                gap_s = float("nan")

            record = {
                "t": float(t),
                "step": int(step),
                "id": int(v.id),
                "x": float(v.x),
                "v": float(v.v),
                "a": float(a_val),
                "gap_s": float(gap_s),
                "type": v.__class__.__name__
            }
            
            # Mesoscopic controller diagnostics.
            if hasattr(v, '_meso_alpha'):
                record["alpha"] = float(v._meso_alpha)
                record["meso_h_c"] = float(getattr(v, '_meso_h_c', 0.0))
                record["meso_k_s"] = float(getattr(v, '_meso_k_s', 0.0))
                record["meso_k_v"] = float(getattr(v, '_meso_k_v', 0.0))
                record["meso_k_v0"] = float(getattr(v, '_meso_k_v0', 0.0))
                record["meso_k_f"] = float(getattr(v, '_meso_k_f', 0.95))
                
                # Detailed diagnostics
                if hasattr(v, '_meso_diagnostics'):
                    diag = v._meso_diagnostics
                    record["meso_mu_v"] = float(diag.get('mu_v', 0.0))
                    record["meso_sigma_v"] = float(diag.get('sigma_v', 0.0))
                    record["meso_beta"] = float(diag.get('beta', 0.0))
                    record["meso_psi"] = float(diag.get('psi', 0.0))
                    record["meso_rho"] = float(diag.get('rho', 0.0))
                    record["meso_stability_margin"] = float(diag.get('stability_margin', 0.0))
                else:
                    record["meso_mu_v"] = 0.0
                    record["meso_sigma_v"] = 0.0
                    record["meso_beta"] = 0.0
                    record["meso_psi"] = 0.0
                    record["meso_rho"] = 0.0
                    record["meso_stability_margin"] = 0.0
            else:
                # Not a CAV or meso disabled - fill with defaults
                record["alpha"] = 1.0
                record["meso_h_c"] = 0.0
                record["meso_k_s"] = 0.0
                record["meso_k_v"] = 0.0
                record["meso_k_v0"] = 0.0
                record["meso_k_f"] = 0.0
                record["meso_mu_v"] = 0.0
                record["meso_sigma_v"] = 0.0
                record["meso_beta"] = 0.0
                record["meso_psi"] = 0.0
                record["meso_rho"] = 0.0
                record["meso_stability_margin"] = 0.0

            # RL diagnostics are logged separately so supervisor-side inspection can
            # distinguish the rule-based alpha from the residual correction.
            if hasattr(v, '_rl_diagnostics'):
                record["rl_alpha_rule"] = float(getattr(v, '_rl_alpha_rule', record["alpha"]))
                record["rl_delta_alpha"] = float(getattr(v, '_rl_delta_alpha', 0.0))
                record["rl_alpha_final"] = float(getattr(v, '_rl_alpha', record["alpha"]))
            else:
                record["rl_alpha_rule"] = float(record["alpha"])
                record["rl_delta_alpha"] = 0.0
                record["rl_alpha_final"] = float(record["alpha"])
            
            self.micro_records.append(record)

    # ---------------------------------------------------------
    # MACROSCOPIC LOGGER
    # ---------------------------------------------------------
    def log_macro(self, rho, u, t, step):
        """Store macroscopic fields (rho, u) per cell."""
        if rho is None or u is None:
            return

        for cell in range(len(rho)):
            self.macro_records.append({
                "t": float(t),
                "step": int(step),
                "cell": int(cell),
                "rho": float(rho[cell]),
                "u": float(u[cell])
            })
    
    # ---------------------------------------------------------
    # MACRO TEACHER LOGGER
    # ---------------------------------------------------------
    def log_macro_teacher(self, rho_input, u_input, rho_target, u_target, t, step):
        if not self.macro_teacher_enabled:
            return
        
        self.macro_input_rho.append(np.array(rho_input, dtype=np.float32))
        self.macro_input_u.append(np.array(u_input, dtype=np.float32))
        self.macro_target_rho.append(np.array(rho_target, dtype=np.float32))
        self.macro_target_u.append(np.array(u_target, dtype=np.float32))
        self.macro_teacher_times.append(float(t))

    # ---------------------------------------------------------
    # SUMMARY TXT
    # ---------------------------------------------------------
    def save_summary_txt(self):
        """Save a compact human-readable summary.txt."""
        with open(self.summary_txt_path, "w") as f:
            f.write("RUN SUMMARY\n")
            f.write("===========\n\n")

            f.write("Metadata\n")
            f.write("--------\n")
            for key, value in self.metadata.items():
                f.write(f"{key}: {value}\n")

            if len(self.micro_records) == 0:
                f.write("\nNo microscopic records logged.\n")
                return

            f.write("\nMicroscopic summary\n")
            f.write("-------------------\n")

            v_vals = np.array([r["v"] for r in self.micro_records], dtype=float)
            a_vals = np.array([r["a"] for r in self.micro_records], dtype=float)

            valid_gaps = np.array(
                [r["gap_s"] for r in self.micro_records if not np.isnan(r["gap_s"])],
                dtype=float
            )

            f.write(f"num_micro_records: {len(self.micro_records)}\n")
            f.write(f"mean_speed: {np.mean(v_vals):.6f}\n")
            f.write(f"speed_variance: {np.var(v_vals):.6f}\n")
            f.write(f"min_speed: {np.min(v_vals):.6f}\n")
            f.write(f"max_speed: {np.max(v_vals):.6f}\n")
            f.write(f"oscillation_amplitude: {(np.max(v_vals) - np.min(v_vals)):.6f}\n")
            f.write(f"rms_acc: {np.sqrt(np.mean(a_vals ** 2)):.6f}\n")

            if len(valid_gaps) > 0:
                f.write(f"min_gap: {np.min(valid_gaps):.6f}\n")
                f.write(f"mean_gap: {np.mean(valid_gaps):.6f}\n")
            else:
                f.write("min_gap: nan\n")
                f.write("mean_gap: nan\n")

            # Jerk estimate from per-vehicle acceleration history
            dt = float(self.metadata.get("dt", 0.0))
            if dt > 0:
                by_id = {}
                for r in self.micro_records:
                    by_id.setdefault(r["id"], []).append(r)

                jerk_vals = []
                for recs in by_id.values():
                    recs = sorted(recs, key=lambda z: z["step"])
                    a_hist = np.array([rr["a"] for rr in recs], dtype=float)
                    if len(a_hist) >= 2:
                        jerk_vals.extend((np.diff(a_hist) / dt).tolist())

                if len(jerk_vals) > 0:
                    jerk_vals = np.array(jerk_vals, dtype=float)
                    f.write(f"rms_jerk: {np.sqrt(np.mean(jerk_vals ** 2)):.6f}\n")
                else:
                    f.write("rms_jerk: nan\n")
            else:
                f.write("rms_jerk: nan\n")

            f.write(f"collision_count: {self.metadata.get('collision_count', 'unknown')}\n")
            f.write(f"collision_clamp_count: {self.metadata.get('collision_clamp_count', 'unknown')}\n")

            if len(self.macro_records) > 0:
                f.write("\nMacroscopic summary\n")
                f.write("------------------\n")
                rho_vals = np.array([r["rho"] for r in self.macro_records], dtype=float)
                u_vals = np.array([r["u"] for r in self.macro_records], dtype=float)
                f.write(f"num_macro_records: {len(self.macro_records)}\n")
                f.write(f"mean_rho: {np.mean(rho_vals):.6f}\n")
                f.write(f"mean_u: {np.mean(u_vals):.6f}\n")

    # ---------------------------------------------------------
    # SAVE ALL OUTPUT
    # ---------------------------------------------------------
    def save(self):

        # --------------------------- MICRO CSV ---------------------------
        with open(self.micro_csv_path, "w", newline="") as f:
            fieldnames = [
                "t", "step", "id", "x", "v", "a", "gap_s", "type",
                "alpha", "meso_h_c", "meso_k_s", "meso_k_v", "meso_k_v0", "meso_k_f",
                "meso_mu_v", "meso_sigma_v", "meso_beta", "meso_psi",
                "meso_rho", "meso_stability_margin",
                "rl_alpha_rule", "rl_delta_alpha", "rl_alpha_final"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.micro_records)

        # --------------------------- MACRO CSV ---------------------------
        if len(self.macro_records) > 0:
            with open(self.macro_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["t", "step", "cell", "rho", "u"]
                )
                writer.writeheader()
                writer.writerows(self.macro_records)

        # --------------------------- METADATA JSON ------------------------
        with open(self.meta_json_path, "w") as f:
            json.dump(self.metadata, f, indent=4)

        # --------------------------- TORCH ARCHIVES -----------------------
        torch.save(self.micro_records, self.micro_pt_path)
        torch.save(self.macro_records, self.macro_pt_path)
        
        # --------------------------- MACRO TEACHER DATASET ----------------
        if self.macro_teacher_enabled and len(self.macro_input_rho) > 0:
            np.savez_compressed(
                self.macro_teacher_input_path,
                rho=np.stack(self.macro_input_rho),
                u=np.stack(self.macro_input_u),
                t=np.array(self.macro_teacher_times)
            )
            
            np.savez_compressed(
                self.macro_teacher_target_path,
                rho=np.stack(self.macro_target_rho),
                u=np.stack(self.macro_target_u),
                t=np.array(self.macro_teacher_times)
            )
            
            print(f"[Logger] Saved macro teacher dataset: {len(self.macro_input_rho)} samples")
            print(f"         Input shape: rho{np.stack(self.macro_input_rho).shape}, u{np.stack(self.macro_input_u).shape}")
            print(f"         Files: macro_teacher_input.npz, macro_teacher_target.npz")

        # --------------------------- SUMMARY TXT -------------------------
        self.save_summary_txt()

        print(f"[Logger] Saved micro.csv, macro.csv, metadata.json, micro.pt, macro.pt, summary.txt in {self.output_dir}")
