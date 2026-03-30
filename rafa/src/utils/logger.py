# -*- coding: utf-8 -*-
"""
Saves:
    - micro.csv  : microscopic trajectories (t, step, id, x, v, a, type)
    - macro.csv  : macroscopic fields (t, step, cell, rho, u)
    - metadata.json : simulation constants and parameters
    - micro.pt / macro.pt : PyTorch archives
"""

import os
import csv
import json
import torch
import numpy as np


class NullLogger:
    """No-op logger used for lightweight live visualization sessions."""

    def __init__(self, output_dir=None, metadata=None):
        self.output_dir = output_dir or ""
        self.metadata = metadata or {}

    def log_micro(self, vehicles, t, step):
        return

    def log_macro(self, rho, u, t, step):
        return

    def log_macro_teacher(self, rho_input, u_input, rho_target, u_target, t, step):
        return

    def save(self):
        return


class Logger:
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
        
        # Macro teacher dataset paths
        if self.macro_teacher_enabled:
            self.macro_teacher_input_path = os.path.join(output_dir, "macro_teacher_input.npz")
            self.macro_teacher_target_path = os.path.join(output_dir, "macro_teacher_target.npz")

    # ---------------------------------------------------------
    # MICROSCOPIC LOGGER
    # ---------------------------------------------------------
    def log_micro(self, vehicles, t, step):
        """Store microscopic data for one timestep."""
        for v in vehicles:
            record = {
                "t": float(t),
                "step": int(step),
                "id": int(v.id),
                "x": float(v.x),
                "v": float(v.v),
                "a": float(getattr(v, "a", 0.0)),  # ensure safe logging
                "type": v.__class__.__name__
            }
            
            # Mesoscopic adaptation data (if CAV with meso enabled)
            if hasattr(v, '_meso_alpha'):
                record["alpha"] = float(v._meso_alpha)
                record["meso_h_c"] = float(getattr(v, '_meso_h_c', 0.0))
                record["meso_k_s"] = float(getattr(v, '_meso_k_s', 0.0))
                record["meso_k_v"] = float(getattr(v, '_meso_k_v', 0.0))
                record["meso_k_v0"] = float(getattr(v, '_meso_k_v0', 0.0))
                record["meso_k_f"] = float(getattr(v, '_meso_k_f', 0.95))  # NEW: adaptive feedforward
                
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
                record["meso_k_f"] = 0.0  # NEW
                record["meso_mu_v"] = 0.0
                record["meso_sigma_v"] = 0.0
                record["meso_beta"] = 0.0
                record["meso_psi"] = 0.0
                record["meso_rho"] = 0.0
                record["meso_stability_margin"] = 0.0
            
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
    # MACRO TEACHER LOGGER (FOR GNN TRAINING DATASET)
    # ---------------------------------------------------------
    def log_macro_teacher(self, rho_input, u_input, rho_target, u_target, t, step):
        """
        Store macro teacher dataset: (rho,u)_t -> (rho,u)_{t+dt} pairs.
        
        This creates a supervised learning dataset where:
        - Input: macroscopic state at time t (from SPH reconstruction)
        - Target: macroscopic state at time t+dt (from PDE solver)
        
        Args:
            rho_input: density array at time t (M,)
            u_input: velocity array at time t (M,)
            rho_target: density array at time t+dt from PDE (M,)
            u_target: velocity array at time t+dt from PDE (M,)
            t: current time
            step: current timestep
        """
        if not self.macro_teacher_enabled:
            return
        
        # Store arrays (convert to numpy if needed)
        self.macro_input_rho.append(np.array(rho_input, dtype=np.float32))
        self.macro_input_u.append(np.array(u_input, dtype=np.float32))
        self.macro_target_rho.append(np.array(rho_target, dtype=np.float32))
        self.macro_target_u.append(np.array(u_target, dtype=np.float32))
        self.macro_teacher_times.append(float(t))

    # ---------------------------------------------------------
    # SAVE ALL OUTPUT
    # ---------------------------------------------------------
    def save(self):

        # --------------------------- MICRO CSV ---------------------------
        with open(self.micro_csv_path, "w", newline="") as f:
            fieldnames = ["t", "step", "id", "x", "v", "a", "type",
                         "alpha", "meso_h_c", "meso_k_s", "meso_k_v", "meso_k_v0", "meso_k_f",
                         "meso_mu_v", "meso_sigma_v", "meso_beta", "meso_psi", 
                         "meso_rho", "meso_stability_margin"]
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
            # Save input arrays (rho, u at time t)
            np.savez_compressed(
                self.macro_teacher_input_path,
                rho=np.stack(self.macro_input_rho),  # Shape: (T, M)
                u=np.stack(self.macro_input_u),      # Shape: (T, M)
                t=np.array(self.macro_teacher_times) # Shape: (T,)
            )
            
            # Save target arrays (rho, u at time t+dt from PDE)
            np.savez_compressed(
                self.macro_teacher_target_path,
                rho=np.stack(self.macro_target_rho),  # Shape: (T, M)
                u=np.stack(self.macro_target_u),      # Shape: (T, M)
                t=np.array(self.macro_teacher_times)  # Shape: (T,)
            )
            
            print(f"[Logger] Saved macro teacher dataset: {len(self.macro_input_rho)} samples")
            print(f"         Input shape: rho{np.stack(self.macro_input_rho).shape}, u{np.stack(self.macro_input_u).shape}")
            print(f"         Files: macro_teacher_input.npz, macro_teacher_target.npz")

        print(f"[Logger] Saved micro.csv, macro.csv, metadata.json, micro.pt, macro.pt in {self.output_dir}")
