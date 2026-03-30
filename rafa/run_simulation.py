# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 09:22:53 2025

@author: shnoz
"""

# -------------------------------------------------------------
# File: run_simulation.py
# Entry point for running the simulator + circular visualizer.
# -------------------------------------------------------------

import argparse
import os

from src.utils.config_manager import ConfigManager
from src.simulation.scenario_manager import ScenarioManager
from src.visualization.visualizer import Visualizer, LiveVisualizer   # Import both visualizers
from src.vehicles.cav_vehicle import CAVVehicle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run mixed-autonomy ring-road simulation.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML file."
    )
    return parser


def main():
    args = build_parser().parse_args()

    # ---------------------------------------------------------
    # Validate config file exists
    # ---------------------------------------------------------
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # ---------------------------------------------------------
    # Load YAML configuration
    # ---------------------------------------------------------
    config = ConfigManager.load(args.config)
    
    # ---------------------------------------------------------
    # Build simulation (vehicles, environment, macro generator)
    # ---------------------------------------------------------
    manager = ScenarioManager(config)
    sim = manager.build(live_viz=None)
    cav_ids = [v.id for v in sim.env.vehicles if isinstance(v, CAVVehicle)]

    # ---------------------------------------------------------
    # Create LIVE visualizer for real-time display (optional)
    # ---------------------------------------------------------
    enable_live_viz = config.get("enable_live_viz", True)  # Default: enabled

    if enable_live_viz:
        sim.live_viz = LiveVisualizer(
            L=config["L"],
            N=config["N"],
            cav_ids=cav_ids,
            R=20.0,
            update_interval=config.get("viz_update_interval", 10)
        )
        print("[Main] Live visualization ENABLED")
    else:
        print("[Main] Live visualization DISABLED")

    # ---------------------------------------------------------
    # Run simulation
    # ---------------------------------------------------------
    sim.run()
    print("Simulation complete. Dataset saved.")

    # ---------------------------------------------------------
    # Launch PLAYBACK visualizer (circular ring-road animation)
    # Shows recorded data after simulation completes
    # ---------------------------------------------------------
    play_recording = config.get("play_recording", True)  # Default: enabled
    
    if play_recording:
        print("[Main] Launching playback visualizer...")
        viz = Visualizer(
            folder=config["output_path"],
            L=config["L"],
            N=config["N"],
            cav_ids=cav_ids
        )
        viz.play()
    else:
        print("[Main] Playback visualization skipped")


if __name__ == "__main__":
    main()
