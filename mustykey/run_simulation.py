# -*- coding: utf-8 -*-
"""Entry point for a single simulation/evaluation run."""

import argparse
import os

from src.utils.config_manager import ConfigManager
from src.simulation.scenario_manager import ScenarioManager
from src.visualization.visualizer import Visualizer, LiveVisualizer


def main():
    parser = argparse.ArgumentParser(description="Run mixed-autonomy ring-road simulation.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML file."
    )
    args = parser.parse_args()

    # ---------------------------------------------------------
    # Validate config file exists
    # ---------------------------------------------------------
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    # ---------------------------------------------------------
    # Load YAML configuration
    # ---------------------------------------------------------
    config = ConfigManager.load(args.config)

    # Keep evaluation artifacts separate from training artifacts.
    base_output = config["output_path"]

    simulation_output = os.path.join(base_output, "simulation")
    training_output = os.path.join(base_output, "training")

    os.makedirs(simulation_output, exist_ok=True)
    os.makedirs(training_output, exist_ok=True)

    # This entry point always writes into the simulation subfolder.
    config["output_path"] = simulation_output
    
    # Vehicle IDs are assigned humans first, then CAVs.
    N = config["N"]
    num_human = int(N * config["human_ratio"])
    cav_ids = list(range(num_human, N))
    
    # ---------------------------------------------------------
    enable_live_viz = config.get("enable_live_viz", True)
    
    if enable_live_viz:
        live_viz = LiveVisualizer(
            L=config["L"],
            N=config["N"],
            cav_ids=cav_ids,
            R=20.0,
            update_interval=config.get("viz_update_interval", 10)
        )
        print("[Main] Live visualization ENABLED")
    else:
        live_viz = None
        print("[Main] Live visualization DISABLED")

    # ---------------------------------------------------------
    # Build simulation (vehicles, environment, macro generator)
    # ---------------------------------------------------------
    manager = ScenarioManager(config)
    sim = manager.build(live_viz=live_viz)

    # ---------------------------------------------------------
    # Run simulation
    # ---------------------------------------------------------
    sim.run()
    print("Simulation complete. Dataset saved.")

    play_recording = config.get("play_recording", True)
    
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
