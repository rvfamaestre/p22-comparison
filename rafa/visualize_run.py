"""
Visualize a trained SAC policy against the current local config.

Pre-computes the simulation, then launches an interactive player
with playback controls (play/pause, speed, frame scrubber).

Usage:
    python visualize_run.py --ckpt "output/sac_train/gpu_sac_final.pt"
    python visualize_run.py --ckpt "output/sac_train/gpu_sac_final.pt" --human_ratio 0.5 --shuffle
    python visualize_run.py --ckpt "output/sac_train/gpu_sac_final.pt" --N 30 --L 400
    python visualize_run.py --replay output/_viz_run   # replay existing data
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

from src.agents.networks import ActorCritic
from src.agents.observation_builder import ObservationBuilder, RunningNormalizer
from src.agents.rl_types import RLConfig, OBS_DIM
from src.simulation.scenario_manager import ScenarioManager
from src.utils.config import get_default_config, merge_config, validate_config
from src.vehicles.cav_vehicle import CAVVehicle
from src.visualization.interactive_player import InteractivePlayer


def extract_normalizer_from_gpu_ckpt(ckpt, obs_dim):
    """Extract normalizer stats from a GPU-trained checkpoint."""
    norm = RunningNormalizer(obs_dim)
    if "normalizer" in ckpt:
        nd = ckpt["normalizer"]
        norm.mean = nd["mean"].numpy().astype(np.float64)
        norm.var = nd["var"].numpy().astype(np.float64)
        norm._M2 = nd["M2"].numpy().astype(np.float64)
        norm.count = int(nd["count"])
    norm.freeze()
    return norm


def run_simulation(config, ckpt_path):
    """Run RL evaluation and save results. Returns output folder path."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Detect obs_dim and hidden_dim from checkpoint for backward compatibility
    ckpt_obs_dim = ckpt["policy_state_dict"]["backbone.0.weight"].shape[1]
    ckpt_hidden_dim = ckpt["policy_state_dict"]["backbone.0.weight"].shape[0]

    rl_cfg = RLConfig(
        delta_alpha_max=0.5,
        alpha_min=0.5,
        alpha_max=2.0,
        hidden_dim=ckpt_hidden_dim,
        num_hidden=2,
    )

    policy = ActorCritic(rl_cfg, obs_dim=ckpt_obs_dim)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    normalizer = extract_normalizer_from_gpu_ckpt(ckpt, ckpt_obs_dim)
    meso_M = config.get("mesoscopic", {}).get("M", 8)
    obs_builder = ObservationBuilder(M=meso_M, normalize=True, obs_dim=ckpt_obs_dim)
    obs_builder.normalizer = normalizer

    # Ensure RL/mesoscopic is active
    if "mesoscopic" not in config:
        config["mesoscopic"] = {}
    config["mesoscopic"]["enabled"] = True
    config.setdefault("rl", {})
    config["rl"]["rl_mode"] = "residual"
    config["rl"]["delta_alpha_max"] = 0.5
    config["rl"]["alpha_min"] = 0.5
    config["rl"]["alpha_max"] = 2.0
    config["rl"]["hidden_dim"] = ckpt_hidden_dim
    config["rl"]["num_hidden"] = 2
    config.setdefault("dx", 1.0)
    config.setdefault("kernel_h", 3.0)

    manager = ScenarioManager(config)
    sim = manager.build(live_viz=None)

    # Discover actual CAV IDs (may differ from sequential if shuffled)
    cav_ids = [v.id for v in sim.env.vehicles if isinstance(v, CAVVehicle)]
    alpha_prev = {cid: 1.0 for cid in cav_ids}

    sim.step()  # priming step

    N = config["N"]
    hr = config["human_ratio"]
    T = config["T"]
    dt = config["dt"]
    shuffle = config.get("shuffle_cav_positions", False)

    print(
        f"Running simulation: N={N}, human_ratio={hr}, T={T}s, dt={dt}s, "
        f"shuffle={shuffle}"
    )

    step_count = 0
    while not sim.done:
        obs, cur_cav_ids = obs_builder.build(sim.env.vehicles, sim.env.L, alpha_prev)
        if len(cur_cav_ids) > 0:
            obs_t = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                actions, _, _, _ = policy.get_action_and_value(
                    obs_t, deterministic=True
                )
            actions_np = actions.cpu().numpy()
            delta_alphas = {
                cid: float(actions_np[k, 0]) for k, cid in enumerate(cur_cav_ids)
            }
            sim.set_rl_actions(delta_alphas)

        sim.step()

        for v in sim.env.vehicles:
            if isinstance(v, CAVVehicle):
                alpha_prev[v.id] = getattr(v, "_meso_alpha", 1.0)

        step_count += 1
        if step_count % 500 == 0:
            print(f"  step {step_count} ...")

    sim.logger.save()
    print(f"Simulation complete ({step_count} steps).")
    return config["output_path"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate & visualize a trained RL checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="output/sac_train/gpu_sac_final.pt",
        help="Path to trained checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/rl_train.yaml",
        help="Base simulation config. Use the training config unless you are intentionally testing a mismatch.",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Skip simulation; replay existing micro.pt from this folder.",
    )
    parser.add_argument(
        "--human_ratio",
        type=float,
        default=None,
        help="Override human ratio (0.0 = all CAV, 1.0 = all HDV).",
    )
    parser.add_argument(
        "--N", type=int, default=None, help="Override number of vehicles."
    )
    parser.add_argument(
        "--L", type=float, default=None, help="Override ring length (m)."
    )
    parser.add_argument(
        "--T", type=float, default=None, help="Override simulation duration (s)."
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly intersperse CAVs among HDVs on the ring.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    return parser


def main():
    args = build_parser().parse_args()

    if args.replay:
        # ---- Replay-only mode: skip simulation ----
        print(f"Replaying data from: {args.replay}")
        player = InteractivePlayer(args.replay)
        player.show()
        return

    # ---- Load config ----
    with open(args.config, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f)
    config = merge_config(user_cfg, get_default_config())
    validate_config(config)

    config["enable_live_viz"] = False
    config["play_recording"] = False
    config["output_path"] = "output/_viz_run"
    config.setdefault(
        "shuffle_cav_positions",
        config.get("rl", {}).get("shuffle_cav_positions", False),
    )
    os.makedirs(config["output_path"], exist_ok=True)

    # ---- Apply CLI overrides ----
    if args.human_ratio is not None:
        config["human_ratio"] = args.human_ratio
    if args.N is not None:
        config["N"] = args.N
    if args.L is not None:
        config["L"] = args.L
    if args.T is not None:
        config["T"] = args.T
    if args.seed is not None:
        config["seed"] = args.seed
    if args.shuffle:
        config["shuffle_cav_positions"] = True

    # ---- Run simulation ----
    out_folder = run_simulation(config, args.ckpt)

    # ---- Launch interactive player ----
    print("Launching interactive player...")
    player = InteractivePlayer(out_folder)
    player.show()


if __name__ == "__main__":
    main()
