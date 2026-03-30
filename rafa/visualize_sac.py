"""
Run the SAC-controlled ring-road simulation in real time with a minimal HUD.

Default behavior is live stepping: the simulator advances only while the HUD
is playing. Replay mode remains available for existing logged trajectories.
"""

import argparse
import os

import yaml

from src.utils.config import get_default_config, merge_config, validate_config
from src.visualization.interactive_player import InteractivePlayer
from src.visualization.live_hud import LiveSACSession, MinimalLiveHUD


def resolve_shuffle_cav_positions(config):
    """Resolve shuffle flag from either top-level or nested RL config."""
    top_level = config.get("shuffle_cav_positions")
    if top_level is not None:
        return bool(top_level)
    return bool(config.get("rl", {}).get("shuffle_cav_positions", False))


def resolve_controller_mode(config):
    """Infer controller mode from config."""
    rl_mode = config.get("rl", {}).get("rl_mode", "off")
    meso_enabled = config.get("mesoscopic", {}).get("enabled", False)
    if rl_mode == "residual":
        return "rl"
    if meso_enabled:
        return "adaptive"
    return "baseline"


def build_scenario_state(config, mode=None):
    """Capture the minimal live-HUD scenario state."""
    return {
        "mode": mode or resolve_controller_mode(config),
        "human_ratio": float(config.get("human_ratio", 0.75)),
        "shuffle_cav_positions": resolve_shuffle_cav_positions(config),
        "initial_conditions": config.get("initial_conditions", "random"),
        "perturbation_enabled": bool(config.get("perturbation_enabled", False)),
        "seed": int(config.get("seed", 42)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a SAC policy in live realtime mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="output/sac_train/gpu_sac_final.pt",
        help="Path to SAC checkpoint.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/rl_train.yaml",
        help="Base simulation config. Training config is the recommended default.",
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Replay an existing saved trajectory folder instead of running live.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/_viz_live",
        help="Working directory for live sessions.",
    )
    parser.add_argument(
        "--window_seconds",
        type=float,
        default=100.0,
        help="Rolling time window shown in the live metrics plot.",
    )
    parser.add_argument("--human_ratio", type=float, default=None)
    parser.add_argument(
        "--mode",
        choices=["rl", "adaptive", "baseline"],
        default=None,
        help="Controller mode for the live session.",
    )
    parser.add_argument(
        "--initial_conditions",
        choices=["uniform", "random"],
        default=None,
        help="Override initial condition generator.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--shuffle",
        dest="shuffle",
        action="store_true",
        help="Enable shuffled CAV placement at reset.",
    )
    parser.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="Disable shuffled CAV placement at reset.",
    )
    parser.add_argument(
        "--perturbation",
        dest="perturbation",
        action="store_true",
        help="Enable the configured perturbation event.",
    )
    parser.add_argument(
        "--no-perturbation",
        dest="perturbation",
        action="store_false",
        help="Disable the perturbation event.",
    )
    parser.set_defaults(shuffle=None, perturbation=None)
    return parser


def main():
    args = build_parser().parse_args()

    if args.replay:
        InteractivePlayer(args.replay).show()
        return

    with open(args.config, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f)

    base_config = merge_config(user_cfg, get_default_config())
    validate_config(base_config)

    os.makedirs(args.output_dir, exist_ok=True)

    scenario_state = build_scenario_state(base_config, mode=args.mode)
    if args.human_ratio is not None:
        scenario_state["human_ratio"] = args.human_ratio
    if args.initial_conditions is not None:
        scenario_state["initial_conditions"] = args.initial_conditions
    if args.seed is not None:
        scenario_state["seed"] = args.seed
    if args.shuffle is not None:
        scenario_state["shuffle_cav_positions"] = args.shuffle
    if args.perturbation is not None:
        scenario_state["perturbation_enabled"] = args.perturbation

    session = LiveSACSession(
        base_config=base_config,
        ckpt_path=args.ckpt,
        scenario_state=scenario_state,
        output_dir=args.output_dir,
        history_window_seconds=args.window_seconds,
    )
    MinimalLiveHUD(session, window_seconds=args.window_seconds).show()


if __name__ == "__main__":
    main()
