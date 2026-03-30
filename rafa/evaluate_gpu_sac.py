"""Evaluate a GPU SAC checkpoint across the formal generalization grid.

Usage:
    python evaluate_gpu_sac.py --config config/rl_train.yaml \
        --checkpoint output/sac_train/gpu_sac_final.pt \
        --device cuda
"""

from __future__ import annotations

import argparse
import os

import torch

from train_gpu_sac import build_initial_obs, build_vec_env_config, load_config
from src.agents.rl_types import OBS_DIM
from src.gpu.gpu_sac import GPUSACTrainer
from src.gpu.hardware_profiles import select_torch_device
from src.utils.gpu_eval_workflow import (
    DEFAULT_EVAL_MODES,
    DEFAULT_PERTURBATION_SETTINGS,
    DEFAULT_SHUFFLE_LAYOUTS,
    build_variation_manifest,
    normalize_eval_modes,
    normalize_human_rates,
    normalize_perturbation_settings,
    normalize_shuffle_layouts,
    resolve_eval_seeds,
    run_generalization_sweep,
    write_generalization_outputs,
)


def build_trainer(
    cfg: dict,
    device: torch.device,
    num_envs: int,
    rollout_steps: int,
) -> GPUSACTrainer:
    """Instantiate a trainer matching the saved checkpoint architecture."""
    rl = cfg.get("rl", {})
    sac_cfg = cfg.get("sac", {})
    env_cfg = build_vec_env_config(cfg)
    warmup_steps = int(sac_cfg.get("normalizer_warmup_steps", 10_000))

    trainer = GPUSACTrainer(
        obs_dim=OBS_DIM,
        hidden_dim=int(rl.get("hidden_dim", 128)),
        num_hidden=int(rl.get("num_hidden", 2)),
        delta_alpha_max=float(rl.get("delta_alpha_max", 0.5)),
        lr_actor=float(sac_cfg.get("lr_actor", rl.get("lr_actor", 3e-4))),
        lr_critic=float(sac_cfg.get("lr_critic", rl.get("lr_critic", 3e-4))),
        gamma=float(sac_cfg.get("gamma", rl.get("gamma", 0.99))),
        tau=float(sac_cfg.get("tau", 0.005)),
        alpha_init=float(sac_cfg.get("alpha_init", 0.2)),
        auto_entropy=bool(sac_cfg.get("auto_entropy", True)),
        target_entropy=float(sac_cfg.get("target_entropy", -1.0)),
        replay_capacity=1000,
        N=env_cfg.N,
        device=device,
        normalizer_warmup=warmup_steps,
    )
    return trainer


@torch.no_grad()
def select_rl_actions(
    trainer: GPUSACTrainer,
    obs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the SAC actor deterministically for one batched observation."""
    obs_norm = trainer.normalizer.normalize(obs)
    actions = trainer.select_actions(obs_norm, mask, deterministic=True)
    return actions.squeeze(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a GPU SAC checkpoint on the vectorized ring-road environment."
    )
    parser.add_argument("--config", type=str, default="config/rl_train.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Parallel environments per evaluation episode.",
    )
    parser.add_argument(
        "--human_rates",
        nargs="+",
        type=float,
        default=None,
        help="Human ratios to evaluate. Defaults to rl.hr_options from the config.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Explicit evaluation seeds. Overrides --num_seeds.",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=5,
        help="Number of seeds when --seeds is omitted (uses 0..num_seeds-1).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_EVAL_MODES),
        help="Evaluation modes to compare. Choices: baseline adaptive rl.",
    )
    parser.add_argument(
        "--shuffle_layouts",
        nargs="+",
        default=list(DEFAULT_SHUFFLE_LAYOUTS),
        help="CAV layout settings to evaluate. Choices include ordered and shuffled.",
    )
    parser.add_argument(
        "--perturbations",
        nargs="+",
        default=list(DEFAULT_PERTURBATION_SETTINGS),
        help="Perturbation settings to evaluate. Choices include off and on.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=1,
        help="Print progress every N completed evaluation cases.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip summary plot generation and only write CSV/JSON outputs.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cfg = load_config(args.config)
    env_cfg = build_vec_env_config(cfg)
    device = select_torch_device(args.device)
    gpu_cfg = cfg.get("gpu_training", {})
    eval_num_envs = int(args.num_envs or gpu_cfg.get("eval_num_envs", 8))

    trainer = build_trainer(
        cfg,
        device,
        num_envs=eval_num_envs,
        rollout_steps=1,
    )
    trainer.load(args.checkpoint)

    human_rates = normalize_human_rates(args.human_rates or env_cfg.hr_options)
    seeds = resolve_eval_seeds(args.seeds, args.num_seeds)
    modes = normalize_eval_modes(args.modes)
    shuffle_layouts = normalize_shuffle_layouts(args.shuffle_layouts)
    perturbation_settings = normalize_perturbation_settings(args.perturbations)

    checkpoint_dir = os.path.dirname(args.checkpoint) or "."
    output_dir = args.output_dir or os.path.join(checkpoint_dir, "evaluation")

    results = run_generalization_sweep(
        trainer,
        env_cfg,
        device,
        build_initial_obs=build_initial_obs,
        action_selector=select_rl_actions,
        modes=modes,
        human_rates=human_rates,
        seeds=seeds,
        shuffle_layouts=shuffle_layouts,
        perturbation_settings=perturbation_settings,
        num_envs=eval_num_envs,
        progress_every=args.progress_every,
    )
    manifest = build_variation_manifest(
        algorithm="sac",
        checkpoint=args.checkpoint,
        config_path=args.config,
        cfg=cfg,
        env_cfg=env_cfg,
        device=device,
        num_envs=eval_num_envs,
        modes=modes,
        human_rates=human_rates,
        seeds=seeds,
        shuffle_layouts=shuffle_layouts,
        perturbation_settings=perturbation_settings,
        raw_rows=results["raw_rows"],
    )
    write_generalization_outputs(
        output_dir,
        raw_rows=results["raw_rows"],
        factor_summary=results["factor_summary"],
        overall_summary=results["overall_summary"],
        manifest=manifest,
        generate_plots=not args.skip_plots,
    )

    print(
        f"[evaluate_gpu_sac] Wrote {len(results['raw_rows'])} raw rows, "
        f"{len(results['factor_summary'])} factor-summary rows, and "
        f"{len(results['overall_summary'])} collapsed summary rows to {output_dir}"
    )
    for row in results["overall_summary"]:
        print(
            f"[summary] HR={row['human_rate']:.2f} mode={row['mode']:<8} "
            f"objective({row['primary_objective_metric']})="
            f"{row['primary_objective_mean']:.3f}"
            f"+-{row['primary_objective_std']:.3f} "
            f"safety_pass={row['safety_constraint_satisfied_rate']:.2f} "
            f"string_stability={row.get('string_stability_value_mean', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
