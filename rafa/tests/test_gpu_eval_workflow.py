from __future__ import annotations

import torch
import pytest

from src.gpu.vec_env import VecEnvConfig, VecRingRoadEnv
from src.utils.gpu_eval_workflow import (
    build_variation_manifest,
    make_eval_cfg,
    summarise_rows,
)
from src.utils.string_stability_metrics import STRING_STABILITY_METRIC


def test_make_eval_cfg_overrides_generalization_axes() -> None:
    env_cfg = VecEnvConfig(
        hr_options=(0.0, 0.25, 0.5),
        hr_weights=(0.2, 0.3, 0.5),
        shuffle_cav_positions=True,
        perturbation_enabled=True,
        perturb_curriculum=True,
    )

    eval_cfg = make_eval_cfg(
        env_cfg,
        0.75,
        meso_enabled=False,
        shuffle_cav_positions=False,
        perturbation_enabled=False,
    )

    assert eval_cfg.hr_options == (0.75,)
    assert eval_cfg.hr_weights is None
    assert eval_cfg.meso_enabled is False
    assert eval_cfg.shuffle_cav_positions is False
    assert eval_cfg.perturbation_enabled is False
    assert eval_cfg.perturb_curriculum is False


def test_summarise_rows_keeps_generalization_dimensions() -> None:
    rows = [
        {
            "mode": "rl",
            "human_rate": 0.75,
            "cav_share": 0.25,
            "seed": 0,
            "shuffle_cav_positions": True,
            "perturbation_enabled": False,
            "primary_objective_value": 3.0,
            "safety_min_gap_threshold": 2.0,
            "mean_reward": 1.0,
            "mean_speed_all": 2.5,
            "mean_speed_last100": 3.0,
            "speed_var_last100": 0.2,
            "min_gap_episode": 2.4,
            "collision_clamp_count": 0,
            "string_stability_metric": STRING_STABILITY_METRIC,
            "string_stability_threshold": 1.0,
            "string_stability_applicable": False,
            "string_stability_metric_valid": None,
            "string_stability_is_stable": None,
            "string_stability_value": float("nan"),
            "safety_min_gap_ok": True,
            "safety_constraint_satisfied": True,
            "safety_no_collision_clamps": True,
        },
        {
            "mode": "rl",
            "human_rate": 0.75,
            "cav_share": 0.25,
            "seed": 1,
            "shuffle_cav_positions": True,
            "perturbation_enabled": False,
            "primary_objective_value": 3.4,
            "safety_min_gap_threshold": 2.0,
            "mean_reward": 1.2,
            "mean_speed_all": 2.7,
            "mean_speed_last100": 3.4,
            "speed_var_last100": 0.4,
            "min_gap_episode": 2.2,
            "collision_clamp_count": 0,
            "string_stability_metric": STRING_STABILITY_METRIC,
            "string_stability_threshold": 1.0,
            "string_stability_applicable": False,
            "string_stability_metric_valid": None,
            "string_stability_is_stable": None,
            "string_stability_value": float("nan"),
            "safety_min_gap_ok": True,
            "safety_constraint_satisfied": False,
            "safety_no_collision_clamps": True,
        },
    ]

    summary = summarise_rows(
        rows,
        group_keys=(
            "mode",
            "human_rate",
            "cav_share",
            "shuffle_cav_positions",
            "perturbation_enabled",
        ),
    )

    assert len(summary) == 1
    row = summary[0]
    assert row["mode"] == "rl"
    assert row["num_trials"] == 2
    assert row["num_seeds"] == 2
    assert row["shuffle_layout"] == "shuffled"
    assert row["perturbation_setting"] == "off"
    assert row["primary_objective_mean"] == pytest.approx(3.2)
    assert row["safety_constraint_satisfied_rate"] == pytest.approx(0.5)
    assert row["string_stability_metric"] == STRING_STABILITY_METRIC
    assert row["string_stability_applicable_rate"] == pytest.approx(0.0)
    assert row["safety_no_collision_clamps_rate"] == pytest.approx(1.0)


def test_summarise_rows_ignores_non_applicable_string_stability_values() -> None:
    rows = [
        {
            "mode": "rl",
            "human_rate": 0.5,
            "cav_share": 0.5,
            "seed": 0,
            "primary_objective_value": 3.0,
            "safety_min_gap_threshold": 2.0,
            "mean_reward": 1.0,
            "mean_speed_all": 2.5,
            "mean_speed_last100": 3.0,
            "speed_var_last100": 0.2,
            "min_gap_episode": 2.4,
            "string_stability_metric": STRING_STABILITY_METRIC,
            "string_stability_threshold": 1.0,
            "string_stability_applicable": False,
            "string_stability_metric_valid": None,
            "string_stability_is_stable": None,
            "string_stability_value": float("nan"),
            "safety_min_gap_ok": True,
            "safety_constraint_satisfied": True,
        },
        {
            "mode": "rl",
            "human_rate": 0.5,
            "cav_share": 0.5,
            "seed": 1,
            "primary_objective_value": 3.2,
            "safety_min_gap_threshold": 2.0,
            "mean_reward": 1.1,
            "mean_speed_all": 2.6,
            "mean_speed_last100": 3.2,
            "speed_var_last100": 0.25,
            "min_gap_episode": 2.5,
            "string_stability_metric": STRING_STABILITY_METRIC,
            "string_stability_threshold": 1.0,
            "string_stability_applicable": True,
            "string_stability_metric_valid": True,
            "string_stability_is_stable": True,
            "string_stability_value": 0.9,
            "safety_min_gap_ok": True,
            "safety_constraint_satisfied": True,
        },
    ]

    summary = summarise_rows(rows, group_keys=("mode", "human_rate", "cav_share"))

    assert len(summary) == 1
    row = summary[0]
    assert row["string_stability_value_mean"] == pytest.approx(0.9)
    assert row["string_stability_applicable_rate"] == pytest.approx(0.5)
    assert row["string_stability_metric_valid_rate"] == pytest.approx(1.0)
    assert row["string_stability_is_stable_rate"] == pytest.approx(1.0)


def test_build_variation_manifest_separates_training_and_evaluation_sources() -> None:
    cfg = {"seed": 42}
    env_cfg = VecEnvConfig(
        hr_options=(0.0, 0.25, 0.5, 0.75),
        shuffle_cav_positions=True,
        perturbation_enabled=True,
        perturb_curriculum=True,
        noise_Q=0.45,
    )

    manifest = build_variation_manifest(
        algorithm="ppo",
        checkpoint="model.pt",
        config_path="config/rl_train.yaml",
        cfg=cfg,
        env_cfg=env_cfg,
        device=torch.device("cpu"),
        num_envs=8,
        modes=("baseline", "adaptive", "rl"),
        human_rates=(0.0, 0.25, 0.5, 0.75),
        seeds=(0, 1, 2),
        shuffle_layouts=(False, True),
        perturbation_settings=(False, True),
        raw_rows=[{"mode": "rl"}],
    )

    training_items = {item["name"]: item for item in manifest["training_variation"]}
    evaluation_items = {item["name"]: item for item in manifest["evaluation_variation"]}

    assert training_items["human_ratio_sampling"]["present"] is True
    assert training_items["cav_layout_shuffling"]["present"] is True
    assert training_items["perturbation_curriculum"]["present"] is True
    assert training_items["stochastic_hdv_noise"]["present"] is True
    assert evaluation_items["matched_seed_sweep"]["present"] is True
    assert evaluation_items["layout_toggle"]["values"] == ["ordered", "shuffled"]
    assert evaluation_items["perturbation_toggle"]["values"] == ["off", "on"]
    assert manifest["string_stability"]["metric"] == STRING_STABILITY_METRIC


def test_vec_env_skips_perturbation_when_disabled() -> None:
    cfg = VecEnvConfig(
        N=6,
        episode_steps=5,
        perturbation_enabled=False,
        perturb_curriculum=False,
        perturbation_time=0.0,
        perturbation_delta_v=-4.0,
        noise_Q=0.0,
        shuffle_cav_positions=False,
    )
    env = VecRingRoadEnv(num_envs=2, cfg=cfg, device=torch.device("cpu"))
    env.reset()
    delta_alphas = torch.zeros(2, cfg.N)

    for _ in range(3):
        env.step(delta_alphas)

    assert bool(env.perturbation_applied.any().item()) is False
    assert int(env.collision_clamp_count.sum().item()) == 0
