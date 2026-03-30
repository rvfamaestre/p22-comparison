# -*- coding: utf-8 -*-
"""Accelerator-native SAC training for the vectorized ring-road env.

Usage:
    # Auto-detect device (CUDA > MPS > CPU)
    python train_gpu_sac.py --config config/rl_train.yaml

    # Explicit device / profile
    python train_gpu_sac.py --config config/rl_train.yaml --device cuda
    python train_gpu_sac.py --config config/rl_train.yaml --device cpu --num_envs 4

    # Bootstrap from a PPO checkpoint (transfers actor weights)
    python train_gpu_sac.py --config config/rl_train.yaml --ppo_ckpt output/gpu_train/gpu_ppo_final.pt

Why SAC?
--------
The ring-road HDV dynamics include Euler-Maruyama noise, making each episode
a different stochastic realisation.  PPO discards every rollout after K epochs.
SAC stores all transitions in a replay buffer and re-samples them many times,
extracting far more gradient signal per collected step — typically 3-5× better
sample efficiency on stochastic continuous-control benchmarks.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import yaml

from src.agents.rl_types import OBS_DIM
from src.gpu.gpu_sac import GPUSACTrainer
from src.gpu.hardware_profiles import (
    empty_accelerator_cache,
    resolve_profile,
    select_torch_device,
)
from src.gpu.vec_env import VecEnvConfig, VecRingRoadEnv
from src.utils.early_stopping import EarlyStoppingConfig, EarlyStopMonitor
from src.utils.primary_objective import PRIMARY_OBJECTIVE_METRIC
from src.utils.training_monitor import TrainingMonitor


# ------------------------------------------------------------------
# Config helpers (reuse same pattern as train_gpu_ppo.py)
# ------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def first_non_none(*values):
    for v in values:
        if v is not None:
            return v
    return None


def normalize_hr_values(values):
    """Validate and de-duplicate human-rate overrides while preserving order."""
    normalized = []
    for value in values:
        hr = float(value)
        if not 0.0 <= hr <= 1.0:
            raise ValueError(f"human_rate must be in [0, 1], got {hr}.")
        if hr not in normalized:
            normalized.append(hr)
    if not normalized:
        raise ValueError("At least one human_rate must be provided.")
    return normalized


def apply_training_overrides(cfg: dict, args: argparse.Namespace) -> None:
    """Apply CLI overrides for seed / human-rate specialization."""
    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    if args.human_rate is not None and args.hr_options is not None:
        raise ValueError("Use either --human_rate or --hr_options, not both.")

    hr_values = None
    if args.human_rate is not None:
        hr_values = normalize_hr_values([args.human_rate])
    elif args.hr_options is not None:
        hr_values = normalize_hr_values(args.hr_options)

    if hr_values is None:
        return

    cfg["human_ratio"] = float(hr_values[0])
    rl = cfg.setdefault("rl", {})
    rl["hr_options"] = hr_values
    rl["hr_weights"] = None


def build_vec_env_config(cfg: dict) -> VecEnvConfig:
    """Translate YAML config into the vectorized environment dataclass."""
    rl = cfg.get("rl", {})
    idm = cfg.get("idm_params", {})
    acc = cfg.get("acc_params", {})
    cth = cfg.get("cth_params", {})
    meso = cfg.get("mesoscopic", {})

    dt = float(cfg.get("dt", 0.1))
    horizon_s = float(cfg.get("T", 150.0))
    episode_steps = max(1, int(round(horizon_s / dt)))
    perturb_enabled = bool(cfg.get("perturbation_enabled", True))

    return VecEnvConfig(
        N=int(cfg.get("N", 22)),
        L=float(cfg.get("L", 300.0)),
        dt=dt,
        episode_steps=episode_steps,
        idm_s0=float(idm.get("s0", 2.0)),
        idm_T=float(idm.get("T", 1.5)),
        idm_a=float(idm.get("a", 0.3)),
        idm_b=float(idm.get("b", 3.0)),
        idm_v0=float(idm.get("v0", 14.0)),
        idm_delta=float(idm.get("delta", 4.0)),
        noise_Q=float(cfg.get("noise_Q", 0.45)),
        cacc_ks=float(acc.get("ks", 0.4)),
        cacc_kv=float(acc.get("kv", 1.3)),
        cacc_kf=float(acc.get("kf", 0.95)),
        cacc_kv0=float(acc.get("kv0", 0.05)),
        cacc_b_max=float(acc.get("b_max", 3.0)),
        cacc_a_max=float(acc.get("a_max", 1.5)),
        cacc_v_max=float(acc.get("v_max", 20.0)),
        cacc_v_des=float(acc.get("v_des", 14.0)),
        cth_d0=float(cth.get("d0", 3.5)),
        cth_hc=float(cth.get("hc", 1.2)),
        meso_enabled=bool(meso.get("enabled", True)),
        meso_M=int(meso.get("M", 8)),
        meso_lambda_rho=float(meso.get("lambda_rho", 0.8)),
        meso_gamma=float(meso.get("gamma", 0.4)),
        meso_alpha_min=float(meso.get("alpha_min", 0.7)),
        meso_alpha_max=float(meso.get("alpha_max", 2.0)),
        meso_max_alpha_rate=float(meso.get("max_alpha_rate", 0.2)),
        meso_sigma_v_ema_lambda=float(meso.get("sigma_v_ema_lambda", 0.9)),
        meso_sigma_v_min_threshold=float(meso.get("sigma_v_min_threshold", 0.2)),
        meso_v_eps_sigma=float(meso.get("v_eps_sigma", 0.5)),
        meso_psi_deadband=float(meso.get("psi_deadband", 0.5)),
        rl_delta_alpha_max=float(rl.get("delta_alpha_max", 0.5)),
        rl_alpha_min=float(rl.get("alpha_min", 0.5)),
        rl_alpha_max=float(rl.get("alpha_max", 2.0)),
        w_s=float(rl.get("w_s", 5.0)),
        w_tau=float(rl.get("w_tau", 3.0)),
        w_v=float(rl.get("w_v", 5.0)),
        w_j=float(rl.get("w_j", 0.15)),
        w_ss=float(rl.get("w_ss", 0.75)),
        w_sigma=float(rl.get("w_sigma", 0.5)),
        w_sigma_cav=float(rl.get("w_sigma_cav", 2.0)),
        w_alpha=float(rl.get("w_alpha", 0.3)),
        s_min_reward=float(rl.get("s_min", 2.0)),
        tau_min_reward=float(rl.get("tau_min", 0.8)),
        v_ref=float(rl.get("v_ref", 5.5)),
        j_ref=float(rl.get("j_ref", 5.0)),
        eps_v=float(rl.get("eps_v", 0.1)),
        fleet_penalty_scaling=bool(rl.get("fleet_penalty_scaling", True)),
        perturbation_enabled=perturb_enabled,
        perturbation_time=float(cfg.get("perturbation_time", 3.0)),
        perturbation_delta_v=float(cfg.get("perturbation_delta_v", -3.0)),
        noise_warmup_time=float(cfg.get("noise_warmup_time", 3.0)),
        perturb_curriculum=perturb_enabled and bool(rl.get("perturb_curriculum", True)),
        perturb_dv_min=float(rl.get("perturb_dv_min", -4.0)),
        perturb_dv_max=float(rl.get("perturb_dv_max", -1.0)),
        perturb_time_min=float(rl.get("perturb_time_min", 2.0)),
        perturb_time_max=float(rl.get("perturb_time_max", 8.0)),
        perturb_random_target=bool(rl.get("perturb_random_target", True)),
        hr_options=tuple(float(x) for x in rl.get("hr_options", [0.0, 0.25, 0.5, 0.75])),
        hr_weights=(
            tuple(float(x) for x in rl["hr_weights"])
            if rl.get("hr_weights") is not None
            else None
        ),
        shuffle_cav_positions=bool(rl.get("shuffle_cav_positions", False)),
        w_damp=float(rl.get("w_damp", 0.3)),
        sigma_ref_damp=float(rl.get("sigma_ref_damp", 1.0)),
        adaptive_v_ref=bool(rl.get("adaptive_v_ref", True)),
        v_ref_delta=float(rl.get("v_ref_delta", 3.0)),
        w_alpha_final=float(rl.get("w_alpha_final", rl.get("w_alpha", 0.3))),
        w_collision_floor=float(rl.get("w_collision_floor", 10.0)),
        collision_gap_critical=float(rl.get("collision_gap_critical", 1.0)),
    )


# ------------------------------------------------------------------
# Observation / reset helpers (mirrors train_gpu_ppo.py)
# ------------------------------------------------------------------


@torch.no_grad()
def build_initial_obs(env: VecRingRoadEnv) -> Tuple[torch.Tensor, torch.Tensor]:
    si, gaps, dv_leader = env._sort_and_gaps()
    mu_v, sigma_v_sq = env._upstream_stats(si)
    a_leader = env.a_prev.gather(1, env._leader_idx)
    return env.build_obs(mu_v, sigma_v_sq, gaps, dv_leader, a_leader)


@torch.no_grad()
def restore_reset_obs(
    env: VecRingRoadEnv,
    obs: torch.Tensor,
    mask: torch.Tensor,
    done: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not done.any():
        return obs, mask
    env.auto_reset(done)
    fresh_obs, fresh_mask = build_initial_obs(env)
    obs = torch.where(done.unsqueeze(1).unsqueeze(2).expand_as(obs), fresh_obs, obs)
    mask = torch.where(done.unsqueeze(1).expand_as(mask), fresh_mask, mask)
    return obs, mask


@torch.no_grad()
def evaluate_policy(
    trainer: GPUSACTrainer,
    env_cfg: VecEnvConfig,
    device: torch.device,
    num_envs: int,
) -> Dict[float, float]:
    """Measure mean speed over the last 100 steps for each configured HR."""
    results: Dict[float, float] = {}
    for hr in env_cfg.hr_options:
        eval_cfg = VecEnvConfig(
            **{
                **env_cfg.__dict__,
                "hr_options": (hr,),
                "hr_weights": None,
                "perturb_curriculum": False,
            }
        )
        eval_env = VecRingRoadEnv(num_envs=num_envs, cfg=eval_cfg, device=device)
        eval_env.reset()
        obs, mask = build_initial_obs(eval_env)

        v_trace = []
        for step in range(eval_cfg.episode_steps):
            obs_norm = trainer.normalizer.normalize(obs)
            actions = trainer.select_actions(obs_norm, mask, deterministic=True)
            obs, _, done, mask, _ = eval_env.step(actions.squeeze(2))
            if step >= eval_cfg.episode_steps - 100:
                v_trace.append(eval_env.v.mean().item())
            obs, mask = restore_reset_obs(eval_env, obs, mask, done)

        results[float(hr)] = float(np.mean(v_trace)) if v_trace else 0.0
    return results


def build_early_stopping_config(
    sac_cfg: dict,
    args: argparse.Namespace,
    total_timesteps: int,
    warmup_steps: int,
    eval_interval: int,
) -> EarlyStoppingConfig:
    """Resolve early-stopping settings from config + CLI overrides."""
    es = sac_cfg.get("early_stopping", {})
    default_metric = "eval_mean_speed" if eval_interval > 0 else "train_reward"
    metric = first_non_none(args.early_stopping_metric, es.get("metric"), default_metric)
    cfg = EarlyStoppingConfig(
        enabled=bool(args.early_stopping or es.get("enabled", False)),
        metric=str(metric),
        mode=str(es.get("mode", "max")),
        patience=int(first_non_none(args.early_stopping_patience, es.get("patience"), 6)),
        min_delta=float(first_non_none(args.early_stopping_min_delta, es.get("min_delta"), 0.02)),
        start_step=int(
            first_non_none(
                args.early_stopping_start_step,
                es.get("start_step"),
                max(total_timesteps // 4, warmup_steps),
            )
        ),
        min_checks=int(first_non_none(args.early_stopping_min_checks, es.get("min_checks"), 4)),
        ema_alpha=float(
            first_non_none(
                args.early_stopping_ema_alpha,
                es.get("ema_alpha"),
                0.3 if metric == "train_reward" else 0.0,
            )
        ),
        restore_best=bool(es.get("restore_best", True)),
    )
    if cfg.enabled and cfg.metric not in {"eval_mean_speed", "train_reward"}:
        raise ValueError(
            f"Unsupported early-stopping metric '{cfg.metric}'. "
            "Use 'eval_mean_speed' or 'train_reward'."
        )
    if cfg.enabled and cfg.metric == "eval_mean_speed" and eval_interval <= 0:
        raise ValueError(
            "Early stopping with metric 'eval_mean_speed' requires eval_interval > 0."
        )
    return cfg


def write_early_stopping_summary(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    apply_training_overrides(cfg, args)
    rl = cfg.get("rl", {})
    sac_cfg = cfg.get("sac", {})
    gpu_cfg = cfg.get("gpu_training", {})

    # Device & profile (same logic as PPO script)
    device_request = first_non_none(args.device, gpu_cfg.get("device"), "auto")
    device = select_torch_device(device_request)

    profile_request = first_non_none(args.profile, gpu_cfg.get("profile"), "auto")
    profile = resolve_profile(device, profile_request)

    num_envs = int(first_non_none(args.num_envs, gpu_cfg.get("num_envs"), profile.num_envs))
    eval_num_envs = int(
        first_non_none(args.eval_num_envs, gpu_cfg.get("eval_num_envs"), profile.eval_num_envs)
    )

    env_cfg = build_vec_env_config(cfg)

    # Training schedule
    total_timesteps = int(
        first_non_none(args.total_timesteps, sac_cfg.get("total_timesteps"), 2_000_000)
    )
    warmup_steps = int(sac_cfg.get("warmup_steps", 10_000))
    update_every = int(sac_cfg.get("update_every", 1))
    updates_per_step = int(sac_cfg.get("updates_per_step", 1))
    batch_size = int(sac_cfg.get("batch_size", 256))
    log_interval = int(sac_cfg.get("log_interval", 10_000))
    eval_interval = int(sac_cfg.get("eval_interval", 0))
    save_interval = int(sac_cfg.get("save_interval", 500_000))
    normalizer_warmup = int(sac_cfg.get("normalizer_warmup_steps", warmup_steps))
    replay_capacity = int(sac_cfg.get("replay_buffer_size", 500_000))

    output_dir = first_non_none(
        args.output_dir, sac_cfg.get("output_path"), os.path.join("output", "sac_train")
    )
    os.makedirs(output_dir, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Build trainer
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
        replay_capacity=replay_capacity,
        N=int(cfg.get("N", 22)),
        device=device,
        normalizer_warmup=normalizer_warmup,
    )

    # Optional: Bootstrap actor weights from a PPO checkpoint
    if args.ppo_ckpt:
        trainer.load_ppo_actor(args.ppo_ckpt)

    # Build environment
    env = VecRingRoadEnv(num_envs=num_envs, cfg=env_cfg, device=device)
    env.reset()
    obs, mask = build_initial_obs(env)

    # Save effective config
    effective_cfg = {
        "config_path": args.config,
        "algorithm": "SAC",
        "device": device.type,
        "profile": profile.name,
        "num_envs": num_envs,
        "total_timesteps": total_timesteps,
        "replay_capacity": replay_capacity,
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        "update_every": update_every,
        "updates_per_step": updates_per_step,
        "output_dir": output_dir,
        "seed": seed,
        "human_ratio": float(cfg.get("human_ratio", 0.75)),
        "hr_options": list(env_cfg.hr_options),
    }
    early_stop_cfg = build_early_stopping_config(
        sac_cfg=sac_cfg,
        args=args,
        total_timesteps=total_timesteps,
        warmup_steps=warmup_steps,
        eval_interval=eval_interval,
    )
    effective_cfg["early_stopping"] = vars(early_stop_cfg)
    effective_cfg["progress_monitoring"] = {
        "training_log": os.path.join(output_dir, "training_log.csv"),
        "eval_log": os.path.join(output_dir, "eval_log.csv"),
        "plots_enabled": not bool(args.disable_progress_plots),
        "plot_refresh_seconds": float(args.plot_refresh_seconds),
    }
    with open(os.path.join(output_dir, "sac_effective.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(effective_cfg, f, sort_keys=False)
    with open(os.path.join(output_dir, "sac_effective.json"), "w", encoding="utf-8") as f:
        json.dump(effective_cfg, f, indent=2)
    monitor = TrainingMonitor(
        output_dir,
        algorithm="sac",
        enable_plots=not args.disable_progress_plots,
        plot_refresh_seconds=args.plot_refresh_seconds,
    )

    # Device info
    print(f"[train_gpu_sac] Algorithm: SAC")
    print(f"[train_gpu_sac] Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(
            f"[train_gpu_sac] CUDA device: {props.name} | "
            f"memory={props.total_memory / 1024**3:.1f} GiB"
        )
    print(
        f"[train_gpu_sac] num_envs={num_envs} | "
        f"warmup={warmup_steps} | update_every={update_every} | "
        f"batch={batch_size} | replay={replay_capacity}"
    )
    if early_stop_cfg.enabled:
        print(
            f"[train_gpu_sac] Early stopping ENABLED | metric={early_stop_cfg.metric} "
            f"| patience={early_stop_cfg.patience} | min_delta={early_stop_cfg.min_delta} "
            f"| start_step={early_stop_cfg.start_step} | min_checks={early_stop_cfg.min_checks}"
        )
        if early_stop_cfg.metric == "train_reward":
            print(
                "[train_gpu_sac] WARNING: train_reward early stopping may keep a "
                "checkpoint that scores well on training reward but not on the "
                "final evaluation benchmark. Prefer eval_mean_speed when feasible."
            )
    else:
        print("[train_gpu_sac] Early stopping DISABLED")

    global_step = 0
    last_eval_step = 0
    last_save_step = 0
    last_log_step = 0
    update_count = 0
    normalizer_frozen = False
    recent_rewards: list = []
    recent_q1: list = []
    recent_actor: list = []
    recent_alphas: list = []
    t0 = time.time()
    early_stop = EarlyStopMonitor(early_stop_cfg)
    best_ckpt_path = os.path.join(output_dir, "gpu_sac_best.pt")
    stopped_early = False
    stop_reason = None

    def maybe_trigger_early_stop(metric_value: float, source: str) -> bool:
        nonlocal stopped_early, stop_reason
        if not early_stop_cfg.enabled or early_stop_cfg.metric != source:
            return False

        status = early_stop.update(metric_value, global_step)
        if not status["tracking_started"]:
            state = f"prestart < {early_stop_cfg.start_step}"
        elif status["ready"]:
            state = f"{status['checks_since_improvement']}/{early_stop_cfg.patience}"
        else:
            state = (
                f"warmup {status['eligible_checks']}/{early_stop_cfg.min_checks}"
            )
        best_display = (
            f"{status['best_metric']:.4f}"
            if status["best_metric"] is not None
            else "n/a"
        )
        print(
            f"[early-stop] source={source} step={global_step} "
            f"value={status['raw_metric']:.4f} smooth={status['smoothed_metric']:.4f} "
            f"best={best_display} stale={state}"
        )

        if status["improved"]:
            trainer.save(best_ckpt_path)
            print(f"[early-stop] new best checkpoint -> {best_ckpt_path}")

        if status["should_stop"]:
            stopped_early = True
            stop_reason = (
                f"No {source} improvement greater than {early_stop_cfg.min_delta} "
                f"for {early_stop_cfg.patience} checks."
            )
            print(
                f"[early-stop] STOP at step={global_step} | "
                f"best_step={status['best_step']} | best={status['best_raw_metric']:.4f}"
            )
            return True

        return False

    # ------------------------------------------------------------------ #
    # w_alpha annealing: same schedule as PPO (linear over total_timesteps)
    # ------------------------------------------------------------------ #
    w_alpha_start = float(rl.get("w_alpha", 0.3))
    w_alpha_end = float(rl.get("w_alpha_final", 0.10))

    while global_step < total_timesteps:
        # w_alpha annealing (passed to env via current_w_alpha)
        progress = global_step / max(total_timesteps, 1)
        env.current_w_alpha = w_alpha_start + (w_alpha_end - w_alpha_start) * progress

        # Freeze normalizer after warm-up
        if not normalizer_frozen and global_step >= normalizer_warmup:
            trainer.normalizer.freeze()
            print(f"[train_gpu_sac] Normalizer frozen at step {global_step}")
            normalizer_frozen = True

        # ---------- Environment step ----------
        obs_norm = trainer.normalizer.normalize(obs)

        # Update normalizer before freezing
        if not normalizer_frozen:
            cav_obs = obs[mask]
            if cav_obs.shape[0] > 0:
                trainer.normalizer.update_batch(cav_obs)

        actions = trainer.select_actions(obs_norm, mask)
        next_obs, reward, done, next_mask, info = env.step(actions.squeeze(2))

        # Store in replay buffer (one row per environment = B rows)
        trainer.replay.add(
            obs=obs_norm,
            next_obs=trainer.normalizer.normalize(next_obs),
            actions=actions,
            rewards=reward,
            dones=done,
            masks=mask,
        )
        recent_rewards.append(float(reward.mean().item()))

        next_obs, next_mask = restore_reset_obs(env, next_obs, next_mask, done)
        obs = next_obs
        mask = next_mask
        global_step += num_envs
        trainer.global_step = global_step

        # ---------- SAC updates ----------
        if global_step >= warmup_steps and global_step % update_every == 0:
            for _ in range(updates_per_step):
                info_u = trainer.update(batch_size)
                if info_u:
                    update_count += 1
                    recent_q1.append(info_u["q1_loss"])
                    recent_actor.append(info_u["actor_loss"])
                    recent_alphas.append(info_u["alpha"])

        # ---------- Logging ----------
        if global_step - last_log_step >= log_interval:
            mean_rew = float(np.mean(recent_rewards)) if recent_rewards else 0.0
            mean_q1 = float(np.mean(recent_q1)) if recent_q1 else float("nan")
            mean_act = float(np.mean(recent_actor)) if recent_actor else float("nan")
            mean_alpha_val = float(np.mean(recent_alphas)) if recent_alphas else float("nan")
            elapsed = time.time() - t0
            sps = log_interval / max(elapsed - (last_log_step / max(global_step, 1) * elapsed), 1e-6)
            monitor.log_training(
                {
                    "step": int(global_step),
                    "update": int(update_count),
                    "elapsed_s": float(elapsed),
                    "steps_per_second": float(sps),
                    "reward": float(mean_rew),
                    "q1_loss": float(mean_q1),
                    "actor_loss": float(mean_act),
                    "alpha": float(mean_alpha_val),
                    "replay_size": int(len(trainer.replay)),
                    "actor_lr": float(trainer.actor_optimizer.param_groups[0]["lr"]),
                    "critic_lr": float(trainer.critic_optimizer.param_groups[0]["lr"]),
                    "alpha_lr": float(trainer.alpha_optimizer.param_groups[0]["lr"]),
                    "w_alpha": float(env.current_w_alpha),
                }
            )
            monitor.maybe_refresh_plots()
            print(
                f"[sac step={global_step}] "
                f"reward={mean_rew:+.4f} q1={mean_q1:.4f} "
                f"actor={mean_act:.4f} alpha={mean_alpha_val:.4f} "
                f"replay={len(trainer.replay)} "
                f"updates={update_count} "
                f"elapsed={elapsed:.0f}s"
            )
            recent_rewards.clear()
            recent_q1.clear()
            recent_actor.clear()
            recent_alphas.clear()
            last_log_step = global_step
            if maybe_trigger_early_stop(mean_rew, "train_reward"):
                break

        # ---------- Evaluation ----------
        if eval_interval > 0 and global_step - last_eval_step >= eval_interval:
            speeds = evaluate_policy(trainer, env_cfg, device, eval_num_envs)
            detail = " | ".join(
                f"HR={hr:.2f}: {speed:.2f} m/s" for hr, speed in sorted(speeds.items())
            )
            print(f"[sac eval] step={global_step} {detail}")
            monitor.log_evaluation(
                step=global_step,
                elapsed_s=time.time() - t0,
                speeds=speeds,
                primary_objective_metric=PRIMARY_OBJECTIVE_METRIC,
            )
            monitor.maybe_refresh_plots()
            last_eval_step = global_step
            eval_score = float(np.mean(list(speeds.values()))) if speeds else 0.0
            if maybe_trigger_early_stop(eval_score, "eval_mean_speed"):
                break

        # ---------- Checkpoint ----------
        if global_step - last_save_step >= save_interval:
            ckpt_path = os.path.join(output_dir, f"gpu_sac_step_{global_step}.pt")
            trainer.save(ckpt_path)
            print(f"[sac save] checkpoint -> {ckpt_path}")
            last_save_step = global_step

    if (
        early_stop_cfg.enabled
        and early_stop_cfg.restore_best
        and early_stop.best_step is not None
        and os.path.exists(best_ckpt_path)
    ):
        trainer.load(best_ckpt_path)
        print(
            f"[train_gpu_sac] Restored best checkpoint from step {early_stop.best_step} "
            f"before writing gpu_sac_final.pt"
        )

    # Final save
    final_ckpt = os.path.join(output_dir, "gpu_sac_final.pt")
    trainer.save(final_ckpt)
    write_early_stopping_summary(
        os.path.join(output_dir, "early_stopping_summary.json"),
        {
            **early_stop.as_dict(),
            "stopped_early": stopped_early,
            "stop_reason": stop_reason,
            "best_checkpoint": best_ckpt_path if os.path.exists(best_ckpt_path) else None,
            "final_checkpoint": final_ckpt,
        },
    )
    monitor.maybe_refresh_plots(force=True)
    empty_accelerator_cache(device)
    elapsed = time.time() - t0
    print(
        f"[train_gpu_sac] Done. "
        f"step={global_step} updates={update_count} "
        f"elapsed={elapsed:.1f}s final_ckpt={final_ckpt}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the vectorized SAC agent on an accelerator"
    )
    parser.add_argument("--config", type=str, default="config/rl_train.yaml")
    parser.add_argument(
        "--device", type=str, default=None, choices=["auto", "cuda", "mps", "cpu"]
    )
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--eval_num_envs", type=int, default=None)
    parser.add_argument("--total_timesteps", type=int, default=None)
    parser.add_argument(
        "--disable_progress_plots",
        action="store_true",
        help="Disable periodic plot refreshes during training. CSV logs are still written.",
    )
    parser.add_argument(
        "--plot_refresh_seconds",
        type=float,
        default=30.0,
        help="Minimum wall-clock time between progress-plot refreshes.",
    )
    parser.add_argument(
        "--human_rate",
        type=float,
        default=None,
        help="Train on a single fixed human ratio; overrides rl.hr_options.",
    )
    parser.add_argument(
        "--hr_options",
        nargs="+",
        type=float,
        default=None,
        help="Override rl.hr_options for domain-randomized training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the training seed.",
    )
    parser.add_argument(
        "--early_stopping",
        action="store_true",
        help="Enable plateau-based early stopping.",
    )
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        choices=["eval_mean_speed", "train_reward"],
        default=None,
        help="Metric used by early stopping.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Number of plateau checks tolerated before stopping.",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=None,
        help="Minimum improvement required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--early_stopping_start_step",
        type=int,
        default=None,
        help="Training step after which early stopping becomes active.",
    )
    parser.add_argument(
        "--early_stopping_min_checks",
        type=int,
        default=None,
        help="Minimum number of metric checks before early stopping can trigger.",
    )
    parser.add_argument(
        "--early_stopping_ema_alpha",
        type=float,
        default=None,
        help="Optional EMA smoothing factor for the monitored metric.",
    )
    parser.add_argument(
        "--ppo_ckpt",
        type=str,
        default=None,
        help="Optional: path to PPO checkpoint to bootstrap the SAC actor",
    )
    return parser


# ------------------------------------------------------------------
if __name__ == "__main__":
    train(build_parser().parse_args())
