"""Automated reward-weight tuning via Bayesian optimisation (Optuna).

Each trial samples a reward-weight configuration, trains PPO for a
reduced budget (~2M steps, ~5 min on T4), evaluates across 4 human
rates, and returns a composite speed-improvement metric.

Optuna's TPE sampler + MedianPruner efficiently explores the space,
killing unpromising trials early based on intermediate training reward.

Usage (GPU):
    import torch
    from src.gpu.auto_tune import run_study
    study = run_study(device=torch.device('cuda'), n_trials=40)
    print(study.best_params)

Usage (CPU / HPC with 16 cores):
    study = run_study(
        device=torch.device('cpu'), n_trials=40,
        num_envs=256, n_jobs=4, threads_per_trial=4,
    )

Usage (CLI):
    python -m src.gpu.auto_tune --n_trials 40 --search_steps 2000000
"""

import dataclasses
import gc
import time
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
import torch

from src.agents.rl_types import OBS_DIM
from src.gpu.gpu_ppo import GPUPPOTrainer
from src.gpu.hardware_profiles import empty_accelerator_cache, select_torch_device
from src.gpu.vec_env import VecEnvConfig, VecRingRoadEnv


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_eval_cfg(env_cfg: VecEnvConfig, hr: float) -> VecEnvConfig:
    """Create eval config inheriting all training params but fixing HR."""
    overrides = dataclasses.asdict(env_cfg)
    overrides["hr_options"] = (hr,)
    overrides["hr_weights"] = None
    overrides["perturb_curriculum"] = False
    return VecEnvConfig(**overrides)


@torch.no_grad()
def compute_baselines(
    device: torch.device,
    hr_values: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75),
    num_envs: int = 16,
) -> Dict[float, float]:
    """Measure adaptive mesoscopic baseline speeds at each HR.

    Run once at study start — baselines don't depend on reward weights.
    """
    cfg = VecEnvConfig()
    baselines = {}

    for hr in hr_values:
        eval_cfg = _make_eval_cfg(cfg, hr)
        env = VecRingRoadEnv(num_envs=num_envs, cfg=eval_cfg, device=device)
        env.reset()

        v_trace = []
        for step in range(eval_cfg.episode_steps):
            delta = torch.zeros(num_envs, eval_cfg.N, device=device)
            obs, reward, done, mask, info = env.step(delta)
            if step >= eval_cfg.episode_steps - 100:
                v_trace.append(env.v.mean().item())
            if done.any():
                env.auto_reset(done)

        baselines[hr] = float(np.mean(v_trace))
        del env

    gc.collect()
    empty_accelerator_cache(device)
    return baselines


@torch.no_grad()
def quick_eval(
    trainer: GPUPPOTrainer,
    env_cfg: VecEnvConfig,
    device: torch.device,
    num_eval_envs: int = 16,
    hr_values: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75),
) -> Dict[float, float]:
    """Fast evaluation: measure RL final speed at each HR."""
    results = {}
    for hr in hr_values:
        eval_cfg = _make_eval_cfg(env_cfg, hr)
        eval_env = VecRingRoadEnv(num_envs=num_eval_envs, cfg=eval_cfg, device=device)
        eval_env.reset()

        si, gi, di = eval_env._sort_and_gaps()
        mu, sig = eval_env._upstream_stats(si)
        al = eval_env.a_prev.gather(1, eval_env._leader_idx)
        obs, mask = eval_env.build_obs(mu, sig, gi, di, al)

        v_trace = []
        for step in range(eval_cfg.episode_steps):
            obs_norm = trainer.normalizer.normalize(obs)
            actions, _, _ = trainer.select_actions(obs_norm, mask, deterministic=True)
            delta_alphas = actions.squeeze(2)
            obs, reward, done, mask, info = eval_env.step(delta_alphas)

            if step >= eval_cfg.episode_steps - 100:
                v_trace.append(eval_env.v.mean().item())

            if done.any():
                eval_env.auto_reset(done)
                si2, gi2, di2 = eval_env._sort_and_gaps()
                mu2, sig2 = eval_env._upstream_stats(si2)
                al2 = eval_env.a_prev.gather(1, eval_env._leader_idx)
                fo, fm = eval_env.build_obs(mu2, sig2, gi2, di2, al2)
                rm = done.unsqueeze(1).unsqueeze(2).expand_as(obs)
                obs = torch.where(rm, fo, obs)
                rm2 = done.unsqueeze(1).expand_as(mask)
                mask = torch.where(rm2, fm, mask)

        results[hr] = float(np.mean(v_trace))
        del eval_env

    gc.collect()
    empty_accelerator_cache(device)
    return results


# ------------------------------------------------------------------ #
# Objective
# ------------------------------------------------------------------ #


def create_objective(
    device: torch.device,
    baselines: Dict[float, float],
    search_timesteps: int = 2_000_000,
    num_envs: int = 1024,
    rollout_steps: int = 128,
    seed: int = 42,
    regression_penalty: float = 5.0,
    threads_per_trial: Optional[int] = None,
):
    """Build the Optuna objective function.

    Parameters
    ----------
    baselines : dict
        Adaptive baseline speeds per HR (from ``compute_baselines``).
    search_timesteps : int
        Training budget per trial.  2M ≈ 5 min on T4.
    regression_penalty : float
        Multiplier on worst-HR regression below baseline.
        metric = mean_improvement - penalty * max(0, -min_improvement)
    threads_per_trial : int, optional
        Number of PyTorch CPU threads per trial.  Set when running
        multiple trials in parallel on CPU (n_jobs > 1) to avoid
        over-subscribing cores.  E.g. 16 cores / 4 jobs = 4 threads.
    """
    hr_values = tuple(sorted(baselines.keys()))
    steps_per_rollout = num_envs * rollout_steps
    num_updates = search_timesteps // steps_per_rollout

    def objective(trial: optuna.Trial) -> float:
        if threads_per_trial is not None:
            torch.set_num_threads(threads_per_trial)
        torch.manual_seed(seed + trial.number)
        t0 = time.time()

        # ---- Sample hyperparameters ----
        w_v = trial.suggest_float("w_v", 2.0, 10.0)
        w_j = trial.suggest_float("w_j", 0.05, 0.5, log=True)
        w_ss = trial.suggest_float("w_ss", 0.1, 2.0)
        w_sigma = trial.suggest_float("w_sigma", 0.1, 2.0)
        w_sigma_cav = trial.suggest_float("w_sigma_cav", 0.5, 4.0)
        w_damp = trial.suggest_float("w_damp", 0.0, 2.0)
        w_alpha_final = trial.suggest_float("w_alpha_final", 0.02, 0.3)
        v_ref_delta = trial.suggest_float("v_ref_delta", 1.0, 5.0)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])

        # ---- Build environment ----
        env_cfg = VecEnvConfig(
            w_v=w_v,
            w_j=w_j,
            w_ss=w_ss,
            w_alpha_final=w_alpha_final,
            w_sigma=w_sigma,
            w_sigma_cav=w_sigma_cav,
            w_damp=w_damp,
            v_ref_delta=v_ref_delta,
            hr_options=hr_values,
            hr_weights=(0.15, 0.25, 0.30, 0.30),
            fleet_penalty_scaling=True,
            adaptive_v_ref=True,
            perturb_curriculum=True,
        )

        env = VecRingRoadEnv(num_envs=num_envs, cfg=env_cfg, device=device)
        trainer = GPUPPOTrainer(
            obs_dim=OBS_DIM,
            hidden_dim=hidden_dim,
            num_hidden=2,
            eps_clip=0.2,
            clip_value_loss=True,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            ppo_epochs=10,
            minibatch_size=4096,
            gamma=0.99,
            lam_gae=0.95,
            rollout_steps=rollout_steps,
            num_envs=num_envs,
            N=env_cfg.N,
            device=device,
            normalizer_warmup=50_000,
        )

        # ---- Shortened training loop ----
        env.reset()
        global_step = 0

        si, gi, di = env._sort_and_gaps()
        mu, sig = env._upstream_stats(si)
        al = env.a_prev.gather(1, env._leader_idx)
        obs, mask = env.build_obs(mu, sig, gi, di, al)

        for update_idx in range(1, num_updates + 1):
            progress = global_step / search_timesteps
            trainer.update_lr(progress)
            env.current_w_alpha = 0.3 + (w_alpha_final - 0.3) * progress

            if global_step >= 50_000 and not trainer.normalizer.frozen:
                trainer.normalizer.freeze()

            rollout_rewards = []
            trainer.buffer.reset()

            for _ in range(rollout_steps):
                obs_norm = trainer.normalizer.normalize(obs)
                if not trainer.normalizer.frozen:
                    cav_obs = obs[mask]
                    if cav_obs.shape[0] > 0:
                        trainer.normalizer.update_batch(cav_obs)

                actions, log_probs, values = trainer.select_actions(obs_norm, mask)
                delta_alphas = actions.squeeze(2)
                next_obs, reward, done, next_mask, info = env.step(delta_alphas)

                trainer.buffer.add(
                    obs=obs_norm,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=reward,
                    values=values,
                    dones=done,
                    masks=mask,
                )
                rollout_rewards.append(reward.mean().item())

                if done.any():
                    env.auto_reset(done)
                    si2, gi2, di2 = env._sort_and_gaps()
                    mu2, sig2 = env._upstream_stats(si2)
                    al2 = env.a_prev.gather(1, env._leader_idx)
                    fo, fm = env.build_obs(mu2, sig2, gi2, di2, al2)
                    rm = done.unsqueeze(1).unsqueeze(2).expand_as(next_obs)
                    next_obs = torch.where(rm, fo, next_obs)
                    rm2 = done.unsqueeze(1).expand_as(next_mask)
                    next_mask = torch.where(rm2, fm, next_mask)

                obs = next_obs
                mask = next_mask
                global_step += num_envs

            trainer.global_step = global_step
            trainer.update()

            # Report intermediate reward for pruning
            mean_reward = float(np.mean(rollout_rewards))
            trial.report(mean_reward, update_idx)
            if trial.should_prune():
                del env, trainer
                gc.collect()
                empty_accelerator_cache(device)
                raise optuna.TrialPruned()

        # ---- Evaluate trained policy ----
        speeds = quick_eval(trainer, env_cfg, device, hr_values=hr_values)

        improvements = [speeds[hr] - baselines[hr] for hr in hr_values]
        mean_imp = float(np.mean(improvements))
        min_imp = float(min(improvements))

        # Penalise regression below adaptive baseline
        penalty = regression_penalty * max(0.0, -min_imp) if min_imp < 0 else 0.0
        metric = mean_imp - penalty

        elapsed = time.time() - t0
        detail = " | ".join(
            f"HR={hr}: {speeds[hr]:.2f} ({speeds[hr] - baselines[hr]:+.2f})"
            for hr in hr_values
        )
        print(
            f"Trial {trial.number:>3}: metric={metric:+.3f} | "
            f"mean_imp={mean_imp:+.3f} | {detail} | {elapsed:.0f}s"
        )

        del env, trainer
        gc.collect()
        empty_accelerator_cache(device)
        return metric

    return objective


# ------------------------------------------------------------------ #
# Public API
# ------------------------------------------------------------------ #


def run_study(
    device: torch.device,
    n_trials: int = 40,
    search_timesteps: int = 2_000_000,
    num_envs: int = 1024,
    seed: int = 42,
    study_name: str = "reward_weight_tuning",
    n_jobs: int = 1,
    threads_per_trial: Optional[int] = None,
) -> optuna.Study:
    """Run the full Optuna study.

    Parameters
    ----------
    n_trials : int
        Number of configurations to evaluate (30–50 recommended).
    search_timesteps : int
        Training budget per trial.  2M ≈ 5 min on T4.
    num_envs : int
        Parallel environments per trial.
    n_jobs : int
        Number of trials to run in parallel.  On CPU, set to
        ``total_cores // threads_per_trial`` for full utilisation.
    threads_per_trial : int, optional
        PyTorch CPU threads per trial.  When *n_jobs* > 1 on CPU,
        set this to ``total_cores // n_jobs`` to avoid over-subscription.

    Returns
    -------
    optuna.Study
        Contains ``best_params``, ``best_value``, per-trial data, etc.
    """
    print("Computing adaptive baselines...")
    baselines = compute_baselines(device)
    for hr, speed in sorted(baselines.items()):
        print(f"  HR={hr:.2f}: {speed:.3f} m/s")
    print()

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
        ),
    )

    objective = create_objective(
        device=device,
        baselines=baselines,
        search_timesteps=search_timesteps,
        num_envs=num_envs,
        seed=seed,
        threads_per_trial=threads_per_trial,
    )

    gpu_rate = 5  # min per trial at 2M steps on T4
    if device.type == "cpu":
        # CPU is ~8-12× slower than T4; parallel trials help
        serial_min = n_trials * search_timesteps / 2_000_000 * 45
        effective_min = serial_min / max(n_jobs, 1)
    else:
        effective_min = n_trials * search_timesteps / 2_000_000 * gpu_rate

    print(f"Starting {n_trials} trials, ~{search_timesteps / 1e6:.1f}M steps each")
    if n_jobs > 1:
        print(
            f"Running {n_jobs} trials in parallel ({threads_per_trial or '?'} threads each)"
        )
    print(f"Estimated wall-clock: ~{effective_min:.0f} min (pruning will shorten this)")
    print("=" * 80)

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # ---- Summary ----
    print("\n" + "=" * 80)
    n_pruned = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    )
    n_complete = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    print(f"Completed: {n_complete}, Pruned: {n_pruned}, Total: {len(study.trials)}")
    print(f"\nBEST TRIAL: #{study.best_trial.number}")
    print(f"Best metric (speed improvement over adaptive): {study.best_value:+.3f} m/s")
    print(f"\nOptimal reward weights:")
    for k, v in sorted(study.best_params.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print("=" * 80)

    return study


def best_params_as_code(study: optuna.Study) -> str:
    """Return a copy-paste Python snippet for the best configuration."""
    p = study.best_params
    lines = [
        "# Auto-tuned reward weights (Optuna)",
        f"# Best metric: {study.best_value:+.4f} m/s improvement",
        f"W_V = {p['w_v']:.4f}",
        f"W_J = {p['w_j']:.4f}",
        f"W_SS = {p['w_ss']:.4f}",
        f"W_SIGMA = {p['w_sigma']:.4f}",
        f"W_SIGMA_CAV = {p['w_sigma_cav']:.4f}",
        f"W_DAMP = {p['w_damp']:.4f}",
        f"W_ALPHA_FINAL = {p['w_alpha_final']:.4f}",
        f"V_REF_DELTA = {p['v_ref_delta']:.4f}",
        f"HIDDEN_DIM = {p['hidden_dim']}",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# CLI entry point
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto-tune RL reward weights via Bayesian optimisation"
    )
    parser.add_argument("--n_trials", type=int, default=40)
    parser.add_argument("--search_steps", type=int, default=2_000_000)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel trials (CPU: set to total_cores // threads_per_trial)",
    )
    parser.add_argument(
        "--threads_per_trial",
        type=int,
        default=None,
        help="PyTorch CPU threads per trial (CPU: total_cores // n_jobs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Accelerator backend to use for the short tuning runs.",
    )
    args = parser.parse_args()

    device = select_torch_device(args.device)
    print(f"Device: {device}")

    study = run_study(
        device=device,
        n_trials=args.n_trials,
        search_timesteps=args.search_steps,
        num_envs=args.num_envs,
        seed=args.seed,
        n_jobs=args.n_jobs,
        threads_per_trial=args.threads_per_trial,
    )

    print("\n" + best_params_as_code(study))



