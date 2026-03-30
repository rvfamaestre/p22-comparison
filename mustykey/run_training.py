# -*- coding: utf-8 -*-
"""
Run RL training for the residual headway controller.
Supports PPO and SAC through a config switch.
Tracks traffic metrics over episodes and saves learning curves.
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils.config_manager import ConfigManager
from src.utils.early_stopping import EarlyStoppingConfig, EarlyStopMonitor
from src.simulation.scenario_manager import ScenarioManager
from src.mesoscopic.rl_training import PPOTrainer
from src.mesoscopic.sac_training import SACTrainer


METRIC_KEYS = (
    "speed_var_global",
    "speed_std_time_mean",
    "oscillation_amplitude",
    "min_gap",
    "rms_acc",
    "rms_jerk",
    "mean_speed",
    "mean_speed_last100",
)


# -------------------------------------------------------------
# Traffic evaluation metrics
# -------------------------------------------------------------
def compute_metrics(df, dt):
    """Compute the scalar traffic metrics used for checkpoint selection."""
    v = df["v"].to_numpy()
    a = df["a"].to_numpy()

    metrics = {}

    # global variance
    metrics["speed_var_global"] = float(np.var(v))

    # wave strength
    metrics["speed_std_time_mean"] = float(
        df.groupby("t")["v"].std().mean()
    )

    # oscillation amplitude
    metrics["oscillation_amplitude"] = float(v.max() - v.min())

    # acceleration comfort
    metrics["rms_acc"] = float(np.sqrt(np.mean(a ** 2)))

    metrics["mean_speed"] = float(np.mean(v))

    # mean speed over last 100 time steps
    times = np.sort(df["t"].unique())
    last100_times = times[-min(100, len(times)):]
    mask = df["t"].isin(last100_times)
    metrics["mean_speed_last100"] = float(df.loc[mask, "v"].mean())

    # jerk calculation
    jerk_vals = []

    for _, g in df.groupby("id"):
        g = g.sort_values("t")
        jerk = np.diff(g["a"].to_numpy()) / dt

        if len(jerk) > 0:
            jerk_vals.append(np.mean(jerk ** 2))

    metrics["rms_jerk"] = float(np.sqrt(np.mean(jerk_vals))) if len(jerk_vals) > 0 else 0.0

    # minimum gap if available
    if "gap_s" in df.columns:
        metrics["min_gap"] = float(df["gap_s"].min())
    else:
        metrics["min_gap"] = np.nan

    return metrics


# -------------------------------------------------------------
# Plot learning curves
# -------------------------------------------------------------
def plot_metrics(metrics_history, model_dir):
    for key in metrics_history:
        plt.figure()
        plt.plot(metrics_history[key], marker="o")
        plt.title(f"Training Curve: {key}")
        plt.xlabel("Episode")
        plt.ylabel(key)
        plt.grid(True)
        plt.savefig(os.path.join(model_dir, f"{key}_curve.png"))
        plt.close()


def build_trainer(config, train_cfg, algorithm):
    """Create the configured PPO or SAC trainer."""
    if algorithm == "ppo":
        return PPOTrainer(
            state_dim=8,
            lr=train_cfg["lr"],
            gamma=train_cfg["gamma"],
            clip=train_cfg["clip"],
            train_epochs=train_cfg["train_epochs"],
            gae_lambda=train_cfg.get("gae_lambda", 0.95),
            minibatch_size=train_cfg.get("minibatch_size", 256),
            value_coef=train_cfg.get("value_coef", 0.5),
            entropy_coef=train_cfg.get("entropy_coef", 1e-3),
            max_grad_norm=train_cfg.get("max_grad_norm", 0.5),
            reward_cfg=train_cfg.get("reward", {}),
            cth_cfg=config.get("cth_params", {}),
            acc_cfg=config.get("acc_params", {}),
        )

    if algorithm == "sac":
        return SACTrainer(
            state_dim=8,
            lr=train_cfg["lr"],
            gamma=train_cfg["gamma"],
            train_epochs=train_cfg["train_epochs"],
            reward_cfg=train_cfg.get("reward", {}),
            cth_cfg=config.get("cth_params", {}),
            acc_cfg=config.get("acc_params", {}),
            tau=train_cfg.get("tau", 0.005),
            entropy_alpha=train_cfg.get("entropy_alpha", 0.2),
            batch_size=train_cfg.get("batch_size", 256),
            action_limit=config.get("rl_layer", {}).get("delta_alpha_max", 0.1),
            replay_size=train_cfg.get("replay_size", 100000),
            replay_warmup=train_cfg.get("replay_warmup", 2048),
            auto_entropy_tuning=train_cfg.get("auto_entropy_tuning", True),
            target_entropy=train_cfg.get("target_entropy", -1.0),
            normalize_states=train_cfg.get("normalize_states", True),
            normalize_rewards=train_cfg.get("normalize_rewards", True),
            reward_scale_epsilon=train_cfg.get("reward_scale_epsilon", 1e-6),
        )

    raise ValueError(f"Unsupported RL algorithm: {algorithm}")


def attach_trainer_policy(sim, trainer, algorithm):
    """
    Reuse the trainer's live policy inside the simulator.

    This keeps action sampling during rollout and parameter updates in sync.
    """
    if sim.rl_layer is None:
        return

    sim.rl_layer.config.algorithm = algorithm
    sim.rl_layer.model = trainer.model
    sim.rl_layer.model.train()
    sim.rl_layer.policy_available = True
    sim.rl_layer.reset()


def collect_rollout_transitions(env, trainer, dt, delta_alpha_max):
    """
    Rebuild the trainer batch from the per-vehicle rollout history saved by the
    simulator. The stored action is always the executed bounded residual action.
    """
    transitions = 0

    for vehicle in env.vehicles:
        history = getattr(vehicle, "_rl_history", None)
        if not history:
            continue

        for idx, step in enumerate(history):
            state = step.get("state", None)
            log_prob = step.get("log_prob", None)
            value = step.get("value", None)

            if state is None or log_prob is None or value is None:
                continue

            done = (idx == len(history) - 1)
            action_raw = step.get("action_raw", 0.0)
            action = step.get("action_applied", action_raw)
            action_low = step.get("action_low", -delta_alpha_max)
            action_high = step.get("action_high", delta_alpha_max)
            next_state = state if done else history[idx + 1].get("state", state)
            next_action_low = action_low if done else history[idx + 1].get("action_low", action_low)
            next_action_high = action_high if done else history[idx + 1].get("action_high", action_high)

            reward = trainer.compute_reward(
                v=step.get("v", 0.0),
                s=step.get("s", 0.0),
                mu_v=step.get("mu_v", 0.0),
                sigma_v2=step.get("sigma_v2", 0.0),
                alpha=step.get("alpha", 1.0),
                a=step.get("a", 0.0),
                a_prev=step.get("a_prev", 0.0),
                dt=dt,
                e_leader=step.get("e_leader", 0.0),
            )

            trainer.store_step(
                state=state,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward,
                next_state=next_state,
                done=done,
                action_low=action_low,
                action_high=action_high,
                next_action_low=next_action_low,
                next_action_high=next_action_high,
            )
            transitions += 1

    return transitions


def main():
    parser = argparse.ArgumentParser(description="Train RL policy for ring-road control")

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML file"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes (overrides config)"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load configuration
    # ---------------------------------------------------------
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = ConfigManager.load(args.config)
    train_cfg = config.get("rl_training", {})

    # Enable RL training
    if "rl_layer" not in config:
        config["rl_layer"] = {}

    config["rl_layer"]["enabled"] = True
    config["rl_layer"]["mode"] = "train"

    # ---------------------------------------------------------
    # Create output folders
    # ---------------------------------------------------------
    base_output = config["output_path"]

    training_output = os.path.join(base_output, "training")
    simulation_output = os.path.join(base_output, "simulation")

    os.makedirs(training_output, exist_ok=True)
    os.makedirs(simulation_output, exist_ok=True)

    # During training, all generated files go into output/.../training
    config["output_path"] = training_output

    model_dir = os.path.join(training_output, "trained_models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "rl_policy.pt")
    best_model_path = os.path.join(model_dir, "rl_policy_best.pt")

    print(f"[Training] Models will be saved to: {model_path}")

    # ---------------------------------------------------------
    # RL trainer
    # ---------------------------------------------------------
    algorithm = config.get("rl_layer", {}).get("algorithm", "ppo").lower()
    trainer = build_trainer(config, train_cfg, algorithm)

    print(f"[Training] Algorithm: {algorithm.upper()}")

    episodes = args.episodes if args.episodes is not None else train_cfg.get("episodes", 200)
    save_every = train_cfg.get("save_every", 20)
    dt = config["dt"]
    delta_alpha_max = config.get("rl_layer", {}).get("delta_alpha_max", 0.1)

    # ---------------------------------------------------------
    # Store metrics over episodes
    # ---------------------------------------------------------
    metrics_history = {key: [] for key in METRIC_KEYS}

    # ---------------------------------------------------------
    # Early stopping (EarlyStopMonitor with EMA smoothing)
    # ---------------------------------------------------------
    es_cfg_dict = train_cfg.get("early_stopping", {})
    es_config = EarlyStoppingConfig(
        enabled=es_cfg_dict.get("enabled", True),
        metric=es_cfg_dict.get("metric", "speed_var_global"),
        mode=es_cfg_dict.get("mode", "min"),
        patience=es_cfg_dict.get("patience", train_cfg.get("early_stop_patience", 20)),
        min_delta=es_cfg_dict.get("min_delta", 0.001),
        start_step=es_cfg_dict.get("start_step", max(episodes // 4, 1)),
        min_checks=es_cfg_dict.get("min_checks", 4),
        ema_alpha=es_cfg_dict.get("ema_alpha", 0.3),
        restore_best=es_cfg_dict.get("restore_best", True),
    )
    es_monitor = EarlyStopMonitor(es_config)
    print(f"[EarlyStop] metric={es_config.metric}, mode={es_config.mode}, "
          f"patience={es_config.patience}, ema_alpha={es_config.ema_alpha}, "
          f"start_step={es_config.start_step}, min_checks={es_config.min_checks}")

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    for episode in tqdm(range(episodes), desc="Training"):
        print("\n========================================")
        print(f"[Training] Episode {episode}")
        print("========================================")

        manager = ScenarioManager(config)
        sim = manager.build(live_viz=None)
        sim.quiet = True

        attach_trainer_policy(sim, trainer, algorithm)

        sim.run()

        # -----------------------------------------------------
        # Collect RL transitions from the simulator-side rollout history
        # -----------------------------------------------------
        transitions = collect_rollout_transitions(sim.env, trainer, dt, delta_alpha_max)

        print(f"[Training] Collected {transitions} transitions")

        # -----------------------------------------------------
        # RL update
        # -----------------------------------------------------
        if transitions > 0:
            trainer.train()
        else:
            print("[Training] WARNING: No transitions collected, skipping RL update")

        # -----------------------------------------------------
        # Load micro trajectory file for evaluation
        # -----------------------------------------------------
        micro_file = os.path.join(training_output, "micro.csv")

        if os.path.isfile(micro_file):
            df = pd.read_csv(micro_file)
            metrics = compute_metrics(df, dt)

            for k in metrics_history:
                metrics_history[k].append(metrics[k])

            print("[Metrics]")
            for k, v_metric in metrics.items():
                print(f"{k:25s}: {v_metric:.4f}")

            current_metric = metrics[es_config.metric]
            es_result = es_monitor.update(current_metric, step=episode)

            if es_result["improved"]:
                trainer.save_model(best_model_path)
                trainer.save_model(model_path)
                print(
                    f"[EarlyStop] New best {es_config.metric} = "
                    f"{es_result['best_raw_metric']:.4f} "
                    f"(smoothed={es_result['smoothed_metric']:.4f}) "
                    f"at episode {episode} -> model saved"
                )
            else:
                print(
                    f"[EarlyStop] No improvement in {es_config.metric} "
                    f"for {es_result['checks_since_improvement']}/{es_config.patience} "
                    f"checks (smoothed={es_result['smoothed_metric']:.4f})"
                )

            if es_config.enabled and es_result["should_stop"]:
                print(
                    f"[EarlyStop] Stopping early at episode {episode}: "
                    f"{es_config.metric} plateaued for {es_config.patience} checks. "
                    f"Best={es_result['best_raw_metric']:.4f} at episode {es_result['best_step']}"
                )
                if es_config.restore_best:
                    print(f"[EarlyStop] Restoring best model from {best_model_path}")
                    trainer.load_model(best_model_path)
                break

        # -----------------------------------------------------
        # Save model periodically
        # -----------------------------------------------------
        if (episode + 1) % save_every == 0:
            trainer.save_model(model_path)
            print(f"[Training] Model saved: {model_path}")

    # ---------------------------------------------------------
    # Final save
    # ---------------------------------------------------------
    trainer.save_model(model_path)

    # ---------------------------------------------------------
    # Save training statistics
    # ---------------------------------------------------------
    stats_file = os.path.join(model_dir, "training_metrics.json")

    with open(stats_file, "w") as f:
        json.dump(metrics_history, f, indent=4)

    es_summary_file = os.path.join(model_dir, "early_stopping_summary.json")
    with open(es_summary_file, "w") as f:
        json.dump(es_monitor.as_dict(), f, indent=4)
    print(f"[Training] Early stopping summary saved to {es_summary_file}")

    print(f"[Training] Metrics saved to {stats_file}")

    # ---------------------------------------------------------
    # Plot learning curves
    # ---------------------------------------------------------
    plot_metrics(metrics_history, model_dir)

    print("\n========================================")
    print("[Training] Training complete")
    print(f"[Training] Final model saved: {model_path}")
    print("========================================")


if __name__ == "__main__":
    main()
