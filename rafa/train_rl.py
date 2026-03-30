# -*- coding: utf-8 -*-
"""
train_rl.py – PPO training loop for residual-headway CAV control (CPU path).

Usage:
    python train_rl.py --config config/rl_smoke.yaml

The script:
  1. builds the ring-road simulator via ScenarioManager
  2. wraps it with ObservationBuilder, RewardFunction, ActionAdapter
  3. collects rollouts and runs PPO updates
  4. periodically evaluates and saves checkpoints
"""

import argparse
import os
import time
from copy import deepcopy

import numpy as np
import yaml

from src.simulation.scenario_manager import ScenarioManager
from src.vehicles.cav_vehicle import CAVVehicle
from src.agents.rl_types import RLConfig, OBS_DIM
from src.agents.observation_builder import ObservationBuilder
from src.agents.reward import RewardFunction
from src.agents.action_adapter import ActionAdapter
from src.agents.ppo_trainer import PPOTrainer


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_rl_config(cfg: dict) -> RLConfig:
    """Construct RLConfig from the 'rl' sub-dict of the YAML config."""
    rc = cfg.get("rl", {})
    return RLConfig(
        # Action
        delta_alpha_max=rc.get("delta_alpha_max", 0.5),
        alpha_min=rc.get("alpha_min", 0.5),
        alpha_max=rc.get("alpha_max", 2.0),
        # Reward weights — run-10 defaults kept in RLConfig
        w_s=rc.get("w_s", 5.0),
        w_tau=rc.get("w_tau", 3.0),
        w_v=rc.get("w_v", 5.0),
        w_j=rc.get("w_j", 0.15),
        w_ss=rc.get("w_ss", 0.75),
        w_sigma=rc.get("w_sigma", 0.5),
        w_sigma_cav=rc.get("w_sigma_cav", 2.0),
        w_alpha=rc.get("w_alpha", 0.3),
        w_alpha_final=rc.get("w_alpha_final", 0.10),
        w_damp=rc.get("w_damp", 0.3),
        sigma_ref_damp=rc.get("sigma_ref_damp", 1.0),
        # Collision floor (run-11)
        w_collision_floor=rc.get("w_collision_floor", 10.0),
        collision_gap_critical=rc.get("collision_gap_critical", 1.0),
        # Reference values
        s_min=rc.get("s_min", 2.0),
        tau_min=rc.get("tau_min", 0.8),
        v_ref=rc.get("v_ref", 5.5),
        adaptive_v_ref=rc.get("adaptive_v_ref", True),
        v_ref_delta=rc.get("v_ref_delta", 3.0),
        j_ref=rc.get("j_ref", 5.0),
        eps_v=rc.get("eps_v", 0.1),
        fleet_penalty_scaling=rc.get("fleet_penalty_scaling", True),
        # PPO hyper-parameters
        gamma=rc.get("gamma", 0.99),
        lam_gae=rc.get("lam_gae", 0.95),
        eps_clip=rc.get("eps_clip", 0.2),
        lr_actor=rc.get("lr_actor", 3e-4),
        lr_critic=rc.get("lr_critic", 1e-3),
        entropy_coef=rc.get("entropy_coef", 0.01),
        value_coef=rc.get("value_coef", 0.5),
        max_grad_norm=rc.get("max_grad_norm", 0.5),
        ppo_epochs=rc.get("ppo_epochs", 10),
        minibatch_size=rc.get("minibatch_size", 64),
        rollout_steps=rc.get("rollout_steps", 1024),
        total_timesteps=rc.get("total_timesteps", 1_000_000),
        eval_interval=rc.get("eval_interval", 10_000),
        log_interval=rc.get("log_interval", 1_000),
        save_interval=rc.get("save_interval", 200_000),
        # Network
        hidden_dim=rc.get("hidden_dim", 128),
        num_hidden=rc.get("num_hidden", 2),
        use_meso_baseline=rc.get("use_meso_baseline", True),
        rl_mode="residual",
        # Domain randomisation
        hr_options=rc.get("hr_options", [0.0, 0.25, 0.5, 0.75]),
        hr_weights=rc.get("hr_weights", None),
        shuffle_cav_positions=rc.get("shuffle_cav_positions", True),
        normalizer_warmup_steps=rc.get("normalizer_warmup_steps", 50_000),
    )


def count_cavs(vehicles) -> int:
    return sum(1 for v in vehicles if isinstance(v, CAVVehicle))


# ------------------------------------------------------------------
# Env wrapper (thin: just resets the simulator for each episode)
# ------------------------------------------------------------------


class SimEnv:
    """Wraps ScenarioManager + Simulator into a step-based RL interface."""

    def __init__(self, base_config: dict, rl_cfg: RLConfig):
        self.base_config = base_config
        self.rl_cfg = rl_cfg
        self.M = base_config.get("mesoscopic", {}).get("M", 8)
        self.obs_builder = ObservationBuilder(M=self.M)
        self.reward_fn = RewardFunction(rl_cfg)
        self.action_adapter = ActionAdapter(rl_cfg)
        self.sim = None
        self.num_cav = 0
        self._alpha_prev = {}
        self._hr_options = rl_cfg.hr_options
        self._hr_weights = rl_cfg.hr_weights
        self._current_hr = base_config.get("human_ratio", 0.75)

    def reset(self) -> np.ndarray:
        """Build a fresh simulator and return the initial observation.

        Domain randomization: samples a random human_rate each episode
        from hr_options (with optional hr_weights) so the policy learns
        conditional behaviour.
        """
        if self._hr_options:
            p = None
            if self._hr_weights is not None:
                w = np.array(self._hr_weights, dtype=np.float64)
                p = w / w.sum()
            self._current_hr = float(np.random.choice(self._hr_options, p=p))

        cfg = deepcopy(self.base_config)
        cfg["human_ratio"] = self._current_hr
        cfg.setdefault("rl", {})["rl_mode"] = "residual"
        # Propagate shuffle option
        cfg["shuffle_cav_positions"] = self.rl_cfg.shuffle_cav_positions
        cfg["enable_live_viz"] = False
        cfg["play_recording"] = False

        mgr = ScenarioManager(cfg)
        self.sim = mgr.build(live_viz=None)
        self.num_cav = count_cavs(self.sim.env.vehicles)
        self._alpha_prev = {
            v.id: 1.0 for v in self.sim.env.vehicles if isinstance(v, CAVVehicle)
        }
        self.reward_fn.reset()

        # Run one step to initialise leaders/gaps
        self.sim.step()

        obs, cav_ids = self.obs_builder.build(
            self.sim.env.vehicles, self.sim.env.L, self._alpha_prev
        )
        self._cav_ids = cav_ids
        return obs

    def step(self, actions: np.ndarray, current_w_alpha: float = None):
        """Apply RL actions and advance one simulation step.

        Parameters
        ----------
        actions : np.ndarray (num_cav, 1)
            Residual delta-alpha for each CAV.
        current_w_alpha : float, optional
            Annealed regularisation weight (passed to reward function).

        Returns
        -------
        obs      : np.ndarray (num_cav, OBS_DIM)
        reward   : float   (global scalar)
        done     : bool
        info     : dict
        """
        delta_alphas = {
            cid: float(actions[k, 0]) for k, cid in enumerate(self._cav_ids)
        }

        self.sim.set_rl_actions(delta_alphas)
        self.sim.step()

        done = self.sim.done

        reward_info = self.reward_fn.compute(
            self.sim.env.vehicles,
            self.sim.env.L,
            self.sim.dt,
            delta_alphas,
            current_w_alpha=current_w_alpha,
        )

        for v in self.sim.env.vehicles:
            if isinstance(v, CAVVehicle):
                self._alpha_prev[v.id] = getattr(v, "_meso_alpha", 1.0)

        obs, cav_ids = self.obs_builder.build(
            self.sim.env.vehicles, self.sim.env.L, self._alpha_prev
        )
        self._cav_ids = cav_ids

        return obs, reward_info["total"], done, reward_info


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------


def train(config_path: str):
    cfg = load_config(config_path)
    rl_cfg = build_rl_config(cfg)

    output_dir = cfg.get("rl", {}).get("output_path", "output/rl_train")
    os.makedirs(output_dir, exist_ok=True)

    # Save effective config
    with open(os.path.join(output_dir, "config_effective.yaml"), "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    env = SimEnv(cfg, rl_cfg)
    obs = env.reset()
    num_cav = env.num_cav
    while num_cav == 0:
        obs = env.reset()
        num_cav = env.num_cav

    print(f"[train_rl] num_cav = {num_cav}, obs_dim = {OBS_DIM}")
    print(f"[train_rl] Domain randomization HR options: {rl_cfg.hr_options}")
    print(f"[train_rl] Shuffle CAV positions: {rl_cfg.shuffle_cav_positions}")

    trainer = PPOTrainer(rl_cfg, num_cav)

    # Store initial learning rates for linear schedule
    initial_lrs = [pg["lr"] for pg in trainer.optimizer.param_groups]

    total_steps = rl_cfg.total_timesteps
    rollout_len = rl_cfg.rollout_steps
    warmup_steps = rl_cfg.normalizer_warmup_steps
    global_step = 0
    episode = 0
    ep_reward = 0.0
    ep_len = 0
    normalizer_frozen = False

    t0 = time.time()
    print(f"[train_rl] Starting training for {total_steps} timesteps ...")
    print(f"[train_rl] Normalizer freezes after {warmup_steps} steps")

    while global_step < total_steps:
        # ---------- Learning rate schedule (linear decay) ----------
        progress = global_step / max(total_steps, 1)
        lr_frac = max(1.0 - progress, 0.0)
        for pg, base_lr in zip(trainer.optimizer.param_groups, initial_lrs):
            pg["lr"] = base_lr * lr_frac

        # ---------- w_alpha annealing ----------
        current_w_alpha = (
            rl_cfg.w_alpha + (rl_cfg.w_alpha_final - rl_cfg.w_alpha) * progress
        )

        # ---------- Freeze normalizer after warm-up ----------
        if not normalizer_frozen and global_step >= warmup_steps:
            if env.obs_builder.normalizer is not None:
                env.obs_builder.normalizer.freeze()
                print(
                    f"  [normalizer] Frozen at step {global_step} "
                    f"(count={env.obs_builder.normalizer.count})"
                )
            normalizer_frozen = True

        # ---------- Collect rollout ----------
        trainer.buffer.reset()
        if trainer.buffer.num_cav != num_cav:
            from src.agents.buffer import RolloutBuffer
            trainer.buffer = RolloutBuffer(rl_cfg, num_cav)

        for _ in range(rollout_len):
            actions, log_probs, values = trainer.select_actions(obs)
            next_obs, reward, done, info = env.step(actions, current_w_alpha=current_w_alpha)

            trainer.buffer.add(obs, actions, log_probs, reward, values, done)

            obs = next_obs
            ep_reward += reward
            ep_len += 1
            global_step += 1

            if done:
                episode += 1
                elapsed = time.time() - t0
                print(
                    f"[ep {episode}] steps={global_step} "
                    f"ep_reward={ep_reward:.2f} ep_len={ep_len} "
                    f"hr={env._current_hr:.2f} ncav={num_cav} "
                    f"w_alpha={current_w_alpha:.4f} elapsed={elapsed:.0f}s"
                )
                ep_reward = 0.0
                ep_len = 0
                obs = env.reset()
                new_num_cav = env.num_cav

                while new_num_cav == 0:
                    obs = env.reset()
                    new_num_cav = env.num_cav

                if new_num_cav != num_cav:
                    if trainer.buffer.ptr > 0:
                        trainer.update()
                    num_cav = new_num_cav
                    from src.agents.buffer import RolloutBuffer
                    trainer.buffer = RolloutBuffer(rl_cfg, num_cav)

            if global_step >= total_steps:
                break

        # ---------- PPO update ----------
        if trainer.buffer.ptr > 0:
            update_info = trainer.update()
            trainer.global_step = global_step

            if global_step % rl_cfg.log_interval < rollout_len:
                print(
                    f"  [update] step={global_step}  "
                    f"pg_loss={update_info['pg_loss']:.4f}  "
                    f"v_loss={update_info['v_loss']:.4f}  "
                    f"entropy={update_info['entropy']:.4f}  "
                    f"w_alpha={current_w_alpha:.4f}"
                )

        # ---------- Save checkpoint ----------
        if global_step % rl_cfg.save_interval < rollout_len:
            ckpt_path = os.path.join(output_dir, f"ckpt_{global_step}.pt")
            trainer.save(ckpt_path)
            print(f"  [save] checkpoint -> {ckpt_path}")

    # Final save
    trainer.save(os.path.join(output_dir, "ckpt_final.pt"))
    if env.obs_builder.normalizer is not None:
        env.obs_builder.normalizer.save(os.path.join(output_dir, "obs_normalizer.npz"))
    elapsed = time.time() - t0
    print(f"[train_rl] Done. {global_step} steps in {elapsed:.0f}s")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RL residual-headway agent (CPU)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/rl_smoke.yaml",
        help="Path to training config YAML",
    )
    return parser


# ------------------------------------------------------------------
if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args.config)
