# -*- coding: utf-8 -*-
"""Minimal real-time HUD for live ring-road simulation control."""

import copy
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from matplotlib import animation
from matplotlib.widgets import Button, Slider

from src.agents.observation_builder import ObservationBuilder
from src.gpu.gpu_ppo import GPURunningNormalizer
from src.gpu.gpu_sac import SACActorGPU
from src.simulation.scenario_manager import ScenarioManager
from src.vehicles.cav_vehicle import CAVVehicle


def compute_steps_per_tick(dt_seconds, redraw_interval_ms, speed_multiplier, accumulator):
    """
    Convert a wall-clock redraw cadence into a number of simulation steps.

    The accumulator preserves fractional steps across redraws so x1, x2, etc.
    remain meaningful even when the redraw interval is shorter than dt.
    """
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive")
    if redraw_interval_ms <= 0:
        raise ValueError("redraw_interval_ms must be positive")
    if speed_multiplier <= 0:
        raise ValueError("speed_multiplier must be positive")

    sim_seconds_per_tick = speed_multiplier * (redraw_interval_ms / 1000.0)
    accumulator += sim_seconds_per_tick / dt_seconds
    steps = int(np.floor(accumulator + 1e-9))
    accumulator -= steps
    return steps, accumulator


def trim_history_window(history, window_seconds):
    """Trim time-series history in-place to a rolling time window."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if not history["t"]:
        return

    cutoff = history["t"][-1] - window_seconds
    while history["t"] and history["t"][0] < cutoff:
        for series in history.values():
            series.popleft()


class LiveSACSession:
    """Runtime wrapper that advances the simulator on demand."""

    def __init__(
        self,
        base_config,
        ckpt_path,
        scenario_state,
        output_dir,
        history_window_seconds=100.0,
    ):
        self.base_config = copy.deepcopy(base_config)
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.history_window_seconds = float(history_window_seconds)
        if self.history_window_seconds <= 0:
            raise ValueError("history_window_seconds must be positive")
        (
            self.policy,
            self.normalizer,
            self.obs_dim,
            self.hidden_dim,
        ) = self._load_policy_components(ckpt_path)
        self.scenario_state = dict(scenario_state)
        self.reset(self.scenario_state)

    @staticmethod
    def _load_policy_components(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        obs_dim = ckpt["actor_state_dict"]["backbone.0.weight"].shape[1]
        hidden_dim = ckpt["actor_state_dict"]["backbone.0.weight"].shape[0]

        policy = SACActorGPU(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            num_hidden=2,
            delta_alpha_max=0.5,
        )
        policy.load_state_dict(ckpt["actor_state_dict"])
        policy.eval()

        normalizer = GPURunningNormalizer(obs_dim, device=torch.device("cpu"))
        if "normalizer" in ckpt:
            nd = ckpt["normalizer"]
            if hasattr(nd["mean"], "numpy"):
                normalizer.mean = torch.tensor(nd["mean"].numpy(), dtype=torch.float32)
                normalizer.var = torch.tensor(nd["var"].numpy(), dtype=torch.float32)
                normalizer._M2 = torch.tensor(nd["M2"].numpy(), dtype=torch.float32)
            else:
                normalizer.mean = torch.tensor(nd["mean"], dtype=torch.float32)
                normalizer.var = torch.tensor(nd["var"], dtype=torch.float32)
                normalizer._M2 = torch.tensor(nd["M2"], dtype=torch.float32)
            normalizer.count = int(nd["count"])
        normalizer.freeze()
        return policy, normalizer, obs_dim, hidden_dim

    def _build_runtime_config(self, state):
        config = copy.deepcopy(self.base_config)
        config.setdefault("mesoscopic", {})
        config.setdefault("rl", {})

        config["human_ratio"] = float(state["human_ratio"])
        config["seed"] = int(state["seed"])
        config["initial_conditions"] = state["initial_conditions"]
        config["perturbation_enabled"] = bool(state["perturbation_enabled"])

        shuffle = bool(state["shuffle_cav_positions"])
        config["shuffle_cav_positions"] = shuffle
        config["rl"]["shuffle_cav_positions"] = shuffle

        mode = state["mode"]
        config["viz_mode"] = mode
        if mode == "baseline":
            config["mesoscopic"]["enabled"] = False
            config["rl"]["rl_mode"] = "off"
        elif mode == "adaptive":
            config["mesoscopic"]["enabled"] = True
            config["rl"]["rl_mode"] = "off"
        elif mode == "rl":
            config["mesoscopic"]["enabled"] = True
            config["rl"]["rl_mode"] = "residual"
            config["rl"]["delta_alpha_max"] = 0.5
            config["rl"]["alpha_min"] = 0.5
            config["rl"]["alpha_max"] = 2.0
            config["rl"]["hidden_dim"] = self.hidden_dim
            config["rl"]["num_hidden"] = 2
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        config["enable_live_viz"] = False
        config["play_recording"] = False
        config["logging_enabled"] = False
        config["compute_macro_fields"] = False
        config["step_log_interval"] = 0
        config["run_forever"] = True
        config["output_path"] = self.output_dir
        config.setdefault("dx", 1.0)
        config.setdefault("kernel_h", 3.0)
        return config

    def reset(self, scenario_state=None):
        if scenario_state is not None:
            self.scenario_state = dict(scenario_state)

        self.config = self._build_runtime_config(self.scenario_state)
        self.mode = self.scenario_state["mode"]
        self.sim = ScenarioManager(self.config).build(live_viz=None)

        meso_M = self.config.get("mesoscopic", {}).get("M", 8)
        self.obs_builder = (
            ObservationBuilder(M=meso_M, normalize=False, obs_dim=self.obs_dim)
            if self.mode == "rl"
            else None
        )

        cav_ids = [v.id for v in self.sim.env.vehicles if isinstance(v, CAVVehicle)]
        self.alpha_prev = {cid: 1.0 for cid in cav_ids}
        self.history = {
            "t": deque(),
            "mean_speed": deque(),
            "min_gap": deque(),
        }
        self._append_history()

    def _append_history(self):
        snap = self.snapshot()
        self.history["t"].append(snap["time"])
        self.history["mean_speed"].append(snap["mean_speed"])
        self.history["min_gap"].append(snap["min_gap"])
        trim_history_window(self.history, self.history_window_seconds)

    def step(self, count=1):
        advanced = 0
        for _ in range(count):
            if self.sim.done:
                break

            if self.mode == "rl" and self.obs_builder is not None:
                obs, cav_ids = self.obs_builder.build(
                    self.sim.env.vehicles, self.sim.env.L, self.alpha_prev
                )
                if len(cav_ids) > 0:
                    obs_t = torch.tensor(obs, dtype=torch.float32)
                    obs_t_norm = self.normalizer.normalize(obs_t)
                    with torch.no_grad():
                        actions, _ = self.policy.get_action(
                            obs_t_norm, deterministic=True
                        )
                    actions_np = actions.cpu().numpy()
                    self.sim.set_rl_actions(
                        {
                            cid: float(actions_np[k, 0])
                            for k, cid in enumerate(cav_ids)
                        }
                    )

            self.sim.step()

            for vehicle in self.sim.env.vehicles:
                if isinstance(vehicle, CAVVehicle):
                    self.alpha_prev[vehicle.id] = getattr(vehicle, "_meso_alpha", 1.0)

            self._append_history()
            advanced += 1

        return advanced

    def snapshot(self):
        by_id = sorted(self.sim.env.vehicles, key=lambda vehicle: vehicle.id)
        positions = np.array([vehicle.x for vehicle in by_id], dtype=np.float32)
        speeds = np.array([vehicle.v for vehicle in by_id], dtype=np.float32)
        alphas = np.array(
            [getattr(vehicle, "_meso_alpha", 1.0) for vehicle in by_id],
            dtype=np.float32,
        )
        is_cav = np.array(
            [isinstance(vehicle, CAVVehicle) for vehicle in by_id], dtype=bool
        )
        min_gap = min(vehicle.compute_gap(self.sim.env.L) for vehicle in self.sim.env.vehicles)
        return {
            "time": float(self.sim.current_time),
            "step": int(self.sim.current_step),
            "positions": positions,
            "speeds": speeds,
            "alphas": alphas,
            "is_cav": is_cav,
            "mean_speed": float(speeds.mean()),
            "min_gap": float(min_gap),
            "done": self.sim.done,
        }


class MinimalLiveHUD:
    """Small HUD that advances the simulator only while playing."""

    MODE_ORDER = ["baseline", "adaptive", "rl"]
    HR_OPTIONS = [0.0, 0.25, 0.5, 0.75]
    INIT_OPTIONS = ["uniform", "random"]

    def __init__(self, session, radius=20.0, window_seconds=100.0):
        self.session = session
        self.radius = float(radius)
        self.window_seconds = float(window_seconds)
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self.playing = False
        self.speed_multiplier = 1.0
        self.step_accumulator = 0.0

        self.fig = plt.figure(figsize=(11, 8))
        self.fig.canvas.manager.set_window_title("Live Ring-Road HUD")
        self.ax_ring = self.fig.add_axes([0.05, 0.22, 0.58, 0.70])
        self.ax_stats = self.fig.add_axes([0.70, 0.42, 0.25, 0.32])
        self.ax_info = self.fig.add_axes([0.70, 0.22, 0.25, 0.15])
        self.ax_info.set_axis_off()

        self._build_ring()
        self._build_stats()
        self._build_controls()

        # Keep rendering cadence fixed and scale the amount of simulated time
        # generated per tick via step accumulation.
        self.redraw_interval_ms = 20
        self.anim = animation.FuncAnimation(
            self.fig,
            self._animate,
            interval=self.redraw_interval_ms,
            blit=False,
            cache_frame_data=False,
        )
        self.anim.pause()
        self._draw()

    def _build_ring(self):
        self.ax_ring.set_aspect("equal", adjustable="box")
        margin = self.radius + 8
        self.ax_ring.set_xlim(-margin, margin)
        self.ax_ring.set_ylim(-margin, margin)
        self.ax_ring.set_xticks([])
        self.ax_ring.set_yticks([])

        road_width = 3.0
        self.ax_ring.add_patch(
            patches.Annulus(
                (0, 0),
                self.radius - road_width / 2,
                road_width,
                facecolor="#e8e6df",
                edgecolor="none",
                alpha=0.8,
            )
        )
        self.ax_ring.add_patch(
            patches.Circle((0, 0), self.radius + road_width / 2, fill=False, linewidth=2)
        )
        self.ax_ring.add_patch(
            patches.Circle((0, 0), self.radius - road_width / 2, fill=False, linewidth=2)
        )

        self.hdv_scatter = self.ax_ring.scatter([], [], s=55, c="#2563eb", label="HDV")
        self.cav_scatter = self.ax_ring.scatter(
            [], [], s=80, c="#ef4444", marker="s", label="CAV"
        )
        self.ax_ring.legend(loc="upper right")
        self.title = self.ax_ring.set_title("", fontsize=13, fontweight="bold")

    def _build_stats(self):
        self.ax_stats.set_title("Live Metrics", fontsize=10, fontweight="bold")
        self.ax_stats.set_xlabel("Time (s)")
        self.ax_stats.set_ylabel("Mean speed (m/s)")
        self.ax_stats.grid(alpha=0.25, linestyle="--")
        self.speed_line, = self.ax_stats.plot(
            [], [], color="#0f766e", linewidth=2.0, label="Mean speed"
        )
        self.cursor = self.ax_stats.axvline(0.0, color="#111827", linestyle="--", linewidth=1.1)
        self.ax_stats_gap = self.ax_stats.twinx()
        self.ax_stats_gap.set_ylabel("Min gap (m)")
        self.gap_line, = self.ax_stats_gap.plot(
            [], [], color="#b91c1c", linewidth=1.5, label="Min gap"
        )
        self.ax_stats.legend(
            [self.speed_line, self.gap_line],
            ["Mean speed", "Min gap"],
            loc="upper left",
            fontsize=8,
            frameon=False,
        )

    def _build_controls(self):
        self.buttons = {}
        specs = [
            ([0.05, 0.08, 0.08, 0.05], "Play", self._toggle_play),
            ([0.14, 0.08, 0.08, 0.05], "Step", self._step_once),
            ([0.23, 0.08, 0.08, 0.05], "Reset", self._reset_session),
            ([0.32, 0.08, 0.08, 0.05], "Mode", self._cycle_mode),
            ([0.41, 0.08, 0.08, 0.05], "HR", self._cycle_hr),
            ([0.50, 0.08, 0.08, 0.05], "Init", self._cycle_init),
            ([0.59, 0.08, 0.08, 0.05], "Shuffle", self._toggle_shuffle),
            ([0.68, 0.08, 0.08, 0.05], "Perturb", self._toggle_perturbation),
            ([0.77, 0.08, 0.08, 0.05], "Seed", self._reseed),
        ]
        for rect, label, callback in specs:
            ax_button = self.fig.add_axes(rect)
            button = Button(ax_button, label)
            button.on_clicked(callback)
            self.buttons[label] = button

        ax_speed = self.fig.add_axes([0.18, 0.02, 0.50, 0.03])
        self.speed_slider = Slider(
            ax_speed, "Speed", 0.25, 20.0, valinit=1.0, valfmt="%.2fx"
        )
        self.speed_slider.on_changed(self._on_speed_change)

        self.info_text = self.ax_info.text(
            0.0,
            1.0,
            "",
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )

    def _snapshot_to_xy(self, positions):
        theta = (2 * np.pi / self.session.config["L"]) * positions
        return self.radius * np.cos(theta), self.radius * np.sin(theta)

    def _draw(self):
        snap = self.session.snapshot()
        x, y = self._snapshot_to_xy(snap["positions"])
        cav_mask = snap["is_cav"]
        hdv_mask = ~cav_mask

        self.hdv_scatter.set_offsets(np.c_[x[hdv_mask], y[hdv_mask]])
        self.cav_scatter.set_offsets(np.c_[x[cav_mask], y[cav_mask]])

        mode = self.session.scenario_state["mode"].upper()
        self.title.set_text(
            f"Live Ring-Road HUD | {mode} | t={snap['time']:.1f}s | step {snap['step']}"
        )

        hist_t = list(self.session.history["t"])
        hist_speed = list(self.session.history["mean_speed"])
        hist_gap = list(self.session.history["min_gap"])
        self.speed_line.set_data(hist_t, hist_speed)
        self.gap_line.set_data(hist_t, hist_gap)
        right = max(self.window_seconds, snap["time"])
        left = max(0.0, right - self.window_seconds)
        self.ax_stats.set_xlim(left, right)
        self.ax_stats.set_ylim(0.0, max(1.0, max(hist_speed, default=0.0) * 1.1))
        self.ax_stats_gap.set_ylim(0.0, max(1.0, max(hist_gap, default=0.0) * 1.1))
        self.cursor.set_xdata([snap["time"], snap["time"]])

        state = self.session.scenario_state
        self.info_text.set_text(
            "\n".join(
                [
                    f"mode     : {state['mode']}",
                    f"human    : {state['human_ratio']:.2f}",
                    f"init     : {state['initial_conditions']}",
                    f"shuffle  : {state['shuffle_cav_positions']}",
                    f"perturb  : {state['perturbation_enabled']}",
                    f"seed     : {state['seed']}",
                    f"window   : {self.window_seconds:.0f}s",
                    f"boost    : x{self.speed_multiplier:.2f}",
                    f"speed    : {snap['mean_speed']:.2f} m/s",
                    f"min gap  : {snap['min_gap']:.2f} m",
                    f"playing  : {self.playing}",
                ]
            )
        )
        self.buttons["Play"].label.set_text("Pause" if self.playing else "Play")
        self.fig.canvas.draw_idle()

    def _reset_to_state(self, new_state):
        self.playing = False
        self.step_accumulator = 0.0
        self.anim.pause()
        self.session.reset(new_state)
        self._draw()

    def _toggle_play(self, event):
        self.playing = not self.playing
        if self.playing:
            self.step_accumulator = 0.0
        if self.playing:
            self.anim.resume()
        else:
            self.anim.pause()
        self._draw()

    def _step_once(self, event):
        if self.playing:
            self.playing = False
            self.anim.pause()
        self.session.step(1)
        self._draw()

    def _reset_session(self, event):
        self._reset_to_state(self.session.scenario_state)

    def _cycle_mode(self, event):
        current = self.session.scenario_state["mode"]
        idx = (self.MODE_ORDER.index(current) + 1) % len(self.MODE_ORDER)
        state = dict(self.session.scenario_state)
        state["mode"] = self.MODE_ORDER[idx]
        self._reset_to_state(state)

    def _cycle_hr(self, event):
        current = self.session.scenario_state["human_ratio"]
        idx = min(
            range(len(self.HR_OPTIONS)),
            key=lambda option_idx: abs(self.HR_OPTIONS[option_idx] - current),
        )
        state = dict(self.session.scenario_state)
        state["human_ratio"] = self.HR_OPTIONS[(idx + 1) % len(self.HR_OPTIONS)]
        self._reset_to_state(state)

    def _cycle_init(self, event):
        current = self.session.scenario_state["initial_conditions"]
        idx = self.INIT_OPTIONS.index(current) if current in self.INIT_OPTIONS else 0
        state = dict(self.session.scenario_state)
        state["initial_conditions"] = self.INIT_OPTIONS[(idx + 1) % len(self.INIT_OPTIONS)]
        self._reset_to_state(state)

    def _toggle_shuffle(self, event):
        state = dict(self.session.scenario_state)
        state["shuffle_cav_positions"] = not state["shuffle_cav_positions"]
        self._reset_to_state(state)

    def _toggle_perturbation(self, event):
        state = dict(self.session.scenario_state)
        state["perturbation_enabled"] = not state["perturbation_enabled"]
        self._reset_to_state(state)

    def _reseed(self, event):
        state = dict(self.session.scenario_state)
        state["seed"] = int(np.random.randint(1, 1_000_000))
        self._reset_to_state(state)

    def _on_speed_change(self, value):
        self.speed_multiplier = float(value)
        self.step_accumulator = 0.0

    def _animate(self, _frame):
        if self.playing:
            steps_to_run, self.step_accumulator = compute_steps_per_tick(
                dt_seconds=self.session.config["dt"],
                redraw_interval_ms=self.redraw_interval_ms,
                speed_multiplier=self.speed_multiplier,
                accumulator=self.step_accumulator,
            )
            advanced = self.session.step(steps_to_run) if steps_to_run > 0 else 0
            if steps_to_run > 0 and advanced == 0:
                self.playing = False
                self.anim.pause()
        self._draw()

    def show(self):
        plt.show()
