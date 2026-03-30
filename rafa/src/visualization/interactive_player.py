# -*- coding: utf-8 -*-
"""
Interactive ring-road playback dashboard.
"""

import copy
import json
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from matplotlib import animation
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.widgets import Button, Slider


class InteractivePlayer:
    def __init__(
        self,
        folder,
        R=20.0,
        on_rerun=None,
        scenario_state=None,
        scenario_options=None,
    ):
        self.folder = folder
        self.R = float(R)
        self.on_rerun = on_rerun
        self.scenario_options = scenario_options or {}

        self.current_frame = 0
        self.playing = False
        self.loop_playback = True
        self.speed_multiplier = 1.0
        self.playback_accumulator = 0.0

        self.color_mode = "type"
        self.heatmap_mode = "speed"
        self.show_labels = False
        self.show_trails = True
        self.show_heatmap = True
        self.trail_length = 30
        self.vehicle_size = 80.0
        self.selected_vehicle = None
        self.status_message = ""

        self._load_folder(folder)

        if scenario_state is None:
            scenario_state = self._scenario_from_metadata()
        self.active_scenario = dict(scenario_state)
        self.pending_scenario = copy.deepcopy(self.active_scenario)

        self._build_figure()
        self._refresh_after_reload()

    def _scenario_from_metadata(self):
        return {
            "mode": self.metadata.get("viz_mode", self._infer_mode_from_metadata()),
            "human_ratio": float(self.metadata.get("human_ratio", 0.8)),
            "shuffle_cav_positions": bool(
                self.metadata.get("shuffle_cav_positions", False)
            ),
            "initial_conditions": self.metadata.get("initial_conditions", "random"),
            "perturbation_enabled": bool(
                self.metadata.get("perturbation_enabled", False)
            ),
            "seed": int(self.metadata.get("seed", 0) or 0),
        }

    def _infer_mode_from_metadata(self):
        rl_mode = self.metadata.get("rl_mode", "off")
        meso_enabled = self.metadata.get("mesoscopic_enabled", False)
        if rl_mode == "residual":
            return "rl"
        if meso_enabled:
            return "adaptive"
        return "baseline"

    def _load_folder(self, folder):
        self.folder = folder

        meta_path = os.path.join(folder, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"metadata.json not found in {folder}. Cannot infer simulation layout."
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.L = float(self.metadata["L"])
        self.N = int(self.metadata["N"])
        self.dt = float(self.metadata["dt"])
        self.human_ratio = float(self.metadata.get("human_ratio", 0.8))

        raw = torch.load(os.path.join(folder, "micro.pt"), map_location="cpu")
        if len(raw) % self.N != 0:
            raise ValueError("micro.pt length is not divisible by N")

        total_steps = len(raw) // self.N
        self.total_frames = total_steps
        self.time_axis = np.arange(total_steps) * self.dt

        self.positions = np.zeros((total_steps, self.N), dtype=np.float32)
        self.velocities = np.zeros((total_steps, self.N), dtype=np.float32)
        self.accelerations = np.zeros((total_steps, self.N), dtype=np.float32)
        self.alpha = np.ones((total_steps, self.N), dtype=np.float32)
        self.meso_sigma = np.zeros((total_steps, self.N), dtype=np.float32)
        self.meso_rho = np.zeros((total_steps, self.N), dtype=np.float32)
        self.vehicle_types = [None] * self.N

        for step in range(total_steps):
            block = raw[step * self.N : (step + 1) * self.N]
            block_sorted = sorted(block, key=lambda b: b["id"])
            for i, record in enumerate(block_sorted):
                self.positions[step, i] = float(record["x"])
                self.velocities[step, i] = float(record["v"])
                self.accelerations[step, i] = float(record.get("a", 0.0))
                self.alpha[step, i] = float(record.get("alpha", 1.0))
                self.meso_sigma[step, i] = float(record.get("meso_sigma_v", 0.0))
                self.meso_rho[step, i] = float(record.get("meso_rho", 0.0))
                if step == 0:
                    self.vehicle_types[i] = record.get("type", "HumanVehicle")

        self.cav_idx = np.array(
            [i for i, name in enumerate(self.vehicle_types) if "CAV" in str(name)],
            dtype=int,
        )
        self.hdv_idx = np.array(
            [i for i, name in enumerate(self.vehicle_types) if "CAV" not in str(name)],
            dtype=int,
        )

        self.mean_speed = self.velocities.mean(axis=1)
        self.std_speed = self.velocities.std(axis=1)
        self.mean_hdv_speed = (
            self.velocities[:, self.hdv_idx].mean(axis=1)
            if len(self.hdv_idx) > 0
            else np.zeros(self.total_frames)
        )
        self.mean_cav_speed = (
            self.velocities[:, self.cav_idx].mean(axis=1)
            if len(self.cav_idx) > 0
            else np.zeros(self.total_frames)
        )
        self.mean_cav_alpha = (
            self.alpha[:, self.cav_idx].mean(axis=1)
            if len(self.cav_idx) > 0
            else np.ones(self.total_frames)
        )
        self.min_gap = np.array(
            [self._compute_min_gap(self.positions[k], self.L) for k in range(self.total_frames)],
            dtype=np.float32,
        )

        self.event_frames = {
            "risk": int(np.argmin(self.min_gap)),
            "wave": int(np.argmax(self.std_speed)),
        }
        if self.metadata.get("perturbation_enabled", False):
            perturb_t = float(self.metadata.get("perturbation_time", 0.0))
            self.event_frames["perturb"] = int(
                np.clip(round(perturb_t / self.dt), 0, self.total_frames - 1)
            )

        self.macro_rho = None
        self.macro_u = None
        macro_path = os.path.join(folder, "macro.pt")
        if os.path.exists(macro_path):
            macro_raw = torch.load(macro_path, map_location="cpu")
            if len(macro_raw) > 0:
                num_cells = int(max(record["cell"] for record in macro_raw)) + 1
                self.macro_rho = np.zeros((self.total_frames, num_cells), dtype=np.float32)
                self.macro_u = np.zeros((self.total_frames, num_cells), dtype=np.float32)
                for record in macro_raw:
                    step = int(record["step"])
                    cell = int(record["cell"])
                    if step < self.total_frames:
                        self.macro_rho[step, cell] = float(record["rho"])
                        self.macro_u[step, cell] = float(record["u"])

        if self.selected_vehicle is None or self.selected_vehicle >= self.N:
            self.selected_vehicle = int(self.cav_idx[0]) if len(self.cav_idx) > 0 else 0

    @staticmethod
    def _compute_min_gap(x, L):
        x_sorted = np.sort(np.mod(x, L))
        gaps = np.diff(np.r_[x_sorted, x_sorted[0] + L])
        return float(np.min(gaps))

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 10), facecolor="#f5f1ea")
        self.fig.canvas.manager.set_window_title("Ring-Road Scenario Lab")

        self.ax = self.fig.add_axes([0.03, 0.23, 0.44, 0.72], facecolor="#fbfaf7")
        self.ax_stats = self.fig.add_axes([0.53, 0.60, 0.43, 0.17], facecolor="#fbfaf7")
        self.ax_heatmap = self.fig.add_axes([0.53, 0.32, 0.43, 0.20], facecolor="#fbfaf7")
        self.ax_status = self.fig.add_axes([0.53, 0.20, 0.43, 0.08], facecolor="#fbfaf7")
        self.ax_status.set_axis_off()

        self._build_ring_axis()
        self._build_stats_axis()
        self._build_heatmap_axis()
        self._build_status_box()
        self._build_sliders()
        self._build_playback_buttons()
        self._build_view_buttons()
        self._build_scenario_buttons()

        self.fig.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key_press)

        self.anim = animation.FuncAnimation(
            self.fig,
            self._animate_frame,
            frames=self._frame_generator,
            interval=40,
            blit=False,
            cache_frame_data=False,
        )
        self.anim.pause()

    def _build_ring_axis(self):
        R = self.R
        road_width = 3.4

        self.ax.set_aspect("equal", adjustable="box")
        margin = R + 9
        self.ax.set_xlim(-margin, margin)
        self.ax.set_ylim(-margin, margin)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        outer = patches.Circle(
            (0, 0),
            R + road_width / 2,
            fill=False,
            linewidth=2.2,
            color="#1f2937",
        )
        inner = patches.Circle(
            (0, 0),
            R - road_width / 2,
            fill=False,
            linewidth=2.2,
            color="#1f2937",
        )
        road = patches.Annulus(
            (0, 0),
            R - road_width / 2,
            road_width,
            facecolor="#dfd7c7",
            edgecolor="none",
            alpha=0.7,
        )
        self.ax.add_patch(road)
        self.ax.add_patch(outer)
        self.ax.add_patch(inner)

        for th in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            x0 = (R + road_width / 2) * np.cos(th)
            y0 = (R + road_width / 2) * np.sin(th)
            x1 = (R + road_width / 2 + 1.3) * np.cos(th)
            y1 = (R + road_width / 2 + 1.3) * np.sin(th)
            self.ax.plot([x0, x1], [y0, y1], color="#1f2937", linewidth=1.6, alpha=0.6)

        self.hdv_scatter = self.ax.scatter(
            [],
            [],
            s=60,
            c="#3b82f6",
            edgecolors="#111827",
            linewidths=0.6,
            zorder=5,
            label="HDV",
        )
        self.cav_scatter = self.ax.scatter(
            [],
            [],
            s=90,
            c="#f97316",
            marker="s",
            edgecolors="#111827",
            linewidths=0.8,
            zorder=6,
            label="CAV",
        )
        self.trail_scatter = self.ax.scatter([], [], s=[], c=[], zorder=4)
        self.selected_marker = self.ax.scatter(
            [],
            [],
            s=200,
            facecolors="none",
            edgecolors="#111827",
            linewidths=2.0,
            zorder=7,
        )

        self.vehicle_labels = [
            self.ax.text(
                0.0,
                0.0,
                str(i),
                fontsize=7,
                color="#111827",
                ha="center",
                va="center",
                visible=False,
                zorder=8,
                bbox=dict(
                    boxstyle="round,pad=0.16",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.85,
                ),
            )
            for i in range(self.N)
        ]
        self.selected_text = self.ax.text(
            0.0,
            0.0,
            "",
            fontsize=8,
            fontweight="bold",
            color="#111827",
            ha="left",
            va="bottom",
            zorder=9,
        )

        self.title_text = self.ax.set_title("", fontsize=14, fontweight="bold", pad=12)
        self.info_text = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            fontsize=9,
            va="top",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", alpha=0.92),
        )
        self.ax.legend(loc="upper right", fontsize=10, framealpha=0.95)

    def _build_stats_axis(self):
        self.ax_stats.set_title("Speed and Alpha Timeline", fontsize=11, fontweight="bold")
        self.ax_stats.set_xlabel("Time (s)")
        self.ax_stats.set_ylabel("Speed (m/s)")
        self.ax_stats.grid(alpha=0.25, linestyle="--")

        self.line_mean_speed, = self.ax_stats.plot([], [], color="#0f766e", linewidth=2.2, label="Fleet speed")
        self.line_hdv_speed, = self.ax_stats.plot([], [], color="#2563eb", linewidth=1.6, alpha=0.7, label="HDV speed")
        self.line_cav_speed, = self.ax_stats.plot([], [], color="#ea580c", linewidth=1.6, alpha=0.8, label="CAV speed")
        self.line_selected_speed, = self.ax_stats.plot([], [], color="#111827", linewidth=2.6, label="Selected speed")
        self.stats_cursor = self.ax_stats.axvline(0.0, color="#111827", linestyle="--", linewidth=1.4, alpha=0.7)

        self.ax_stats_alpha = self.ax_stats.twinx()
        self.ax_stats_alpha.set_ylabel("Alpha")
        self.line_selected_alpha, = self.ax_stats_alpha.plot(
            [],
            [],
            color="#a21caf",
            linewidth=1.7,
            alpha=0.85,
            label="Selected alpha",
        )

        self.event_markers = {}
        self.ax_stats.legend(loc="upper left", fontsize=8, ncol=2)

    def _build_heatmap_axis(self):
        self.ax_heatmap.set_title("Telemetry Heatmap", fontsize=11, fontweight="bold")
        self.ax_heatmap.set_xlabel("Time (s)")
        self.ax_heatmap.set_ylabel("Vehicle id")
        self.heatmap_im = None
        self.heatmap_cursor = self.ax_heatmap.axvline(
            0.0, color="white", linestyle="--", linewidth=1.0, alpha=0.85
        )
        self.heatmap_colorbar = None
        self.heatmap_fallback_text = self.ax_heatmap.text(
            0.5,
            0.5,
            "",
            transform=self.ax_heatmap.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="#6b7280",
            visible=False,
        )

    def _build_status_box(self):
        self.status_text = self.ax_status.text(
            0.01,
            0.96,
            "",
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )

    def _build_sliders(self):
        self.timeline_rect = [0.08, 0.17, 0.34, 0.026]
        self.speed_rect = [0.08, 0.12, 0.34, 0.024]
        self.trail_rect = [0.08, 0.07, 0.15, 0.022]
        self.size_rect = [0.27, 0.07, 0.15, 0.022]

        self._rebuild_timeline_slider()

        ax_speed = self.fig.add_axes(self.speed_rect, facecolor="#ece6db")
        self.slider_speed = Slider(
            ax_speed,
            "Playback",
            0.1,
            8.0,
            valinit=self.speed_multiplier,
            valfmt="%.2fx",
        )
        self.slider_speed.on_changed(self._on_speed_changed)

        ax_trail = self.fig.add_axes(self.trail_rect, facecolor="#ece6db")
        self.slider_trail = Slider(
            ax_trail,
            "Trail",
            0,
            120,
            valinit=self.trail_length,
            valstep=1,
            valfmt="%d",
        )
        self.slider_trail.on_changed(self._on_trail_changed)

        ax_size = self.fig.add_axes(self.size_rect, facecolor="#ece6db")
        self.slider_size = Slider(
            ax_size,
            "Size",
            30,
            180,
            valinit=self.vehicle_size,
            valfmt="%.0f",
        )
        self.slider_size.on_changed(self._on_size_changed)

    def _rebuild_timeline_slider(self):
        if hasattr(self, "slider_timeline"):
            self.slider_timeline.ax.remove()

        ax_timeline = self.fig.add_axes(self.timeline_rect, facecolor="#ece6db")
        self.slider_timeline = Slider(
            ax_timeline,
            "Frame",
            0,
            max(0, self.total_frames - 1),
            valinit=np.clip(self.current_frame, 0, max(0, self.total_frames - 1)),
            valstep=1,
            valfmt="%d",
        )
        self.slider_timeline.on_changed(self._on_timeline_changed)

    def _build_playback_buttons(self):
        labels = [
            ("|<", self._on_jump_start),
            ("-10", self._on_step_back),
            ("Play", self._on_play_clicked),
            ("+10", self._on_step_fwd),
            (">|", self._on_jump_end),
            ("Loop", self._on_toggle_loop),
            ("Pert", lambda event: self._jump_to_event("perturb")),
            ("Risk", lambda event: self._jump_to_event("risk")),
            ("Wave", lambda event: self._jump_to_event("wave")),
        ]

        self.playback_buttons = {}
        x = 0.05
        width = 0.045
        gap = 0.007
        for label, callback in labels:
            btn = self._add_button([x, 0.02, width, 0.034], label, callback)
            self.playback_buttons[label] = btn
            x += width + gap

        self.btn_play = self.playback_buttons["Play"]
        self.btn_loop = self.playback_buttons["Loop"]

    def _build_view_buttons(self):
        self.color_buttons = {}
        self.toggle_buttons = {}
        self.heatmap_buttons = {}

        for idx, mode in enumerate(["type", "speed", "alpha", "accel"]):
            btn = self._add_button(
                [0.53 + 0.105 * idx, 0.92, 0.09, 0.032],
                mode.title(),
                lambda event, value=mode: self._set_color_mode(value),
            )
            self.color_buttons[mode] = btn

        toggle_specs = [
            ("Labels", self._toggle_labels),
            ("Trails", self._toggle_trails),
            ("Heatmap", self._toggle_heatmap),
            ("Prev Veh", self._select_prev_vehicle),
            ("Next Veh", self._select_next_vehicle),
        ]
        for idx, (label, callback) in enumerate(toggle_specs):
            btn = self._add_button(
                [0.53 + 0.092 * idx, 0.875, 0.082, 0.032],
                label,
                callback,
            )
            self.toggle_buttons[label] = btn

        for idx, mode in enumerate(["speed", "alpha", "rho", "u"]):
            btn = self._add_button(
                [0.53 + 0.105 * idx, 0.83, 0.09, 0.032],
                mode.upper(),
                lambda event, value=mode: self._set_heatmap_mode(value),
            )
            self.heatmap_buttons[mode] = btn

    def _build_scenario_buttons(self):
        self.hr_buttons = {}
        self.mode_buttons = {}
        self.scenario_toggle_buttons = {}

        if self.on_rerun is None:
            return

        human_rates = self.scenario_options.get("human_rates", [0.0, 0.25, 0.5, 0.75])
        modes = self.scenario_options.get("modes", ["baseline", "adaptive", "rl"])

        for idx, rate in enumerate(human_rates):
            label = f"HR {int(rate * 100):02d}"
            btn = self._add_button(
                [0.60 + 0.08 * idx, 0.12, 0.07, 0.035],
                label,
                lambda event, value=rate: self._set_pending_human_ratio(value),
            )
            self.hr_buttons[rate] = btn

        for idx, mode in enumerate(modes):
            btn = self._add_button(
                [0.60 + 0.12 * idx, 0.075, 0.105, 0.035],
                mode.title(),
                lambda event, value=mode: self._set_pending_mode(value),
            )
            self.mode_buttons[mode] = btn

        self.scenario_toggle_buttons["shuffle"] = self._add_button(
            [0.60, 0.03, 0.10, 0.035],
            "Shuffle",
            self._toggle_pending_shuffle,
        )
        self.scenario_toggle_buttons["init"] = self._add_button(
            [0.71, 0.03, 0.10, 0.035],
            "Init",
            self._cycle_pending_init,
        )
        self.scenario_toggle_buttons["perturb"] = self._add_button(
            [0.82, 0.03, 0.10, 0.035],
            "Perturb",
            self._toggle_pending_perturbation,
        )
        self.btn_reseed = self._add_button([0.60, 0.165, 0.10, 0.03], "Reseed", self._reseed_pending)
        self.btn_rerun = self._add_button([0.82, 0.165, 0.10, 0.03], "Rerun", self._rerun_pending)

    def _add_button(self, rect, label, callback):
        ax_btn = self.fig.add_axes(rect)
        btn = Button(ax_btn, label, color="#e8dece", hovercolor="#d9cbb5")
        btn.label.set_fontsize(8.5)
        btn.on_clicked(callback)
        return btn

    def _x_to_xy(self, x):
        theta = (2 * np.pi / self.L) * np.asarray(x)
        return self.R * np.cos(theta), self.R * np.sin(theta)

    def _get_frame_values(self, frame_idx, mode):
        if mode == "speed":
            return self.velocities[frame_idx], plt.get_cmap("viridis"), Normalize(
                vmin=float(np.min(self.velocities)), vmax=float(np.max(self.velocities))
            )
        if mode == "alpha":
            return self.alpha[frame_idx], plt.get_cmap("plasma"), Normalize(
                vmin=float(np.min(self.alpha)), vmax=float(np.max(self.alpha))
            )
        if mode == "accel":
            limit = float(np.max(np.abs(self.accelerations))) or 1.0
            return self.accelerations[frame_idx], plt.get_cmap("coolwarm"), TwoSlopeNorm(
                vcenter=0.0,
                vmin=-limit,
                vmax=limit,
            )
        return None, None, None

    def _update_ring_colors(self, frame_idx):
        if self.color_mode == "type":
            self.hdv_scatter.set_facecolors("#3b82f6")
            self.cav_scatter.set_facecolors("#f97316")
            return

        values, cmap, norm = self._get_frame_values(frame_idx, self.color_mode)
        colors = cmap(norm(values))
        if len(self.hdv_idx) > 0:
            self.hdv_scatter.set_facecolors(colors[self.hdv_idx])
        if len(self.cav_idx) > 0:
            self.cav_scatter.set_facecolors(colors[self.cav_idx])

    def _update_vehicle_markers(self, frame_idx):
        X, Y = self._x_to_xy(self.positions[frame_idx])

        self.hdv_scatter.set_sizes(np.full(max(1, len(self.hdv_idx)), self.vehicle_size * 0.8))
        self.cav_scatter.set_sizes(np.full(max(1, len(self.cav_idx)), self.vehicle_size))

        if len(self.hdv_idx) > 0:
            self.hdv_scatter.set_offsets(np.c_[X[self.hdv_idx], Y[self.hdv_idx]])
        else:
            self.hdv_scatter.set_offsets(np.empty((0, 2)))

        if len(self.cav_idx) > 0:
            self.cav_scatter.set_offsets(np.c_[X[self.cav_idx], Y[self.cav_idx]])
        else:
            self.cav_scatter.set_offsets(np.empty((0, 2)))

        self._update_ring_colors(frame_idx)

        self.selected_marker.set_offsets([[X[self.selected_vehicle], Y[self.selected_vehicle]]])
        self.selected_text.set_position((X[self.selected_vehicle] + 0.9, Y[self.selected_vehicle] + 0.9))
        self.selected_text.set_text(f"veh {self.selected_vehicle}")

        if self.show_labels:
            for idx, txt in enumerate(self.vehicle_labels):
                txt.set_position((X[idx], Y[idx]))
                txt.set_visible(True)
        else:
            for txt in self.vehicle_labels:
                txt.set_visible(False)

        if self.show_trails and self.trail_length > 0:
            lo = max(0, frame_idx - self.trail_length)
            trail_x = self.positions[lo : frame_idx + 1, self.selected_vehicle]
            tx, ty = self._x_to_xy(trail_x)
            n = len(tx)
            rgba = np.tile(np.array([17 / 255, 24 / 255, 39 / 255, 1.0]), (n, 1))
            rgba[:, 3] = np.linspace(0.12, 0.95, n)
            sizes = np.linspace(10.0, 55.0, n)
            self.trail_scatter.set_offsets(np.c_[tx, ty])
            self.trail_scatter.set_sizes(sizes)
            self.trail_scatter.set_facecolors(rgba)
        else:
            self.trail_scatter.set_offsets(np.empty((0, 2)))
            self.trail_scatter.set_sizes(np.array([]))

    def _update_titles_and_info(self, frame_idx):
        t = self.time_axis[frame_idx]
        mode = self.active_scenario.get("mode", self._infer_mode_from_metadata()).upper()
        self.title_text.set_text(
            f"Ring-Road Scenario Lab | {mode} | t={t:.1f}s | frame {frame_idx}/{self.total_frames - 1}"
        )

        mean_v = self.mean_speed[frame_idx]
        std_v = self.std_speed[frame_idx]
        min_gap = self.min_gap[frame_idx]
        selected_v = self.velocities[frame_idx, self.selected_vehicle]
        selected_a = self.accelerations[frame_idx, self.selected_vehicle]
        selected_alpha = self.alpha[frame_idx, self.selected_vehicle]

        self.info_text.set_text(
            f"Selection: vehicle {self.selected_vehicle} ({self.vehicle_types[self.selected_vehicle]})\n"
            f"Mean speed: {mean_v:.2f} m/s | sigma(v): {std_v:.2f} m/s | min gap: {min_gap:.2f} m\n"
            f"Selected v: {selected_v:.2f} m/s | a: {selected_a:.2f} m/s^2 | alpha: {selected_alpha:.2f}\n"
            f"Color: {self.color_mode} | Heatmap: {self.heatmap_mode} | Trail: {self.trail_length} frames"
        )

    def _update_stats_cursor(self, frame_idx):
        t = self.time_axis[frame_idx]
        self.stats_cursor.set_xdata([t, t])
        self.heatmap_cursor.set_xdata([t, t])

    def _update_status_box(self):
        active = self.active_scenario
        pending = self.pending_scenario
        selected_type = self.vehicle_types[self.selected_vehicle]
        selected_sigma = self.meso_sigma[self.current_frame, self.selected_vehicle]
        selected_rho = self.meso_rho[self.current_frame, self.selected_vehicle]

        lines = [
            f"Active  : mode={active.get('mode', '-'):<8} HR={active.get('human_ratio', 0.0):.2f} "
            f"shuffle={str(active.get('shuffle_cav_positions', False)):<5} "
            f"init={active.get('initial_conditions', '-'):<7} pert={str(active.get('perturbation_enabled', False)):<5} "
            f"seed={active.get('seed', '-')}",
            f"Pending : mode={pending.get('mode', '-'):<8} HR={pending.get('human_ratio', 0.0):.2f} "
            f"shuffle={str(pending.get('shuffle_cav_positions', False)):<5} "
            f"init={pending.get('initial_conditions', '-'):<7} pert={str(pending.get('perturbation_enabled', False)):<5} "
            f"seed={pending.get('seed', '-')}",
            f"Selected: veh={self.selected_vehicle:02d} type={selected_type:<22} "
            f"meso_sigma={selected_sigma:5.2f} meso_rho={selected_rho:5.2f}",
            "Shortcuts: space play/pause | left/right +/-1 | up/down +/-10 | c cycle color | h cycle heatmap | l labels | t trails",
        ]
        if self.status_message:
            lines.append(f"Status  : {self.status_message}")
        self.status_text.set_text("\n".join(lines))

    def _update_button_styles(self):
        for mode, btn in self.color_buttons.items():
            btn.ax.set_facecolor("#cbe7dd" if mode == self.color_mode else "#e8dece")

        for mode, btn in self.heatmap_buttons.items():
            btn.ax.set_facecolor("#d9d3f3" if mode == self.heatmap_mode else "#e8dece")

        self.toggle_buttons["Labels"].ax.set_facecolor("#f6d9a8" if self.show_labels else "#e8dece")
        self.toggle_buttons["Trails"].ax.set_facecolor("#f6d9a8" if self.show_trails else "#e8dece")
        self.toggle_buttons["Heatmap"].ax.set_facecolor("#f6d9a8" if self.show_heatmap else "#e8dece")
        self.btn_loop.ax.set_facecolor("#f6d9a8" if self.loop_playback else "#e8dece")

        if self.on_rerun is None:
            return

        pending_rate = float(self.pending_scenario["human_ratio"])
        for rate, btn in self.hr_buttons.items():
            btn.ax.set_facecolor(
                "#cfe7ff" if abs(pending_rate - float(rate)) < 1e-9 else "#e8dece"
            )

        for mode, btn in self.mode_buttons.items():
            btn.ax.set_facecolor("#cfe7ff" if self.pending_scenario["mode"] == mode else "#e8dece")

        shuffle = self.pending_scenario["shuffle_cav_positions"]
        self.scenario_toggle_buttons["shuffle"].label.set_text(
            "Shuffle On" if shuffle else "Shuffle Off"
        )
        self.scenario_toggle_buttons["shuffle"].ax.set_facecolor(
            "#f6d9a8" if shuffle else "#e8dece"
        )

        init_mode = self.pending_scenario["initial_conditions"]
        self.scenario_toggle_buttons["init"].label.set_text(f"Init {init_mode[:4]}")

        perturb = self.pending_scenario["perturbation_enabled"]
        self.scenario_toggle_buttons["perturb"].label.set_text(
            "Perturb On" if perturb else "Perturb Off"
        )
        self.scenario_toggle_buttons["perturb"].ax.set_facecolor(
            "#f6d9a8" if perturb else "#e8dece"
        )

    def _refresh_stats_plot(self):
        self.line_mean_speed.set_data(self.time_axis, self.mean_speed)
        self.line_hdv_speed.set_data(self.time_axis, self.mean_hdv_speed)
        self.line_cav_speed.set_data(self.time_axis, self.mean_cav_speed)
        self.line_selected_speed.set_data(
            self.time_axis, self.velocities[:, self.selected_vehicle]
        )
        self.line_selected_alpha.set_data(self.time_axis, self.alpha[:, self.selected_vehicle])

        x_max = float(self.time_axis[-1] if len(self.time_axis) > 1 else self.dt)
        self.ax_stats.set_xlim(float(self.time_axis[0]), x_max)
        max_speed = float(np.max(self.velocities)) if self.velocities.size else 1.0
        self.ax_stats.set_ylim(0.0, max(1.0, max_speed * 1.08))

        alpha_min = float(np.min(self.alpha)) if self.alpha.size else 0.5
        alpha_max = float(np.max(self.alpha)) if self.alpha.size else 1.5
        if abs(alpha_max - alpha_min) < 1e-6:
            alpha_min -= 0.1
            alpha_max += 0.1
        pad = 0.08 * (alpha_max - alpha_min)
        self.ax_stats_alpha.set_ylim(alpha_min - pad, alpha_max + pad)

        for marker in self.event_markers.values():
            marker.remove()
        self.event_markers = {}

        event_specs = {
            "perturb": ("#ef4444", "Perturb"),
            "risk": ("#dc2626", "Worst gap"),
            "wave": ("#7c3aed", "Peak wave"),
        }
        for key, (color, label) in event_specs.items():
            if key not in self.event_frames:
                continue
            t = self.time_axis[self.event_frames[key]]
            self.event_markers[key] = self.ax_stats.axvline(
                t, color=color, linestyle=":", linewidth=1.1, alpha=0.7, label=label
            )

    def _get_heatmap_payload(self):
        if self.heatmap_mode == "speed":
            return self.velocities.T, "viridis", "Vehicle id", "Speed (m/s)", ""
        if self.heatmap_mode == "alpha":
            return self.alpha.T, "plasma", "Vehicle id", "Alpha", ""
        if self.heatmap_mode == "rho":
            if self.macro_rho is not None:
                return self.macro_rho.T, "magma", "Cell index", "Density", ""
            return self.velocities.T, "viridis", "Vehicle id", "Speed (m/s)", "macro rho unavailable; showing speed"
        if self.heatmap_mode == "u":
            if self.macro_u is not None:
                return self.macro_u.T, "cividis", "Cell index", "Velocity", ""
            return self.velocities.T, "viridis", "Vehicle id", "Speed (m/s)", "macro u unavailable; showing speed"
        return self.velocities.T, "viridis", "Vehicle id", "Speed (m/s)", ""

    def _refresh_heatmap(self):
        data, cmap, ylabel, cbar_label, fallback = self._get_heatmap_payload()
        extent = [
            float(self.time_axis[0]),
            float(self.time_axis[-1] if len(self.time_axis) > 1 else self.dt),
            -0.5,
            data.shape[0] - 0.5,
        ]

        vmin = float(np.min(data))
        vmax = float(np.max(data))
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1.0

        if self.heatmap_im is None:
            self.heatmap_im = self.ax_heatmap.imshow(
                data,
                aspect="auto",
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            self.heatmap_colorbar = self.fig.colorbar(
                self.heatmap_im, ax=self.ax_heatmap, fraction=0.035, pad=0.02
            )
        else:
            self.heatmap_im.set_data(data)
            self.heatmap_im.set_extent(extent)
            self.heatmap_im.set_cmap(cmap)
            self.heatmap_im.set_clim(vmin, vmax)

        self.heatmap_colorbar.set_label(cbar_label)
        self.ax_heatmap.set_ylabel(ylabel)
        self.heatmap_fallback_text.set_text(fallback)
        self.heatmap_fallback_text.set_visible(bool(fallback))
        self.ax_heatmap.set_visible(self.show_heatmap)
        self.heatmap_colorbar.ax.set_visible(self.show_heatmap)

    def _draw_frame(self, frame_idx):
        frame_idx = int(np.clip(frame_idx, 0, self.total_frames - 1))
        self.current_frame = frame_idx

        self._update_vehicle_markers(frame_idx)
        self._update_titles_and_info(frame_idx)
        self._update_stats_cursor(frame_idx)
        self._update_status_box()
        self._update_button_styles()

        self.slider_timeline.eventson = False
        self.slider_timeline.set_val(frame_idx)
        self.slider_timeline.eventson = True

        self.fig.canvas.draw_idle()

    def _refresh_after_reload(self):
        self.current_frame = int(np.clip(self.current_frame, 0, self.total_frames - 1))
        self._rebuild_timeline_slider()
        self._refresh_stats_plot()
        self._refresh_heatmap()
        self._update_button_styles()
        self._draw_frame(self.current_frame)

    def _frame_generator(self):
        while True:
            if self.playing:
                self.playback_accumulator += self.speed_multiplier
                step = int(self.playback_accumulator)
                if step > 0:
                    self.playback_accumulator -= step
                    self.current_frame += step
                    if self.current_frame >= self.total_frames:
                        if self.loop_playback:
                            self.current_frame %= self.total_frames
                        else:
                            self.current_frame = self.total_frames - 1
                            self.playing = False
                            self.anim.pause()
                            self.btn_play.label.set_text("Play")
            yield self.current_frame

    def _animate_frame(self, frame_idx):
        self._draw_frame(frame_idx)

    def _jump_to_frame(self, frame_idx):
        self._draw_frame(frame_idx)

    def _jump_to_event(self, key):
        if key not in self.event_frames:
            self.status_message = f"No '{key}' event available in this run."
            self._update_status_box()
            self.fig.canvas.draw_idle()
            return
        self.status_message = f"Jumped to {key} event."
        self._jump_to_frame(self.event_frames[key])

    def _set_color_mode(self, mode):
        self.color_mode = mode
        self._draw_frame(self.current_frame)

    def _set_heatmap_mode(self, mode):
        self.heatmap_mode = mode
        self._refresh_heatmap()
        self._draw_frame(self.current_frame)

    def _toggle_labels(self, event):
        self.show_labels = not self.show_labels
        self._draw_frame(self.current_frame)

    def _toggle_trails(self, event):
        self.show_trails = not self.show_trails
        self._draw_frame(self.current_frame)

    def _toggle_heatmap(self, event):
        self.show_heatmap = not self.show_heatmap
        self._refresh_heatmap()
        self._draw_frame(self.current_frame)

    def _select_prev_vehicle(self, event):
        self.selected_vehicle = (self.selected_vehicle - 1) % self.N
        self._refresh_stats_plot()
        self._draw_frame(self.current_frame)

    def _select_next_vehicle(self, event):
        self.selected_vehicle = (self.selected_vehicle + 1) % self.N
        self._refresh_stats_plot()
        self._draw_frame(self.current_frame)

    def _on_timeline_changed(self, val):
        self._jump_to_frame(int(val))

    def _on_speed_changed(self, val):
        self.speed_multiplier = float(val)

    def _on_trail_changed(self, val):
        self.trail_length = int(val)
        self._draw_frame(self.current_frame)

    def _on_size_changed(self, val):
        self.vehicle_size = float(val)
        self._draw_frame(self.current_frame)

    def _on_play_clicked(self, event):
        self.playing = not self.playing
        self.playback_accumulator = 0.0
        if self.playing:
            self.btn_play.label.set_text("Pause")
            self.anim.resume()
        else:
            self.btn_play.label.set_text("Play")
            self.anim.pause()
        self.fig.canvas.draw_idle()

    def _on_step_back(self, event):
        self._jump_to_frame(max(0, self.current_frame - 10))

    def _on_step_fwd(self, event):
        self._jump_to_frame(min(self.total_frames - 1, self.current_frame + 10))

    def _on_jump_start(self, event):
        self._jump_to_frame(0)

    def _on_jump_end(self, event):
        self._jump_to_frame(self.total_frames - 1)

    def _on_toggle_loop(self, event):
        self.loop_playback = not self.loop_playback
        self._update_button_styles()
        self.fig.canvas.draw_idle()

    def _set_pending_human_ratio(self, value):
        self.pending_scenario["human_ratio"] = float(value)
        self.status_message = f"Pending human ratio set to {value:.2f}."
        self._update_button_styles()
        self._update_status_box()
        self.fig.canvas.draw_idle()

    def _set_pending_mode(self, value):
        self.pending_scenario["mode"] = value
        self.status_message = f"Pending controller mode set to {value}."
        self._update_button_styles()
        self._update_status_box()
        self.fig.canvas.draw_idle()

    def _toggle_pending_shuffle(self, event):
        self.pending_scenario["shuffle_cav_positions"] = not self.pending_scenario[
            "shuffle_cav_positions"
        ]
        self.status_message = "Pending shuffle toggled."
        self._update_button_styles()
        self._update_status_box()
        self.fig.canvas.draw_idle()

    def _cycle_pending_init(self, event):
        current = self.pending_scenario["initial_conditions"]
        self.pending_scenario["initial_conditions"] = (
            "random" if current == "uniform" else "uniform"
        )
        self.status_message = "Pending initial-condition mode updated."
        self._update_button_styles()
        self._update_status_box()
        self.fig.canvas.draw_idle()

    def _toggle_pending_perturbation(self, event):
        self.pending_scenario["perturbation_enabled"] = not self.pending_scenario[
            "perturbation_enabled"
        ]
        self.status_message = "Pending perturbation toggle updated."
        self._update_button_styles()
        self._update_status_box()
        self.fig.canvas.draw_idle()

    def _reseed_pending(self, event):
        self.pending_scenario["seed"] = int(np.random.randint(1, 1_000_000))
        self.status_message = f"Pending seed changed to {self.pending_scenario['seed']}."
        self._update_status_box()
        self.fig.canvas.draw_idle()

    def _rerun_pending(self, event):
        if self.on_rerun is None:
            return

        self.playing = False
        self.playback_accumulator = 0.0
        self.anim.pause()
        self.btn_play.label.set_text("Play")
        self.status_message = "Running new scenario..."
        self._update_status_box()
        self.fig.canvas.draw_idle()

        try:
            new_folder = self.on_rerun(dict(self.pending_scenario))
            self._load_folder(new_folder)
            self.active_scenario = copy.deepcopy(self.pending_scenario)
            self.current_frame = 0
            self.status_message = f"Reloaded scenario from {new_folder}."
            self._refresh_after_reload()
        except Exception as exc:
            self.status_message = f"Scenario rerun failed: {exc}"
            self._update_status_box()
            self.fig.canvas.draw_idle()

    def _on_canvas_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        X, Y = self._x_to_xy(self.positions[self.current_frame])
        distances = np.hypot(X - event.xdata, Y - event.ydata)
        self.selected_vehicle = int(np.argmin(distances))
        self.status_message = f"Selected vehicle {self.selected_vehicle} from ring view."
        self._refresh_stats_plot()
        self._draw_frame(self.current_frame)

    def _on_key_press(self, event):
        if event.key == " ":
            self._on_play_clicked(None)
        elif event.key == "left":
            self._jump_to_frame(max(0, self.current_frame - 1))
        elif event.key == "right":
            self._jump_to_frame(min(self.total_frames - 1, self.current_frame + 1))
        elif event.key == "up":
            self._jump_to_frame(min(self.total_frames - 1, self.current_frame + 10))
        elif event.key == "down":
            self._jump_to_frame(max(0, self.current_frame - 10))
        elif event.key == "c":
            modes = ["type", "speed", "alpha", "accel"]
            idx = (modes.index(self.color_mode) + 1) % len(modes)
            self._set_color_mode(modes[idx])
        elif event.key == "h":
            modes = ["speed", "alpha", "rho", "u"]
            idx = (modes.index(self.heatmap_mode) + 1) % len(modes)
            self._set_heatmap_mode(modes[idx])
        elif event.key == "l":
            self._toggle_labels(None)
        elif event.key == "t":
            self._toggle_trails(None)

    def show(self):
        plt.show()
