# Branch Comparison Report: `/rafa/` vs `/mustykey/`

> Project: Ring-road CAV (Connected Autonomous Vehicle) traffic control via Reinforcement Learning
> Context: Both branches share the same original problem — a ring-road with mixed HDV/CAV traffic, where CAVs learn residual corrections to a mesoscopic headway rule (CACC/IDM) to reduce stop-and-go waves.

---

## 1. High-Level Summary

| | **rafa** | **mustykey** |
| --- | --- | --- |
| Execution model | Vectorized GPU (N envs in parallel) | Sequential CPU (1 episode at a time) |
| Primary RL algorithm | SAC (GPU-native), PPO (GPU-native) | PPO (CPU), SAC (CPU) |
| Observation space | 15-dimensional | 8-dimensional |
| Reward terms | 10+ weighted terms | 6 weighted terms |
| RL/mesoscopic coupling | Decoupled — RL in `src/agents/` | Coupled — RL inside `src/mesoscopic/` |
| String stability metric | Amplification ratio (formal, continuous) | Boolean validity flag |
| Checkpoint ranking | Safety-first policy | Traffic metrics (variance, oscillation) |
| Scale target | HPC/GPU clusters, 1M+ timesteps | CPU workstation, iterative development |
| Code volume | ~13 400 Python lines | ~18 100 Python lines |

---

## 2. Repository Structure

### `rafa/`

```text
rafa/
├── train_gpu_ppo.py          # vectorized GPU PPO entry point
├── train_gpu_sac.py          # vectorized GPU SAC entry point
├── train_rl.py               # legacy CPU PPO path
├── evaluate_gpu_ppo.py       # checkpoint evaluation + generalisation sweep
├── evaluate_gpu_sac.py
├── run_experiments.py        # multi-mode orchestration
├── generate_summary_compare_plots.py
├── config/
│   ├── rl_train.yaml         # master GPU training config
│   └── rl_smoke.yaml         # quick smoke test
├── src/
│   ├── gpu/
│   │   ├── vec_env.py        # VecRingRoadEnv — entire physics as parallel tensors
│   │   ├── gpu_ppo.py        # on-device GAE, rollout management
│   │   ├── gpu_sac.py        # twin-Q, replay buffer, entropy tuning
│   │   ├── hardware_profiles.py
│   │   ├── auto_tune.py
│   │   └── training_monitor.py
│   ├── agents/
│   │   ├── ppo_trainer.py
│   │   ├── networks.py
│   │   ├── buffer.py
│   │   ├── reward.py         # multi-objective reward (10+ terms)
│   │   ├── observation_builder.py  # 15-dim obs construction
│   │   └── rl_types.py       # OBS_DIM=15, reward weight dataclasses
│   ├── mesoscopic/
│   │   └── meso_adapter.py   # baseline only, not integrated with RL
│   └── visualization/
│       ├── live_hud.py
│       └── interactive_player.py
├── run_eval_gpu_checkpoints.slurm  # SLURM batch script
└── Reports/MASTER.md         # mathematical formulation, decision history
```

### `mustykey/`

```text
mustykey/
├── run_training.py           # single-episode training loop
├── run_simulation.py         # baseline simulation (no RL)
├── run_experiments.py        # sweep over human rates, modes, seeds
├── config/
│   └── *.yaml                # single-level config, rl embedded in main config
├── src/
│   ├── simulation/
│   │   └── simulator.py      # sequential CPU simulator (Python loops)
│   ├── mesoscopic/
│   │   ├── meso_adapter.py   # deep variance-based alpha adaptation
│   │   ├── rl_layer.py       # ActorCritic (64-dim hidden), residual action
│   │   ├── rl_common.py
│   │   ├── rl_training.py    # CPU PPO with GAE
│   │   ├── sac_training.py   # CPU SAC with dual-Q
│   │   └── rl_rewards.py     # 6-term reward function
│   └── visualization/
│       ├── visualizer.py
│       └── transition_visualizer.py
├── scripts/
│   └── build_rl_comparison_report.py
└── output/
    └── method_compare_random_rerun_20260323_ep800/  # latest results
```

**Key structural divergence:** `rafa` separates RL into its own `src/agents/` layer, treating the environment as a standalone vectorised tensor object. `mustykey` embeds RL directly inside `src/mesoscopic/`, making RL a sub-component of the control stack.

---

## 3. Simulation Engine

### `rafa` — `VecRingRoadEnv` (GPU tensor simulation)

- The entire ring-road physics (IDM, CACC, mesoscopic headway rules) is implemented as **PyTorch tensor operations** with no Python loops over vehicles.
- `num_envs` (e.g. 16–128) independent simulations run in a **single GPU kernel call** via broadcasting.
- Each `step()` call returns a batch of `(obs, reward, done, info)` tensors of shape `[num_envs, ...]`.
- Physics parameters (HDV aggressiveness, CAV density, noise) are **domain-randomised** across the batch to improve generalisation.

### `mustykey` — `simulator.py` (CPU sequential simulation)

- Standard Python class with a `step()` loop over all vehicles.
- One environment instance per training episode.
- No parallelism — HDV noise, spacing, and CAV placement are fixed per episode but vary across episodes via a random seed.

**Consequence:** `rafa` can collect `num_envs × T` transitions per wall-clock second vs. `mustykey`'s `T` transitions. This difference drives many of the downstream design choices.

---

## 4. RL Algorithms

### PPO

Both branches implement Proximal Policy Optimisation with GAE. The key differences:

| | `rafa` `gpu_ppo.py` | `mustykey` `rl_training.py` |
| --- | --- | --- |
| Device | GPU (all tensors stay on device) | CPU |
| Rollout source | `num_envs` parallel envs | 1 env |
| GAE computation | Vectorised, in-place on device | Python loop |
| Normaliser | Running mean/var normaliser | None |
| Early stopping | Patience-based `early_stopping.py` | Fixed episode count |

### SAC

Both implement Soft Actor-Critic with twin-Q critics.

| | `rafa` `gpu_sac.py` | `mustykey` `sac_training.py` |
| --- | --- | --- |
| Device | GPU | CPU |
| Replay buffer | On-device circular buffer | Python list |
| Entropy temperature | Learnable α (auto-tuned) | Fixed |
| Target networks | Polyak update, GPU | Polyak update, CPU |
| Pre-training | Can bootstrap from PPO checkpoint | Independent start |

**Algorithmic preference:** Both branches arrive at SAC being competitive or preferred for the stochastic HDV noise regime. `rafa` explicitly states SAC is preferred over PPO for off-policy advantages on stochastic systems. `mustykey`'s latest results show PPO winning at 75% CAV penetration (h75) but SAC competitive at lower penetration rates.

---

## 5. Observation Space

### `rafa` — 15-dimensional (`rl_types.py`: `OBS_DIM = 15`)

Captures both **local** and **global** context:

- Local state: ego speed, gap, relative speed, previous alpha
- Neighbourhood: mean/std speed of N nearest CAVs, leader acceleration
- Global context: fleet mean speed, CAV penetration rate, density
- Temporal: recent acceleration history

### `mustykey` — 8-dimensional

Local state only:

- `μ_v` (fleet mean speed), `σ_v²` (speed variance), `δ_v` (speed deviation)
- `v` (ego speed), `s` (gap), `Δv` (relative speed)
- `a_leader` (leader acceleration), `α_prev` (previous headway parameter)

**Design philosophy difference:** `rafa` includes global context to help the policy generalise across human rates and traffic regimes. `mustykey` keeps the observation minimal for interpretability and fewer hyperparameters.

---

## 6. Reward Function

### `rafa` — `src/agents/reward.py` (10+ terms)

```text
r = w_safety   · r_safety      # collision penalty
  + w_gap      · r_gap         # spacing error vs. desired
  + w_speed    · r_speed       # mean speed reward
  + w_variance · r_variance    # speed variance penalty
  + w_comfort  · r_comfort     # acceleration smoothness
  + w_jerk     · r_jerk        # jerk penalty
  + w_ss       · r_string_stab # string stability reward
  + w_damping  · r_damping     # oscillation damping
  + w_action   · r_action      # action magnitude regularisation
  + w_alpha    · r_alpha        # headway parameter regularisation
  + ...
```

Reward weights are dataclass fields in `rl_types.py`, overridable via YAML/CLI.

### `mustykey` — `src/mesoscopic/rl_rewards.py` (6 terms)

```text
r = w_safety   · r_safety      # collision / gap clamp penalty
  + w_spacing  · r_spacing     # spacing error
  + w_speed    · r_speed       # efficiency
  + w_comfort  · r_comfort     # acceleration
  + w_ss       · r_string_stab # string stability
  + w_variance · r_variance    # speed variance
```

**Consequence:** `rafa`'s richer reward encodes more domain knowledge (jerk, oscillation damping, action regularisation) at the cost of more hyperparameters to tune. `mustykey`'s simpler reward is easier to reason about but may miss fine-grained comfort or stability signals.

---

## 7. Mesoscopic Layer Integration

### `rafa` — Decoupled

The mesoscopic adapter (`meso_adapter.py`) provides the **baseline** but is not in the RL loop. The RL agent directly outputs `Δα` corrections to the headway parameter. The mesoscopic layer and the RL layer are independent modules.

### `mustykey` — Coupled (layered stack)

RL sits **on top of** the mesoscopic layer:

1. The mesoscopic adapter computes an `α` value using variance-based rules.
2. The RL policy outputs a residual `Δα ∈ [-0.1, 0.1]`.
3. Final `α = α_meso + Δα`.

This makes the mesoscopic logic a persistent, active component at runtime, not just a baseline for comparison. `mustykey`'s mesoscopic adapter (`meso_adapter.py`) is more deeply developed with detailed variance-based adaptation logic.

---

## 8. String Stability Metric

### `rafa` — Amplification ratio (continuous, formal)

A perturbation is injected into the ring road. The metric is:

```text
amplification = max downstream speed error / upstream speed error
```

A value > 1 means perturbations grow (unstable). This is a **continuous, quantitative** metric that can be compared across runs and plotted as a function of CAV penetration rate. Computed in the dedicated evaluation scripts.

### `mustykey` — Boolean validity flag

A run is classified as "valid" if no minimum-gap clamp (0.3 m threshold) occurs during the episode. This is a **binary** pass/fail criterion. Simple and cheap to compute per episode, but provides no gradient of stability.

---

## 9. Checkpoint Selection & Safety

### `rafa` — Safety-first ranking policy

Checkpoints are ranked by a multi-factor policy:

1. Worst-case collision count (hard constraint — lower is better)
2. Minimum gap statistics (lower tail — must not violate)
3. Traffic performance metrics (mean speed, variance)

This ensures that a checkpoint with better traffic metrics but worse safety cannot outrank a safer one.

### `mustykey` — Traffic-metric-based selection

Best checkpoints saved by:

- Primary: speed variance reduction
- Secondary: oscillation amplitude
- Tertiary: acceleration comfort (RMS acc/jerk)

No formal hard-constraint enforcement in ranking — safety is monitored as a separate logged quantity.

---

## 10. Evaluation Protocol

### `rafa`

Formal generalisation sweep:

- 4 modes: `baseline`, `adaptive`, `PPO`, `SAC`
- 4 human rates: h25, h50, h75, h100
- Multiple seeds
- With/without CAV shuffle (random vs. uniform spacing)
- With/without perturbation injection

Outputs: raw CSV, aggregate summary, publication-quality comparison plots.

### `mustykey`

Lighter sweep:

- 3 modes: `no_rl`, `PPO`, `SAC`
- 3 human rates: h25, h50, h75
- Multiple seeds
- Post-processed into a comparison report via `scripts/build_rl_comparison_report.py`

Latest results (800 episodes, `output/method_compare_random_rerun_20260323_ep800/`):

- **h25**: PPO best mean speed, SAC best variance
- **h50**: SAC best mean speed, PPO best variance
- **h75**: PPO wins 5/7 metrics; SAC struggles (42 collisions vs. 1 for PPO)

---

## 11. Configuration System

### `rafa` — Dual-level precedence

```text
CLI flags  >  YAML file  >  dataclass defaults
```

All parameters (env physics, RL hyperparameters, reward weights, GPU tuning) share a unified config namespace. Separate YAML files for full training (`rl_train.yaml`) and smoke tests (`rl_smoke.yaml`).

### `mustykey` — Single-level YAML

RL configuration embedded in the main simulation config under an `rl_layer` / `rl_training` section. Simpler but less flexible for sweeps.

---

## 12. Infrastructure & Tooling

| | `rafa` | `mustykey` |
| --- | --- | --- |
| GPU support | Core (everything runs on GPU) | None |
| HPC / SLURM | `run_eval_gpu_checkpoints.slurm` | None |
| Auto-tuning | `auto_tune.py` (batch size, lr schedule) | None |
| Live training HUD | `live_hud.py` | None |
| Interactive replay | `interactive_player.py` | None |
| Early stopping | `early_stopping.py` (patience-based) | Fixed episode budget |
| Transition visualiser | No | `transition_visualizer.py` |
| Report builder | `generate_summary_compare_plots.py` | `scripts/build_rl_comparison_report.py` |

---

## 13. Where the Branches Diverge Most

### Divergence point 1 — Execution model choice

`rafa` made an early architectural decision to rewrite the entire simulator as GPU tensor ops. This unlocked massive parallelism but required abandoning Python-loop-based simulators entirely. `mustykey` kept the conventional simulator, which is easier to debug and modify but cannot be parallelised on GPU.

### Divergence point 2 — RL placement in the stack

`rafa` treats RL as a top-level module that directly controls the environment. `mustykey` embeds RL as a sub-component of the mesoscopic control hierarchy. This means `mustykey`'s mesoscopic logic remains active and useful at runtime (not just as a baseline), while `rafa`'s mesoscopic layer becomes a research baseline only.

### Divergence point 3 — Observation richness vs. interpretability

`rafa` expanded the observation to 15 dimensions to capture global context, betting that richer input improves generalisation across penetration rates. `mustykey` kept 8 dimensions for interpretability. The trade-off is generalisation vs. debuggability.

### Divergence point 4 — Safety formalisation

`rafa` formalised safety as a hard constraint in checkpoint ranking and introduced a continuous amplification-ratio metric. `mustykey` monitors safety as a logged quantity with a Boolean pass/fail string stability criterion.

### Divergence point 5 — Primary objective

`rafa` optimises `mean_speed_last100` (fleet throughput over the final phase). `mustykey` optimises `speed_var_global` (homogeneity of speeds). These reflect different interpretations of what "good" CAV control means: throughput vs. smoothness.

---

## 14. Complementarity

The two branches are not competing implementations of the same design — they represent **two different research tracks** from the same starting point:

- **`rafa`** is a production-grade framework aimed at scale, formal evaluation, and eventual publication. Its complexity is justified by the rigour required for scientific claims about generalisation and safety.
- **`mustykey`** is an exploratory codebase for algorithm development and control-layer integration. Its simplicity makes it faster to iterate on reward shaping, mesoscopic design, and algorithm comparisons.

A synthesis would take `mustykey`'s deep mesoscopic integration and layered control logic, combined with `rafa`'s vectorised simulation, formal safety ranking, and richer observation space.
