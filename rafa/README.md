# P22-MCFMT Codebase

This repository contains a mixed-autonomy ring-road traffic simulator with a reinforcement-learning layer for residual CAV headway control. The inherited core is the object-based simulator, vehicle models, mesoscopic adaptation logic, and visualization stack. The RL work adds residual PPO/SAC training, vectorized GPU simulation, evaluation tooling, and comparison workflows.

The canonical docs are now:

- [`README.md`](README.md): operational entrypoint
- [`Reports/MASTER.md`](Reports/MASTER.md): rigorous technical reference

Historical notes and superseded reports live under [`Reports/archive/`](Reports/archive/).

## What This Repo Is For

- Simulate a ring-road traffic system with human drivers modeled by stochastic IDM.
- Control CAVs with a string-stable CACC + constant-time-headway baseline.
- Adapt CAV headway through a mesoscopic rule-based layer.
- Learn a bounded residual correction to that headway rule using RL.
- Compare `baseline`, `adaptive`, `ppo`, and `sac` control modes across CAV penetration rates.

## Main Entry Points

### Local GUI

Launch the local GUI with:

```powershell
python -m src
```

The GUI is a local launcher for the existing CLI entrypoints. It does not replace the scripts used for HPC or automation.

### Simulator / Control Path

- `run_simulation.py`: baseline simulator execution from a YAML config
- `run_experiments.py`: sweep over controller modes, human rates, and seeds
- `visualize_run.py`: replay/precompute visualizer for saved rollouts

### RL / GPU Path

- `train_rl.py`: older CPU PPO path
- `train_gpu_ppo.py`: vectorized GPU PPO path
- `train_gpu_sac.py`: vectorized GPU SAC path
- `evaluate_gpu_ppo.py`: evaluate GPU PPO checkpoints
- `evaluate_gpu_sac.py`: evaluate GPU SAC checkpoints
- `visualize_sac.py`: live SAC visualization or replay
- `generate_summary_compare_plots.py`: generate publication-style comparison plots from `summary_runs.csv`

## Environment Setup

### Local

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### HPC

Use the cluster Python/CUDA environment you already rely on, then install:

```bash
pip install -r requirements_hpc.txt
```

The batch helper script remains at the repo root:

```bash
sbatch run_eval_gpu_checkpoints.slurm
```

## Where To Run What

| Workflow | Recommended Location | Inputs | Main Outputs |
| --- | --- | --- | --- |
| `run_simulation.py` | Local CPU | YAML config | `output/<run>/micro.csv`, `macro.csv`, `metadata.json` |
| `train_rl.py` | Local CPU | `config/rl_smoke.yaml` or `config/rl_train.yaml` | PPO checkpoints under `output/rl_*` |
| `train_gpu_ppo.py` | HPC GPU | `config/rl_train.yaml` + optional CLI overrides | GPU PPO checkpoints and effective config |
| `train_gpu_sac.py` | HPC GPU | `config/rl_train.yaml` + optional PPO bootstrap checkpoint | SAC checkpoints and effective config |
| `evaluate_gpu_ppo.py` | Local or HPC | config + PPO checkpoint | evaluation CSV/JSON summaries |
| `evaluate_gpu_sac.py` | Local or HPC | config + SAC checkpoint | evaluation CSV/JSON summaries |
| `visualize_run.py` | Local with display | config + optional checkpoint or replay dir | replay data under `output/_viz_run` |
| `visualize_sac.py` | Local with display | SAC checkpoint or replay dir | live HUD / replay session |
| `run_experiments.py` | Local or HPC | config + mode list + checkpoints as needed | per-run outputs + summary CSVs |
| `generate_summary_compare_plots.py` | Local | `summary_runs.csv` | PNG/SVG comparison plots |

## Recommended Working Order

### 1. Baseline sanity check

```powershell
python run_simulation.py --config config\config.yaml
```

Use this when you want to verify the inherited simulator/control stack independently of RL.

### 2. Train the main RL model

Preferred path:

```powershell
python train_gpu_sac.py --config config\rl_train.yaml --device cuda
```

PPO baseline path:

```powershell
python train_gpu_ppo.py --config config\rl_train.yaml --device cuda
```

Legacy CPU PPO path:

```powershell
python train_rl.py --config config\rl_smoke.yaml
```

The GPU training entrypoints now also write lightweight scalar histories during training:

- `training_log.csv`: reward, loss, entropy / alpha, throughput, learning-rate scalars
- `eval_log.csv`: periodic evaluation `mean_speed_last100` by human rate
- `plots/`: refreshed static PNG/SVG progress figures such as training reward, diagnostics, and eval convergence

This is intended for low-overhead monitoring on local or HPC runs. If you want CSV logs without plot refreshes, pass `--disable_progress_plots`.

### 3. Evaluate checkpoints

```powershell
python evaluate_gpu_sac.py --config config\rl_train.yaml --checkpoint output\sac_train\gpu_sac_final.pt
python evaluate_gpu_ppo.py --config config\rl_train.yaml --checkpoint output\gpu_train_8core\gpu_ppo_final.pt
```

The GPU evaluation entrypoints now run the formal generalization sweep by default. For each checkpoint they evaluate the Cartesian product of:

- human ratios from `rl.hr_options` unless `--human_rates` overrides them
- seeds `0..num_seeds-1` unless `--seeds` is provided explicitly
- `ordered` and `shuffled` CAV layouts
- perturbation `off` and `on`
- controller modes `baseline`, `adaptive`, and `rl`

Each run writes:

- `gpu_eval_raw.csv` / `gpu_eval_raw.json`: one row per `(mode, human_rate, seed, shuffle, perturbation)` case
- `gpu_eval_summary.csv` / `gpu_eval_summary.json`: factor-aware aggregates over seeds
- `gpu_eval_overall_summary.csv` / `gpu_eval_overall_summary.json`: collapsed aggregates over all generalization conditions
- `gpu_eval_manifest.json`: evaluation metadata plus a clear split between training-time variation and evaluation-only variation
- `gpu_eval_report.json`: combined structured JSON payload
- `summary_plots/`: factor-aware generalization plots, including perturbation-on string-stability figures
- `summary_plots_overall/`: the existing comparison plots generated from the raw rows collapsed over the new factors

For long checkpoint sweeps, the GPU evaluation entrypoints now print case-by-case progress with ETA. If you want the fastest artifact-writing path and do not need figures immediately, pass `--skip_plots`. You can also reduce terminal noise with `--progress_every N`.

The canonical RL objective is `mean_speed_last100`: mean fleet speed over the last 100 simulation steps. This is the per-scenario metric behind training-time `eval_mean_speed` and GPU checkpoint ranking, so it is the headline number to report. `mean_speed` remains a whole-run descriptive metric, and `mean_reward` remains a training/internal shaping signal.

String stability is now reported as a formal perturbation-response metric rather than as the old Boolean validity flag. When perturbation is on, the evaluation keeps the existing controlled experiment: start from the standard uniform setup, apply the configured one-shot speed drop to the designated vehicle at `perturbation_time`, and analyze how that disturbance propagates to downstream followers. For vehicle `i`, define the speed error after perturbation as `e_i(t) = v_i(t) - mean_pre(v_i)`, where `mean_pre(v_i)` is the mean speed over the 20 steps immediately before the perturbation. Let `A_i = max_{t >= t_p} |e_i(t)|`. Ordering vehicles from the perturbed vehicle to its downstream followers, the canonical metric is `string_stability_value = max_{j >= 1} A_j / A_0`.

Interpret `string_stability_value` as follows:

- `<= 1.0`: downstream followers never exceed the perturbed vehicle's peak speed error, so the disturbance does not amplify downstream
- `> 1.0`: at least one downstream follower amplifies the disturbance, so the run is string-unstable under this controlled perturbation protocol

The outputs also include `string_stability_max_pairwise_speed_gain` and a spacing-error amplification diagnostic, but `string_stability_value` is the main reportable number.

Sources of variation already present during training:

- human-ratio sampling from `rl.hr_options` when more than one value is configured
- shuffled CAV placement when `rl.shuffle_cav_positions: true`
- perturbation timing / magnitude / target randomization when `rl.perturb_curriculum: true`
- stochastic HDV process noise when `noise_Q > 0`

Sources of variation introduced only by the formal evaluation workflow:

- an explicit matched seed sweep over the requested evaluation seeds
- toggling ordered versus shuffled layouts in the same report, even if training used only one layout policy
- toggling perturbation off versus on in the same report
- replacing perturbation curriculum with a fixed perturbation protocol whenever perturbation is on, so controller comparisons stay matched within each seed
- enumerating the human-rate grid exhaustively rather than sampling it

Safety is reported as a constraint, not as a replacement objective. When you compare controllers, pair the primary objective with:

- `min_gap` or `min_gap_episode` against the 0.3 m collision-clamp threshold
- `collision_count == 0`
- `collision_clamp_count == 0`

For string-stability reporting specifically, keep the historical `string_stability_valid` flag as the collision-clamp validity condition, not as the metric itself. The formal amplification metric is only valid when perturbation is enabled, the perturbation occurs within the episode horizon, a pre-perturbation baseline window exists, the perturbed vehicle exhibits a measurable response, and `string_stability_valid == True`.

The evaluation CSV/JSON outputs and sweep summaries now surface this explicitly via `primary_objective_metric`, `primary_objective_value` or `primary_objective_mean`, `string_stability_metric`, `string_stability_value` or `string_stability_value_mean`, the historical clamp-validity flag `string_stability_valid`, the formal metric-validity fields `string_stability_metric_valid` or `string_stability_metric_valid_rate`, and `safety_constraint_satisfied` or `safety_constraint_satisfied_rate`. In figures and tables, put the primary objective first and show safety pass rate, minimum-gap diagnostics, and the string-stability amplification metric alongside it.

Checkpoint ranking is now safety-first by default:

```powershell
python rank_gpu_checkpoints.py --eval_root output\hpc_gpu_train\evaluation
```

The default `safety_first` policy ranks checkpoints by worst-case safety pass rate first, then worst-case minimum gap and string-stability robustness, and only then by the primary objective. Use `--ranking_policy objective_first` only for sensitivity checks. In the synced PPO evaluation snapshot under `output/hpc_gpu_train/evaluation`, this rule selects `gpu_ppo_step_229376.pt` as the best available PPO checkpoint; `gpu_ppo_final.pt` is faster on average but too safety-degraded to treat as the main candidate.

### 4. Visualize behavior locally

```powershell
python visualize_sac.py --ckpt output\sac_train\gpu_sac_final.pt --config config\rl_train.yaml --mode rl
python visualize_run.py --replay output\_viz_run
```

### 5. Run experiment sweeps

```powershell
python run_experiments.py --config config\config.yaml --modes baseline adaptive ppo sac --ppo_checkpoint output\rl_train\ckpt_final.pt --sac_checkpoint output\sac_train\gpu_sac_final.pt --results_dir Results_compare_all
```

### 6. Generate comparison plots

```powershell
python generate_summary_compare_plots.py --summary_csv Results_compare_all\summary_runs.csv --output_dir Results_compare_all\summary_plots
```

## Inputs and Outputs

### Inputs

- YAML configs in [`config/`](config/)
- Optional checkpoints from previous PPO/SAC runs
- Optional CLI overrides for device, profile, env count, total timesteps, checkpoint mapping, and output locations

### Outputs

Generated outputs are intentionally not tracked anymore. By default they go to:

- `output/` for direct runs, training, evaluation, and temporary visualization data
- `Results/` or `Results_compare_all/` for sweep-style experiment results
- `Google Colab/run-*` for notebook-generated training artifacts if those notebooks are used

## Configuration Precedence

The repo uses the following precedence when values overlap:

1. CLI overrides on the entrypoint script
2. YAML values from the selected config file
3. Dataclass defaults in the relevant runtime layer, especially [`src/agents/rl_types.py`](src/agents/rl_types.py) and [`src/gpu/vec_env.py`](src/gpu/vec_env.py)

This matters most for:

- device/profile selection
- output directories
- training horizons
- human-rate domain randomization
- reward weights and action bounds

## Current Top-Level Layout

- `config/`: baseline and RL configs
- `Reports/`: canonical docs plus archived historical report sources
- `src/`: simulator, RL modules, GPU environment, visualization, utilities
- `tests/`: regression and unit tests
- `colab_gpu_training.ipynb`: historical Colab notebook
- `hpc_training.ipynb`: historical HPC notebook
- `run_eval_gpu_checkpoints.slurm`: HPC batch helper
- `requirements_hpc.txt`: HPC-specific dependency list

## Current Project Status

- The March 9, 2026 commit `ed8dc9a` is treated as the inherited baseline snapshot.
- The main RL implementation begins with the March 10, 2026 commit `ec93cef`.
- SAC is the preferred training path for current work.
- The repo now keeps source, configs, notebooks, and docs under version control, but not generated training/evaluation outputs.

## More Detail

For the rigorous mathematical formulation, decision history, current parameter tables, and inherited-vs-developed file map, use [`Reports/MASTER.md`](Reports/MASTER.md).
