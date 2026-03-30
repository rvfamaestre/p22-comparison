# Detailed Repository Change Log

This document records the evolution of the repository from its initial import to the current local workspace state. It is deliberately more detailed than the main report section and is meant as a technical trace of what changed, where it changed, and why those changes mattered.

## Scope

- Repository root: `p22-comparison`
- Main implementation used for the final reported results: `mustykey/`
- Historical companion code and experiments also present in the repository: `rafa/`
- Report sources: `mid-report/`

The main report no longer presents the work as parallel development lines. This file, however, keeps the historical repository view because its purpose is to document how the codebase evolved over time.

## 1. Initial import

Commit: `961ef05`  
Message: `initialization`

This first state introduced the complete project skeleton:

- `mid-report/`
  - added the LaTeX report structure:
    - `main.tex`
    - `references.bib`
    - all initial section files
    - `images/logo.png`
    - `images/system_architecture.png`
  - established the first narrative structure of the report.

- `mustykey/`
  - introduced the initial simulation and control implementation:
    - environment
    - vehicle models
    - mesoscopic adapter
    - PPO/SAC training modules
    - simulation manager
    - plotting/report scripts
  - added many YAML configurations for:
    - base scenarios
    - evaluation runs
    - training runs
    - transition experiments
    - string-stability checks
  - added the first orchestrated outputs under `mustykey/output/`, including:
    - training artifacts
    - evaluation artifacts
    - plots
    - summary CSV/XLSX/Markdown files

- `rafa/`
  - imported a more GPU-oriented experimental codebase with:
    - GPU vectorized environments
    - GPU PPO/SAC trainers
    - utility modules for checkpoint ranking and safety-aware evaluation
    - its own evaluation outputs and plots

- Repository hygiene
  - the initialization still tracked many compiled Python artifacts (`__pycache__`, `.pyc`) and large generated outputs, which later became a maintenance issue.

## 2. First consolidation pass

Commit: `c6c69a9`  
Message: `few changes`

This was the first major convergence step around the main implementation used later for the updated experiments.

### 2.1 Report changes

- `mid-report/sections/current_progress_and_future_work.tex`
  - expanded substantially.
  - introduced the first comparison-oriented narrative and experimental summary.
- `mid-report/main.tex`
  - small updates associated with the rebuilt PDF.
- generated LaTeX outputs were also added:
  - `main.bbl`
  - `main.pdf`
  - `main.synctex.gz`

### 2.2 Metric and early-stopping upgrades in `mustykey/`

New utility modules were added:

- `mustykey/src/utils/early_stopping.py`
  - added a configurable plateau-based early stopping monitor.
  - supports:
    - metric choice
    - min/max mode
    - patience
    - minimum delta
    - delayed tracking start
    - EMA smoothing
    - best-checkpoint restoration metadata

- `mustykey/src/utils/string_stability_metrics.py`
  - introduced a formal downstream amplification metric.
  - added validity handling for perturbation experiments.
  - explicitly invalidates the metric when collision clamps occur.

- `mustykey/src/utils/primary_objective.py`
  - formalized `mean_speed_last100` as the primary late-run objective.
  - added safety-annotation helpers for:
    - minimum-gap checks
    - collision-free checks
    - collision-clamp checks
    - string-stability safety checks

### 2.3 Training pipeline updates

- `mustykey/run_training.py`
  - replaced simplistic stopping logic with the new early-stop monitor.
  - added `mean_speed_last100` to tracked metrics.
  - began writing `early_stopping_summary.json`.

- `mustykey/src/mesoscopic/rl_training.py`
- `mustykey/src/mesoscopic/sac_training.py`
  - both received `load_model()` support so that best-checkpoint restoration could work cleanly.

### 2.4 Evaluation/reporting updates

- `mustykey/run_experiments.py`
  - added missing metrics to the experiment summary:
    - `mean_speed_last100`
    - string-stability outputs
    - safety summary fields
    - additional plots for the new metrics

- `mustykey/scripts/build_rl_comparison_report.py`
  - expanded the exported tables and comparison plots.
  - included the newly added metrics in the report outputs.

### 2.5 Training config coverage

Originally only `h50` train configs existed. This pass added:

- `mustykey/config/compare_train_ppo_h25_random.yaml`
- `mustykey/config/compare_train_ppo_h75_random.yaml`
- `mustykey/config/compare_train_sac_h25_random.yaml`
- `mustykey/config/compare_train_sac_h75_random.yaml`

These files completed the training-config matrix across human-driver ratios.

### 2.6 Repository hygiene

- `.gitignore`
  - updated to ignore Python cache and bytecode files more generally.
- a large number of tracked cache files were removed from the index in the same consolidation period.

## 3. Vectorized GPU training backend

Commit: `a894e62`  
Message: `gpu run`

This was the most significant implementation jump for the main codebase.

### 3.1 New GPU/vectorized package

Added:

- `mustykey/src/gpu/__init__.py`
- `mustykey/src/gpu/vec_env.py`
- `mustykey/src/gpu/vectorized_training.py`

These files introduced a batched torch-native training path while preserving compatibility with the existing controller design.

### 3.2 What this backend changed technically

- the simulator state for training was rewritten into tensor form;
- multiple environments could now be stepped in parallel;
- PPO and SAC could both train against the vectorized rollout path;
- the existing 8-feature observation format was preserved, so models remained compatible with the standard `mustykey` inference/evaluation path.

### 3.3 Training entry-point rewrite

- `mustykey/run_training.py`
  - was restructured around two execution modes:
    - the original CPU/sequential path;
    - the new vectorized backend.
  - added:
    - backend selection
    - device resolution
    - parallel environment count
    - vectorized rollout/update orchestration

### 3.4 Config changes

The training configs were updated to opt into the new backend:

- `mustykey/config/compare_train_ppo_h25_random.yaml`
- `mustykey/config/compare_train_ppo_h50_random.yaml`
- `mustykey/config/compare_train_ppo_h75_random.yaml`
- `mustykey/config/compare_train_sac_h25_random.yaml`
- `mustykey/config/compare_train_sac_h50_random.yaml`
- `mustykey/config/compare_train_sac_h75_random.yaml`

Key added fields:

- `backend: "vectorized"`
- `device: "auto"`
- `num_envs: 8`

### 3.5 Why this mattered

This made GPU training practical on the DCE cluster while keeping the final evaluation path simple and understandable. It also enabled the later training runs whose results are now discussed in the report.

## 4. Dependency split for a cleaner environment

Commit: `9002d0f`  
Message: `Add vectorized training backend and report metrics`

This commit cleaned the Python dependency story:

- `mustykey/requirements.txt`
  - reduced to the core packages actually required for the main pipeline.

- `mustykey/requirements-optional.txt`
  - added for optional or heavier extras such as graph-related tooling and TensorBoard-like features.

This was especially useful for remote execution because it reduced environment friction on the GPU cluster.

## 5. Post-run fixes after the first DCE evaluation attempt

Commit: `d01687a`  
Message: `fixes`

This pass addressed issues discovered during remote execution and report generation.

### 5.1 Evaluation now uses the best checkpoint

Updated:

- `mustykey/config/compare_eval_ppo_h25_random.yaml`
- `mustykey/config/compare_eval_ppo_h50_random.yaml`
- `mustykey/config/compare_eval_ppo_h75_random.yaml`
- `mustykey/config/compare_eval_sac_h25_random.yaml`
- `mustykey/config/compare_eval_sac_h50_random.yaml`
- `mustykey/config/compare_eval_sac_h75_random.yaml`

Change:

- model paths now point to `rl_policy_best.pt` instead of the last checkpoint.

### 5.2 Simulation outputs became more reproducible

- `mustykey/run_simulation.py`
  - now writes `config_effective.yml` into each simulation output directory;
  - added `--seed`;
  - added `--output-path`.

These changes make evaluation reruns traceable and support holdout-seed studies.

### 5.3 Report-builder robustness

- `mustykey/scripts/build_rl_comparison_report.py`
  - fixed the `KeyError: 'string_stability_value'` issue seen when the metric column was absent;
  - added safer handling of missing or invalid string-stability fields;
  - made the report builder less brittle when some runs do not produce a valid amplification ratio.

### 5.4 Repository hygiene

- `.gitignore`
  - adjusted again to keep Python cache files out of version control.

## 6. Cluster execution and operational documentation

This part is not a single historical commit theme; it is the operational layer that became important once the GPU pipeline was actually exercised on DCE.

### 6.1 DCE workflow documentation

Added:

- `mustykey/DCE_GPU_PIPELINE.md`

This document records the concrete working flow used on the university infrastructure:

- SSH to `dce.metz.centralesupelec.fr`
- request a GPU node with `srun -p gpu_inter`
- derive a virtual environment from `/mounts/datasets/venvs/torch-2.7.1/`
- verify CUDA visibility through PyTorch
- install the core requirements
- run training, evaluation, and report generation
- copy results back locally

### 6.2 Confirmed remote runtime behavior

The DCE run validated that:

- the vectorized backend selected `cuda` automatically;
- training really ran as `vectorized (cuda)`;
- PPO and SAC both completed under the new main pipeline;
- results could be exported back into the local repository and analyzed from there.

## 7. Current local workspace state after pulling back the results

The following items are part of the present local state, even if not all of them correspond to historical commits yet.

### 7.1 Report-local result bundle

Created inside `mid-report/`:

- `mid-report/results/comparison_metrics_snapshot.csv`
- `mid-report/results/training_summary_snapshot.csv`
- `mid-report/results/experiment_report_snapshot.md`
- `mid-report/results/ppo_training_metrics_snapshot.json`
- `mid-report/results/sac_training_metrics_snapshot.json`
- `mid-report/results/ppo_early_stopping_snapshot.json`
- `mid-report/results/sac_early_stopping_snapshot.json`

Purpose:

- make the report folder self-contained;
- keep the report narrative decoupled from `mustykey/output/`;
- preserve the exact numerical snapshots used to write the final conclusions.

### 7.2 Report-local images

Copied or generated under `mid-report/images/results/`:

- copied snapshots:
  - `overview_comparison.png`
  - `training_overview_comparison.png`
  - `mean_speed_last100_comparison.png`
  - `speed_var_last100_comparison.png`
  - `collision_count_comparison.png`
  - `collision_clamp_count_comparison.png`

- new report-focused dashboards:
  - `evaluation_dashboard.png`
  - `training_diagnostics.png`
  - `performance_tradeoff_map.png`

Purpose:

- ensure all figures referenced by the report live under `mid-report/`;
- avoid report dependencies on the experiment output tree;
- improve the readability of the results section with cleaner summary plots.

### 7.3 Report narrative rewrite

Updated:

- `mid-report/sections/current_progress_and_future_work.tex`

Main changes:

- removed the old two-line development framing from the report narrative;
- rewrote the section around the current consolidated implementation only;
- added:
  - updated experiment protocol;
  - training summary table;
  - updated evaluation table from the latest local results;
  - a qualitative synthesis table;
  - clearer reasoning about safety, invalid string-stability runs, and the intermediate-regime benefit of RL;
  - an explicit limitation note about the lack of unseen-seed evaluation.

### 7.4 Consistency fixes in the technical sections

Updated:

- `mid-report/sections/reinforcement_learning.tex`
- `mid-report/sections/system_architecture.tex`
- `mid-report/main.tex`

Main changes:

- aligned the report with the actual 8-feature policy input:
  - upstream mean speed;
  - upstream variance;
  - macro speed mismatch;
  - ego speed;
  - spacing;
  - relative speed;
  - leader acceleration;
  - previous headway state;
- clarified the residual-policy input description;
- updated the report date in the page header.

### 7.5 This file itself

Added:

- `mid-report/repository_changes_detailed.md`

Purpose:

- provide a durable technical memory of how the repository evolved;
- separate implementation history from the cleaner scientific narrative of the report.

## 8. Net effect of the repository evolution

From initialization to the current state, the repository moved through four major phases:

1. initial import of a broad mixed-traffic RL project with report skeleton and multiple experiment trees;
2. first consolidation of metrics, safety reporting, and early stopping into the main implementation;
3. introduction of a practical vectorized GPU training path within the same main controller design;
4. post-run hardening of evaluation/reporting, followed by a report rewrite grounded in the actual latest results.

The most important technical outcome is that the main implementation now supports:

- GPU-assisted batched training;
- checkpoint-aware training summaries;
- late-run objective reporting;
- explicit safety flags;
- string-stability validity checks;
- self-contained report assets and conclusions tied to the pulled-back local results.
