# DCE GPU Pipeline

Concrete runbook for the current `p22-comparison` repo layout on the CentraleSupelec DCE cluster.

Validated against the current repo structure on March 30, 2026:
- front node: `chome`
- interactive GPU node example: `tx12`
- repo root on DCE: `~/p22-comparison`

## 1. Sync the repo on the front node

```bash
ssh maestr_raf@dce.metz.centralesupelec.fr
cd ~/p22-comparison
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse --short HEAD
git log -1 --oneline
```

## 2. Request an interactive GPU shell

```bash
srun --nodes=1 --time=01:00:00 -p gpu_inter --pty /bin/bash
hostname
nvidia-smi
cd ~/p22-comparison
```

## 3. Activate Python on the GPU node

Recommended self-contained DCE venv:

```bash
export MPLCONFIGDIR="/tmp/$USER-mpl"
mkdir -p "$MPLCONFIGDIR"
/opt/dce/dce_venv.sh /mounts/datasets/venvs/torch-2.7.1/ "$TMPDIR/venv"
source "$TMPDIR/venv/bin/activate"
pip install -r requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

If you already maintain the conda env used by the long script, you can use this instead:

```bash
export MPLCONFIGDIR="/tmp/$USER-mpl"
mkdir -p "$MPLCONFIGDIR"
conda activate mixed_traffic
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

## 4. Commands to run right now from `tx12`

You are already at:

```bash
maestr_raf@tx12:~/p22-comparison$
```

Run the corrected 200-episode comparison pipeline from there:

```bash
export MPLCONFIGDIR="/tmp/$USER-mpl"
mkdir -p "$MPLCONFIGDIR"

python run_training.py --config config/compare_train_ppo_h50_random.yaml
python run_training.py --config config/compare_train_sac_h50_random.yaml

python run_simulation.py --config config/compare_eval_no_rl_h25_random.yaml
python run_simulation.py --config config/compare_eval_no_rl_h50_random.yaml
python run_simulation.py --config config/compare_eval_no_rl_h75_random.yaml

python run_simulation.py --config config/compare_eval_ppo_h25_random.yaml
python run_simulation.py --config config/compare_eval_ppo_h50_random.yaml
python run_simulation.py --config config/compare_eval_ppo_h75_random.yaml

python run_simulation.py --config config/compare_eval_sac_h25_random.yaml
python run_simulation.py --config config/compare_eval_sac_h50_random.yaml
python run_simulation.py --config config/compare_eval_sac_h75_random.yaml

python scripts/build_rl_comparison_report.py \
  --base-output output/method_compare_random \
  --methods no_rl ppo sac \
  --training-methods ppo sac \
  --human-ratios 0.25 0.5 0.75
```

## 5. Optional long rerun script

The repo already contains a convenience script for the 800-episode rerun configs:

```bash
bash scripts/run_method_compare_ep800.sh
```

Notes:
- this script expects a working Python environment before launch;
- if you use conda, it also works with `ENV_NAME=mixed_traffic bash scripts/run_method_compare_ep800.sh`;
- outputs go under `output/method_compare_random_rerun_20260323_ep800`.

## 6. Main output paths

For the corrected 200-episode comparison:

```bash
output/method_compare_random/ppo/training/trained_models/
output/method_compare_random/sac/training/trained_models/
output/method_compare_random/comparison_plots/comparison_metrics.csv
output/method_compare_random/comparison_plots/experiment_report.md
```

For the 800-episode rerun script:

```bash
output/method_compare_random_rerun_20260323_ep800/
```

## 7. Copy results back to the local machine

Run this from your local machine, not from DCE:

```bash
scp -r maestr_raf@dce.metz.centralesupelec.fr:~/p22-comparison/output/method_compare_random ./output/
```

Or for the long rerun:

```bash
scp -r maestr_raf@dce.metz.centralesupelec.fr:~/p22-comparison/output/method_compare_random_rerun_20260323_ep800 ./output/
```

## 8. Sanity checks after training

Useful quick checks:

```bash
python - <<'PY'
import json
from pathlib import Path
for method in ["ppo", "sac"]:
    path = Path("output/method_compare_random") / method / "training" / "trained_models" / "early_stopping_summary.json"
    data = json.loads(path.read_text())
    print(method, data["metric"], data["best_step"], data["best_raw_metric"], data["stop_step"])
PY
```

The expected training metric is now `training_objective`, not `speed_var_global`.
