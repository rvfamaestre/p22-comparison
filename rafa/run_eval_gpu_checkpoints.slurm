#!/bin/bash
#SBATCH -p gpu_inter
#SBATCH --job-name=p22-gpu-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

set -euo pipefail

module purge
module load python/3.11.2/gcc-13.1.0
module load cuda/12.1.1/gcc-13.1.0

cd "$HOME/src/p22-mcfmt-code"
source .venv-hpc/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1
unset PYTHONPATH || true
unset PYTHONHOME || true
export PYTHONNOUSERSITE=1

mkdir -p output/hpc_gpu_train/evaluation

python --version
python -c "import sys, numpy as np, torch; print(f'python={sys.executable}'); print(f'numpy={np.__version__} {np.__file__}'); print(f'torch={torch.__version__}')"

for ckpt in \
    output/hpc_gpu_train/gpu_ppo_step_229376.pt \
    output/hpc_gpu_train/gpu_ppo_step_458752.pt \
    output/hpc_gpu_train/gpu_ppo_step_688128.pt \
    output/hpc_gpu_train/gpu_ppo_step_917504.pt \
    output/hpc_gpu_train/gpu_ppo_final.pt

do
    name=$(basename "$ckpt" .pt)
    python evaluate_gpu_ppo.py \
        --config config/rl_train.yaml \
        --checkpoint "$ckpt" \
        --device cuda \
        --num_seeds 5 \
        --shuffle_layouts ordered shuffled \
        --perturbations off on \
        --output_dir "output/hpc_gpu_train/evaluation/$name"
done

python rank_gpu_checkpoints.py \
    --eval_root output/hpc_gpu_train/evaluation \
    | tee output/hpc_gpu_train/evaluation/checkpoint_ranking.txt
