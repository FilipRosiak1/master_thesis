#!/bin/bash
#SBATCH -J opt_vertpos_10
#SBATCH -A inz_csolso
#SBATCH -p hgx
#SBATCH -w hgx2
#SBATCH -n1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:30:00

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate f1vae

cd "$HOME/master_thesis"

STAMP="$(date +%Y%m%d_%H%M%S)"

python scripts/benchmark_selected_latents.py \
  --checkpoints-root exports/f1_selected_10_ckpts_20260520_184738 \
  --data-path datasets/f1/f1_dataset.txt \
  --framsticks-path src/framsticks/Framsticks54 \
  --framsticks-sim src/framsticks/framspy/eval-allcriteria.sim \
  --algorithms cem,cmaes \
  --seed-source dataset_top \
  --seed-max-fitness 1.2 \
  --seed-candidates 5 \
  --iterations 100000 \
  --time-budget-seconds 600 \
  --population-size 16 \
  --output "exports/f1_selected_10_ckpts_20260520_184738/latent_optimization_all10_cap1p2_${STAMP}.csv"
