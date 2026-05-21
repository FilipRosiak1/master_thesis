#!/bin/bash
#SBATCH -J train_fit_lhs
#SBATCH -A inz_csolso
#SBATCH -p hgx
#SBATCH -n1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate f1vae

cd "$HOME/master_thesis"

python scripts/train_fitness_guided_vae.py \
  --mode auxiliary \
  --model tree_vae_masked_lhs \
  --data-path datasets/f1/f1_dataset.txt \
  --run-dir models/f1_guided/fitness_aware_lhs_seed42 \
  --latent-dim 128 \
  --hidden-dim 512 \
  --embedding-dim 64 \
  --batch-size 32 \
  --epochs 1200 \
  --learning-rate 0.001 \
  --max-length 500 \
  --checkpoint-every 10 \
  --seed 42 \
  --val-split 0.2 \
  --val-every 10 \
  --schedule-epochs 1200 \
  --fitness-weight 100
