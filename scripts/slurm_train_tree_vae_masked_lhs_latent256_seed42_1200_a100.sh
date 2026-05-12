#!/bin/bash
#SBATCH -J lhs_z256_s42
#SBATCH -A inz_csolso
#SBATCH -p hgx
#SBATCH -w hgx1
#SBATCH -n1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate f1vae

cd "$HOME/master_thesis"

python scripts/train.py \
  --model tree_vae_masked_lhs \
  --data-path datasets/f1/f1_dataset.txt \
  --output-root models/f1_sweep_lhs/seed42_latent256_1200 \
  --latent-dim 256 \
  --hidden-dim 512 \
  --embedding-dim 64 \
  --epochs 1200 \
  --schedule-epochs 1200 \
  --learning-rate 0.001 \
  --seed 42 \
  --val-split 0.2 \
  --val-every 10 \
  --checkpoint-every 10
