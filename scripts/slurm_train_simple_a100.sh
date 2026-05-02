#!/bin/bash
#SBATCH -J f1vae_simple_a100
#SBATCH -A REPLACE_WITH_ACCOUNT
#SBATCH -p hgx
#SBATCH -w hgx1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate f1vae

cd "$HOME/Magisterka"

python scripts/train_simple_models_with_validation.py \
  --data-path datasets/f1/f1_dataset_1k_same.txt \
  --output-root models/f1_val \
  --epochs 300 \
  --val-split 0.2 \
  --val-every 10 \
  --checkpoint-every 10 \
  --seed 42
