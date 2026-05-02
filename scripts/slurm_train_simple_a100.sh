#!/bin/bash
#SBATCH -A inz_csolso
#SBATCH -w hgx2
#SBATCH -n1
#SBATCH -c1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate f1vae

cd "$HOME/Magisterka"

python scripts/train_simple_models_with_validation.py \
  --data-path datasets/f1/f1_dataset.txt \
  --output-root models/f1_val \
  --epochs 1000 \
  --val-split 0.2 \
  --val-every 10 \
  --checkpoint-every 10 