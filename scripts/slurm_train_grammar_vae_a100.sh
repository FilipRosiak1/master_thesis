#!/bin/bash
#SBATCH -J grammar_vae_a100
#SBATCH -A inz_csolso
#SBATCH -p hgx
#SBATCH -w hgx2
#SBATCH -n1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate f1vae

cd "$HOME/master_thesis"

python scripts/train.py \
  --model grammar_vae \
  --data-path datasets/f1/f1_dataset.txt \
  --output-root models/f1_val \
  --epochs 1000 \
  --val-split 0.2 \
  --val-every 10 \
  --checkpoint-every 10
