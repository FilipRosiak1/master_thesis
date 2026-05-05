#!/bin/bash
#SBATCH -J tree_masked_recon
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
  --model tree_vae_masked \
  --data-path datasets/f1/f1_dataset.txt \
  --output-root models/f1_sweep/reconstruction_ceiling \
  --epochs 500 \
  --schedule-epochs 100000 \
  --learning-rate 0.003 \
  --val-split 0.2 \
  --val-every 10 \
  --checkpoint-every 10
  