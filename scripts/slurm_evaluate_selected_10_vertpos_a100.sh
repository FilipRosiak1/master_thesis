#!/bin/bash
#SBATCH -J eval_vertpos_10
#SBATCH -A inz_csolso
#SBATCH -p hgx
#SBATCH -w hgx2
#SBATCH -n1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate f1vae

cd "$HOME/master_thesis"

python scripts/evaluate_selected_checkpoints.py \
  --checkpoints-root exports/f1_selected_10_ckpts_20260520_184738 \
  --data-path datasets/f1/f1_dataset.txt \
  --framsticks-path src/framsticks/Framsticks54 \
  --framsticks-sim src/framsticks/framspy/eval-allcriteria.sim \
  --generation-samples 100 \
  --latent-samples 1000 \
  --max-recon-items 512 \
  --batch-size 32 \
  --output exports/f1_selected_10_ckpts_20260520_184738/evaluation_framsticks.csv
