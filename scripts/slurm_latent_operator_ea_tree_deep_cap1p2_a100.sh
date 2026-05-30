#!/bin/bash
#SBATCH -J latent09_deep
#SBATCH -A inz_csolso
#SBATCH -p hgx
#SBATCH -n1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate f1vae

cd "$HOME/master_thesis"

STAMP="$(date +%Y%m%d_%H%M%S)"

python scripts/compare_latent_operators_ea.py \
  --checkpoints-root exports/f1_selected_10_ckpts_20260520_184738 \
  --labels 09_tree_vae_epoch80 \
  --methods latent \
  --data-path datasets/f1/f1_dataset.txt \
  --framsticks-path src/framsticks/Framsticks54 \
  --framsticks-sim src/framsticks/framspy/eval-allcriteria.sim \
  --seed-source dataset_top \
  --seed-max-fitness 1.2 \
  --seed-candidates 5 \
  --population-size 64 \
  --offspring-size 64 \
  --elite-size 10 \
  --tournament-size 4 \
  --generations 1000 \
  --max-evaluations 20000 \
  --time-budget-seconds 39600 \
  --split-time-budget-across-runs \
  --latent-mutation-stds 0.08,0.16,0.32,0.64,1.0 \
  --latent-crossover-prob 0.5 \
  --latent-extrapolate-prob 0.3 \
  --latent-directional-prob 0.35 \
  --latent-line-scale 0.8 \
  --latent-random-immigrant-prob 0.08 \
  --seed 779 \
  --output "exports/latent_operator_ea_tree_deep_cap1p2_${STAMP}.csv"
