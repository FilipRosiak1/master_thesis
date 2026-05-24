#!/bin/bash
#SBATCH -J evo_new_slp
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

STAMP="$(date +%Y%m%d_%H%M%S)"

python scripts/benchmark_selected_latents.py \
  --checkpoints-root models/f1_guided \
  --labels conditional_lhs_seed42/tree_vae_masked_lhs_cond/2026-05-21_21-10-52,fitness_aware_lhs_seed42/tree_vae_masked_lhs/2026-05-21_19-54-36,struct_conditional_lhs_seed42/tree_vae_masked_lhs_struct_cond/2026-05-21_21-15-40 \
  --data-path datasets/f1/f1_dataset.txt \
  --framsticks-path src/framsticks/Framsticks54 \
  --framsticks-sim src/framsticks/framspy/eval-allcriteria.sim \
  --algorithms evolution \
  --seed-source dataset_top \
  --seed-max-fitness 1.2 \
  --seed-candidates 5 \
  --iterations 100000 \
  --time-budget-seconds 1200 \
  --population-size 24 \
  --elite-fraction 0.25 \
  --initial-std 0.25 \
  --min-std 0.02 \
  --mutation-decay 0.997 \
  --crossover-rate 0.35 \
  --slp-interval 1 \
  --condition-fitness 2.0 \
  --condition-length 44 \
  --condition-segments 6 \
  --seed 423 \
  --output "exports/new_neural_latent_evolution_slp_cap1p2_${STAMP}.csv"
