#!/bin/bash
#SBATCH -J evo_gen_cap
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
  --checkpoints-root exports/f1_selected_10_ckpts_20260520_184738 \
  --labels 02_lhs_latent256_seed42_best,09_tree_vae_epoch80,10_transformer_vae_epoch90 \
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
  --initial-std 0.45 \
  --min-std 0.03 \
  --mutation-decay 0.995 \
  --crossover-rate 0.35 \
  --slp-interval 0 \
  --seed 421 \
  --output "exports/f1_selected_10_ckpts_20260520_184738/latent_evolution_generators_cap1p2_${STAMP}.csv"
