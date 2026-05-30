#!/bin/bash
#SBATCH -J latent_evo_deep
#SBATCH -A inz_csolso
#SBATCH -p hgx
#SBATCH -n1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate f1vae

cd "$HOME/master_thesis"

STAMP="$(date +%Y%m%d_%H%M%S)"

python scripts/benchmark_selected_latents.py \
  --checkpoints-root exports/f1_selected_10_ckpts_20260520_184738 \
  --labels 09_tree_vae_epoch80,10_transformer_vae_epoch90 \
  --data-path datasets/f1/f1_dataset.txt \
  --framsticks-path src/framsticks/Framsticks54 \
  --framsticks-sim src/framsticks/framspy/eval-allcriteria.sim \
  --algorithms evolution \
  --seed-source dataset_top \
  --seed-max-fitness 1.2 \
  --seed-candidates 5 \
  --iterations 100000 \
  --time-budget-seconds 10800 \
  --population-size 48 \
  --elite-fraction 0.25 \
  --initial-std 0.6 \
  --min-std 0.03 \
  --mutation-decay 0.998 \
  --crossover-rate 0.4 \
  --slp-interval 0 \
  --seed 780 \
  --output "exports/latent_evolution_generators_deep_no_slp_cap1p2_${STAMP}.csv"
