#!/bin/bash
#SBATCH -J cmp_latent_ops
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
  --labels 09_tree_vae_epoch80,10_transformer_vae_epoch90 \
  --methods frams,latent \
  --data-path datasets/f1/f1_dataset.txt \
  --framsticks-path src/framsticks/Framsticks54 \
  --framsticks-sim src/framsticks/framspy/eval-allcriteria.sim \
  --seed-source dataset_top \
  --seed-max-fitness 1.2 \
  --seed-candidates 5 \
  --population-size 48 \
  --offspring-size 48 \
  --elite-size 8 \
  --tournament-size 3 \
  --generations 500 \
  --max-evaluations 5000 \
  --time-budget-seconds 21600 \
  --split-time-budget-across-runs \
  --frams-crossover-prob 0.25 \
  --frams-mutate-after-crossover-prob 0.25 \
  --latent-mutation-stds 0.12,0.25,0.5,0.9 \
  --latent-crossover-prob 0.45 \
  --latent-extrapolate-prob 0.2 \
  --latent-directional-prob 0.25 \
  --latent-line-scale 0.6 \
  --latent-random-immigrant-prob 0.05 \
  --seed 777 \
  --output "exports/latent_operator_ea_compare_generators_cap1p2_${STAMP}.csv"
