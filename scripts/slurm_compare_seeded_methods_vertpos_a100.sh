#!/bin/bash
#SBATCH -J compare_seeded
#SBATCH -A inz_csolso
#SBATCH -p hgx
#SBATCH -n1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate f1vae

cd "$HOME/master_thesis"

STAMP="$(date +%Y%m%d_%H%M%S)"

python scripts/compare_seeded_methods.py \
  --checkpoints-root exports/f1_selected_10_ckpts_20260520_184738 \
  --labels 08_masked_reconstruction_ceiling_best,09_tree_vae_epoch80,10_transformer_vae_epoch90 \
  --methods framsticks,latent_random,cem,cmaes \
  --data-path datasets/f1/f1_dataset.txt \
  --framsticks-path src/framsticks/Framsticks54 \
  --framsticks-sim src/framsticks/framspy/eval-allcriteria.sim \
  --seed-source dataset_top \
  --seed-max-fitness 1.2 \
  --seed-candidates 5 \
  --iterations 100 \
  --population-size 32 \
  --output "exports/f1_selected_10_ckpts_20260520_184738/seeded_method_comparison_top3_cap1p2_${STAMP}.csv"
