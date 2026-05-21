#!/bin/bash
#SBATCH -J surr_mut
#SBATCH -A inz_csolso
#SBATCH -p hgx
#SBATCH -n1
#SBATCH -c8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate f1vae

cd "$HOME/master_thesis"

STAMP="$(date +%Y%m%d_%H%M%S)"

python scripts/surrogate_guided_mutation_search.py \
  --surrogate models/f1_surrogate/structural_ensemble_full/surrogate.pkl \
  --data-path datasets/f1/f1_dataset.txt \
  --framsticks-path src/framsticks/Framsticks54 \
  --framsticks-sim src/framsticks/framspy/eval-allcriteria.sim \
  --seed-max-fitness 1.2 \
  --seed-candidates 5 \
  --generations 20 \
  --pool-size 2048 \
  --true-evals-per-generation 32 \
  --parents 8 \
  --selection-modes surrogate,random \
  --output "exports/surrogate_guided_mutation_cap1p2_${STAMP}.csv"
