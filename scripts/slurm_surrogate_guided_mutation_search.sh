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
SURROGATE="models/f1_surrogate/structural_ensemble_full/surrogate.pkl"

if [[ ! -f "$SURROGATE" ]]; then
  echo "Missing surrogate: $SURROGATE" >&2
  echo "Run scripts/slurm_train_structural_surrogate.sh first, or submit this job with afterok dependency." >&2
  exit 1
fi

if [[ ! -f "src/framsticks/framspy/FramsticksLib.py" ]]; then
  echo "Missing src/framsticks/framspy/FramsticksLib.py on cluster." >&2
  exit 1
fi

if [[ ! -d "src/framsticks/Framsticks54" ]]; then
  echo "Missing src/framsticks/Framsticks54 runtime on cluster." >&2
  exit 1
fi

python scripts/surrogate_guided_mutation_search.py \
  --surrogate "$SURROGATE" \
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
