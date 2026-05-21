#!/bin/bash
#SBATCH -J grammar_cem
#SBATCH -A inz_csolso
#SBATCH -p hgx
#SBATCH -n1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

set -euo pipefail

source "$HOME/miniconda3/bin/activate"
conda activate f1vae

cd "$HOME/master_thesis"

STAMP="$(date +%Y%m%d_%H%M%S)"

if [[ ! -f "src/framsticks/framspy/FramsticksLib.py" ]]; then
  echo "Missing src/framsticks/framspy/FramsticksLib.py on cluster." >&2
  exit 1
fi

if [[ ! -d "src/framsticks/Framsticks54" ]]; then
  echo "Missing src/framsticks/Framsticks54 runtime on cluster." >&2
  exit 1
fi

python scripts/grammar_cem_search.py \
  --data-path datasets/f1/f1_dataset.txt \
  --framsticks-path src/framsticks/Framsticks54 \
  --framsticks-sim src/framsticks/framspy/eval-allcriteria.sim \
  --init-max-fitness 1.2 \
  --init-top-count 200 \
  --generations 50 \
  --population-size 512 \
  --elite-fraction 0.1 \
  --smoothing 0.5 \
  --output "exports/grammar_cem_cap1p2_${STAMP}.csv"
