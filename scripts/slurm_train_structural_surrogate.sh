#!/bin/bash
#SBATCH -J train_surrogate
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

python scripts/train_structural_surrogate.py \
  --data-path datasets/f1/f1_dataset.txt \
  --output-dir models/f1_surrogate/structural_ensemble_full \
  --test-size 0.2 \
  --seed 42 \
  --estimators 800 \
  --ensemble-size 8 \
  --top-fracs 0.01,0.05,0.10
