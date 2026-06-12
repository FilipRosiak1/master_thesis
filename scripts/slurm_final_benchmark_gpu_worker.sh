#!/bin/bash
#SBATCH -J final_latent
#SBATCH -A inz_csolso
#SBATCH -p hgx
#SBATCH -n1
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00

set -euo pipefail

RUN_DIR="${1:?Usage: slurm_final_benchmark_gpu_worker.sh RUN_DIR [JOB_PLAN] [JOB_INDEX]}"
JOB_PLAN="${2:-$RUN_DIR/job_plan.gpu.csv}"
JOB_INDEX="${3:-${SLURM_ARRAY_TASK_ID:-}}"

if [ -z "$JOB_INDEX" ]; then
  echo "Missing JOB_INDEX argument" >&2
  exit 2
fi

set --
source "$HOME/miniconda3/bin/activate"
conda activate f1vae

cd "$HOME/master_thesis"

python scripts/run_final_benchmark_job.py \
  --job-plan "$JOB_PLAN" \
  --job-index "$JOB_INDEX" \
  --seed-manifest "$RUN_DIR/seed_manifest.csv" \
  --require-job-type latent \
  --checkpoints-root exports/f1_selected_10_ckpts_20260520_184738 \
  --data-path datasets/f1/f1_dataset.txt \
  --framsticks-path src/framsticks/Framsticks54 \
  --framsticks-sim "src/framsticks/framspy/eval-allcriteria.sim;src/framsticks/framspy/deterministic.sim;src/framsticks/framspy/sample-period-2.sim;src/framsticks/framspy/only-body.sim" \
  --device cuda \
  --generations 300 \
  --iterations 300 \
  --population-size 100 \
  --offspring-size 100 \
  --elite-size 10 \
  --tournament-size 3 \
  --latent-mutation-stds 0.12,0.25,0.5,0.9 \
  --latent-crossover-prob 0.45 \
  --latent-extrapolate-prob 0.2 \
  --latent-directional-prob 0.25 \
  --latent-line-scale 0.6 \
  --latent-random-immigrant-prob 0.05 \
  --elite-fraction 0.25 \
  --initial-std 0.35 \
  --smoothing 0.35 \
  --min-std 0.02 \
  --seed 777 \
  --skip-existing
