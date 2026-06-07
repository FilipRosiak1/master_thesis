#!/bin/bash
#SBATCH -J final_frams
#SBATCH -A inz_csolso
#SBATCH -p hgx
#SBATCH -n1
#SBATCH --mem=64G
#SBATCH --time=72:00:00

set -euo pipefail

RUN_DIR="${1:?Usage: slurm_final_benchmark_cpu_worker.sh RUN_DIR [JOB_PLAN] [JOB_INDEX]}"
JOB_PLAN="${2:-$RUN_DIR/job_plan.cpu.csv}"
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
  --require-job-type baseline \
  --data-path datasets/f1/f1_dataset.txt \
  --framsticks-path src/framsticks/Framsticks54 \
  --framsticks-sim src/framsticks/framspy/eval-allcriteria.sim \
  --generations 300 \
  --population-size 100 \
  --offspring-size 100 \
  --elite-size 10 \
  --tournament-size 3 \
  --frams-crossover-prob 0.25 \
  --frams-mutate-after-crossover-prob 0.25 \
  --seed 777 \
  --skip-existing
