#!/bin/bash

set -euo pipefail

cd "$HOME/master_thesis"

RUN_DIR="${1:?Usage: submit_final_benchmark_jobs.sh RUN_DIR}"
GPU_PLAN="$RUN_DIR/job_plan.gpu.csv"
CPU_PLAN="$RUN_DIR/job_plan.cpu.csv"

mkdir -p "$RUN_DIR/logs"

plan_indices() {
  python - "$1" <<'PY'
import csv
import sys
with open(sys.argv[1], newline='', encoding='utf-8') as handle:
    for idx, _ in enumerate(csv.DictReader(handle)):
        print(idx)
PY
}

GPU_SUBMITTED=0
while IFS= read -r JOB_INDEX; do
  sbatch \
    --output="$RUN_DIR/logs/gpu_${JOB_INDEX}_%j.out" \
    scripts/slurm_final_benchmark_gpu_worker.sh "$RUN_DIR" "$GPU_PLAN" "$JOB_INDEX"
  GPU_SUBMITTED=$((GPU_SUBMITTED + 1))
done < <(plan_indices "$GPU_PLAN")

CPU_SUBMITTED=0
while IFS= read -r JOB_INDEX; do
  sbatch \
    --output="$RUN_DIR/logs/cpu_${JOB_INDEX}_%j.out" \
    scripts/slurm_final_benchmark_cpu_worker.sh "$RUN_DIR" "$CPU_PLAN" "$JOB_INDEX"
  CPU_SUBMITTED=$((CPU_SUBMITTED + 1))
done < <(plan_indices "$CPU_PLAN")

echo "Submitted GPU jobs: $GPU_SUBMITTED"
echo "Submitted CPU jobs: $CPU_SUBMITTED"
