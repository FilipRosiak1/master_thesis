#!/bin/bash

set -euo pipefail

cd "$HOME/master_thesis"

RUN_DIR="${1:?Usage: submit_final_benchmark_cpu_mini_jobs.sh RUN_DIR}"
CPU_PLAN="$RUN_DIR/job_plan.cpu.csv"

if [ ! -f "$RUN_DIR/seed_manifest.csv" ]; then
  echo "Missing seed manifest: $RUN_DIR/seed_manifest.csv" >&2
  exit 2
fi
if [ ! -f "$CPU_PLAN" ]; then
  echo "Missing CPU job plan: $CPU_PLAN" >&2
  exit 2
fi

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

CPU_SUBMITTED=0
while IFS= read -r JOB_INDEX; do
  sbatch \
    --output="$RUN_DIR/logs/cpu_mini_${JOB_INDEX}_%j.out" \
    scripts/slurm_final_benchmark_cpu_mini_worker.sh "$RUN_DIR" "$CPU_PLAN" "$JOB_INDEX"
  CPU_SUBMITTED=$((CPU_SUBMITTED + 1))
done < <(plan_indices "$CPU_PLAN")

echo "Submitted mini CPU jobs: $CPU_SUBMITTED"
