#!/bin/bash

set -euo pipefail

REPO_ROOT="${1:-$HOME/master_thesis}"
EXPORTS_DIR="$REPO_ROOT/exports"
SINCE_DAYS="${SINCE_DAYS:-14}"
LATEST_RUNS_PER_ROOT="${LATEST_RUNS_PER_ROOT:-6}"
INCLUDE_LATEST_EPOCH="${INCLUDE_LATEST_EPOCH:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXPORT_DIR="$EXPORTS_DIR/latest_results_$STAMP"
ARCHIVE="$EXPORT_DIR.tar.gz"
MANIFEST="$EXPORT_DIR/manifest.txt"

mkdir -p "$EXPORT_DIR"
: > "$MANIFEST"

add_path() {
  local path="$1"
  if [[ -e "$path" ]]; then
    printf '%s\n' "${path#$REPO_ROOT/}" >> "$MANIFEST"
  fi
}

add_recent_files() {
  local root="$1"
  shift
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  find "$root" -maxdepth 1 -type f "$@" -mtime -"$SINCE_DAYS" -print 2>/dev/null | while IFS= read -r path; do
    add_path "$path"
  done
}

add_recent_export_files_recursive() {
  local root="$1"
  shift
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  find "$root" \
    \( -path "$root/cluster_export*" -o -path "$root/latest_results_*" -o -name '*.tar.gz' \) -prune \
    -o -type f "$@" -mtime -"$SINCE_DAYS" -print 2>/dev/null | while IFS= read -r path; do
      add_path "$path"
    done
}

add_latest_epoch() {
  local run_dir="$1"
  local checkpoint_dir="$run_dir/checkpoints"
  if [[ "$INCLUDE_LATEST_EPOCH" != "1" || ! -d "$checkpoint_dir" ]]; then
    return 0
  fi
  local latest_epoch
  latest_epoch="$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'epoch_*.pth' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
  if [[ -n "$latest_epoch" ]]; then
    add_path "$latest_epoch"
  fi
}

add_run_summary() {
  local run_dir="$1"
  add_path "$run_dir/run_config.json"
  add_path "$run_dir/training.log"
  add_path "$run_dir/best_val_recon.pth"
  add_path "$run_dir/best_val_fitness.pth"
  add_path "$run_dir/fitness_head.pth"
  find "$run_dir" -maxdepth 1 -type f -name '*.pth' -print 2>/dev/null | while IFS= read -r path; do
    add_path "$path"
  done
  add_latest_epoch "$run_dir"
}

add_recent_runs_under() {
  local root="$1"
  if [[ ! -d "$root" ]]; then
    return 0
  fi
  find "$root" -type f -name 'run_config.json' -mtime -"$SINCE_DAYS" -printf '%T@ %h\n' 2>/dev/null \
    | sort -rn \
    | head -n "$LATEST_RUNS_PER_ROOT" \
    | cut -d' ' -f2- \
    | while IFS= read -r run_dir; do
        add_run_summary "$run_dir"
      done
}

write_command_snapshot() {
  local target="$EXPORT_DIR/commands.txt"
  {
    printf 'Export created: %s\n' "$(date --iso-8601=seconds)"
    printf 'Repo root: %s\n' "$REPO_ROOT"
    printf 'Since days: %s\n' "$SINCE_DAYS"
    printf 'Latest runs per root: %s\n' "$LATEST_RUNS_PER_ROOT"
    printf 'Include latest epoch: %s\n' "$INCLUDE_LATEST_EPOCH"
    printf '\nSuggested import command after download:\n'
    printf '  tar -xzf %s\n' "$(basename "$ARCHIVE")"
  } > "$target"
  add_path "$target"
}

write_git_snapshot() {
  local target="$EXPORT_DIR/git_snapshot.txt"
  {
    printf 'git rev-parse HEAD:\n'
    git -C "$REPO_ROOT" rev-parse HEAD 2>&1 || true
    printf '\n\ngit status --short:\n'
    git -C "$REPO_ROOT" status --short 2>&1 || true
    printf '\n\ngit diff --stat:\n'
    git -C "$REPO_ROOT" diff --stat 2>&1 || true
  } > "$target"
  add_path "$target"
}

write_slurm_snapshot() {
  local target="$EXPORT_DIR/slurm_accounting.txt"
  local start_date
  start_date="$(date -d "$SINCE_DAYS days ago" +%F 2>/dev/null || date +%F)"
  {
    printf 'sacct from %s:\n' "$start_date"
    sacct -S "$start_date" --format=JobID,JobName%45,State,ExitCode,Elapsed,MaxRSS,ReqMem,Nodelist%30 2>&1 || true
  } > "$target"
  add_path "$target"
}

# Recent result tables/details/plots written by benchmark and search scripts.
add_recent_export_files_recursive "$EXPORTS_DIR" \
  \( -name '*.csv' -o -name '*.json' -o -name '*.png' -o -name '*.log' \)

# Slurm logs are usually emitted in the repository root, but older exports may also
# contain copied logs directly under exports.
add_recent_files "$REPO_ROOT" \( -name 'slurm-*.out' \)
add_recent_files "$EXPORTS_DIR" \( -name 'slurm-*.out' \)

# Recent AE/surrogate runs. Only compact run summaries are exported by default:
# config, logs, best/final checkpoints, and the latest epoch checkpoint.
add_recent_runs_under "$REPO_ROOT/models/f1_guided"
add_recent_runs_under "$REPO_ROOT/models/f1_surrogate"

# Include exact scripts needed to reproduce the latest batch.
add_path "$REPO_ROOT/scripts/train_fitness_guided_vae.py"
add_path "$REPO_ROOT/scripts/optimize_selected_latents.py"
add_path "$REPO_ROOT/scripts/benchmark_selected_latents.py"
add_path "$REPO_ROOT/scripts/surrogate_guided_mutation_search.py"
add_path "$REPO_ROOT/scripts/grammar_cem_search.py"
add_path "$REPO_ROOT/scripts/export_latest_results.sh"
add_path "$REPO_ROOT/scripts/slurm_train_conditional_lhs_elite1p2_seed43_a100.sh"
add_path "$REPO_ROOT/scripts/slurm_train_fitness_aware_lhs_oversample1p5_seed43_a100.sh"
add_path "$REPO_ROOT/scripts/slurm_train_struct_conditional_lhs_elite1p2_seed44_a100.sh"
add_path "$REPO_ROOT/scripts/slurm_latent_evolution_selected_generators_cap1p2_a100.sh"
add_path "$REPO_ROOT/scripts/slurm_latent_evolution_selected_generators_slp_cap1p2_a100.sh"
add_path "$REPO_ROOT/scripts/slurm_latent_evolution_new_neural_slp_cap1p2_a100.sh"

write_command_snapshot
write_git_snapshot
write_slurm_snapshot

sort -u "$MANIFEST" -o "$MANIFEST"
add_path "$MANIFEST"
sort -u "$MANIFEST" -o "$MANIFEST"

if [[ ! -s "$MANIFEST" ]]; then
  echo "No exportable files found." >&2
  exit 1
fi

tar -czf "$ARCHIVE" -C "$REPO_ROOT" -T "$MANIFEST"

printf 'Latest results export ready\n'
printf -- '- Folder : %s\n' "$EXPORT_DIR"
printf -- '- Archive: %s\n' "$ARCHIVE"
printf -- '- Manifest: %s\n' "$MANIFEST"
printf -- '- Since  : last %s days\n' "$SINCE_DAYS"
printf -- '- Files  : %s\n' "$(wc -l < "$MANIFEST")"
