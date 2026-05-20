#!/bin/bash

set -euo pipefail

REPO_ROOT="${1:-$HOME/master_thesis}"
EXPORTS_DIR="$REPO_ROOT/exports"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXPORT_DIR="$EXPORTS_DIR/f1_selected_10_ckpts_$STAMP"
ARCHIVE="$EXPORT_DIR.tar.gz"

mkdir -p "$EXPORT_DIR"

manifest="$EXPORT_DIR/manifest.txt"
: > "$manifest"

missing=()

copy_run() {
  local label="$1"
  local run_dir="$2"
  local checkpoint="$3"
  local target_dir="$EXPORT_DIR/$label"
  local checkpoint_path="$run_dir/$checkpoint"

  mkdir -p "$target_dir"

  if [[ ! -f "$checkpoint_path" ]]; then
    missing+=("$checkpoint_path")
    return 0
  fi

  mkdir -p "$target_dir/$(dirname "$checkpoint")"
  cp -p "$checkpoint_path" "$target_dir/$checkpoint"

  if [[ -f "$run_dir/run_config.json" ]]; then
    cp -p "$run_dir/run_config.json" "$target_dir/run_config.json"
  fi
  if [[ -f "$run_dir/training.log" ]]; then
    cp -p "$run_dir/training.log" "$target_dir/training.log"
  fi

  printf '%s\n' "$run_dir" > "$target_dir/source_run.txt"
  printf '%s/%s <- %s\n' "$label" "$checkpoint" "$checkpoint_path" >> "$manifest"
}

copy_run \
  "01_lhs_standard_null_best" \
  "$REPO_ROOT/models/f1_sweep_lhs/standard_vae/tree_vae_masked_lhs/2026-05-09_00-46-21" \
  "best_val_recon.pth"

copy_run \
  "02_lhs_latent256_seed42_best" \
  "$REPO_ROOT/models/f1_sweep_lhs/seed42_latent256_1200/tree_vae_masked_lhs/2026-05-12_23-03-46" \
  "best_val_recon.pth"

copy_run \
  "03_masked_standard_null_best" \
  "$REPO_ROOT/models/f1_sweep/standard_vae/tree_vae_masked/2026-05-05_23-15-42" \
  "best_val_recon.pth"

copy_run \
  "04_lhs_standard_seed42_best" \
  "$REPO_ROOT/models/f1_sweep_lhs/seed42_standard_1200/tree_vae_masked_lhs/2026-05-12_23-03-46" \
  "best_val_recon.pth"

copy_run \
  "05_masked_slow_null_best" \
  "$REPO_ROOT/models/f1_sweep/slow_vae/tree_vae_masked/2026-05-05_23-15-42" \
  "best_val_recon.pth"

copy_run \
  "06_lhs_depth_seed42_best" \
  "$REPO_ROOT/models/f1_sweep_lhs/seed42_depth_1200/tree_vae_masked_lhs_depth/2026-05-12_23-03-46" \
  "best_val_recon.pth"

copy_run \
  "07_lhs_slow_null_best" \
  "$REPO_ROOT/models/f1_sweep_lhs/slow_vae/tree_vae_masked_lhs/2026-05-09_00-46-21" \
  "best_val_recon.pth"

copy_run \
  "08_masked_reconstruction_ceiling_best" \
  "$REPO_ROOT/models/f1_sweep/reconstruction_ceiling/tree_vae_masked/2026-05-05_23-15-42" \
  "best_val_recon.pth"

copy_run \
  "09_tree_vae_epoch80" \
  "$REPO_ROOT/models/f1_val/tree_vae/2026-05-02_18-43-27" \
  "checkpoints/epoch_80.pth"

copy_run \
  "10_transformer_vae_epoch90" \
  "$REPO_ROOT/models/f1_val/transformer_vae/2026-05-02_18-43-22" \
  "checkpoints/epoch_90.pth"

if (( ${#missing[@]} > 0 )); then
  printf '\nMissing checkpoint files:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  printf '\nExport folder was left at: %s\n' "$EXPORT_DIR" >&2
  exit 1
fi

tar -czf "$ARCHIVE" -C "$EXPORTS_DIR" "$(basename "$EXPORT_DIR")"

printf '\nExport ready\n'
printf -- '- Folder : %s\n' "$EXPORT_DIR"
printf -- '- Archive: %s\n' "$ARCHIVE"
printf -- '- Contents:\n'
find "$EXPORT_DIR" -type f | sort
