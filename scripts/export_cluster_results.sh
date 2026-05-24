#!/bin/bash

set -euo pipefail

REPO_ROOT="${1:-$HOME/master_thesis}"
EXPORTS_DIR="$REPO_ROOT/exports"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXPORT_DIR="$EXPORTS_DIR/cluster_export_$STAMP"
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

add_glob() {
  local pattern="$1"
  shopt -s nullglob
  local matches=( $pattern )
  shopt -u nullglob
  for path in "${matches[@]}"; do
    add_path "$path"
  done
}

add_path "$REPO_ROOT/models/f1_guided/fitness_aware_lhs_seed42"
add_path "$REPO_ROOT/models/f1_guided/conditional_lhs_seed42"
add_path "$REPO_ROOT/models/f1_guided/struct_conditional_lhs_seed42"
add_path "$REPO_ROOT/models/f1_surrogate/structural_ensemble_full"

add_glob "$REPO_ROOT/exports/surrogate_guided_mutation_cap1p2_*"
add_glob "$REPO_ROOT/exports/grammar_cem_cap1p2_*"
add_glob "$REPO_ROOT/slurm-*.out"

sort -u "$MANIFEST" -o "$MANIFEST"

if [[ ! -s "$MANIFEST" ]]; then
  echo "No exportable files found." >&2
  echo "Checked models/f1_guided, models/f1_surrogate, exports/*cap1p2*, and slurm logs." >&2
  exit 1
fi

tar -czf "$ARCHIVE" -C "$REPO_ROOT" -T "$MANIFEST"

printf 'Export ready\n'
printf -- '- Folder : %s\n' "$EXPORT_DIR"
printf -- '- Archive: %s\n' "$ARCHIVE"
printf -- '- Manifest: %s\n' "$MANIFEST"
printf -- '- Contents:\n'
sed 's/^/  /' "$MANIFEST"
