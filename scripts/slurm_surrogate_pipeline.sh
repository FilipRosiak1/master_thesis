#!/bin/bash

set -euo pipefail

train_job="$(sbatch --parsable scripts/slurm_train_structural_surrogate.sh)"
echo "Submitted structural surrogate training: ${train_job}"

search_job="$(sbatch --parsable --dependency=afterok:${train_job} scripts/slurm_surrogate_guided_mutation_search.sh)"
echo "Submitted surrogate-guided mutation search: ${search_job}"
echo "Search will start after surrogate training finishes successfully."
