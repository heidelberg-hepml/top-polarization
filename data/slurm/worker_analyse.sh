#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --error=logs/pbs/%j.log
#SBATCH --output=logs/pbs/%j.log
#SBATCH --mail-type=NONE

# Run the parquet analyzer for one (process, nrun, label, semilep) combination.
# Iterates over MASS_WINDOWS (default: 165:190 and 120:220).
#
# Usage: sbatch worker_analyse.sh <process> <nrun> <label> <semilep>

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <process> <nrun> <label> <semilep>" >&2
    exit 2
fi

: "${SLURM_DIR:=$(dirname "${BASH_SOURCE[0]}")}"
source "${SLURM_DIR}/_env.sh"
source "${SLURM_DIR}/_lib.sh"

# conda activate/deactivate hooks reference unset vars (CONDA_BACKUP_*).
# shellcheck disable=SC1091
set +u
source "${CONDA_SETUP}"
conda activate "${CONDA_ENV_ANA}"
set -u

process="$1"
nrun="$2"
label="$3"
semilep="$4"

cd "${REPO_ROOT}"
run_analysis "${process}" "${nrun}" "${label}" "${semilep}"
