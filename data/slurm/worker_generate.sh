#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=4:00:00
#SBATCH --error=logs/pbs/%j.log
#SBATCH --output=logs/pbs/%j.log
#SBATCH --mail-type=NONE

# Run MadGraph for one (process, nrun, label, semilep) combination.
#
# Usage: sbatch worker_generate.sh <process> <nrun> <label> <semilep>
#   process: tt | tth
#   label:   polarization (0|L|R|LR|RL) for tt, alpha (0|45|-45) for tth

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <process> <nrun> <label> <semilep>" >&2
    exit 2
fi

# Slurm copies the worker to a spool dir, so ${BASH_SOURCE[0]} no longer
# points at data/slurm/. submit.sh exports SLURM_DIR with the real path.
: "${SLURM_DIR:=$(dirname "${BASH_SOURCE[0]}")}"
source "${SLURM_DIR}/_env.sh"
source "${SLURM_DIR}/_lib.sh"

# conda activate/deactivate hooks reference unset vars (CONDA_BACKUP_*).
# shellcheck disable=SC1091
set +u
source "${CONDA_SETUP}"
conda activate "${CONDA_ENV_GEN}"
set -u
unset FC F77 F90 F95 FCFLAGS FFLAGS CC CFLAGS CXX CXXFLAGS CPPFLAGS LDFLAGS \
      AR AS LD NM RANLIB STRIP OBJCOPY OBJDUMP MAKE
_host="$(hostname)"
while IFS= read -r _v; do
    if [[ "${!_v:-}" == "${_host}" ]]; then
        echo "Unsetting polluted env var ${_v}=${_host}"
        unset "${_v}"
    fi
done < <(compgen -e)
unset _host _v

process="$1"
nrun="$2"
label="$3"
semilep="$4"

terminate_children() { pkill -P $$ 2>/dev/null || true; }
cleanup() {
    [[ -n "${SCRATCH_PATH:-}" ]] && rm -rf "${SCRATCH_PATH}"
}
trap 'rc=$?; trap - EXIT; terminate_children; cleanup; exit $rc' EXIT

run_generation "${process}" "${nrun}" "${label}" "${semilep}"
echo "Generated ${ROOT_FILE} and ${LHE_FILE}."
