#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=4:00:00
#SBATCH --error=logs/pbs/%j.log
#SBATCH --output=logs/pbs/%j.log
#SBATCH --mail-type=NONE

# Generate + analyse for one (process, nrun, label, semilep) combination.
# Removes the intermediate root/lhe files after analysis succeeds.
#
# Usage: sbatch worker_gen_ana.sh <process> <nrun> <label> <semilep>

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
conda activate "${CONDA_ENV_GEN}"
set -u
# Drop any build-toolchain pollution inherited from the submitting shell
# (something in the user's shell is exporting these as "$HOSTNAME"). Belt
# AND suspenders: sweep any remaining env var whose value equals the
# current hostname, since we don't know what else got mis-set.
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

produced_outputs=0
analysis_succeeded=0
terminate_children() { pkill -P $$ 2>/dev/null || true; }
cleanup() {
    local rc="${1:-1}"
    # Keep the .root/.lhe.gz when analysis crashed — they're what you need to
    # debug. Only purge them once analysis completes successfully.
    if [[ ${produced_outputs} -eq 1 && ${analysis_succeeded} -eq 1 \
          && ${rc} -eq 0 && -n "${ROOT_FILE:-}" ]]; then
        rm -f "${ROOT_FILE}" "${LHE_FILE}"
    fi
    [[ -n "${SCRATCH_PATH:-}" ]] && rm -rf "${SCRATCH_PATH}"
}
trap 'rc=$?; trap - EXIT; terminate_children; cleanup "$rc"; exit $rc' EXIT

run_generation "${process}" "${nrun}" "${label}" "${semilep}"
produced_outputs=1
echo "Generated ${ROOT_FILE}; starting analysis."

# conda activate references unset vars; relax `-u` briefly.
set +u
conda activate "${CONDA_ENV_ANA}"
set -u

cd "${REPO_ROOT}"
run_analysis "${process}" "${nrun}" "${label}" "${semilep}"
analysis_succeeded=1
