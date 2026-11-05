#!/usr/bin/env bash
#
# Submit a batch of Slurm jobs for MC production / analysis.
#
# Examples:
#   ./submit.sh --process tt  --mode gen+ana --range 1:100
#   ./submit.sh --process tth --mode gen+ana --range 1:600 --wave-size 10
#   ./submit.sh --process tt  --mode ana     --range 1:40  --labels 0
#   ./submit.sh --process tth --mode gen     --range 1:20  --labels "0 45 -45"
#
# Modes:
#   gen      -> sbatch worker_generate.sh
#   ana      -> sbatch worker_analyse.sh
#   gen+ana  -> sbatch worker_gen_ana.sh
#
# Labels:
#   tt:  defaults to "0 L R LR RL"
#   tth: defaults to "0 45 -45"
#   tt_ctg / tt_ctg_lin: required, one or more "re,im" ctG points
#                        (e.g. --labels "0.5,0.3 1.0,0.0"). tt_ctg keeps the
#                        ctG^2 term, tt_ctg_lin truncates at linear order.
#
# Wave dependency:
#   --wave-size N submits in waves of N nruns; each wave is afterany-dependent
#   on the previous wave. Useful for large submissions that must serialize on
#   shared MG state, or that would exhaust the scratch quota if every nrun ran
#   at once.
#
# --skip-missing (ana only): skip combinations where the input root/lhe file
# does not exist (preserves behavior of the legacy submit_ana_ttH.sh).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Exported so worker scripts can find _env.sh / _lib.sh — slurm copies the
# worker out of data/slurm/ into a spool dir, breaking ${BASH_SOURCE[0]} self-resolution.
export SLURM_DIR="${SCRIPT_DIR}"
# shellcheck source=_env.sh
source "${SCRIPT_DIR}/_env.sh"

usage() {
    sed -n '/^# /,/^$/p' "$0" | head -40
    exit "${1:-2}"
}

process=""
mode=""
range=""
labels=""
semileps="1 2"
wave_size=0
skip_missing=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --process)      process="$2"; shift 2 ;;
        --mode)         mode="$2"; shift 2 ;;
        --range)        range="$2"; shift 2 ;;
        --labels)       labels="$2"; shift 2 ;;
        --semileps)     semileps="$2"; shift 2 ;;
        --wave-size)    wave_size="$2"; shift 2 ;;
        --skip-missing) skip_missing=1; shift ;;
        -h|--help)      usage 0 ;;
        *)              echo "Unknown arg: $1" >&2; usage ;;
    esac
done

[[ -z "${process}" ]] && { echo "--process required" >&2; usage; }
[[ -z "${mode}"    ]] && { echo "--mode required" >&2; usage; }
[[ -z "${range}"   ]] && { echo "--range required (e.g. 1:100)" >&2; usage; }

case "${mode}" in
    gen)     worker="${SCRIPT_DIR}/worker_generate.sh" ;;
    ana)     worker="${SCRIPT_DIR}/worker_analyse.sh" ;;
    gen+ana) worker="${SCRIPT_DIR}/worker_gen_ana.sh" ;;
    *) echo "Unknown mode: ${mode}" >&2; usage ;;
esac

if [[ -z "${labels}" ]]; then
    case "${process}" in
        tt)     labels="0 L R LR RL" ;;
        tth)    labels="0 45 -45" ;;
        tt_ctg|tt_ctg_lin)
            echo "--labels is required for ${process} (e.g. '0.5,0.3 1.0,0.0')" >&2; exit 2 ;;
        *)      echo "Unknown process: ${process}" >&2; exit 2 ;;
    esac
fi

nrun_start="${range%%:*}"
nrun_end="${range##*:}"
if ! [[ "${nrun_start}" =~ ^[0-9]+$ && "${nrun_end}" =~ ^[0-9]+$ ]]; then
    echo "Invalid --range '${range}', expected A:B with integers." >&2
    exit 2
fi

raw_dir="$(raw_dir_for "${process}")"

# Wave-dependency state
wave_ids=()
prev_wave_dep=""
wave_count=0

sbatch_one() {
    local jid
    if [[ "${wave_size}" -gt 0 && -n "${prev_wave_dep}" ]]; then
        # afterany, not afterok: the wave chain exists to cap how many jobs run
        # concurrently, not to gate on success. With afterok a single failure
        # anywhere in a wave leaves every later wave DependencyNeverSatisfied.
        jid=$(sbatch --dependency="afterany:${prev_wave_dep}" "${worker}" "$@" | awk '{print $4}')
    else
        jid=$(sbatch "${worker}" "$@" | awk '{print $4}')
    fi
    # if-block, not `[[ ... ]] && ...`: as the last statement of the function the
    # && form returns 1 when wave_size is 0, which trips `set -e` in the caller.
    if [[ "${wave_size}" -gt 0 ]]; then
        wave_ids+=("${jid}")
    fi
}

flush_wave() {
    if [[ "${wave_size}" -gt 0 && ${#wave_ids[@]} -gt 0 ]]; then
        prev_wave_dep="$(IFS=:; echo "${wave_ids[*]}")"
        wave_ids=()
        wave_count=0
    fi
}

for ((nrun=nrun_start; nrun<=nrun_end; nrun++)); do
    for label in ${labels}; do
        for semilep in ${semileps}; do
            if [[ "${mode}" == "ana" && "${skip_missing}" -eq 1 ]]; then
                tag="$(tag_for "${process}" "${label}" "${semilep}" "${nrun}")"
                if [[ ! -f "${raw_dir}/${tag}.root" || ! -f "${raw_dir}/${tag}.lhe.gz" ]]; then
                    echo "Skip ${tag}: missing root/lhe."
                    continue
                fi
            fi
            echo "Submit: ${process} nrun=${nrun} label=${label} semilep=${semilep}"
            sbatch_one "${process}" "${nrun}" "${label}" "${semilep}"
        done
    done
    if [[ "${wave_size}" -gt 0 ]]; then
        # Use assignment form: ((wave_count++)) returns exit 1 when wave_count
        # was 0 (post-increment yields the old value), which trips `set -e`.
        wave_count=$((wave_count + 1))
        if [[ ${wave_count} -eq ${wave_size} ]]; then
            flush_wave
        fi
    fi
done

flush_wave
