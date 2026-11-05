# Shared environment for data/slurm workers and submitters.
# Source this file with `source "$(dirname "$0")/_env.sh"`.
#
# Override any of the variables below by exporting them before sbatch.

_env_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_DIR="$(dirname "${_env_script_dir}")"
export REPO_ROOT="$(dirname "${DATA_DIR}")"

: "${CARDS_DIR:=${DATA_DIR}/cards}"
: "${MODELS_DIR:=${DATA_DIR}/models}"
: "${MG_BIN:=${DATA_DIR}/MG/bin/mg5_aMC}"
: "${DELPHES_LIB:=$(dirname "$(dirname "${MG_BIN}")")/Delphes/libDelphes.so}"
: "${CONDA_SETUP:=${HOME}/miniconda3/etc/profile.d/conda.sh}"
: "${CONDA_ENV_GEN:=base}"
: "${CONDA_ENV_ANA:=rootpy}"
# If unset, scratch defaults to a per-process MG_proc_folders/ inside the raw
# sample dir; that path was on the previously working /work mount via the
# samples/ symlink. /home is often slow NFS and MG's parallel refine races on
# it ("input_sg.txt: No such file or directory" / missing results.dat).
# Override by exporting SCRATCH_BASE — workers use ${SCRATCH_BASE}/${SLURM_JOB_ID}.
: "${MASS_WINDOWS:=165:190 120:220}"
export CARDS_DIR MODELS_DIR MG_BIN DELPHES_LIB CONDA_SETUP CONDA_ENV_GEN CONDA_ENV_ANA MASS_WINDOWS

# Process-specific paths.
raw_dir_for() {
    case "$1" in
        tt)     echo "${DATA_DIR}/samples" ;;
        tth)    echo "${DATA_DIR}/samples_ttH" ;;
        tt_ctg) echo "${DATA_DIR}/samples_tt_ctg" ;;
        tt_ctg_lin) echo "${DATA_DIR}/samples_tt_ctg_lin" ;;
        *)      echo "Unknown process: $1" >&2; return 2 ;;
    esac
}

parquet_dir_for() {
    case "$1" in
        tt)     echo "${DATA_DIR}/samples_awk" ;;
        tth)    echo "${DATA_DIR}/samples_ttH_awk" ;;
        tt_ctg) echo "${DATA_DIR}/samples_tt_ctg_awk" ;;
        tt_ctg_lin) echo "${DATA_DIR}/samples_tt_ctg_lin_awk" ;;
        *)      echo "Unknown process: $1" >&2; return 2 ;;
    esac
}

combined_dir_for() {
    case "$1" in
        tt)     echo "${DATA_DIR}/samples_awk_comb" ;;
        tth)    echo "${DATA_DIR}/samples_ttH_awk_comb" ;;
        tt_ctg) echo "${DATA_DIR}/samples_tt_ctg_awk_comb" ;;
        tt_ctg_lin) echo "${DATA_DIR}/samples_tt_ctg_lin_awk_comb" ;;
        *)      echo "Unknown process: $1" >&2; return 2 ;;
    esac
}

# Replace '.' with 'p' so floats are safe in filenames; preserve leading '-'.
_filesafe() { echo "${1//./p}"; }

tag_for() {
    local process="$1" label="$2" semilep="$3" nrun="$4"
    case "${process}" in
        tt)  echo "top_pol_${label}_semilep_${semilep}_nrun_${nrun}" ;;
        tth) echo "ttH_alpha_${label}_semilep_${semilep}_nrun_${nrun}" ;;
        tt_ctg|tt_ctg_lin)
            local re="${label%%,*}" im="${label##*,}"
            if [[ "${re}" == "${label}" || "${im}" == "${label}" ]]; then
                echo "${process} label must be 're,im' (e.g. 0.5,0.3); got '${label}'" >&2
                return 2
            fi
            echo "${process}_re_$(_filesafe "${re}")_im_$(_filesafe "${im}")_semilep_${semilep}_nrun_${nrun}"
            ;;
        *)   echo "Unknown process: ${process}" >&2; return 2 ;;
    esac
}

# Emit the MadGraph command list for one (process, label, semilep, proc_folder, seed)
# combination on stdout. Caller redirects to a file.
emit_mg_cmds() {
    local process="$1" label="$2" semilep="$3" proc_folder="$4" seed="$5"
    case "${process}" in
        tt)
            case "${label}" in
                0)  echo "generate p p > t t~" ;;
                L)  echo "generate p p > t{L} t~{L}" ;;
                R)  echo "generate p p > t{R} t~{R}" ;;
                LR) echo "generate p p > t{L} t~{R}" ;;
                RL) echo "generate p p > t{R} t~{L}" ;;
                *)  echo "Invalid polarization '${label}'. Expected: 0|L|R|LR|RL" >&2; return 2 ;;
            esac
            ;;
        tth)
            echo "import model ${MODELS_DIR}/HC_NLO_X0_UFO"
            echo "generate p p > t t~ x0"
            ;;
        tt_ctg|tt_ctg_lin)
            # The -ctG_massless restriction zeros every Wilson coefficient except
            # ctGRe/ctGIm, so diagram generation only picks up the chromomagnetic
            # ctG vertex. The complex ctG value comes from the per-job param card
            # written by prepare_smeft_param_card.
            echo "import model ${MODELS_DIR}/SMEFTsim_topU3l_alphaScheme_UFO-ctG_massless"
            if [[ "${process}" == "tt_ctg_lin" ]]; then
                # NP^2<=1 constrains the SQUARED amplitude: it keeps |SM|^2
                # (NP^2=0) and the SM x ctG interference (NP^2=1) but drops the
                # ctG^2 term (NP^2=2), i.e. sigma = sigma_SM + c*sigma_int.
                echo "generate p p > t t~ NP^2<=1"
            else
                # NP<=1 constrains the AMPLITUDE: at most one dim-6 insertion per
                # diagram, so |M|^2 still holds SM + ctG/SM interference + ctG^2.
                echo "generate p p > t t~ NP<=1"
            fi
            ;;
        *)
            echo "Unknown process: ${process}" >&2; return 2 ;;
    esac
    echo "output ${proc_folder}"
    if [[ "${process}" == "tt_ctg_lin" ]]; then
        # MadSpin rebuilds its matrix elements from the process line recorded in
        # Cards/proc_card_mg5.dat, as a DECAY CHAIN:
        #   generate p p > t t~ NP^2<=1, (t > b w+, w+ > j j), (t~ > b~ w-, ...)
        # and MG5 rejects that outright ("Decay processes cannot specify squared
        # orders constraints", madgraph_interface.py). Rewriting the recorded
        # line to the amplitude-order form *after* output lets MadSpin run: the
        # production matrix element is already exported and keeps the truncation
        # (CHOSEN_SO_CONFIGS = .TRUE.,.TRUE.,.FALSE. in SubProcesses/*/matrix*.f),
        # so the generated events stay linear in ctG. Only MadSpin's decay and
        # spin-correlation MEs fall back to the full |M|^2 -- see data/README.md.
        echo "shell sed -i -e 's|NP\^2<=1|NP<=1|g' ${proc_folder}/Cards/proc_card_mg5.dat"
        # The truthful line is put back into the shipped LHE banner after
        # MadSpin has run, by restore_linear_proc_card in _lib.sh. A comment
        # cannot be used instead: MG re-serialises the proc card through its
        # ProcCard parser when writing the banner, which drops comment lines.
    fi
    echo "launch"
    echo "analysis=OFF"
    echo "madspin=ON"
    echo "detector=Delphes"
    echo "shower=Pythia8"
    echo "done"
    case "${process}" in
        tt)
            echo "cards/run_card_top.dat"
            echo "cards/madspin_card_semilep_${semilep}.dat"
            ;;
        tth)
            echo "cards/run_card_ttH.dat"
            echo "cards/paramcard_ttH_alpha_${label}.dat"
            echo "cards/madspin_card_ttH_semilep_${semilep}.dat"
            ;;
        tt_ctg|tt_ctg_lin)
            echo "cards/run_card_top.dat"
            echo "cards/paramcard_smeft_ctg.dat"
            echo "cards/madspin_card_semilep_${semilep}.dat"
            ;;
    esac
    echo "cards/delphes_card_R04.dat"
    echo "set HEPMCoutput:file=hepmcremove"
    echo "set iseed ${seed}"
    # NEVENTS overrides the run-card nevents value — useful for quick
    # pipeline smoke tests (e.g. NEVENTS=200 ./submit.sh ...).
    if [[ -n "${NEVENTS:-}" ]]; then
        echo "set nevents ${NEVENTS}"
    fi
}
