# Shared helpers for generation and analysis stages.
# Source _env.sh first.

# Build a SMEFTsim param card with a complex ctG = ctGRe + i*ctGIm.
# Takes the model's default massless param card as a base, substitutes the
# ctGRe line (Block SMEFT, lhacode 15), and appends a Block SMEFTcpv with
# ctGIm (lhacode 8) so MadGraph picks up the CP-odd part.
prepare_smeft_param_card() {
    local re="$1" im="$2" out="$3"
    local src="${MODELS_DIR}/SMEFTsim_topU3l_alphaScheme_UFO/param_card_massless.dat"
    if [[ ! -f "${src}" ]]; then
        echo "SMEFTsim param card not found at ${src}" >&2
        return 2
    fi
    sed -E "s|^([[:space:]]+15[[:space:]]+)[0-9eE.+-]+([[:space:]]+#[[:space:]]*ctGRe.*)\$|\1${re}\2|" \
        "${src}" > "${out}"
    cat >> "${out}" <<EOF

Block SMEFTcpv
    8 ${im} # ctGIm
EOF
}

# Put the real process line back into a tt_ctg_lin LHE banner.
# emit_mg_cmds rewrites "NP^2<=1" -> "NP<=1" in Cards/proc_card_mg5.dat before
# launch, because MadSpin rebuilds its decay matrix elements from that line as a
# decay chain and MG5 rejects squared-order constraints there. MadSpin has run by
# the time this is called, so the shipped banner can state what was really
# generated instead of the form MadSpin needed to see.
restore_linear_proc_card() {
    # Separate declarations: bash expands the whole `local` command line before
    # running the builtin, so "tmp=${lhe}..." on the same line trips `set -u`.
    local lhe="$1"
    local tmp="${lhe}.tmp.$$"
    zcat "${lhe}" \
        | sed 's|^generate p p > t t~ NP<=1[[:space:]]*$|generate p p > t t~ NP^2<=1|' \
        | gzip > "${tmp}"
    # Subshell without pipefail: `head` closing the pipe early is expected here.
    if ( set +o pipefail; zcat "${tmp}" | head -400 | grep -q '^generate p p > t t~ NP\^2<=1$' ); then
        mv "${tmp}" "${lhe}"
        echo "LHE banner: restored 'generate p p > t t~ NP^2<=1'."
    else
        rm -f "${tmp}"
        echo "Warning: could not restore NP^2<=1 in $(basename "${lhe}"); its banner still reads NP<=1." >&2
    fi
}

# Count negative-weight events in an unweighted LHE and print the fraction.
# xwgtup is the third field of the line following each <event> tag.
report_negative_weights() {
    local lhe="$1"
    zcat "${lhe}" | awk -v f="$(basename "${lhe}")" '
        /^[[:space:]]*<event>/ { getline; t++; if ($3 + 0 < 0) n++ }
        END {
            if (t == 0) { printf "%s: no events found.\n", f; exit }
            printf "%s: %d/%d events (%.3f%%) carry a negative weight.\n", f, n, t, 100.0 * n / t
            if (n > 0) {
                print "  These come from the linear (NP^2<=1) truncation going negative where"
                print "  the ctG interference overwhelms the SM term. The parquet pipeline does"
                print "  not read weights, so it counts every event as +1."
            }
        }' || echo "Could not scan ${lhe} for negative weights." >&2
}

run_generation() {
    # Run MadGraph for one (process, label, semilep, nrun) combination and
    # move the resulting .root / .lhe.gz into the per-process raw_dir.
    # On success: exports ROOT_FILE and LHE_FILE.
    local process="$1" nrun="$2" label="$3" semilep="$4"
    local tag raw_dir scratch script_folder mg_cmds proc_folder seed run_dir

    tag="$(tag_for "${process}" "${label}" "${semilep}" "${nrun}")"
    raw_dir="$(raw_dir_for "${process}")"
    mkdir -p "${raw_dir}"

    seed="${SLURM_JOB_ID:-$$}"
    # Default scratch sits next to the raw outputs so it inherits the same
    # (typically fast /work) NFS mount. SCRATCH_BASE overrides.
    : "${SCRATCH_BASE:=${raw_dir}/MG_proc_folders}"
    scratch="${SCRATCH_BASE}/${seed}"
    mkdir -p "${scratch}/MG_procs" "${scratch}/scripts"

    script_folder="${scratch}/scripts"
    mg_cmds="${script_folder}/${tag}.txt"
    proc_folder="${scratch}/MG_procs/temp_${label}_${semilep}_${nrun}"

    emit_mg_cmds "${process}" "${label}" "${semilep}" "${proc_folder}" "${seed}" > "${mg_cmds}"

    echo "MadGraph command file:"
    cat "${mg_cmds}"

    # MG reads `cards/...` paths relative to its working dir.
    cp -r "${CARDS_DIR}" "${scratch}/cards"

    # The tt_ctg* processes build their param card per-job from (Re, Im) in the label.
    if [[ "${process}" == tt_ctg* ]]; then
        local re="${label%%,*}" im="${label##*,}"
        prepare_smeft_param_card "${re}" "${im}" "${scratch}/cards/paramcard_smeft_ctg.dat"
        echo "Wrote SMEFT param card with ctGRe=${re} ctGIm=${im}."
    fi

    pushd "${scratch}" > /dev/null
    "${MG_BIN}" "${mg_cmds}"
    popd > /dev/null

    ROOT_FILE="${raw_dir}/${tag}.root"
    LHE_FILE="${raw_dir}/${tag}.lhe.gz"

    if [[ -e "${ROOT_FILE}" || -e "${LHE_FILE}" ]]; then
        echo "Overwriting existing outputs for ${tag}."
        rm -f "${ROOT_FILE}" "${LHE_FILE}"
    fi

    run_dir="${proc_folder}/Events/run_01_decayed_1"
    if [[ ! -d "${run_dir}" ]]; then
        run_dir="$(find "${proc_folder}/Events" -maxdepth 1 -name 'run_*_decayed_*' | sort | head -n1)"
        echo "run_01_decayed_1 not found, using: ${run_dir}"
    fi

    mv "${run_dir}/tag_1_delphes_events.root" "${ROOT_FILE}"
    mv "${run_dir}/unweighted_events.lhe.gz" "${LHE_FILE}"

    # A linear-truncated |M|^2 is not positive definite, and MadEvent unweights
    # on abs(wgt) while keeping the sign (Template/LO/SubProcesses/unwgt.f).
    # Report the negative fraction rather than letting it pass silently.
    if [[ "${process}" == "tt_ctg_lin" ]]; then
        restore_linear_proc_card "${LHE_FILE}"
        report_negative_weights "${LHE_FILE}"
    fi

    # Caller is responsible for cleanup of ${scratch} via its EXIT trap.
    SCRATCH_PATH="${scratch}"
    export ROOT_FILE LHE_FILE SCRATCH_PATH
}

run_analysis() {
    # Run the parquet writer for one combination. Loops over MASS_WINDOWS.
    local process="$1" nrun="$2" label="$3" semilep="$4"
    local raw_dir parquet_dir mw mlo mhi
    raw_dir="$(raw_dir_for "${process}")"
    parquet_dir="$(parquet_dir_for "${process}")"
    mkdir -p "${parquet_dir}"

    for mw in ${MASS_WINDOWS}; do
        mlo="${mw%%:*}"
        mhi="${mw##*:}"
        python -m data.pipeline \
            --process "${process}" \
            --label "${label}" \
            --nrun "${nrun}" \
            --semilep "${semilep}" \
            --top-mass "${mlo}" "${mhi}" \
            --input-dir "${raw_dir}" \
            --output-dir "${parquet_dir}" \
            --train
    done
}
