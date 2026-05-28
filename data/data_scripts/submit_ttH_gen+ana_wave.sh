#!/usr/bin/env bash

# ---- USER SETTINGS ----------------------------------------------------------

nrun_start=201          # first nrun value
nrun_end=600           # last nrun value (inclusive)

nruns_per_wave=10      # how many nrun values per wave
                       # each nrun → 3 alpha × 2 semilep = 6 jobs
                       # so 10 nrun → 60 jobs per wave

# ---- INTERNAL STATE ---------------------------------------------------------

ids=()                 # array to store job IDs of the current wave
dep=""                 # dependency string for the NEXT wave
count=0                # how many nrun values we've submitted in this wave
nrun=$nrun_start       # initialize nrun counter

# ---- FUNCTION: submit a single job -----------------------------------------

submit_job() {
    local jid

    # If this is the first wave (no dependency yet),
    # submit jobs normally.
    if [ -z "$dep" ]; then
        jid=$(sbatch ttH_gen+ana.sh "$@" | awk '{print $4}')
    else
        # Otherwise, make this job depend on ALL jobs from the previous wave.
        # "afterok" → only run if all previous jobs finished successfully.
        jid=$(sbatch --dependency=afterok:$dep ttH_gen+ana.sh "$@" | awk '{print $4}')
    fi

    # Store job ID so we can build the dependency for the next wave
    ids+=($jid)
}

# ---- FUNCTION: finalize a wave ---------------------------------------------

flush_wave() {
    # Only do something if we actually collected job IDs
    if [ ${#ids[@]} -gt 0 ]; then
        # Join all job IDs with ":" → required format for Slurm dependencies
        # Example: "123:124:125:..."
        dep=$(IFS=:; echo "${ids[*]}")

        # Reset for the next wave
        ids=()
        count=0
    fi
}

# ---- MAIN LOOP --------------------------------------------------------------

# Loop over nrun values manually (instead of {..}) so we can control incrementing
while [ $nrun -le $nrun_end ]; do

    # For each nrun, submit all parameter combinations
    # This produces 6 jobs per nrun:
    #   alpha ∈ {0, 45, -45}
    #   semilep ∈ {1, 2}
    for alpha in 0 45 -45; do
        for semilep in 1 2; do
            submit_job ${nrun} ${alpha} ${semilep}
        done
    done

    # We finished one "nrun block" (6 jobs)
    ((count++))

    # Increment nrun explicitly (instead of relying on bash brace expansion)
    ((nrun++))

    # If we've reached the configured wave size:
    # → lock in dependencies so the NEXT wave waits for THIS one
    if [ $count -eq $nruns_per_wave ]; then
        flush_wave
    fi
done

# After the loop, flush any remaining jobs (last partial wave)
flush_wave