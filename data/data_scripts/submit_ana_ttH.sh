#!/usr/bin/env bash

# loop over nrun values
for nrun in {1..100}; do
    # loop over alpha and semilep values
    for alpha in 0 45 -45; do
        for semilep in 1 2; do
            # test if root and lhe files exist for the given nrun, alpha, and semilep values
            root_file="samples_ttH/ttH_alpha_${alpha}_semilep_${semilep}_nrun_${nrun}.root"
            lhe_file="samples_ttH/ttH_alpha_${alpha}_semilep_${semilep}_nrun_${nrun}.lhe.gz"
            if [[ -f "$root_file" && -f "$lhe_file" ]]; then
                echo "Submitting job for nrun=${nrun}, alpha=${alpha}, semilep=${semilep}"
                sbatch ttH_analysis.sh ${nrun} ${alpha} ${semilep}
            else
                echo "Skipping nrun=${nrun}, alpha=${alpha}, semilep=${semilep} due to missing files"
            fi
        done
    done
done
