#!/usr/bin/env bash

# loop over nrun values
for nrun in {11..20}; do
    # loop over alpha and semilep values
    # for alpha in 0; do
    for alpha in 0 45 -45; do
        for semilep in 1 2; do
            sbatch ttH_gen+ana.sh ${nrun} ${alpha} ${semilep}
        done
    done
done
