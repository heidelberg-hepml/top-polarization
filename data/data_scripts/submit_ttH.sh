#!/usr/bin/env bash

# loop over nrun values
for nrun in {1..25}; do
    # loop over alpha and semilep values
    for alpha in 0; do
    # for alpha in 45 -45; do
        for semilep in 1 2; do
            sbatch ttH_delphes.sh ${nrun} ${alpha} ${semilep}
        done
    done
done
