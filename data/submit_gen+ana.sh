#!/usr/bin/env bash

# loop over nrun values
for nrun in {1..100}; do
    # loop over pol and semilep values
    # for pol in 0 L R LR RL; do
    for pol in 0; do
        for semilep in 1 2; do
            sbatch top_polarization_gen+ana.sh ${nrun} ${pol} ${semilep}
        done
    done
done
