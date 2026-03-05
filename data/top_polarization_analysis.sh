#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --error=logs/pbs/%j.log
#SBATCH --output=logs/pbs/%j.log
#SBATCH --mail-type=END,FAIL

# source ~/.zshrc
# conda activate base
source ~/codes/miniconda3/etc/profile.d/conda.sh
conda activate rootpy

# positional arguments
nrun="$1"
pol="$2"
semilep="$3"

./analyse_root_files.py --pol=${pol} --nrun=${nrun} --semilep=${semilep} --train_data --top_mass_lower=120 --top_mass_upper=220

hi im stupid