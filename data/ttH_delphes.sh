#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=4:00:00
#SBATCH --error=logs/pbs/%j.log
#SBATCH --output=logs/pbs/%j.log
#SBATCH --mail-type=NONE

# source ~/.zshrc
# conda activate base
source ~/codes/miniconda3/etc/profile.d/conda.sh
conda activate base

# positional arguments
nrun="$1"
alpha="$2"
semilep="$3"

RANDOM=$$

scratch=/tmp/$USER/$SLURM_JOB_ID
mkdir -p "$scratch"

script_folder="${scratch}/scripts"
mkdir -p ${script_folder}
mg_cmds="${script_folder}/ttH_alpha_${alpha}_semilep_${semilep}_nrun_${nrun}.txt"

proc_folder="${scratch}/MG_procs/temp_${alpha}_${semilep}_${nrun}"
mkdir -p "${scratch}/MG_procs"

data_folder="/home/hd/hd_ve298/projects/top_polarization/top-polarization/data/samples_ttH"

echo "import model HC_NLO_X0_UFO" > ${mg_cmds}
echo "generate p p > t t~ x0" >> ${mg_cmds}

echo "output ${proc_folder}" >> ${mg_cmds}
echo "launch" >> ${mg_cmds}
echo "analysis=OFF" >> ${mg_cmds}
echo "madspin=ON" >> ${mg_cmds}
echo "detector=Delphes" >> ${mg_cmds}
echo "shower=Pythia8" >> ${mg_cmds}
echo "done" >> ${mg_cmds}
echo "cards/run_card_ttH.dat" >> ${mg_cmds}
echo "cards/paramcard_ttH_alpha_${alpha}.dat" >> ${mg_cmds}
echo "cards/madspin_card_ttH_semilep_${semilep}.dat" >> ${mg_cmds}
echo "cards/delphes_card_R04.dat" >> ${mg_cmds}
echo "set HEPMCoutput:file=hepmcremove" >> ${mg_cmds}
echo set iseed ${RANDOM} >> ${mg_cmds}

echo "Running MG5 with the following commands:"
cat ${mg_cmds}
./MG/bin/mg5_aMC ${mg_cmds}

#clean up
mv ${proc_folder}/Events/run_01_decayed_1/tag_1_delphes_events.root ${data_folder}/ttH_alpha_${alpha}_semilep_${semilep}_nrun_${nrun}.root
mv ${proc_folder}/Events/run_01_decayed_1/unweighted_events.lhe.gz ${data_folder}/ttH_alpha_${alpha}_semilep_${semilep}_nrun_${nrun}.lhe.gz
rm -r ${proc_folder}
rm ${mg_cmds}