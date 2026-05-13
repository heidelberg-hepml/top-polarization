#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=4:00:00
#SBATCH --error=logs/pbs/%j.log
#SBATCH --output=logs/pbs/%j.log
#SBATCH --mail-type=NONE

set -euo pipefail

# source ~/.zshrc
# conda activate base
source ~/codes/miniconda3/etc/profile.d/conda.sh
conda activate base

# positional arguments
if [[ $# -ne 3 ]]; then
	echo "Usage: $0 <nrun> <alpha> <semilep>"
	exit 2
fi

nrun="$1"
alpha="$2"
semilep="$3"

seed="${SLURM_JOB_ID:-$$}"

# scratch=/tmp/$USER/$SLURM_JOB_ID
scratch=/home/hd/hd_ve298/projects/top_polarization/top-polarization/data/samples_ttH/MG_proc_folders/$SLURM_JOB_ID
mkdir -p "$scratch"

script_folder="${scratch}/scripts"
mkdir -p "${script_folder}"
mg_cmds="${script_folder}/ttH_alpha_${alpha}_semilep_${semilep}_nrun_${nrun}.txt"

proc_folder="${scratch}/MG_procs/temp_${alpha}_${semilep}_${nrun}"
mkdir -p "${scratch}/MG_procs"

main_dir="/home/hd/hd_ve298/projects/top_polarization/top-polarization/data"
data_folder="/home/hd/hd_ve298/projects/top_polarization/top-polarization/data/samples_ttH"
root_file="${data_folder}/ttH_alpha_${alpha}_semilep_${semilep}_nrun_${nrun}.root"
lhe_file="${data_folder}/ttH_alpha_${alpha}_semilep_${semilep}_nrun_${nrun}.lhe.gz"
produced_outputs=0

terminate_children() {
	# Ensure no orphaned children keep the Slurm allocation alive.
	pkill -P $$ 2>/dev/null || true
}

cleanup() {
	echo "Cleaning up temporary files..."
	if [[ ${produced_outputs} -eq 1 ]]; then
		rm -f "${root_file}" "${lhe_file}"
	fi
	rm -f "${mg_cmds}"
	rm -rf "${proc_folder}"
	rm -rf "${scratch}"
	echo "Cleaned up."
}

on_exit() {
	rc=$?
	trap - EXIT
	terminate_children
	cleanup
	exit $rc
}
trap on_exit EXIT

echo "import model HC_NLO_X0_UFO" > "${mg_cmds}"
echo "generate p p > t t~ x0" >> "${mg_cmds}"

echo "output ${proc_folder}" >> "${mg_cmds}"
echo "launch" >> "${mg_cmds}"
echo "analysis=OFF" >> "${mg_cmds}"
echo "madspin=ON" >> "${mg_cmds}"
echo "detector=Delphes" >> "${mg_cmds}"
echo "shower=Pythia8" >> "${mg_cmds}"
echo "done" >> "${mg_cmds}"
echo "cards/run_card_ttH.dat" >> "${mg_cmds}"
echo "cards/paramcard_ttH_alpha_${alpha}.dat" >> "${mg_cmds}"
echo "cards/madspin_card_ttH_semilep_${semilep}.dat" >> "${mg_cmds}"
echo "cards/delphes_card_R04.dat" >> "${mg_cmds}"
echo "set HEPMCoutput:file=hepmcremove" >> "${mg_cmds}"
echo "set iseed ${seed}" >> "${mg_cmds}"

echo "Running MG5 with the following commands:"
cat "${mg_cmds}"

cp -r cards "${scratch}/"
cd "${scratch}"
"${main_dir}/MG/bin/mg5_aMC" "${mg_cmds}"
cd "${main_dir}"

# Replace pre-existing outputs so analysis can proceed for repeated runs.
if [[ -e "${root_file}" || -e "${lhe_file}" ]]; then
	echo "Overwriting existing output files for alpha=${alpha}, semilep=${semilep}, nrun=${nrun}."
	rm -f "${root_file}" "${lhe_file}"
fi

echo "MadGraph Events directory contents:"
ls -lR "${proc_folder}/Events/" || echo "Events dir missing: ${proc_folder}/Events/"

run_dir="${proc_folder}/Events/run_01_decayed_1"
if [[ ! -d "${run_dir}" ]]; then
	# Fall back to first available decayed run directory
	run_dir="$(find "${proc_folder}/Events" -maxdepth 1 -name 'run_*_decayed_*' | sort | head -n1)"
	echo "run_01_decayed_1 not found, using: ${run_dir}"
fi

mv "${run_dir}/tag_1_delphes_events.root" "${root_file}"
mv "${run_dir}/unweighted_events.lhe.gz" "${lhe_file}"
produced_outputs=1

# analysis
echo "Starting analysis step..."
# conda activate scripts may reference unset variables; disable -u around activation.
set +u
conda activate rootpy
set -u
echo "Activated conda env: rootpy"
./analyse_root_files.py --sample_folder='samples_ttH' --alpha=${alpha} --nrun=${nrun} --semilep=${semilep} --train_data --top_mass_lower=120 --top_mass_upper=220