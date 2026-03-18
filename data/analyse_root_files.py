#!/usr/bin/env python3

import os
from glob import glob
from delphes_reader import DelphesReader
import argparse
import awkward as ak

import ROOT
import fastjet as fj
print(ROOT.gSystem.GetLibraries())
ROOT.gSystem.Load(
"MG/Delphes/libDelphes.so"
)

args = argparse.ArgumentParser()
args.add_argument('--sample_folder', type=str, default='samples', help='top polarization state')
args.add_argument('--pol', type=str, help='top polarization state')
args.add_argument('--alpha', type=str, help='top polarization state')
args.add_argument('--nrun', type=str, help='top polarization state')
args.add_argument('--semilep', type=str, help='top polarization state')
args.add_argument('--top_mass_lower', type=float, default=165, help='lower top mass')
args.add_argument('--top_mass_upper', type=float, default=190, help='upper top mass')
args.add_argument('--train_data', action='store_true', help='flag to indicate training data')

args = args.parse_args()

is_ttH = args.alpha is not None
print(f"Running analysis for {'ttH' if is_ttH else 'ttbar'} samples with args {args}")

top_mass = [args.top_mass_lower, args.top_mass_upper]
if is_ttH:
    tag = f"ttH_alpha_{args.alpha}_semilep_{args.semilep}_nrun_{args.nrun}"
else:
    tag = f"top_pol_{args.pol}_semilep_{args.semilep}_nrun_{args.nrun}"
delphes_file = f"{args.sample_folder}/{tag}.root"
lhe_file = f"{args.sample_folder}/{tag}.lhe.gz"

delphes_reader = DelphesReader(pnet=True, top_mass=top_mass, ttH=is_ttH)
# delphes_reader.change_clustering()
delphes_reader._get_single_delphes(delphes_file, lhe_file, train_data=args.train_data)
delphes_reader.inspect_fails()

data = delphes_reader.jet_data.snapshot()

if args.train_data:
    ak.to_parquet(data, f"samples_awk/{tag}_{top_mass[0]}_{top_mass[1]}_train.parquet")
else:
    ak.to_parquet(data, f"samples_awk/{tag}_{top_mass[0]}_{top_mass[1]}_test.parquet")
