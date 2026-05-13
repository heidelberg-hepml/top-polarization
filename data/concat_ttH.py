#!/usr/bin/env python3

import awkward as ak
from tqdm import tqdm
import os

top_mass_window = (120.0, 220.0)
# top_mass_window = (165.0, 190.0)

for alpha in tqdm(["0", "45", "-45"]):
    # for pol in tqdm(['0']):
    for semilep in ["1", "2"]:
        comb_array = []
        nrun_range = range(1, 601)
        for nrun in nrun_range:
            file = f"/home/hd/hd_ve298/projects/top_polarization/top-polarization/data/samples_ttH_awk/ttH_alpha_{alpha}_semilep_{semilep}_nrun_{nrun}_{top_mass_window[0]}_{top_mass_window[1]}_train.parquet"
            if os.path.exists(file):
                array = ak.from_parquet(file)
            else:
                continue
            # print lengths
            print(f"Alpha: {alpha}, Semilep: {semilep}, Length: {len(array)}")
            comb_array.append(array)
        comb_array = ak.concatenate(comb_array)
        print(f"Combined length for alpha {alpha} semilep {semilep}: {len(comb_array)}")
        ak.to_parquet(
            comb_array,
            f"/home/hd/hd_ve298/projects/top_polarization/top-polarization/data/samples_ttH_awk_comb/ttH_alpha_{alpha}_semilep_{semilep}_allruns_{top_mass_window[0]}_{top_mass_window[1]}_train.parquet",
        )
