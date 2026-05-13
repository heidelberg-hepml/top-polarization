#!/usr/bin/env python3

import awkward as ak
from tqdm import tqdm

# top_mass_window = (120.0, 220.0)
top_mass_windows = [(165.0, 190.0), (120.0, 220.0)]

# for pol in tqdm(['0', 'L', 'R', 'LR', 'RL']):
for pol in tqdm(['0']):
    for semilep in ['1', '2']:
        for top_mass_window in top_mass_windows:
            comb_array = []
            nrun_range = range(1, 101) if pol == '0' else range(1, 41)
            for nrun in nrun_range:
                array = ak.from_parquet(
                    f'/home/hd/hd_ve298/projects/top_polarization/top-polarization/data/samples_awk/top_pol_{pol}_semilep_{semilep}_nrun_{nrun}_{top_mass_window[0]}_{top_mass_window[1]}_train.parquet'
                )
                # print lengths
                print(
                    f"Pol: {pol}, Semilep: {semilep}, Top mass window: {top_mass_window} Length: {len(array)}"
                )
                comb_array.append(array)
            ak.to_parquet(ak.concatenate(comb_array),
                        f'/home/hd/hd_ve298/projects/top_polarization/top-polarization/data/samples_awk_comb/top_pol_{pol}_semilep_{semilep}_allruns_{top_mass_window[0]}_{top_mass_window[1]}_train.parquet'
                        )
