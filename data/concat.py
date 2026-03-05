#!/usr/bin/env python3

import awkward as ak
from tqdm import tqdm

for pol in tqdm(['0', 'L', 'R', 'LR', 'RL']):
    for semilep in ['1', '2']:
        array = ak.from_parquet(
            f'/home/hd/hd_ve298/projects/top_polarization/top-polarization/data/samples_awk/top_pol_{pol}_semilep_{semilep}_nrun_*_120.0_220.0_train.parquet'
        )
        # print lengths
        print(f'Pol: {pol}, Semilep: {semilep}, Length: {len(array)}')
        ak.to_parquet(array,
                      f'/home/hd/hd_ve298/projects/top_polarization/top-polarization/data/samples_awk_comb/top_pol_{pol}_semilep_{semilep}_allruns_120.0_220.0_train.parquet'
                      )
