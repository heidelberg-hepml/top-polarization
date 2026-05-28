import argparse
import os

import requests

parser = argparse.ArgumentParser(description="Download data.")
parser.add_argument("--out", type=str)
parser.add_argument("--process", type=str, default="tt_pol")
parser.add_argument("--mass", type=str, default="loose")

args = parser.parse_args()

OUT = args.out
PROC = args.process
MASS = args.mass

if PROC == "tt_pol":
    BASE_URL = "https://seafile.cloud.uni-hannover.de/d/053ac05f748d4f0583fa/files"
    LABELS = ["0", ("L", "L"), ("L", "R"), ("R", "L"), ("R", "R")]
elif PROC == "tt_smeft":
    BASE_URL = "https://seafile.cloud.uni-hannover.de/d/3f6addd27e8645ae9f70/files"
    LABELS  = []
elif PROC == "tth":
    BASE_URL = "https://seafile.cloud.uni-hannover.de/d/de5e4b2fda5546c1be46/files"
    LABELS = ["0", "45", "-45"]
elif PROC == "vbf":
    BASE_URL = "https://seafile.cloud.uni-hannover.de/d/bc0c7097c00949ca822c/files"
    LABELS = ["pos", "neg", "zero"]
else:
    raise ValueError(f"Unknown process: {PROC}. Available: tt_pol, tt_smeft, tth, vbf")
MASS_WINDOWS = {"strict": (165, 190), "loose": (120, 220)}


def load(filename, out):
    params = {"p": filename, "dl": "1"}
    print(f"Downloading {filename}")
    if out is None:
        out = os.path.join("data", PROC)
    os.makedirs(out, exist_ok=True)
    target_path = os.path.join(out, filename)

    with requests.get(BASE_URL, params=params, stream=True) as r:
        r.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Successfully downloaded {filename}")
    print("")


def main():
    m = MASS_WINDOWS[MASS]
    for lbl in LABELS:
        if PROC == "vbf":
            load(f"cHWtil_{lbl}.npy", OUT)
        else:
            for decay in [1, 2]:
                if PROC == "tt_pol":
                    if lbl == "0":
                        for nr in [1,2]:
                            load(
                                f"top_pol_{lbl}_semilep_{decay}_{m[0]}_{m[1]}_train_{nr}.parquet",
                                OUT,
                            )
                    else:
                        for nr in [1]:
                            load(
                                f"top_pol_had_{lbl[0]}_lep_{lbl[1]}_semilep_{decay}_{m[0]}_{m[1]}_train_{nr}.parquet",
                                OUT,
                            )
                elif PROC == "tt_smeft":
                    pass
                elif PROC == "tth":
                    for nr in range(1, 3):
                        load(
                            f"ttH_alpha_{lbl}_semilep_{decay}_{m[0]}_{m[1]}_train_{nr}.parquet", OUT
                        )


if __name__ == "__main__":
    main()
