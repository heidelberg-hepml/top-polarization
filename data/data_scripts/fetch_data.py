import argparse
import os
from urllib.parse import quote

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
    BASE_URL = "https://tu-dortmund.sciebo.de/public.php/dav/files/nKYKyDm23TabWQy"
    LABELS = ["0", ("L", "L"), ("L", "R"), ("R", "L"), ("R", "R")]
elif PROC == "tt_smeft":
    BASE_URL = "https://tu-dortmund.sciebo.de/public.php/dav/files/SwSxDJeNkTDxKyr"
    LABELS  = ["0", "0p1", "-0p1", "2p6", "-2p6", "10", "-10", "100", "-100"]
elif PROC == "tth":
    BASE_URL = "https://tu-dortmund.sciebo.de/public.php/dav/files/3SprM35zsfWgqHs"
    LABELS = ["0", "45", "-45"]
elif PROC == "vbf":
    BASE_URL = "https://tu-dortmund.sciebo.de/public.php/dav/files/FKzPdYLLF3ePdy8"
    LABELS = ["pos", "neg", "zero"]
else:
    raise ValueError(f"Unknown process: {PROC}. Available: tt_pol, tt_smeft, tth, vbf")
MASS_WINDOWS = {"strict": (165, 190), "loose": (120, 220)}


def load(filename, out=None):
    print(f"Downloading {filename}")

    if out is None:
        out = os.path.join("data", PROC)

    os.makedirs(out, exist_ok=True)
    target_path = os.path.join(out, filename)
    url = f"{BASE_URL}/{quote(filename)}"

    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        with open(target_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"Successfully downloaded {filename}")
    print("")


def main():
    m = MASS_WINDOWS[MASS]
    for lbl in LABELS:
        if PROC == "vbf":
            load(f"cHWtil_{lbl}.npy", OUT)
        elif PROC == "tt_smeft":
            load(f"tt_ctg_re_0_im_{lbl}_semilep_1_{m[0]}_{m[1]}_train.parquet",
                 OUT,
            )
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
                elif PROC == "tth":
                    for nr in range(1, 3):
                        load(
                            f"ttH_alpha_{lbl}_semilep_{decay}_{m[0]}_{m[1]}_train_{nr}.parquet",
                            OUT,
                        )


if __name__ == "__main__":
    main()
