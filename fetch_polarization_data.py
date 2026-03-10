import argparse
import os

import requests

parser = argparse.ArgumentParser(description="Download data.")
parser.add_argument("--out", type=str)
parser.add_argument("--mass", type=str, default="loose")

args = parser.parse_args()

OUT = args.out
MASS = args.mass

BASE_URL = "https://seafile.cloud.uni-hannover.de/d/053ac05f748d4f0583fa/files"
POLARIZATIONS = ["0", ("L","L"), ("L","R"), ("R","L"), ("R","R")]
MASS_WINDOWS = {"strict": (165, 190), "loose": (120, 220)}


def load(filename, out):
    params = {"p": filename, "dl": "1"}
    print(f"Downloading {filename}")
    target_path = os.path.join(out, filename) if out else os.path.join("data", filename)

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
    for pol in POLARIZATIONS:
        for decay in [1, 2]:
            if pol == "0":
                for nr in range(1,4):
                    load(f"top_pol_{pol}_semilep_{decay}_{m[0]}_{m[1]}_train_{nr}.parquet", OUT)
            else:
                for nr in [1]:
                    load(f"top_pol_had_{pol[0]}_lep_{pol[1]}_semilep_{decay}_{m[0]}_{m[1]}_train_{nr}.parquet", OUT)

if __name__ == "__main__":
    main()
