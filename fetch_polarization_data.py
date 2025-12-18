import argparse
import os

import requests

parser = argparse.ArgumentParser(description="Download data.")
parser.add_argument("--out", type=str)
parser.add_argument("--mass", type=str, default="strict")

args = parser.parse_args()

OUT = args.out
MASS = args.mass

BASE_URL = "https://seafile.cloud.uni-hannover.de/d/bed29fcc630340d4a131/files"
POLARIZATIONS = ["L", "R", "0"]
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
        for nr in range(1, 4):
            load(f"polarization_top_{pol}_{m[0]}_{m[1]}_train_{nr}.parquet", OUT)
        load(f"polarization_top_{pol}_{m[0]}_{m[1]}_test.parquet", OUT)


if __name__ == "__main__":
    main()
