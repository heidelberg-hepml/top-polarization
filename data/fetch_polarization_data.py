import numpy as np
import os
import wget

# Does not work right now, to be fixed

BASE_URL = "https://cernbox.cern.ch/files/spaces/eos/user/m/mmenen/top-polarization"
POLARIZATIONS = ["L", "R", "0"]
MASS_WINDOWS = {
    "strict": (165, 190),
    "loose": (120, 220)
}


def load(filename):
    url = os.path.join(BASE_URL, filename)
    print(f"Started to download {url}")
    target_path = filename
    wget.download(url, out=target_path)
    print("")
    print(f"Successfully downloaded {target_path}")


def main():
    for pol in POLARIZATIONS:
        for mass in ["strict", "loose"]:
            for nr in range(1, 2):
                load(f"polarization_top_{pol}_{MASS_WINDOWS[mass][0]}_{MASS_WINDOWS[mass][1]}_train_{nr}.parquet")
            load(f"polarization_top_{pol}_{MASS_WINDOWS[mass][0]}_{MASS_WINDOWS[mass][1]}_test.parquet")


if __name__ == "__main__":
    main()
