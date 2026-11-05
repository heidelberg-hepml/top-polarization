"""Concatenate per-run parquet files into one combined file per (label, semilep,
mass window) configuration.

Examples:
    python -m data.postprocess.concat --process tt     --range 1:69
    python -m data.postprocess.concat --process tth    --range 1:600 --top-mass 120 220
    python -m data.postprocess.concat --process tt_ctg --range 1:50  --labels 0.5,0.3 1.0,0.0
    python -m data.postprocess.concat --process tt_ctg_lin --range 1:50 --labels 0.5,0.3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
from tqdm import tqdm

from ..pipeline.config import PROCESS_CONFIGS, make_tag

PROCESS_LABEL_DEFAULTS = {
    "tt": ["0", "L", "R", "LR", "RL"],
    "tth": ["0", "45", "-45"],
    "tt_ctg": [],  # no sensible default — user must pass --labels
    "tt_ctg_lin": [],  # idem
}


def _parquet_filename(
    process: str, label: str, semilep: str, nrun: int, mlo: float, mhi: float, split: str = "train"
) -> str:
    tag = make_tag(process, label, semilep, nrun)
    return f"{tag}_{mlo}_{mhi}_{split}.parquet"


def _combined_filename(
    process: str, label: str, semilep: str, mlo: float, mhi: float, split: str = "train"
) -> str:
    # Combined file shares the same per-run tag but with "nrun_X" replaced by
    # "allruns" so it lives next to the shards under a recognisable name.
    per_run_tag = make_tag(process, label, semilep, nrun=1)
    base = per_run_tag.replace("_nrun_1", "_allruns")
    return f"{base}_{mlo}_{mhi}_{split}.parquet"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--process", required=True, choices=sorted(PROCESS_CONFIGS))
    p.add_argument(
        "--input-dir", required=True, type=Path, help="Directory with per-run parquet files."
    )
    p.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to write combined parquet files."
    )
    p.add_argument(
        "--range", required=True, dest="nrun_range", help="Inclusive nrun range A:B, e.g. 1:69"
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels to combine (default: all for process; required for tt_ctg).",
    )
    p.add_argument("--semileps", nargs="+", default=["1", "2"])
    p.add_argument(
        "--top-mass",
        nargs=2,
        type=float,
        action="append",
        metavar=("LO", "HI"),
        help="Mass window LO HI. Can be repeated. Default: both (165 190) and (120 220).",
    )
    p.add_argument("--split", default="train", choices=["train", "test"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    labels = args.labels or PROCESS_LABEL_DEFAULTS[args.process]
    if not labels:
        raise SystemExit(f"--labels is required for process {args.process}; got none.")
    mass_windows = args.top_mass or [(165.0, 190.0), (120.0, 220.0)]
    parts = args.nrun_range.split(":")
    nrun_start, nrun_end = int(parts[0]), int(parts[1])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for label in tqdm(labels, desc="labels"):
        for semilep in args.semileps:
            for mlo, mhi in mass_windows:
                shards = []
                for nrun in range(nrun_start, nrun_end + 1):
                    src = args.input_dir / _parquet_filename(
                        args.process,
                        label,
                        semilep,
                        nrun,
                        mlo,
                        mhi,
                        args.split,
                    )
                    if not src.exists():
                        print(f"File not found: {src}")
                        continue
                    arr = ak.from_parquet(str(src))
                    print(
                        f"label={label} semilep={semilep} mass=({mlo},{mhi}) "
                        f"nrun={nrun} length={len(arr)}"
                    )
                    shards.append(arr)
                if not shards:
                    print(
                        f"No shards found for label={label} semilep={semilep} "
                        f"mass=({mlo},{mhi}); skipping."
                    )
                    continue
                comb = ak.concatenate(shards)
                print(
                    f"label={label} semilep={semilep} mass=({mlo},{mhi}) "
                    f"combined length={len(comb)}"
                )
                dst = args.output_dir / _combined_filename(
                    args.process,
                    label,
                    semilep,
                    mlo,
                    mhi,
                    args.split,
                )
                ak.to_parquet(comb, str(dst))


if __name__ == "__main__":
    main()
