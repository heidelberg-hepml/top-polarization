"""Command-line entry point: read one Delphes+LHE pair, write parquet.

Invoke via:
    python -m data.pipeline --process tt     --label 0       --nrun 1 --semilep 1 \\
                          --top-mass 165 190 --input-dir ... --output-dir ... --train

    python -m data.pipeline --process tth    --label -45     --nrun 1 --semilep 1 \\
                          --top-mass 120 220 --input-dir ... --output-dir ... --train

    python -m data.pipeline --process tt_ctg --label 0.5,0.3 --nrun 1 --semilep 1 \\
                          --top-mass 165 190 --input-dir ... --output-dir ... --train
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import awkward as ak

from .config import PROCESS_CONFIGS, make_tag
from .reader import DelphesReader


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Delphes+LHE -> parquet.")
    p.add_argument(
        "--process",
        required=True,
        choices=sorted(PROCESS_CONFIGS),
        help="Process tag (tt, tth, tt_ctg, tt_ctg_lin).",
    )
    p.add_argument(
        "--label",
        required=True,
        help="Polarization (0|L|R|LR|RL) for tt, alpha (0|45|-45) for tth, "
        "ctG 're,im' (e.g. 0.5,0.3) for tt_ctg / tt_ctg_lin.",
    )
    p.add_argument("--nrun", required=True, type=int)
    p.add_argument("--semilep", required=True, type=int, choices=[1, 2])
    p.add_argument(
        "--top-mass",
        required=True,
        type=float,
        nargs=2,
        metavar=("LO", "HI"),
        help="Top mass window, e.g. 165 190.",
    )
    p.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing the .root and .lhe.gz inputs.",
    )
    p.add_argument(
        "--output-dir", required=True, type=Path, help="Directory to write the .parquet output."
    )
    p.add_argument(
        "--train",
        action="store_true",
        help="Match subjets to truth partons (default: Delphes b-tag).",
    )
    p.add_argument("--clustering", choices=["pnet", "ca"], default="pnet")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    process_cfg = PROCESS_CONFIGS[args.process]
    top_mass = (float(args.top_mass[0]), float(args.top_mass[1]))

    tag = make_tag(args.process, args.label, args.semilep, args.nrun)
    delphes_file = args.input_dir / f"{tag}.root"
    lhe_file = args.input_dir / f"{tag}.lhe.gz"

    print(f"Running {args.process} analysis with: {vars(args)}")

    reader = DelphesReader(process_cfg, clustering=args.clustering, top_mass=top_mass)
    reader.process_pair(str(delphes_file), str(lhe_file), train_data=args.train)
    reader.inspect_fails()

    print("Analysis done. Creating snapshot...")
    data = reader.writer.snapshot()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split = "train" if args.train else "test"
    out_path = args.output_dir / f"{tag}_{top_mass[0]}_{top_mass[1]}_{split}.parquet"
    print(f"Saving {len(data)} events to {out_path}")
    ak.to_parquet(data, str(out_path))

    print("Done!")

    # ROOT teardown can hang at interpreter shutdown on some clusters; flush
    # logs and force-exit so the batch script can continue cleanup.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
