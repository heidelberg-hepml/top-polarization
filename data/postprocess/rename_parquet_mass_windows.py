"""Rename parquet files with integer mass-window tokens to decimal tokens.

Replacements:
- _120_220_  -> _120.0_220.0_
- _165_190_  -> _165.0_190.0_

Examples:
    python -m data.postprocess.rename_parquet_mass_windows
    python -m data.postprocess.rename_parquet_mass_windows --apply
    python -m data.postprocess.rename_parquet_mass_windows --root data/samples_awk
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("_120_220_", "_120.0_220.0_"),
    ("_165_190_", "_165.0_190.0_"),
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/samples_awk"),
        help="Directory containing parquet files (searched recursively).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform renames. If not set, prints a dry run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination file if it already exists.",
    )
    return parser.parse_args(argv)


def _target_name(name: str) -> str:
    updated = name
    for old, new in REPLACEMENTS:
        updated = updated.replace(old, new)
    return updated


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    root = args.root.expanduser().resolve()

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    planned = 0
    skipped_existing = 0
    renamed = 0

    for src in sorted(root.rglob("*.parquet")):
        dst_name = _target_name(src.name)
        if dst_name == src.name:
            continue

        planned += 1
        dst = src.with_name(dst_name)

        if dst.exists() and not args.overwrite:
            skipped_existing += 1
            print(f"SKIP (exists): {src} -> {dst}")
            continue

        action = "RENAME" if args.apply else "DRYRUN"
        print(f"{action}: {src} -> {dst}")

        if args.apply:
            if dst.exists() and args.overwrite:
                dst.unlink()
            src.rename(dst)
            renamed += 1

    if args.apply:
        print(f"Done. planned={planned} renamed={renamed} skipped_existing={skipped_existing}")
    else:
        print(f"Dry run complete. candidates={planned} would_skip_existing={skipped_existing}")


if __name__ == "__main__":
    main()
