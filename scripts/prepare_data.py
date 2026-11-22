"""CLI for Step 1A: download/verify raw MATH sources and generate zero-leakage splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep import (
    DEFAULT_SEED,
    build_splits,
    ensure_raw_datasets,
    write_splits,
)


DEFAULT_COURSE_MATH500 = (
    PROJECT_ROOT.parent / "math550_2026fall_class" / "data" / "reasoning" / "math500_test.json"
)


def run(
    *,
    raw_dir: Path = PROJECT_ROOT / "data" / "raw",
    splits_dir: Path = PROJECT_ROOT / "data" / "splits",
    course_math500_path: Path = DEFAULT_COURSE_MATH500,
    seed: int = DEFAULT_SEED,
    force_download: bool = False,
) -> Path:
    raw_meta = ensure_raw_datasets(
        raw_dir,
        course_math500_path=course_math500_path,
        force=force_download,
    )
    raw_train = json.loads((raw_dir / "math_full_minus_math500.json").read_text(encoding="utf-8"))
    raw_test = json.loads((raw_dir / "math500_test.json").read_text(encoding="utf-8"))

    splits, manifest = build_splits(raw_train, raw_test, seed=seed)
    manifest_path = write_splits(splits, manifest, splits_dir, raw_metadata=raw_meta)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    manifest_path = run(seed=args.seed, force_download=args.force_download)
    print(f"Wrote split manifest: {manifest_path}")
    print(manifest_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
