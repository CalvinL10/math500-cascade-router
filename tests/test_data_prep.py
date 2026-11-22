"""Verification tests for Step 1A data splits: counts, stratification, determinism, zero leakage."""

from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

from src.data_prep import (
    DEFAULT_SEED,
    PILOT_COUNTS,
    ROUTER_TRAIN_COUNTS,
    ROUTER_VAL_COUNTS,
    build_splits,
    normalize_problem_text,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def test_splits_exist_and_match_manifest() -> None:
    manifest_path = SPLITS_DIR / "splits_manifest.json"
    assert manifest_path.is_file(), "splits_manifest.json missing; run scripts/prepare_data.py"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    expected_counts = {
        "pilot_50": 50,
        "router_train_600": 600,
        "router_val_200": 200,
        "math500_test_500": 500,
    }
    expected_levels = {
        "pilot_50": PILOT_COUNTS,
        "router_train_600": ROUTER_TRAIN_COUNTS,
        "router_val_200": ROUTER_VAL_COUNTS,
        "math500_test_500": {1: 43, 2: 90, 3: 105, 4: 128, 5: 134},
    }

    loaded: dict[str, list[dict]] = {}
    for name, count in expected_counts.items():
        path = SPLITS_DIR / f"{name}.json"
        assert path.is_file()
        assert sha256_file(path) == manifest["files"][f"{name}.json"]["sha256"]
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert len(rows) == count
        assert dict(Counter(r["level"] for r in rows)) == expected_levels[name]
        for row in rows:
            assert set(row.keys()) == {
                "unique_id",
                "problem",
                "answer",
                "level",
                "subject",
                "source_split",
                "split_name",
            }
            assert row["problem"].strip()
            assert str(row["answer"]).strip()
            assert 1 <= row["level"] <= 5
        loaded[name] = rows

    bundle_path = SPLITS_DIR / "splits_bundle.json"
    assert bundle_path.is_file()
    assert sha256_file(bundle_path) == manifest["files"]["splits_bundle.json"]["sha256"]

    # Zero leakage & pairwise disjointness across all 4 splits
    names = list(loaded)
    for i, left in enumerate(names):
        left_uids = {r["unique_id"] for r in loaded[left]}
        left_probs = {normalize_problem_text(r["problem"]) for r in loaded[left]}
        assert len(left_uids) == len(loaded[left])
        assert len(left_probs) == len(loaded[left])
        for right in names[i + 1 :]:
            right_uids = {r["unique_id"] for r in loaded[right]}
            right_probs = {normalize_problem_text(r["problem"]) for r in loaded[right]}
            assert not (left_uids & right_uids), f"uid overlap: {left} vs {right}"
            assert not (left_probs & right_probs), f"problem overlap: {left} vs {right}"


def test_split_generation_is_deterministic() -> None:
    raw_train = json.loads((RAW_DIR / "math_full_minus_math500.json").read_text(encoding="utf-8"))
    raw_test = json.loads((RAW_DIR / "math500_test.json").read_text(encoding="utf-8"))
    splits_a, _ = build_splits(raw_train, raw_test, seed=DEFAULT_SEED)
    splits_b, _ = build_splits(raw_train, raw_test, seed=DEFAULT_SEED)
    for name in splits_a:
        assert [r["unique_id"] for r in splits_a[name]] == [r["unique_id"] for r in splits_b[name]]
        disk_rows = json.loads((SPLITS_DIR / f"{name}.json").read_text(encoding="utf-8"))
        assert [r["unique_id"] for r in splits_a[name]] == [r["unique_id"] for r in disk_rows]
