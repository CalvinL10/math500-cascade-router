"""Zero-leakage data download, schema normalization, and stratified split generation (Step 1A)."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
from typing import Any
from urllib.request import Request, urlopen


MATH_TRAIN_PRIMARY_URL = (
    "https://raw.githubusercontent.com/rasbt/math_full_minus_math500/"
    "refs/heads/main/math_full_minus_math500.json"
)
MATH_TRAIN_BACKUP_URL = (
    "https://f001.backblazeb2.com/file/reasoning-from-scratch/MATH/math_full_minus_math500.json"
)
MATH500_PRIMARY_URL = (
    "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
    "main/ch03/01_main-chapter-code/math500_test.json"
)

DEFAULT_SEED = 550
PILOT_COUNTS = {1: 10, 2: 10, 3: 10, 4: 10, 5: 10}
ROUTER_TRAIN_COUNTS = {1: 52, 2: 108, 3: 126, 4: 154, 5: 160}
ROUTER_VAL_COUNTS = {1: 17, 2: 36, 3: 42, 4: 51, 5: 54}

_USER_AGENT = "MATH550-Final-Project/1.0"


def normalize_problem_text(text: str) -> str:
    """Canonicalize problem whitespace and casing for leakage/duplicate checks."""

    return re.sub(r"\s+", " ", str(text).strip().lower())


def parse_level(raw_level: object) -> int | None:
    """Convert `'Level 3'` or `3` into integer `1..5`, returning None if invalid."""

    if isinstance(raw_level, bool):
        return None
    if isinstance(raw_level, int):
        return raw_level if 1 <= raw_level <= 5 else None
    text = str(raw_level).strip()
    match = re.fullmatch(r"(?:Level\s*)?([1-5])", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def sha256_bytes(payload: bytes) -> str:
    """Return hexadecimal SHA-256 digest of bytes."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Return hexadecimal SHA-256 digest of a file."""

    return sha256_bytes(path.read_bytes())


def _download_with_fallback(urls: list[str], dest: Path) -> str:
    last_error: Exception | None = None
    for url in urls:
        try:
            req = Request(url, headers={"User-Agent": _USER_AGENT})
            with urlopen(req, timeout=60) as resp:
                payload = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(payload)
            return url
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"failed to download from {urls}: {last_error}")


def ensure_raw_datasets(
    raw_dir: Path,
    *,
    course_math500_path: Path | None = None,
    force: bool = False,
) -> dict[str, dict[str, object]]:
    """Ensure both `math_full_minus_math500.json` and `math500_test.json` exist locally."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, dict[str, object]] = {}

    train_path = raw_dir / "math_full_minus_math500.json"
    if force or not train_path.is_file():
        src_url = _download_with_fallback([MATH_TRAIN_PRIMARY_URL, MATH_TRAIN_BACKUP_URL], train_path)
        train_status = f"downloaded:{src_url}"
    else:
        train_status = "existing"
    metadata["math_full_minus_math500"] = {
        "path": str(train_path),
        "bytes": train_path.stat().st_size,
        "sha256": sha256_file(train_path),
        "status": train_status,
    }

    test_path = raw_dir / "math500_test.json"
    if force or not test_path.is_file():
        if course_math500_path is not None and course_math500_path.is_file():
            shutil.copy2(course_math500_path, test_path)
            test_status = f"copied:{course_math500_path}"
        else:
            src_url = _download_with_fallback([MATH500_PRIMARY_URL], test_path)
            test_status = f"downloaded:{src_url}"
    else:
        test_status = "existing"
    metadata["math500_test"] = {
        "path": str(test_path),
        "bytes": test_path.stat().st_size,
        "sha256": sha256_file(test_path),
        "status": test_status,
    }
    return metadata


def normalize_records(
    raw_rows: list[dict[str, Any]],
    *,
    source_split: str,
    drop_invalid: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Normalize raw MATH / MATH-500 rows to the unified project schema and deduplicate."""

    normalized: list[dict[str, Any]] = []
    seen_problems: set[str] = set()
    stats = {
        "raw_rows": len(raw_rows),
        "dropped_invalid_level": 0,
        "dropped_empty_field": 0,
        "dropped_duplicate_problem": 0,
    }

    for idx, row in enumerate(raw_rows):
        level = parse_level(row.get("level"))
        if level is None:
            if drop_invalid:
                stats["dropped_invalid_level"] += 1
                continue
            raise ValueError(f"invalid level in {source_split} row {idx}: {row.get('level')!r}")

        problem = str(row.get("problem", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not problem or not answer:
            if drop_invalid:
                stats["dropped_empty_field"] += 1
                continue
            raise ValueError(f"empty problem/answer in {source_split} row {idx}")

        norm_prob = normalize_problem_text(problem)
        if norm_prob in seen_problems:
            if drop_invalid:
                stats["dropped_duplicate_problem"] += 1
                continue
            raise ValueError(f"duplicate problem in {source_split} row {idx}")
        seen_problems.add(norm_prob)

        subject = str(row.get("subject") or row.get("type") or "Unknown").strip()
        raw_uid = row.get("unique_id", idx)
        uid = f"{source_split}:{raw_uid}" if source_split == "math_train" else str(raw_uid)
        normalized.append(
            {
                "unique_id": uid,
                "problem": problem,
                "answer": answer,
                "level": level,
                "subject": subject,
                "source_split": source_split,
            }
        )
    return normalized, stats


def _stratified_take(
    pool_by_level: dict[int, list[dict[str, Any]]],
    counts_by_level: dict[int, int],
    rng: random.Random,
    *,
    split_name: str,
) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for level in sorted(counts_by_level):
        need = counts_by_level[level]
        available = pool_by_level[level]
        if len(available) < need:
            raise ValueError(
                f"insufficient rows for {split_name} level {level}: need {need}, have {len(available)}"
            )
        selected = available[:need]
        del available[:need]
        for row in selected:
            item = dict(row)
            item["split_name"] = split_name
            chosen.append(item)
    rng.shuffle(chosen)
    return chosen


def build_splits(
    raw_train_rows: list[dict[str, Any]],
    raw_test_rows: list[dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build mutually disjoint `pilot_50`, `router_train_600`, `router_val_200`, `math500_test_500`."""

    test_rows, test_stats = normalize_records(raw_test_rows, source_split="math500_test", drop_invalid=False)
    test_norm_problems = {normalize_problem_text(row["problem"]) for row in test_rows}
    test_uids = {row["unique_id"] for row in test_rows}

    train_rows, train_stats = normalize_records(raw_train_rows, source_split="math_train", drop_invalid=True)
    # Enforce hard exclusion of any test-overlapping problem just in case an upstream file changes.
    filtered_train: list[dict[str, Any]] = []
    excluded_overlap = 0
    for row in train_rows:
        if normalize_problem_text(row["problem"]) in test_norm_problems or row["unique_id"] in test_uids:
            excluded_overlap += 1
            continue
        filtered_train.append(row)
    train_stats["dropped_test_overlap"] = excluded_overlap

    rng = random.Random(seed)
    pool_by_level: dict[int, list[dict[str, Any]]] = {lv: [] for lv in range(1, 6)}
    for row in filtered_train:
        pool_by_level[row["level"]].append(row)
    for lv in sorted(pool_by_level):
        pool_by_level[lv].sort(key=lambda r: r["unique_id"])
        rng.shuffle(pool_by_level[lv])

    pilot_50 = _stratified_take(pool_by_level, PILOT_COUNTS, rng, split_name="pilot_50")
    router_train_600 = _stratified_take(
        pool_by_level, ROUTER_TRAIN_COUNTS, rng, split_name="router_train_600"
    )
    router_val_200 = _stratified_take(
        pool_by_level, ROUTER_VAL_COUNTS, rng, split_name="router_val_200"
    )
    math500_test_500 = [dict(row, split_name="math500_test_500") for row in test_rows]

    splits = {
        "pilot_50": pilot_50,
        "router_train_600": router_train_600,
        "router_val_200": router_val_200,
        "math500_test_500": math500_test_500,
    }

    # Explicit cross-split leakage & disjointness audit.
    all_uid_sets = {name: {r["unique_id"] for r in rows} for name, rows in splits.items()}
    all_prob_sets = {
        name: {normalize_problem_text(r["problem"]) for r in rows} for name, rows in splits.items()
    }
    split_names = list(splits)
    pairwise_overlaps: dict[str, dict[str, int]] = {}
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            pairwise_overlaps[f"{left}__vs__{right}"] = {
                "unique_id_overlap": len(all_uid_sets[left] & all_uid_sets[right]),
                "normalized_problem_overlap": len(all_prob_sets[left] & all_prob_sets[right]),
            }

    manifest = {
        "seed": seed,
        "cleaning": {"math_train": train_stats, "math500_test": test_stats},
        "pairwise_overlaps": pairwise_overlaps,
        "splits": {
            name: {
                "count": len(rows),
                "levels": dict(sorted(Counter(r["level"] for r in rows).items())),
                "subjects": dict(sorted(Counter(r["subject"] for r in rows).items())),
            }
            for name, rows in splits.items()
        },
    }
    return splits, manifest


def write_splits(
    splits: dict[str, list[dict[str, Any]]],
    manifest: dict[str, Any],
    splits_dir: Path,
    raw_metadata: dict[str, dict[str, object]] | None = None,
) -> Path:
    """Persist split JSONs, combined `splits_bundle.json`, and SHA-256 `splits_manifest.json`."""

    splits_dir.mkdir(parents=True, exist_ok=True)
    file_hashes: dict[str, dict[str, object]] = {}

    for name, rows in splits.items():
        path = splits_dir / f"{name}.json"
        raw_bytes = (json.dumps(rows, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        path.write_bytes(raw_bytes)
        file_hashes[f"{name}.json"] = {
            "bytes": len(raw_bytes),
            "sha256": sha256_bytes(raw_bytes),
            "count": len(rows),
        }

    bundle_path = splits_dir / "splits_bundle.json"
    bundle_bytes = (json.dumps(splits, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    bundle_path.write_bytes(bundle_bytes)
    file_hashes["splits_bundle.json"] = {
        "bytes": len(bundle_bytes),
        "sha256": sha256_bytes(bundle_bytes),
    }

    full_manifest = dict(manifest)
    full_manifest["raw_sources"] = raw_metadata or {}
    full_manifest["files"] = file_hashes
    manifest_path = splits_dir / "splits_manifest.json"
    manifest_bytes = (json.dumps(full_manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    return manifest_path
