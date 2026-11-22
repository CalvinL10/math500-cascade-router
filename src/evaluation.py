"""Phase 3: MATH-500 offline replay, statistical testing, robustness checks, and plotting.

Replays all atomic baselines (`Atom 1-5`), multi-path Self-Consistency (`Adaptive SC 2->3`
and `3-Path Majority SC` with representative-first symmetric equivalence), true Pre-Gen
Direct Dispatch baselines, Truncation-Only fallback, and Val-locked Router policies under
a unified repaired `src/grading.py` contract (without mutating archival JSONL files).
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, roc_curve

from .grading import classify_response, grade_answer
from .router import (
    CostAwareRouter,
    ONLINE_FEATURE_NAMES,
    compute_curve_auac,
    evaluate_calibration_methods,
    extract_feature_matrix,
    lock_val_thresholds,
    run_dev_split_stability_check,
)


def _read_jsonl_strict(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load JSONL records strictly, failing fast on malformed JSON or duplicate `unique_id`."""

    if not jsonl_path.is_file():
        raise FileNotFoundError(f"Missing JSONL file: {jsonl_path}")
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON in {jsonl_path} at line {line_no}: {exc}"
                ) from exc
            uid = rec.get("unique_id")
            if uid is not None:
                uid_str = str(uid)
                if uid_str in seen_ids:
                    raise ValueError(
                        f"Duplicate unique_id '{uid_str}' in {jsonl_path} at line {line_no}"
                    )
                seen_ids.add(uid_str)
            records.append(rec)
    return records


def load_latency_calibration(jsonl_path: Path) -> dict[str, dict[str, float]]:
    """Aggregate calibrated single-stream (`bs=1`) throughput from `latency_calibration_v2.jsonl`."""

    rows = [
        json.loads(line)
        for line in jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in rows:
        grouped[str(rec["mode_label"])].append(rec)

    summary: dict[str, dict[str, float]] = {}
    for mode, items in grouped.items():
        total_tok = float(sum(float(r["out_tokens"]) for r in items))
        total_lat = float(sum(float(r["bs1_latency_s"]) for r in items))
        tps = total_tok / total_lat if total_lat > 0 else 0.0
        summary[mode] = {
            "count": float(len(items)),
            "total_out_tokens": total_tok,
            "total_bs1_latency_s": total_lat,
            "tokens_per_sec": tps,
            "ms_per_token": (1000.0 / tps) if tps > 0 else 0.0,
            "mean_bs1_latency_s": total_lat / len(items),
        }
    return summary


def _regrade_records_in_memory(
    split_name: str,
    atom_name: str,
    recs: list[dict[str, Any]],
    gt_by_uid: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Re-grade records using current `classify_response` without mutating disk files."""

    regraded: list[dict[str, Any]] = []
    diffs: list[dict[str, Any]] = []

    for rec in recs:
        r = dict(rec)
        uid = str(r["unique_id"])
        gt = str(r.get("ground_truth", gt_by_uid[uid]))

        if atom_name == "atom2":
            g = classify_response(
                str(r["refine_text"]),
                gt,
                truncated=bool(r["refine_truncated"]),
                strip_thinking=False,
            )
            if g.correct != bool(r["refine_correct"]) or g.status != r["refine_status"]:
                diffs.append(
                    {
                        "split": split_name,
                        "atom": atom_name,
                        "unique_id": uid,
                        "sample_idx": -1,
                        "old_status": str(r["refine_status"]),
                        "new_status": str(g.status),
                        "old_correct": bool(r["refine_correct"]),
                        "new_correct": bool(g.correct),
                        "extracted_answer": g.extracted_answer,
                        "ground_truth": gt,
                    }
                )
            r["refine_status"] = g.status
            r["refine_correct"] = g.correct
            r["refine_extracted_answer"] = g.extracted_answer
            r["refine_has_boxed"] = g.has_boxed
            r["refine_extracted_correct_ignoring_truncation"] = (
                g.extracted_correct_ignoring_truncation
            )
        elif atom_name == "atom3":
            new_samples: list[dict[str, Any]] = []
            for s in r["samples"]:
                sc = dict(s)
                g = classify_response(
                    str(sc["response_text"]),
                    gt,
                    truncated=bool(sc["truncated"]),
                    strip_thinking=False,
                )
                if g.correct != bool(sc["correct"]) or g.status != sc["status"]:
                    diffs.append(
                        {
                            "split": split_name,
                            "atom": atom_name,
                            "unique_id": uid,
                            "sample_idx": int(sc["sample_idx"]),
                            "old_status": str(sc["status"]),
                            "new_status": str(g.status),
                            "old_correct": bool(sc["correct"]),
                            "new_correct": bool(g.correct),
                            "extracted_answer": g.extracted_answer,
                            "ground_truth": gt,
                        }
                    )
                sc["status"] = g.status
                sc["correct"] = g.correct
                sc["extracted_answer"] = g.extracted_answer
                sc["has_boxed"] = g.has_boxed
                sc["extracted_correct_ignoring_truncation"] = (
                    g.extracted_correct_ignoring_truncation
                )
                new_samples.append(sc)
            r["samples"] = new_samples
        elif atom_name == "atom5":
            g = classify_response(
                str(r["final_body"]),
                gt,
                truncated=bool(r["truncated"]),
                strip_thinking=True,
                think_closed=bool(r["think_closed"]),
            )
            if g.correct != bool(r["correct"]) or g.status != r["status"]:
                diffs.append(
                    {
                        "split": split_name,
                        "atom": atom_name,
                        "unique_id": uid,
                        "sample_idx": -1,
                        "old_status": str(r["status"]),
                        "new_status": str(g.status),
                        "old_correct": bool(r["correct"]),
                        "new_correct": bool(g.correct),
                        "extracted_answer": g.extracted_answer,
                        "ground_truth": gt,
                    }
                )
            r["status"] = g.status
            r["correct"] = g.correct
            r["extracted_answer"] = g.extracted_answer
            r["has_boxed"] = g.has_boxed
            r["extracted_correct_ignoring_truncation"] = (
                g.extracted_correct_ignoring_truncation
            )
        else:
            g = classify_response(
                str(r["response_text"]),
                gt,
                truncated=bool(r["truncated"]),
                strip_thinking=False,
            )
            if g.correct != bool(r["correct"]) or g.status != r["status"]:
                diffs.append(
                    {
                        "split": split_name,
                        "atom": atom_name,
                        "unique_id": uid,
                        "sample_idx": -1,
                        "old_status": str(r["status"]),
                        "new_status": str(g.status),
                        "old_correct": bool(r["correct"]),
                        "new_correct": bool(g.correct),
                        "extracted_answer": g.extracted_answer,
                        "ground_truth": gt,
                    }
                )
            r["status"] = g.status
            r["correct"] = g.correct
            r["extracted_answer"] = g.extracted_answer
            r["has_boxed"] = g.has_boxed
            r["extracted_correct_ignoring_truncation"] = (
                g.extracted_correct_ignoring_truncation
            )

        regraded.append(r)

    return regraded, diffs


EXPECTED_SPLIT_COUNTS: dict[str, int] = {
    "pilot_50": 50,
    "router_train_600": 600,
    "router_val_200": 200,
    "math500_test_500": 500,
}


def load_aligned_split(
    bundle_dir: Path,
    split_name: str,
    *,
    regrade: bool = True,
) -> dict[str, Any]:
    """Load split JSONL files strictly and aligned by sorted `unique_id`."""

    split_dir = bundle_dir / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    def _load_map(filename: str) -> dict[str, dict[str, Any]]:
        path = split_dir / filename
        if not path.is_file():
            return {}
        recs = _read_jsonl_strict(path)
        return {str(r["unique_id"]): r for r in recs}

    a1_map = _load_map("atom1_1.7b_first.jsonl")
    a4_map = _load_map("atom4_8b_first.jsonl")
    uids = sorted(a1_map.keys())
    if split_name in EXPECTED_SPLIT_COUNTS and len(uids) != EXPECTED_SPLIT_COUNTS[split_name]:
        raise ValueError(
            f"Expected {EXPECTED_SPLIT_COUNTS[split_name]} records in {split_name}, got {len(uids)}"
        )
    if set(uids) != set(a4_map.keys()):
        raise ValueError(f"ID mismatch between atom1 and atom4 in {split_name}")

    a1_raw = [a1_map[u] for u in uids]
    a4_raw = [a4_map[u] for u in uids]
    gt_by_uid = {u: str(a1_map[u]["ground_truth"]) for u in uids}

    regrade_diffs: list[dict[str, Any]] = []
    if regrade:
        a1, d1 = _regrade_records_in_memory(split_name, "atom1", a1_raw, gt_by_uid)
        a4, d4 = _regrade_records_in_memory(split_name, "atom4", a4_raw, gt_by_uid)
        regrade_diffs.extend(d1)
        regrade_diffs.extend(d4)
    else:
        a1, a4 = a1_raw, a4_raw

    out: dict[str, Any] = {
        "split_name": split_name,
        "uids": uids,
        "levels": np.array([int(r["level"]) for r in a1], dtype=np.int64),
        "X": extract_feature_matrix(a1, ONLINE_FEATURE_NAMES),
        "y17": np.array([int(bool(r["correct"])) for r in a1], dtype=np.int64),
        "y8": np.array([int(bool(r["correct"])) for r in a4], dtype=np.int64),
        "y17_raw": np.array([int(bool(r["correct"])) for r in a1_raw], dtype=np.int64),
        "y8_raw": np.array([int(bool(r["correct"])) for r in a4_raw], dtype=np.int64),
        "trunc17": np.array([bool(r["truncated"]) for r in a1], dtype=bool),
        "trunc8": np.array([bool(r["truncated"]) for r in a4], dtype=bool),
        "tok17": np.array([float(r["out_tokens"]) for r in a1], dtype=np.float64),
        "tok8": np.array([float(r["out_tokens"]) for r in a4], dtype=np.float64),
        "bat17": np.array([float(r["latency_s"]) for r in a1], dtype=np.float64),
        "bat8": np.array([float(r["latency_s"]) for r in a4], dtype=np.float64),
        "atom1": a1,
        "atom4": a4,
        "regrade_diffs": regrade_diffs,
    }

    for key, fname in [
        ("atom2", "atom2_1.7b_self_refine.jsonl"),
        ("atom3", "atom3_1.7b_samples3.jsonl"),
        ("atom5", "atom5_1.7b_thinking.jsonl"),
    ]:
        sub_map = _load_map(fname)
        if sub_map:
            if set(uids) != set(sub_map.keys()):
                raise ValueError(f"ID mismatch between atom1 and {fname} in {split_name}")
            raw_list = [sub_map[u] for u in uids]
            if regrade:
                re_list, d_sub = _regrade_records_in_memory(
                    split_name, key, raw_list, gt_by_uid
                )
                out[key] = re_list
                regrade_diffs.extend(d_sub)
            else:
                out[key] = raw_list

    return out


class _SymmetricEquivalenceCache:
    """Memoized symmetric answer equivalence checker (never accesses ground truth)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], bool] = {}

    def equal(self, ans_a: str | None, ans_b: str | None) -> bool:
        if ans_a is None or ans_b is None:
            return False
        sa = str(ans_a).strip()
        sb = str(ans_b).strip()
        if not sa or not sb:
            return False
        if sa == sb:
            return True
        key = (sa, sb) if sa <= sb else (sb, sa)
        if key not in self._cache:
            self._cache[key] = bool(grade_answer(sa, sb) and grade_answer(sb, sa))
        return self._cache[key]


def _majority_vote_cluster(
    samples: Sequence[dict[str, Any]],
    ground_truth: str,
    eq_cache: _SymmetricEquivalenceCache,
) -> tuple[bool, str, str | None]:
    """Cluster valid answers symmetrically, pick winner, and grade representative once."""

    valid_samples = [
        s
        for s in samples
        if (not bool(s["truncated"])) and (s.get("extracted_answer") is not None)
    ]
    if not valid_samples:
        all_trunc = all(bool(s["truncated"]) for s in samples)
        return False, ("truncated" if all_trunc else "invalid"), None

    clusters: list[dict[str, Any]] = []
    for s in valid_samples:
        ans = str(s["extracted_answer"])
        placed = False
        for cl in clusters:
            if eq_cache.equal(ans, cl["rep_ans"]):
                cl["count"] += 1
                placed = True
                break
        if not placed:
            clusters.append({"rep_ans": ans, "count": 1})

    winner = max(clusters, key=lambda cl: cl["count"])
    rep_ans = str(winner["rep_ans"])
    is_corr = bool(grade_answer(rep_ans, ground_truth))
    return is_corr, ("correct" if is_corr else "wrong"), rep_ans


def evaluate_atomic_baselines(
    test_data: dict[str, Any],
    tps_17b: float,
    tps_8b: float,
    tps_think: float,
) -> dict[str, dict[str, Any]]:
    """Compute aligned 500-item metrics and arrays for all Atom 1-5 baselines."""

    a1 = test_data["atom1"]
    a2 = test_data["atom2"]
    a3 = test_data["atom3"]
    a4 = test_data["atom4"]
    a5 = test_data["atom5"]
    n = len(a1)
    eq_cache = _SymmetricEquivalenceCache()

    def _pack(
        label: str,
        correct_arr: np.ndarray,
        statuses: list[str],
        tokens_arr: np.ndarray,
        bs1_arr: np.ndarray,
        bat_arr: np.ndarray,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        counts = dict(Counter(statuses))
        res: dict[str, Any] = {
            "label": label,
            "count": int(n),
            "accuracy": float(correct_arr.mean()),
            "correct_count": int(correct_arr.sum()),
            "statuses": {
                k: int(counts.get(k, 0))
                for k in ("correct", "wrong", "invalid", "truncated")
            },
            "truncated_rate": float(counts.get("truncated", 0) / n),
            "mean_tokens": float(tokens_arr.mean()),
            "mean_bs1_latency_s": float(bs1_arr.mean()),
            "mean_bat_latency_s": float(bat_arr.mean()),
            "correct_arr": correct_arr.astype(np.int64),
            "statuses_list": statuses,
            "tokens_arr": tokens_arr,
            "bs1_arr": bs1_arr,
            "bat_arr": bat_arr,
        }
        if extra:
            res.update(extra)
        return res

    # Atom 1: 1.7B Non-Think
    y1 = np.array([int(bool(r["correct"])) for r in a1], dtype=np.int64)
    st1 = [str(r["status"]) for r in a1]
    tok1 = np.array([float(r["out_tokens"]) for r in a1], dtype=np.float64)
    bs1_1 = tok1 / float(tps_17b)
    bat_1 = np.array([float(r["latency_s"]) for r in a1], dtype=np.float64)

    # Atom 2: 1.7B Self-Refine (includes Atom 1 draft + critique + refine)
    y2 = np.array([int(bool(r["refine_correct"])) for r in a2], dtype=np.int64)
    st2 = [str(r["refine_status"]) for r in a2]
    tok2 = tok1 + np.array(
        [float(r["critique_out_tokens"]) + float(r["refine_out_tokens"]) for r in a2],
        dtype=np.float64,
    )
    bs1_2 = tok2 / float(tps_17b)
    bat_2 = bat_1 + np.array(
        [float(r["critique_latency_s"]) + float(r["refine_latency_s"]) for r in a2],
        dtype=np.float64,
    )

    completed_wrong = [(r1, r2) for r1, r2 in zip(a1, a2) if r1["status"] == "wrong"]
    completed_all = [(r1, r2) for r1, r2 in zip(a1, a2) if r1["status"] != "truncated"]
    truncated_a1 = [(r1, r2) for r1, r2 in zip(a1, a2) if r1["status"] == "truncated"]
    no_err_all = sum("no errors found" in r["critique_text"].lower() for r in a2)
    no_err_completed_wrong = sum(
        "no errors found" in r2["critique_text"].lower() for _, r2 in completed_wrong
    )
    unchanged_completed_strings = sum(
        str(r1.get("extracted_answer") or "") == str(r2.get("refine_extracted_answer") or "")
        for r1, r2 in completed_all
    )
    atom2_extra = {
        "transition_counts": {
            "right_to_right": int(sum(r1["correct"] and r2["refine_correct"] for r1, r2 in zip(a1, a2))),
            "right_to_wrong": int(sum(r1["correct"] and not r2["refine_correct"] for r1, r2 in zip(a1, a2))),
            "wrong_to_right": int(sum((not r1["correct"]) and r2["refine_correct"] for r1, r2 in zip(a1, a2))),
            "wrong_to_wrong": int(sum((not r1["correct"]) and (not r2["refine_correct"]) for r1, r2 in zip(a1, a2))),
        },
        "completed_total": int(len(completed_all)),
        "completed_unchanged_answer_strings": int(unchanged_completed_strings),
        "completed_wrong_total": int(len(completed_wrong)),
        "completed_wrong_fixed": int(sum(r2["refine_correct"] for _, r2 in completed_wrong)),
        "truncated_total": int(len(truncated_a1)),
        "truncated_fixed": int(sum(r2["refine_correct"] for _, r2 in truncated_a1)),
        "critique_no_errors_all": int(no_err_all),
        "critique_no_errors_completed_wrong": int(no_err_completed_wrong),
        "critique_boxed_count": int(sum(r"\boxed" in r["critique_text"] for r in a2)),
    }

    # Atom 3a (Adaptive SC: 2->3 independent paths) & Atom 3b (3-Path Majority SC)
    y3a = np.zeros(n, dtype=np.int64)
    st3a: list[str] = []
    tok3a = np.zeros(n, dtype=np.float64)
    bat3a = np.zeros(n, dtype=np.float64)
    stop2_flags = np.zeros(n, dtype=np.int64)
    stop2_correct = 0

    y3b = np.zeros(n, dtype=np.int64)
    st3b: list[str] = []
    tok3b = np.zeros(n, dtype=np.float64)
    bat3b = np.zeros(n, dtype=np.float64)
    pass3 = np.zeros(n, dtype=np.int64)

    for i, rec in enumerate(a3):
        samples = rec["samples"]
        gt = str(rec["ground_truth"])
        s0, s1, s2 = samples[0], samples[1], samples[2]

        both_valid = (
            (not bool(s0["truncated"]))
            and (not bool(s1["truncated"]))
            and (s0.get("extracted_answer") is not None)
            and (s1.get("extracted_answer") is not None)
        )
        agree_early = both_valid and eq_cache.equal(
            s0["extracted_answer"], s1["extracted_answer"]
        )
        if agree_early:
            stop2_flags[i] = 1
            corr_a = bool(grade_answer(str(s0["extracted_answer"]), gt))
            if corr_a:
                stop2_correct += 1
            st_a = "correct" if corr_a else "wrong"
            tok_a = float(s0["out_tokens"]) + float(s1["out_tokens"])
            bat_a = float(s0["latency_s"]) + float(s1["latency_s"])
        else:
            corr_a, st_a, _ = _majority_vote_cluster(samples, gt, eq_cache)
            tok_a = sum(float(s["out_tokens"]) for s in samples)
            bat_a = sum(float(s["latency_s"]) for s in samples)

        y3a[i] = int(corr_a)
        st3a.append(st_a)
        tok3a[i] = tok_a
        bat3a[i] = bat_a

        corr_b, st_b, _ = _majority_vote_cluster(samples, gt, eq_cache)
        y3b[i] = int(corr_b)
        st3b.append(st_b)
        tok3b[i] = sum(float(s["out_tokens"]) for s in samples)
        bat3b[i] = sum(float(s["latency_s"]) for s in samples)
        pass3[i] = int(any(bool(s["correct"]) for s in samples))

    bs1_3a = tok3a / float(tps_17b)
    bs1_3b = tok3b / float(tps_17b)
    atom3a_extra = {
        "stop_at_2_count": int(stop2_flags.sum()),
        "stop_at_2_rate": float(stop2_flags.mean()),
        "stop_at_2_correct": int(stop2_correct),
        "stop_at_2_both_wrong": int(stop2_flags.sum() - stop2_correct),
    }
    atom3b_extra = {
        "pass_at_3": float(pass3.mean()),
        "sample_single_accs": [
            float(np.mean([bool(r["samples"][k]["correct"]) for r in a3]))
            for k in range(3)
        ],
    }

    # Atom 4: 8B Non-Think Direct (and 100% Cascade 1.7B -> 8B)
    y4 = np.array([int(bool(r["correct"])) for r in a4], dtype=np.int64)
    st4 = [str(r["status"]) for r in a4]
    tok4 = np.array([float(r["out_tokens"]) for r in a4], dtype=np.float64)
    bs1_4 = tok4 / float(tps_8b)
    bat_4 = np.array([float(r["latency_s"]) for r in a4], dtype=np.float64)

    # Atom 5: 1.7B Thinking (max_new_tokens=4096) with exact 289/37/174 decomposition
    y5 = np.array([int(bool(r["correct"])) for r in a5], dtype=np.int64)
    st5 = [str(r["status"]) for r in a5]
    tok5 = np.array([float(r["out_tokens"]) for r in a5], dtype=np.float64)
    bs1_5 = tok5 / float(tps_think)
    bat_5 = np.array([float(r["latency_s"]) for r in a5], dtype=np.float64)

    output_completed_mask = np.array([not bool(r["truncated"]) for r in a5], dtype=bool)
    think_closed_mask = np.array([bool(r["think_closed"]) for r in a5], dtype=bool)
    think_closed_body_trunc_mask = think_closed_mask & (~output_completed_mask)
    think_unclosed_mask = ~think_closed_mask
    atom5_extra = {
        "output_completed_count": int(output_completed_mask.sum()),
        "output_completed_correct": int(y5[output_completed_mask].sum()),
        "conditional_acc_on_completed_output": (
            float(y5[output_completed_mask].mean()) if output_completed_mask.any() else 0.0
        ),
        "think_closed_count": int(think_closed_mask.sum()),
        "think_closed_but_body_truncated_count": int(think_closed_body_trunc_mask.sum()),
        "think_unclosed_truncated_count": int(think_unclosed_mask.sum()),
    }

    return {
        "atom1_1.7b_nonthink": _pack("Atom 1: 1.7B Non-Think (1-Pass)", y1, st1, tok1, bs1_1, bat_1),
        "atom2_1.7b_self_refine": _pack("Atom 2: 1.7B Self-Refine", y2, st2, tok2, bs1_2, bat_2, atom2_extra),
        "atom3a_1.7b_adaptive_sc": _pack("Atom 3a: 1.7B Adaptive SC (2->3 paths)", y3a, st3a, tok3a, bs1_3a, bat3a, atom3a_extra),
        "atom3b_1.7b_majority_sc3": _pack("Atom 3b: 1.7B 3-Path Majority SC", y3b, st3b, tok3b, bs1_3b, bat3b, atom3b_extra),
        "atom4_8b_direct": _pack("Atom 4: 8B Non-Think (Direct)", y4, st4, tok4, bs1_4, bat_4),
        "cascade_100pct": _pack(
            "100% Cascade (1.7B -> 8B)",
            y4,
            st4,
            tok1 + tok4,
            bs1_1 + bs1_4,
            bat_1 + bat_4,
        ),
        "atom5_1.7b_thinking": _pack("Atom 5: 1.7B Thinking (max=4096)", y5, st5, tok5, bs1_5, bat_5, atom5_extra),
    }


def trace_routing_curve(
    scores: np.ndarray,
    y17: np.ndarray,
    y8: np.ndarray,
    tok17: np.ndarray,
    tok8: np.ndarray,
    bat17: np.ndarray,
    bat8: np.ndarray,
    tps_17b: float,
    tps_8b: float,
    *,
    dispatch_mode: str = "cascade",
) -> dict[str, Any]:
    """Sweep `k = 0..N` under either `'cascade'` (`C17 + I*C8`) or `'direct'` (`(1-I)*C17 + I*C8`)."""

    if dispatch_mode not in {"cascade", "direct"}:
        raise ValueError(f"Unsupported dispatch_mode: {dispatch_mode}")

    n = len(scores)
    lat17 = tok17 / float(tps_17b)
    lat8 = tok8 / float(tps_8b)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="mergesort")

    esc_fracs = np.arange(n + 1, dtype=np.float64) / float(n)
    accs = np.empty(n + 1, dtype=np.float64)
    toks = np.empty(n + 1, dtype=np.float64)
    bs1_lats = np.empty(n + 1, dtype=np.float64)
    bat_lats = np.empty(n + 1, dtype=np.float64)

    cur_y = np.asarray(y17, dtype=np.float64).copy()
    cur_t = np.asarray(tok17, dtype=np.float64).copy()
    cur_l = lat17.copy()
    cur_b = np.asarray(bat17, dtype=np.float64).copy()

    accs[0], toks[0], bs1_lats[0], bat_lats[0] = (
        float(cur_y.mean()),
        float(cur_t.mean()),
        float(cur_l.mean()),
        float(cur_b.mean()),
    )

    for k in range(1, n + 1):
        idx = int(order[k - 1])
        cur_y[idx] = float(y8[idx])
        if dispatch_mode == "direct":
            cur_t[idx] = float(tok8[idx])
            cur_l[idx] = float(lat8[idx])
            cur_b[idx] = float(bat8[idx])
        else:
            cur_t[idx] += float(tok8[idx])
            cur_l[idx] += float(lat8[idx])
            cur_b[idx] += float(bat8[idx])
        accs[k] = float(cur_y.mean())
        toks[k] = float(cur_t.mean())
        bs1_lats[k] = float(cur_l.mean())
        bat_lats[k] = float(cur_b.mean())

    lat_auac, lat_excess = compute_curve_auac(bs1_lats, accs)
    tok_auac, tok_excess = compute_curve_auac(
        toks, accs, allow_non_monotone=(dispatch_mode != "direct")
    )

    return {
        "dispatch_mode": dispatch_mode,
        "esc_frac": esc_fracs,
        "accuracy": accs,
        "mean_tokens": toks,
        "mean_bs1_latency_s": bs1_lats,
        "mean_bat_latency_s": bat_lats,
        "lat_auac": lat_auac,
        "lat_excess_over_endpoints_line": lat_excess,
        "tok_auac": tok_auac,
        "tok_excess_over_endpoints_line": tok_excess,
    }


def evaluate_val_locked_mask(
    mask: np.ndarray,
    test_data: dict[str, Any],
    tps_17b: float,
    tps_8b: float,
    *,
    dispatch_mode: str = "cascade",
    tau: float | None = None,
) -> dict[str, Any]:
    """Evaluate a boolean escalation mask under `'cascade'` or `'direct'` cost accounting."""

    m = np.asarray(mask, dtype=bool)
    y_ro = np.where(m, test_data["y8"], test_data["y17"]).astype(np.int64)
    lat17 = test_data["tok17"] / float(tps_17b)
    lat8 = test_data["tok8"] / float(tps_8b)

    if dispatch_mode == "direct":
        tok_ro = np.where(m, test_data["tok8"], test_data["tok17"])
        bs1_ro = np.where(m, lat8, lat17)
        bat_ro = np.where(m, test_data["bat8"], test_data["bat17"])
    else:
        tok_ro = test_data["tok17"] + np.where(m, test_data["tok8"], 0.0)
        bs1_ro = lat17 + np.where(m, lat8, 0.0)
        bat_ro = test_data["bat17"] + np.where(m, test_data["bat8"], 0.0)

    statuses = [
        str(test_data["atom4"][i]["status"] if m[i] else test_data["atom1"][i]["status"])
        for i in range(len(m))
    ]
    counts = dict(Counter(statuses))
    n = len(m)
    return {
        "dispatch_mode": dispatch_mode,
        "tau": float(tau) if tau is not None else None,
        "escalation_count": int(m.sum()),
        "escalation_rate": float(m.mean()),
        "accuracy": float(y_ro.mean()),
        "correct_count": int(y_ro.sum()),
        "statuses": {
            k: int(counts.get(k, 0))
            for k in ("correct", "wrong", "invalid", "truncated")
        },
        "truncated_rate": float(counts.get("truncated", 0) / n),
        "mean_tokens": float(tok_ro.mean()),
        "mean_bs1_latency_s": float(bs1_ro.mean()),
        "mean_bat_latency_s": float(bat_ro.mean()),
        "mask": m,
        "correct_arr": y_ro,
        "statuses_list": statuses,
        "tokens_arr": tok_ro,
        "bs1_arr": bs1_ro,
        "bat_arr": bat_ro,
    }


def evaluate_val_locked_operating_point(
    test_scores: np.ndarray,
    tau: float,
    test_data: dict[str, Any],
    tps_17b: float,
    tps_8b: float,
    *,
    dispatch_mode: str = "cascade",
) -> dict[str, Any]:
    """Apply a Val-locked threshold `tau` (`score >= tau`) onto `math500_test_500`."""

    mask = np.asarray(test_scores, dtype=np.float64) >= float(tau)
    return evaluate_val_locked_mask(
        mask,
        test_data,
        tps_17b,
        tps_8b,
        dispatch_mode=dispatch_mode,
        tau=tau,
    )


def compute_pareto_envelope(
    points: Sequence[dict[str, Any]],
    *,
    cost_key: str = "mean_bs1_latency_s",
    acc_key: str = "accuracy",
) -> list[dict[str, Any]]:
    """Return strictly non-dominated points sorted by ascending `cost_key`."""

    sorted_pts = sorted(points, key=lambda p: (float(p[cost_key]), -float(p[acc_key])))
    envelope: list[dict[str, Any]] = []
    best_acc = -1.0
    for pt in sorted_pts:
        acc = float(pt[acc_key])
        if acc > best_acc + 1e-12:
            envelope.append(dict(pt))
            best_acc = acc
    return envelope


def paired_bootstrap_test(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_boot: int = 5000,
    seed: int = 550,
) -> dict[str, float]:
    """Compute paired bootstrap mean difference `mean(A - B)` with 95% CI."""

    a = np.asarray(metric_a, dtype=np.float64)
    b = np.asarray(metric_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch in paired bootstrap: {a.shape} vs {b.shape}")
    diff_vec = a - b
    n = len(diff_vec)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot_diffs = diff_vec[idx].mean(axis=1)
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": float(diff_vec.mean()),
        "ci_low": float(np.percentile(boot_diffs, 2.5)),
        "ci_high": float(np.percentile(boot_diffs, 97.5)),
        "boot_pos_share": float(np.mean(boot_diffs > 0.0)),
    }


def paired_bootstrap_matched_random_dispatch(
    y_router: np.ndarray,
    lat_router: np.ndarray,
    y17: np.ndarray,
    y8: np.ndarray,
    lat17: np.ndarray,
    lat8: np.ndarray,
    *,
    mode: str = "direct",
    n_boot: int = 5000,
    seed: int = 550,
) -> dict[str, float]:
    """Paired bootstrap vs Rao-Blackwellized random dispatch matched to Router latency in each resample."""

    n = len(y_router)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))

    y_r_mean = y_router[idx].mean(axis=1)
    l_r_mean = lat_router[idx].mean(axis=1)
    l17_mean = lat17[idx].mean(axis=1)
    l8_mean = lat8[idx].mean(axis=1)
    y17_mean = y17[idx].mean(axis=1)
    y8_mean = y8[idx].mean(axis=1)

    if mode == "direct":
        p_boot = np.clip((l_r_mean - l17_mean) / np.maximum(l8_mean - l17_mean, 1e-9), 0.0, 1.0)
    elif mode == "cascade":
        p_boot = np.clip((l_r_mean - l17_mean) / np.maximum(l8_mean, 1e-9), 0.0, 1.0)
    else:
        raise ValueError(f"Unknown random dispatch mode: {mode}")

    y_rand_boot = (1.0 - p_boot) * y17_mean + p_boot * y8_mean
    boot_diffs = y_r_mean - y_rand_boot

    if mode == "direct":
        p_full = float(np.clip((lat_router.mean() - lat17.mean()) / (lat8.mean() - lat17.mean()), 0.0, 1.0))
    else:
        p_full = float(np.clip((lat_router.mean() - lat17.mean()) / lat8.mean(), 0.0, 1.0))
    y_rand_full = float((1.0 - p_full) * y17.mean() + p_full * y8.mean())

    return {
        "p_matched": p_full,
        "mean_a": float(y_router.mean()),
        "mean_b": y_rand_full,
        "diff": float(y_router.mean() - y_rand_full),
        "ci_low": float(np.percentile(boot_diffs, 2.5)),
        "ci_high": float(np.percentile(boot_diffs, 97.5)),
        "boot_pos_share": float(np.mean(boot_diffs > 0.0)),
    }


def mcnemar_test(y_a: np.ndarray, y_b: np.ndarray) -> dict[str, float | int]:
    """Compute exact two-sided McNemar binomial test on paired binary correctness."""

    a = np.asarray(y_a, dtype=np.int64)
    b = np.asarray(y_b, dtype=np.int64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch in McNemar test: {a.shape} vs {b.shape}")
    n_a_only = int(np.sum((a == 1) & (b == 0)))
    n_b_only = int(np.sum((a == 0) & (b == 1)))
    n_disc = n_a_only + n_b_only
    pvalue = (
        float(binomtest(min(n_a_only, n_b_only), n_disc, 0.5).pvalue)
        if n_disc > 0
        else 1.0
    )
    return {
        "n_a_only": n_a_only,
        "n_b_only": n_b_only,
        "n_discordant": n_disc,
        "p_value": pvalue,
    }


def apply_holm_bonferroni(p_values: Sequence[float | None]) -> list[float | None]:
    """Apply Holm-Bonferroni step-down adjustment across non-None p-values."""

    indexed = [(i, float(p)) for i, p in enumerate(p_values) if p is not None]
    out: list[float | None] = [None] * len(p_values)
    if not indexed:
        return out
    indexed.sort(key=lambda pair: pair[1])
    m = len(indexed)
    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(indexed):
        adj = min(1.0, (m - rank) * p_val)
        running_max = max(running_max, adj)
        out[orig_idx] = running_max
    return out


def compute_throughput_stress_test(
    router_pt: dict[str, Any],
    a1: dict[str, Any],
    a3a: dict[str, Any],
    a4: dict[str, Any],
    *,
    tps_8b: float,
    observed_speed_ratio: float,
    ratios: Sequence[float] = (1.5, 2.0, 3.0, 4.0, 5.0, 6.0),
) -> dict[str, Any]:
    """Evaluate Router estimated latency vs `8B Direct` and `Adaptive SC` across speed ratios `r = TPS_1.7B / TPS_8B`."""

    tok17_r = float(a1["mean_tokens"])
    tok8_added_r = float(router_pt["mean_tokens"] - a1["mean_tokens"])
    tok8_direct = float(a4["mean_tokens"])
    tok17_sc = float(a3a["mean_tokens"])

    # Router faster than Direct8: tok17_r / (r * tps8) + tok8_added_r / tps8 < tok8_direct / tps8
    denom_8b = tok8_direct - tok8_added_r
    r_crit_8b = float(tok17_r / denom_8b) if denom_8b > 0 else float("inf")

    # Router faster than Adaptive SC: tok17_r / (r * tps8) + tok8_added_r / tps8 < tok17_sc / (r * tps8)
    r_crit_sc = float((tok17_sc - tok17_r) / tok8_added_r) if tok8_added_r > 0 else float("inf")

    sweep: list[dict[str, float | bool]] = []
    for r_val in [float(x) for x in ratios] + [float(observed_speed_ratio)]:
        tps17_hyp = r_val * float(tps_8b)
        lat_r = tok17_r / tps17_hyp + tok8_added_r / float(tps_8b)
        lat_8 = tok8_direct / float(tps_8b)
        lat_sc = tok17_sc / tps17_hyp
        sweep.append(
            {
                "speed_ratio": float(r_val),
                "is_observed_l4": bool(abs(r_val - observed_speed_ratio) < 1e-9),
                "router_est_lat_s": float(lat_r),
                "direct8_est_lat_s": float(lat_8),
                "adaptive_sc_est_lat_s": float(lat_sc),
                "faster_than_direct8": bool(lat_r < lat_8),
                "faster_than_adaptive_sc": bool(lat_r < lat_sc),
            }
        )
    sweep.sort(key=lambda row: float(row["speed_ratio"]))

    return {
        "observed_speed_ratio": float(observed_speed_ratio),
        "r_crit_faster_than_direct8": r_crit_8b,
        "r_crit_faster_than_adaptive_sc": r_crit_sc,
        "sweep": sweep,
    }


def compute_decile_analysis(
    p17: np.ndarray,
    p8: np.ndarray,
    test_data: dict[str, Any],
    n_deciles: int = 10,
) -> list[dict[str, float | int]]:
    """Bucket `math500_test_500` by ascending `p17` confidence into 10 equal deciles (`n=50`)."""

    order = np.argsort(np.asarray(p17, dtype=np.float64), kind="mergesort")
    chunks = np.array_split(order, n_deciles)
    rows: list[dict[str, float | int]] = []
    fmt_valid = test_data["X"][:, ONLINE_FEATURE_NAMES.index("format_valid")]
    min_lp = test_data["X"][:, ONLINE_FEATURE_NAMES.index("min_token_logprob")]
    for d_idx, idx in enumerate(chunks, start=1):
        acc17 = float(test_data["y17"][idx].mean())
        acc8 = float(test_data["y8"][idx].mean())
        rows.append(
            {
                "decile": int(d_idx),
                "count": int(len(idx)),
                "mean_pred_p17": float(p17[idx].mean()),
                "mean_pred_p8": float(p8[idx].mean()),
                "actual_17b_acc": acc17,
                "actual_8b_acc": acc8,
                "net_gain": acc8 - acc17,
                "mean_tok17": float(test_data["tok17"][idx].mean()),
                "mean_tok8": float(test_data["tok8"][idx].mean()),
                "format_valid_rate": float(fmt_valid[idx].mean()),
                "mean_min_logprob": float(min_lp[idx].mean()),
                "mean_level": float(test_data["levels"][idx].mean()),
            }
        )
    return rows


def extract_case_studies(
    test_data: dict[str, Any],
    comp: dict[str, np.ndarray],
    router_mask_val20: np.ndarray,
    conf_matched_mask: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Select concrete, illustrative examples from `math500_test_500` for reporting."""

    a1 = test_data["atom1"]
    a2 = test_data["atom2"]
    a4 = test_data["atom4"]
    p17 = comp["p17"]
    p8 = comp["p8"]
    pred_lat8 = comp["pred_lat8"]

    def _summarize(idx: int) -> dict[str, Any]:
        return {
            "unique_id": str(a1[idx]["unique_id"]),
            "level": int(a1[idx]["level"]),
            "subject": str(a1[idx].get("subject", "")),
            "problem_preview": str(a1[idx]["problem"])[:260],
            "ground_truth": str(a1[idx]["ground_truth"]),
            "features": a1[idx]["features"],
            "pred_p17": float(p17[idx]),
            "pred_p8": float(p8[idx]),
            "pred_lat8_s": float(pred_lat8[idx]),
            "atom1_status": str(a1[idx]["status"]),
            "atom1_answer": a1[idx]["extracted_answer"],
            "atom1_out_tokens": int(a1[idx]["out_tokens"]),
            "atom4_status": str(a4[idx]["status"]),
            "atom4_answer": a4[idx]["extracted_answer"],
            "atom4_out_tokens": int(a4[idx]["out_tokens"]),
            "atom2_refine_status": str(a2[idx]["refine_status"]),
            "atom2_refine_answer": a2[idx]["refine_extracted_answer"],
            "atom2_critique_preview": str(a2[idx]["critique_text"])[:260],
        }

    n = len(a1)
    kept_candidates = [
        i for i in range(n)
        if (not router_mask_val20[i]) and a1[i]["correct"] and a4[i]["correct"]
    ]
    kept_idx = max(kept_candidates, key=lambda i: p17[i])

    completed_fixed = [
        i for i in range(n)
        if router_mask_val20[i] and a1[i]["status"] == "wrong" and a4[i]["correct"]
    ]
    fixed_idx = max(completed_fixed, key=lambda i: p8[i] - p17[i])

    trunc_rescued = [
        i for i in range(n)
        if router_mask_val20[i] and a1[i]["status"] == "truncated" and a4[i]["correct"]
    ]
    trunc_idx = max(trunc_rescued, key=lambda i: p8[i])

    avoided_shared = [
        i for i in range(n)
        if (not router_mask_val20[i])
        and conf_matched_mask[i]
        and (not a1[i]["correct"])
        and (not a4[i]["correct"])
    ]
    avoided_idx = (
        max(avoided_shared, key=lambda i: pred_lat8[i])
        if avoided_shared
        else max(
            [
                i
                for i in range(n)
                if (not router_mask_val20[i])
                and (not a1[i]["correct"])
                and (not a4[i]["correct"])
            ],
            key=lambda i: pred_lat8[i],
        )
    )

    sycophancy = [
        i for i in range(n)
        if a1[i]["status"] == "wrong"
        and (not a2[i]["refine_correct"])
        and "no errors found" in a2[i]["critique_text"].lower()
    ]
    syc_idx = sycophancy[0]

    return {
        "kept_easy_correct": _summarize(kept_idx),
        "escalated_completed_fixed": _summarize(fixed_idx),
        "escalated_truncated_rescued": _summarize(trunc_idx),
        "avoided_shared_failure": _summarize(avoided_idx),
        "self_refine_false_confirmation": _summarize(syc_idx),
    }


def _relaxed_format_grade(pred_text: str | None, gt_text: str | None) -> bool:
    """Upper-bound formatting check (tuple LHS & MC suffix stripping) for sensitivity auditing."""

    if pred_text is None or gt_text is None:
        return False
    if grade_answer(pred_text, gt_text):
        return True

    def _clean_variants(raw: str) -> list[str]:
        s = str(raw).strip()
        variants = [s]
        m_tuple = re.match(r"^\([A-Za-z,\s]+\)\s*=\s*(\(.+\))$", s)
        if m_tuple:
            variants.append(m_tuple.group(1).strip())
        m_mc = re.match(r"^(?:\\text\s*\{)?\(([A-E])\)\s+.+$", s)
        if m_mc:
            variants.append(f"({m_mc.group(1)})")
        return variants

    for pv in _clean_variants(pred_text):
        for gv in _clean_variants(gt_text):
            if grade_answer(pv, gv):
                return True
    return False


def audit_grading_formatting_sensitivity(
    bundle_dir: Path,
) -> dict[str, Any]:
    """Compare raw bundle vs unified fixed grader vs relaxed upper-bound formatting across splits & Atom 3."""

    audit_splits: dict[str, Any] = {}
    targets = [
        ("router_train_600", ["atom1_1.7b_first.jsonl", "atom4_8b_first.jsonl"]),
        ("router_val_200", ["atom1_1.7b_first.jsonl", "atom4_8b_first.jsonl"]),
        (
            "math500_test_500",
            [
                "atom1_1.7b_first.jsonl",
                "atom2_1.7b_self_refine.jsonl",
                "atom3_1.7b_samples3.jsonl",
                "atom4_8b_first.jsonl",
                "atom5_1.7b_thinking.jsonl",
            ],
        ),
    ]

    for split_name, files in targets:
        split_dir = bundle_dir / split_name
        a1_gt = {
            str(r["unique_id"]): str(r["ground_truth"])
            for r in _read_jsonl_strict(split_dir / "atom1_1.7b_first.jsonl")
        }
        split_res: dict[str, Any] = {}
        for fname in files:
            recs = _read_jsonl_strict(split_dir / fname)
            raw_corr = 0
            fixed_corr = 0
            relaxed_corr = 0
            total_units = 0
            fixed_flipped_ids: list[str] = []
            relaxed_flipped_ids: list[str] = []

            for r in recs:
                uid = str(r["unique_id"])
                gt = str(r.get("ground_truth", a1_gt[uid]))
                if fname.startswith("atom3"):
                    for s in r["samples"]:
                        total_units += 1
                        is_raw = bool(s["correct"])
                        g = classify_response(
                            str(s["response_text"]),
                            gt,
                            truncated=bool(s["truncated"]),
                            strip_thinking=False,
                        )
                        raw_corr += int(is_raw)
                        fixed_corr += int(g.correct)
                        if g.correct != is_raw:
                            fixed_flipped_ids.append(f"{uid}#s{s['sample_idx']}")
                        if g.correct:
                            relaxed_corr += 1
                        elif g.status == "wrong" and _relaxed_format_grade(g.extracted_answer, gt):
                            relaxed_corr += 1
                            relaxed_flipped_ids.append(f"{uid}#s{s['sample_idx']}")
                else:
                    total_units += 1
                    if fname.startswith("atom2"):
                        is_raw = bool(r["refine_correct"])
                        g = classify_response(
                            str(r["refine_text"]),
                            gt,
                            truncated=bool(r["refine_truncated"]),
                            strip_thinking=False,
                        )
                    elif fname.startswith("atom5"):
                        is_raw = bool(r["correct"])
                        g = classify_response(
                            str(r["final_body"]),
                            gt,
                            truncated=bool(r["truncated"]),
                            strip_thinking=True,
                            think_closed=bool(r["think_closed"]),
                        )
                    else:
                        is_raw = bool(r["correct"])
                        g = classify_response(
                            str(r["response_text"]),
                            gt,
                            truncated=bool(r["truncated"]),
                            strip_thinking=False,
                        )
                    raw_corr += int(is_raw)
                    fixed_corr += int(g.correct)
                    if g.correct != is_raw:
                        fixed_flipped_ids.append(uid)
                    if g.correct:
                        relaxed_corr += 1
                    elif g.status == "wrong" and _relaxed_format_grade(g.extracted_answer, gt):
                        relaxed_corr += 1
                        relaxed_flipped_ids.append(uid)

            split_res[fname] = {
                "count": int(total_units),
                "raw_acc": float(raw_corr / total_units),
                "repaired_acc": float(fixed_corr / total_units),
                "relaxed_upper_acc": float(relaxed_corr / total_units),
                "repaired_flipped_count": int(len(fixed_flipped_ids)),
                "repaired_flipped_ids": fixed_flipped_ids,
                "relaxed_extra_flipped_count": int(len(relaxed_flipped_ids)),
                "relaxed_extra_flipped_ids": relaxed_flipped_ids,
            }
        audit_splits[split_name] = split_res
    return audit_splits


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.94,
            "legend.edgecolor": "#D0D7DE",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png_path), "pdf": str(pdf_path)}


def generate_all_figures(
    *,
    figures_dir: Path,
    baselines: dict[str, dict[str, Any]],
    curves: dict[str, dict[str, Any]],
    val_locked_points: dict[str, dict[str, dict[str, Any]]],
    pareto_latency: list[dict[str, Any]],
    pareto_tokens: list[dict[str, Any]],
    test_data: dict[str, Any],
    calib_17b_report: dict[str, Any],
    calib_8b_report: dict[str, Any],
    router: CostAwareRouter,
    deciles: list[dict[str, float | int]],
    stat_tests: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Render and save the 4 audit-hardened publication-grade figures (300 DPI PNG + PDF)."""

    _configure_plot_style()
    paths: dict[str, dict[str, str]] = {}

    a1 = baselines["atom1_1.7b_nonthink"]
    a2 = baselines["atom2_1.7b_self_refine"]
    a3a = baselines["atom3a_1.7b_adaptive_sc"]
    a3b = baselines["atom3b_1.7b_majority_sc3"]
    a4 = baselines["atom4_8b_direct"]
    a5 = baselines["atom5_1.7b_thinking"]
    c100 = baselines["cascade_100pct"]
    trunc_only = val_locked_points["truncation_only"]["fixed"]
    r_v12 = val_locked_points["tier1_trunc_cost_utility"]["0.12"]
    r_v20 = val_locked_points["tier1_trunc_cost_utility"]["0.20"]
    r_v30 = val_locked_points["tier1_trunc_cost_utility"]["0.30"]
    cg_v20 = val_locked_points["conf_gate"]["0.20"]
    cg_k = val_locked_points["conf_gate"]["matched_k_v20"]
    oracle_acc = 100.0 * float(np.mean((test_data["y17"] == 1) | (test_data["y8"] == 1)))

    # ------------------------------------------------------------------
    # Figure 1: Cost-Accuracy Tradeoffs & Non-Dominated Pareto Envelope
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15.4, 6.4), sharey=True)
    c_cu = curves["tier1_trunc_cost_utility"]
    c_cg = curves["conf_gate"]
    c_pre = curves["pre_gen_qlen"]  # true direct-dispatch curve

    panel_specs = [
        (
            "mean_bs1_latency_s",
            "Estimated Serial Latency (s, out_tokens / calibrated L4 TPS)",
            (10.0, 60.5),
            pareto_latency,
        ),
        (
            "mean_tokens",
            "Mean Total Output Tokens per Query (1.7B + 8B raw sum)",
            (560.0, 3300.0),
            pareto_tokens,
        ),
    ]

    for ax, (x_key, xlabel, xlim, env_pts) in zip(axes, panel_specs):
        x_a1 = a1[x_key]
        x_a4 = a4[x_key]
        x_c100 = c100[x_key]
        ax.axhline(oracle_acc, color="#6E7781", linestyle=":", linewidth=1.3, label=f"Oracle Union ({oracle_acc:.1f}%)")
        ax.plot(
            [x_a1, x_a4],
            [100.0 * a1["accuracy"], 100.0 * a4["accuracy"]],
            color="#24292F",
            linestyle="--",
            linewidth=1.6,
            alpha=0.75,
            label="Random Direct Dispatch",
            zorder=2,
        )
        ax.plot(
            [x_a1, x_c100],
            [100.0 * a1["accuracy"], 100.0 * c100["accuracy"]],
            color="#8C959F",
            linestyle=":",
            linewidth=1.5,
            label="Random Cascade",
            zorder=2,
        )
        ax.plot(
            c_pre[x_key],
            100.0 * c_pre["accuracy"],
            color="#9A6700",
            linestyle="-.",
            linewidth=1.7,
            alpha=0.9,
            label="Pre-Gen Direct Router (q_len)",
            zorder=2,
        )
        ax.plot(
            c_cg[x_key],
            100.0 * c_cg["accuracy"],
            color="#0969DA",
            linewidth=1.8,
            alpha=0.85,
            label="Naive ConfGate Cascade (1 - p17)",
            zorder=3,
        )
        ax.plot(
            c_cu[x_key],
            100.0 * c_cu["accuracy"],
            color="#CF222E",
            linewidth=2.4,
            alpha=0.92,
            label="Two-Tier Cost-Aware Router (Ours)",
            zorder=4,
        )

        # Highlight non-dominated Pareto envelope across evaluated points
        ex = [pt[x_key] for pt in env_pts]
        ey = [100.0 * pt["accuracy"] for pt in env_pts]
        ax.step(
            ex,
            ey,
            where="post",
            color="#111827",
            linewidth=1.3,
            alpha=0.45,
            linestyle="-",
            label="Non-Dominated Envelope",
            zorder=3,
        )

        # Highlight Truncation-Only and Val-locked Router operating points
        xt = trunc_only[x_key]
        yt = 100.0 * trunc_only["accuracy"]
        ax.scatter([xt], [yt], color="#D97706", edgecolor="white", s=105, marker="d", zorder=7)
        if x_key == "mean_bs1_latency_s":
            trunc_off = (-86, 12)
        else:
            trunc_off = (-6, -56)
        ax.annotate(
            f"Trunc-Only\n({yt:.1f}%)",
            xy=(xt, yt),
            xytext=trunc_off,
            textcoords="offset points",
            fontsize=8.0,
            fontweight="semibold",
            color="#92400E",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#D97706", alpha=0.92),
            arrowprops=dict(arrowstyle="-", color="#D97706", lw=0.8),
        )

        if x_key == "mean_bs1_latency_s":
            val_specs = [
                ("Val 12%", r_v12, (-14, -26)),
                ("Val 20%", r_v20, (-34, 34)),
                ("Val 30%", r_v30, (14, 28)),
            ]
        else:
            val_specs = [
                ("Val 12%", r_v12, (46, -12)),
                ("Val 20%", r_v20, (16, 36)),
                ("Val 30%", r_v30, (46, -2)),
            ]
        for vlbl, pt, off in val_specs:
            xv = pt[x_key]
            yv = 100.0 * pt["accuracy"]
            ax.scatter([xv], [yv], color="#CF222E", edgecolor="white", s=90, marker="D", zorder=6)
            ax.annotate(
                f"{vlbl} ({yv:.1f}%)",
                xy=(xv, yv),
                xytext=off,
                textcoords="offset points",
                fontsize=8.0,
                fontweight="bold",
                color="#A40E26",
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#CF222E", alpha=0.92),
                arrowprops=dict(arrowstyle="-", color="#CF222E", lw=0.8),
            )

        if x_key == "mean_bs1_latency_s":
            b_specs = [
                (a1, "#1F2328", "o", "1.7B 1-Pass", (-12, -24)),
                (a2, "#8250DF", "s", "Self-Refine", (0, -28)),
                (a3a, "#1A7F37", "^", "Adaptive SC", (12, -46)),
                (a3b, "#2DA44E", "v", "3-Path SC", (16, -18)),
                (a4, "#0550AE", "P", "8B Direct", (-56, 12)),
                (a5, "#BF8700", "X", "Thinking 4K", (-44, 12)),
            ]
        else:
            b_specs = [
                (a1, "#1F2328", "o", "1.7B 1-Pass", (-36, -32)),
                (a2, "#8250DF", "s", "Self-Refine", (24, -26)),
                (a3a, "#1A7F37", "^", "Adaptive SC", (-26, -26)),
                (a3b, "#2DA44E", "v", "3-Path SC", (-10, 14)),
                (a4, "#0550AE", "P", "8B Direct", (-18, 16)),
                (a5, "#BF8700", "X", "Thinking 4K", (-60, 12)),
            ]
        for b_info, col, mk, short_name, txt_off in b_specs:
            xb = b_info[x_key]
            yb = 100.0 * b_info["accuracy"]
            ax.scatter([xb], [yb], color=col, marker=mk, s=88, edgecolor="white", linewidth=0.8, zorder=5)
            ax.annotate(
                f"{short_name}\n({yb:.1f}%)",
                xy=(xb, yb),
                xytext=txt_off,
                textcoords="offset points",
                fontsize=8.0,
                color=col,
                fontweight="semibold",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.65, alpha=0.65),
            )

        ax.scatter([x_c100], [100.0 * c100["accuracy"]], color="#57606A", marker="*", s=105, zorder=5)
        ax.annotate(
            "100% Cascade",
            xy=(x_c100, 100.0 * c100["accuracy"]),
            xytext=(-58 if x_key == "mean_bs1_latency_s" else 10, 16 if x_key == "mean_bs1_latency_s" else 10),
            textcoords="offset points",
            fontsize=8.0,
            color="#57606A",
            arrowprops=dict(arrowstyle="-", color="#57606A", lw=0.65, alpha=0.65),
        )
        ax.set_xlabel(xlabel)
        ax.set_xlim(*xlim)
        ax.set_ylim(54.0, 85.0)

    tok_overhead_pct = 100.0 * (r_v20["mean_tokens"] / a4["mean_tokens"] - 1.0)
    axes[1].text(
        0.52,
        0.25,
        "Note on Token Space:\n"
        f"Cascade pays 1.7B upfront (+{tok_overhead_pct:.1f}% raw\n"
        "tokens vs 8B Direct at Val 20%), so 8B\n"
        "Direct dominates in raw token count.\n"
        "Advantage is in calibrated serial latency.",
        transform=axes[1].transAxes,
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8C5", ec="#D4A72C", alpha=0.92),
    )
    axes[0].set_ylabel("MATH-500 Accuracy (%, Unified Repaired Grader)")
    axes[0].set_title("(A) Accuracy vs. Estimated Serial Latency")
    axes[1].set_title("(B) Accuracy vs. Mean Raw Output Tokens")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8.4, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle(
        "Figure 1. MATH-500 Cost-Accuracy Tradeoffs Under Fair Direct-Dispatch & Cascade Accounting",
        fontsize=13.0,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.07, 1.0, 0.94])
    paths["fig1_pareto_frontier"] = _save_fig(fig, figures_dir, "fig1_pareto_frontier")

    # ------------------------------------------------------------------
    # Figure 2: Mechanism Analysis Under Fixed Decoding Configurations
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.6))

    # Panel A: Detailed outcome decomposition including Atom 5 (289/37/174) & Trunc-Only
    n_test = len(test_data["y17"])
    n_tot = float(n_test)
    a5_closed_trunc = a5["think_closed_but_body_truncated_count"]
    a5_unclosed = a5["think_unclosed_truncated_count"]
    stack_defs = [
        ("1.7B 1-Pass", a1["statuses"]["correct"], a1["statuses"]["wrong"], 0, a1["statuses"]["truncated"]),
        ("Trunc-Only", trunc_only["statuses"]["correct"], trunc_only["statuses"]["wrong"], 0, trunc_only["statuses"]["truncated"]),
        ("Self-Refine", a2["statuses"]["correct"], a2["statuses"]["wrong"], 0, a2["statuses"]["truncated"]),
        ("Adap SC", a3a["statuses"]["correct"], a3a["statuses"]["wrong"], 0, a3a["statuses"]["truncated"]),
        ("1.7B Think 4K", a5["statuses"]["correct"], a5["statuses"]["wrong"], a5_closed_trunc, a5_unclosed),
        ("Router @20%", r_v20["statuses"]["correct"], r_v20["statuses"]["wrong"], 0, r_v20["statuses"]["truncated"]),
        ("8B Direct", a4["statuses"]["correct"], a4["statuses"]["wrong"], 0, a4["statuses"]["truncated"]),
    ]
    x_pos = np.arange(len(stack_defs))
    c_corr = np.array([100.0 * s[1] / n_tot for s in stack_defs])
    c_wrg = np.array([100.0 * s[2] / n_tot for s in stack_defs])
    c_ct = np.array([100.0 * s[3] / n_tot for s in stack_defs])
    c_ut = np.array([100.0 * s[4] / n_tot for s in stack_defs])

    seg_specs = [
        (c_corr, "Correct", "#1A7F37"),
        (c_wrg, "Wrong (Completed)", "#CF222E"),
        (c_ct, "Think Closed / Body Trunc.", "#E09F3E"),
        (c_ut, "Truncated (or Think Unclosed)", "#B45309"),
    ]
    bottoms = np.zeros(len(stack_defs))
    for vals, s_lbl, s_col in seg_specs:
        axes[0].bar(x_pos, vals, bottom=bottoms, color=s_col, label=s_lbl, width=0.62, edgecolor="white", linewidth=0.5)
        for xi, v in enumerate(vals):
            if v >= 7.0:
                axes[0].text(
                    xi, bottoms[xi] + v / 2.0, f"{v:.1f}%",
                    ha="center", va="center", color="white", fontsize=7.8, fontweight="bold",
                )
        bottoms += vals
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels([s[0] for s in stack_defs], rotation=24, ha="right", fontsize=8.6)
    axes[0].set_ylabel("Share of MATH-500 Problems (%)")
    axes[0].set_ylim(0, 124)
    axes[0].set_yticks([0, 20, 40, 60, 80, 100])
    axes[0].set_title("(A) Outcome & Truncation Decomposition")
    axes[0].legend(loc="upper center", ncol=2, fontsize=7.6)

    # Panel B: Descriptive stratification by difficulty Level 1-5
    levels = np.array([1, 2, 3, 4, 5])
    lvl_arr = test_data["levels"]
    acc_17_lvl = [100.0 * a1["correct_arr"][lvl_arr == lv].mean() for lv in levels]
    acc_8_lvl = [100.0 * a4["correct_arr"][lvl_arr == lv].mean() for lv in levels]
    acc_ro_lvl = [100.0 * r_v20["correct_arr"][lvl_arr == lv].mean() for lv in levels]
    acc_th_lvl = [100.0 * a5["correct_arr"][lvl_arr == lv].mean() for lv in levels]
    trunc_th_lvl = [
        100.0 * np.mean([r["truncated"] for r, l in zip(test_data["atom5"], lvl_arr) if l == lv])
        for lv in levels
    ]
    ax_b = axes[1]
    ax_b.plot(levels, acc_8_lvl, marker="P", color="#0550AE", linewidth=2.0, label="8B Direct")
    ax_b.plot(levels, acc_ro_lvl, marker="D", color="#CF222E", linewidth=2.2, label="Router (@Val 20%)")
    ax_b.plot(levels, acc_17_lvl, marker="o", color="#1F2328", linewidth=1.8, label="1.7B 1-Pass")
    ax_b.plot(levels, acc_th_lvl, marker="X", color="#B45309", linewidth=1.8, linestyle="--", label="1.7B Think (4K cap)")
    ax_b.bar(levels, trunc_th_lvl, color="#B45309", alpha=0.18, width=0.45, label="Think 4K Trunc. %")
    ax_b.set_xticks(levels)
    ax_b.set_xticklabels([f"L{lv}\n(n={int((lvl_arr == lv).sum())})" for lv in levels], fontsize=8.6)
    ax_b.set_ylabel("Accuracy / Truncation Rate (%)")
    ax_b.set_ylim(0, 126)
    ax_b.set_yticks([0, 20, 40, 60, 80, 100])
    ax_b.set_title("(B) Descriptive Breakdown by Difficulty (Not Perturbation Test)")
    ax_b.legend(loc="upper center", ncol=2, fontsize=7.6)

    # Panel C: Escalation rate by difficulty + exact scoped Self-Refine/Thinking notes
    esc_ro_lvl = [100.0 * r_v20["mask"][lvl_arr == lv].mean() for lv in levels]
    esc_cg_lvl = [100.0 * cg_v20["mask"][lvl_arr == lv].mean() for lv in levels]
    w = 0.34
    axes[2].bar(
        levels - w / 2, esc_ro_lvl, width=w, color="#CF222E",
        label=f"Cost-Aware (@Val 20%: k={r_v20['escalation_count']}, {r_v20['mean_bs1_latency_s']:.1f}s)",
    )
    axes[2].bar(
        levels + w / 2, esc_cg_lvl, width=w, color="#0969DA", alpha=0.85,
        label=f"ConfGate (@Val 20%: k={cg_v20['escalation_count']}, {cg_v20['mean_bs1_latency_s']:.1f}s)",
    )
    for lv, rv, cv in zip(levels, esc_ro_lvl, esc_cg_lvl):
        axes[2].text(lv - w / 2, rv + 1.4, f"{rv:.0f}%", ha="center", va="bottom", fontsize=7.8, color="#A40E26")
        axes[2].text(lv + w / 2, cv + 1.4, f"{cv:.0f}%", ha="center", va="bottom", fontsize=7.8, color="#0550AE")
    axes[2].set_xticks(levels)
    axes[2].set_xticklabels([f"Level {lv}" for lv in levels], fontsize=8.6)
    axes[2].set_ylabel("Escalation Rate to 8B (%)")
    axes[2].set_ylim(0, 88)
    axes[2].set_title("(C) Escalation by Level & Single-Model Observations")
    axes[2].legend(loc="upper left", fontsize=7.8)
    axes[2].text(
        0.04,
        0.74,
        f"Self-Refine (Prompt/Greedy Config):\n"
        f"• Completed wrong fixed: {a2['completed_wrong_fixed']}/{a2['completed_wrong_total']}\n"
        f"• Truncated drafts fixed: {a2['truncated_fixed']}/{a2['truncated_total']}\n"
        f"• False confirmations: {a2['critique_no_errors_completed_wrong']}/{a2['completed_wrong_total']} wrong\n\n"
        f"1.7B Thinking (4K Budget Point):\n"
        f"• Completed output: {a5['output_completed_correct']}/{a5['output_completed_count']} ({100*a5['conditional_acc_on_completed_output']:.1f}%)\n"
        f"• Closed <think> total: {a5['think_closed_count']}/{n_test} ({a5['think_closed_but_body_truncated_count']} body trunc.)",
        transform=axes[2].transAxes,
        fontsize=7.7,
        va="top",
        bbox=dict(boxstyle="round,pad=0.32", fc="#F6F8FA", ec="#D0D7DE"),
    )

    fig.suptitle(
        "Figure 2. Observed Outcomes Under Fixed Qwen3 Decoding Configurations on MATH-500",
        fontsize=13.0,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    paths["fig2_mechanism_decomposition"] = _save_fig(fig, figures_dir, "fig2_mechanism_decomposition")

    # ------------------------------------------------------------------
    # Figure 3: Router Diagnostics (ROC, p17+p8 Calibration, Weights, Deciles)
    # ------------------------------------------------------------------
    fig, axes_2d = plt.subplots(2, 2, figsize=(14.4, 10.2))
    ax_roc, ax_cal = axes_2d[0, 0], axes_2d[0, 1]
    ax_wts, ax_dec = axes_2d[1, 0], axes_2d[1, 1]

    y17_te = test_data["y17"]
    comp_te = router.predict_components(test_data["X"], truncated_mask=test_data["trunc17"])
    roc_specs = [
        ("Post-1.7B 6-Feat Router", comp_te["p17"], "#CF222E", "-", 2.4),
        ("min_token_logprob only", test_data["X"][:, ONLINE_FEATURE_NAMES.index("min_token_logprob")], "#0969DA", "--", 1.8),
        ("output_length only", -test_data["X"][:, ONLINE_FEATURE_NAMES.index("output_length")], "#1A7F37", "-.", 1.8),
        ("Pre-Gen question_length", comp_te["p17_pre_qlen"], "#9A6700", ":", 1.9),
    ]
    for r_lbl, r_sc, r_col, r_ls, r_lw in roc_specs:
        fpr, tpr, _ = roc_curve(y17_te, r_sc)
        auc_val = float(roc_auc_score(y17_te, r_sc))
        ax_roc.plot(fpr, tpr, color=r_col, linestyle=r_ls, linewidth=r_lw, label=f"{r_lbl} (AUC={auc_val:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="#8C959F", linestyle="--", linewidth=1.0, label="Chance (0.500)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("(A) Discrimination on MATH-500: Predicting 1.7B Correctness")
    ax_roc.legend(loc="lower right", fontsize=8.4)

    # Panel B: Reliability diagram for BOTH p17 (selected) and p8 (selected) with error bars & bin n
    ax_cal.plot([0, 1], [0, 1], color="#6E7781", linestyle="--", linewidth=1.2, label="Perfect Calibration")
    m17_sel = router.selected_calibration_17b_
    m8_sel = router.selected_calibration_8b_
    cal_curves = [
        (calib_17b_report["methods"][m17_sel]["val"], f"p17 Val 200 ({m17_sel}, ECE={calib_17b_report['methods'][m17_sel]['val']['ece']:.3f})", "#0969DA", "s", False),
        (calib_17b_report["methods"][m17_sel]["test"], f"p17 Test 500 ({m17_sel}, ECE={calib_17b_report['methods'][m17_sel]['test']['ece']:.3f})", "#CF222E", "o", True),
        (calib_8b_report["methods"][m8_sel]["test"], f"p8 Test 500 ({m8_sel}, ECE={calib_8b_report['methods'][m8_sel]['test']['ece']:.3f})", "#1A7F37", "^", False),
    ]
    for rep_split, c_lbl, col, mk, show_n in cal_curves:
        bins = rep_split["reliability_bins"]
        xs = [b["mean_pred"] for b in bins]
        ys = [b["empirical_acc"] for b in bins]
        ses = [b["se"] for b in bins]
        ax_cal.errorbar(xs, ys, yerr=ses, color=col, marker=mk, linewidth=1.8, markersize=5.5, capsize=2.5, label=c_lbl)
        if show_n:
            n_offsets = [
                (16, 1),
                (6, -15),
                (12, -12),
                (12, -10),
                (14, 4),
                (14, -6),
                (-18, -11),
                (14, -8),
                (-12, 10),
                (-16, 8),
            ]
            for idx_b, b in enumerate(bins):
                off_b = n_offsets[idx_b] if idx_b < len(n_offsets) else (0, 8)
                ax_cal.annotate(
                    f"n={int(b['count'])}",
                    xy=(b["mean_pred"], b["empirical_acc"]),
                    xytext=off_b,
                    textcoords="offset points",
                    ha="center",
                    fontsize=6.8,
                    color="#A40E26",
                    bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=0.72),
                )
    ax_cal.set_xlim(0.0, 1.03)
    ax_cal.set_ylim(0.0, 1.08)
    ax_cal.set_xlabel("Predicted Probability (with ±1 Binomial SE)")
    ax_cal.set_ylabel("Empirical Accuracy")
    ax_cal.set_title("(B) Calibration Reliability for Both p17 and p8")
    ax_cal.legend(loc="upper left", fontsize=8.0)

    # Panel C: Associative Standardized LR Weights
    coefs = router.get_standardized_coefficients()
    feat_order = list(reversed(ONLINE_FEATURE_NAMES))
    y_idx = np.arange(len(feat_order))
    w17 = [coefs["clf_17b"]["weights"][f] for f in feat_order]
    w8 = [coefs["clf_8b"]["weights"][f] for f in feat_order]
    bh = 0.36
    ax_wts.barh(y_idx + bh / 2, w17, height=bh, color="#1F2328", label="P(1.7B Correct | z_x)")
    ax_wts.barh(y_idx - bh / 2, w8, height=bh, color="#0550AE", alpha=0.85, label="P(8B Correct | z_x, base LR)")
    ax_wts.axvline(0.0, color="#6E7781", linewidth=1.0)
    ax_wts.set_yticks(y_idx)
    ax_wts.set_yticklabels(feat_order, fontsize=9.0)
    ax_wts.set_xlabel("Standardized LR Coefficient (Associative; Correlated Features)")
    ax_wts.set_title("(C) Standardized Linear Weights (Non-Causal Associations)")
    ax_wts.legend(loc="lower right", fontsize=8.2)

    # Panel D: Confidence Decile Anatomy with clean z-order (gray bars strictly behind lines!)
    dec_ids = np.array([d["decile"] for d in deciles])
    dec_17 = np.array([100.0 * d["actual_17b_acc"] for d in deciles])
    dec_8 = np.array([100.0 * d["actual_8b_acc"] for d in deciles])
    dec_tok8 = np.array([d["mean_tok8"] for d in deciles])
    dec_counts = sorted({int(d["count"]) for d in deciles})
    dec_bin_lbl = str(dec_counts[0]) if len(dec_counts) == 1 else f"{dec_counts[0]}-{dec_counts[-1]}"
    dec1_n = int(deciles[0]["count"])

    ax_dec2 = ax_dec.twinx()
    ax_dec2.spines["top"].set_visible(False)
    ax_dec.set_zorder(ax_dec2.get_zorder() + 2)
    ax_dec.patch.set_visible(False)

    ax_dec2.bar(dec_ids, dec_tok8, color="#D8DEE4", alpha=0.65, width=0.62, label=f"Mean 8B Tokens (n={dec_bin_lbl}/bin)", zorder=1)
    ax_dec.plot(dec_ids, dec_17, color="#1F2328", marker="o", linewidth=2.2, label="1.7B Empirical Acc (%)", zorder=5)
    ax_dec.plot(dec_ids, dec_8, color="#CF222E", marker="D", linewidth=2.4, label="8B Empirical Acc (%)", zorder=6)
    ax_dec.fill_between(dec_ids, dec_17, dec_8, where=(dec_8 >= dec_17), color="#CF222E", alpha=0.14, label="8B Rescue Gain", zorder=3)

    ax_dec.annotate(
        f"Decile 1: Truncated\n(+{dec_8[0]-dec_17[0]:.0f} pp, n={dec1_n})",
        xy=(1, dec_8[0]),
        xytext=(1.15, 86.0),
        fontsize=8.0,
        fontweight="semibold",
        color="#A40E26",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CF222E", alpha=0.96),
        arrowprops=dict(arrowstyle="->", color="#A40E26", lw=1.1),
        zorder=8,
    )
    ax_dec.annotate(
        f"Decile 2: Shared Hard Trap\n(+{dec_8[1]-dec_17[1]:.0f} pp, {int(round(dec_tok8[1]))} tok)",
        xy=(2, dec_8[1]),
        xytext=(3.2, 14.0),
        fontsize=8.0,
        fontweight="semibold",
        color="#1F2328",
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#6E7781", alpha=0.96),
        arrowprops=dict(arrowstyle="->", color="#57606A", lw=1.1),
        zorder=8,
    )
    ax_dec.set_xticks(dec_ids)
    ax_dec.set_xlabel(f"Predicted 1.7B Confidence Decile (n={dec_bin_lbl} per bin)")
    ax_dec.set_ylabel("Empirical Accuracy (%)")
    ax_dec2.set_ylabel("Mean 8B Output Tokens", color="#57606A")
    ax_dec.set_ylim(0, 108)
    ax_dec2.set_ylim(0, 2150)
    ax_dec.set_title(f"(D) MATH-500 Confidence Decile Anatomy ({dec_bin_lbl} Questions/Decile)")
    h1, l1 = ax_dec.get_legend_handles_labels()
    h2, l2 = ax_dec2.get_legend_handles_labels()
    leg = ax_dec.legend(
        h1 + h2,
        l1 + l2,
        loc="center right",
        bbox_to_anchor=(0.99, 0.54),
        fontsize=7.9,
        facecolor="white",
        framealpha=0.96,
    )
    leg.set_zorder(10)

    fig.suptitle(
        "Figure 3. Router Diagnostics: Discrimination, Calibration & Decile Economics",
        fontsize=13.0,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    paths["fig3_router_diagnostics"] = _save_fig(fig, figures_dir, "fig3_router_diagnostics")

    # ------------------------------------------------------------------
    # Figure 4: Head-to-Head Across Matching Protocols (Dot-CI + Forest Plot)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.9))
    ax_dot, ax_for = axes[0], axes[1]

    pre_v20 = val_locked_points["pre_gen_qlen"]["0.20"]
    summary_specs = [
        ("1.7B Thinking (4K)", a5, "#B45309"),
        ("1.7B 1-Pass", a1, "#57606A"),
        ("Self-Refine", a2, "#8250DF"),
        ("Trunc-Only Fallback", trunc_only, "#D97706"),
        ("Pre-Gen Direct @Val 20%", pre_v20, "#9A6700"),
        (f"ConfGate @Val 20% (k={cg_v20['escalation_count']})", cg_v20, "#0969DA"),
        (f"ConfGate @Matched k={cg_k['escalation_count']}", cg_k, "#1F6FEB"),
        ("Adaptive SC (2->3)", a3a, "#1A7F37"),
        (f"Router @Val 20% (k={r_v20['escalation_count']})", r_v20, "#CF222E"),
        ("8B Direct", a4, "#0550AE"),
    ]
    sy = np.arange(len(summary_specs))
    for yi, (lbl, info, col) in enumerate(summary_specs):
        p_hat = float(info["accuracy"])
        se = float(np.sqrt(p_hat * (1.0 - p_hat) / n_tot))
        ci = 1.96 * 100.0 * se
        acc_pct = 100.0 * p_hat
        ax_dot.errorbar(
            acc_pct, yi, xerr=ci, fmt="o", color=col, ecolor=col,
            elinewidth=1.9, capsize=3.5, markersize=7.0,
        )
        ax_dot.text(
            acc_pct + ci + 0.8,
            yi,
            f"{acc_pct:.1f}% | {int(round(info['mean_tokens']))}t | {info['mean_bs1_latency_s']:.1f}s",
            va="center",
            fontsize=8.0,
            color=col,
            fontweight="semibold",
        )
    ax_dot.axvline(100.0 * a1["accuracy"], color="#57606A", linestyle=":", linewidth=1.1)
    ax_dot.set_yticks(sy)
    ax_dot.set_yticklabels([s[0] for s in summary_specs], fontsize=8.4)
    ax_dot.set_xlim(48.0, 96.0)
    ax_dot.set_xlabel("MATH-500 Accuracy (% ± 1.96·SE) with Tokens & Est. Serial Latency")
    ax_dot.set_title("(A) Accuracy & Actual Cost Across Strategies (Not Equal-Budget)")

    # Panel B: Complete honest forest plot (including @Matched k, Random Direct Dispatch, and 8B Direct!)
    forest_tests = [t for t in stat_tests if t.get("show_in_forest", True)]
    fy = np.arange(len(forest_tests))[::-1]
    diffs = [100.0 * t["diff"] for t in forest_tests]
    lows = [100.0 * (t["diff"] - t["ci_low"]) for t in forest_tests]
    highs = [100.0 * (t["ci_high"] - t["diff"]) for t in forest_tests]

    ax_for.axvline(0.0, color="#1F2328", linestyle="--", linewidth=1.2)
    for yv, df, lo, hi, t in zip(fy, diffs, lows, highs, forest_tests):
        is_random_baseline = t.get("mcnemar_p") is None
        sig_holm = (not is_random_baseline) and (t.get("holm_p") is not None) and float(t["holm_p"]) < 0.05
        sig_raw = (not is_random_baseline) and float(t["mcnemar_p"]) < 0.05
        if is_random_baseline:
            col = "#B45309"
            mk = "D"
        elif sig_holm:
            col = "#CF222E"
            mk = "o"
        elif sig_raw:
            col = "#0969DA"
            mk = "o"
        else:
            col = "#57606A"
            mk = "o"
        ax_for.errorbar(
            df, yv, xerr=[[lo], [hi]], fmt=mk, color=col, ecolor=col,
            elinewidth=1.9, capsize=3.5, markersize=6.2 if mk == "D" else 6.5,
        )
        if not is_random_baseline:
            p_txt = f"p={t['mcnemar_p']:.2e}, p_H={t['holm_p']:.2e}"
        else:
            p_txt = f"boot>0={100.0*t['boot_pos_share']:.2f}% (unadj.)"
        ax_for.text(
            df + hi + 0.5,
            yv,
            f"{df:+.1f} pp ({p_txt})",
            va="center",
            ha="left",
            fontsize=7.6,
            color=col,
            fontweight="semibold",
        )

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#CF222E", marker="o", lw=1.9, markersize=6.2, label="Holm p < 0.05 (Exact McNemar)"),
        Line2D([0], [0], color="#0969DA", marker="o", lw=1.9, markersize=6.2, label="Raw McNemar p < 0.05 (Holm NS)"),
        Line2D([0], [0], color="#57606A", marker="o", lw=1.9, markersize=6.2, label="Not significant (raw p ≥ 0.05)"),
        Line2D([0], [0], color="#B45309", marker="D", lw=1.9, markersize=5.8, label="Unadjusted bootstrap vs E[Random]"),
    ]
    ax_for.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.48, -0.14),
        ncol=2,
        fontsize=8.0,
        facecolor="white",
        framealpha=0.96,
    )

    ax_for.set_yticks(fy)
    ax_for.set_yticklabels([t["comparison"] for t in forest_tests], fontsize=8.1)
    ax_for.set_xlabel("Paired Accuracy Difference ΔAcc (pp, 95% Percentile Bootstrap CI)")
    max_x = max(41.5, max(d + h for d, h in zip(diffs, highs)) + 17.5)
    min_x = min(-7.5, min(d - l for d, l in zip(diffs, lows)) - 1.5)
    ax_for.set_xlim(min_x, max_x)
    ax_for.set_title("(B) Paired Bootstrap (B=5,000) & Exact McNemar (Raw + Holm p)")

    fig.suptitle(
        "Figure 4. Head-to-Head Comparisons Across Budget & Matching Protocols (MATH-500)",
        fontsize=13.0,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.94])
    paths["fig4_matched_budget_stat_tests"] = _save_fig(fig, figures_dir, "fig4_matched_budget_stat_tests")

    return paths


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _rows_to_markdown(rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> str:
    header = "| " + " | ".join(fieldnames) + " |"
    sep = "| " + " | ".join("---" for _ in fieldnames) + " |"
    lines = [header, sep]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in fieldnames) + " |")
    return "\n".join(lines) + "\n"


def run_phase2_phase3_pipeline(
    bundle_dir: Path,
    output_dir: Path,
    *,
    seed: int = 550,
    n_boot: int = 5000,
    regrade: bool = True,
) -> dict[str, Any]:
    """Execute end-to-end Phase 2 training/calibration and Phase 3 evaluation/plotting."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"

    # 1. Load latency calibration & strictly aligned splits (re-graded uniformly by default)
    lat_cal = load_latency_calibration(bundle_dir / "latency_calibration_v2.jsonl")
    tps_17b = lat_cal["qwen3_1.7b_non_think"]["tokens_per_sec"]
    tps_8b = lat_cal["qwen3_8b_non_think"]["tokens_per_sec"]
    tps_think = lat_cal["qwen3_1.7b_think"]["tokens_per_sec"]

    train_data = load_aligned_split(bundle_dir, "router_train_600", regrade=regrade)
    val_data = load_aligned_split(bundle_dir, "router_val_200", regrade=regrade)
    test_data = load_aligned_split(bundle_dir, "math500_test_500", regrade=regrade)

    all_regrade_diffs = (
        train_data["regrade_diffs"] + val_data["regrade_diffs"] + test_data["regrade_diffs"]
    )

    # 2. Phase 2: Calibration diagnostics (p17 & p8), auto-wired CostAwareRouter, lambda & Val locking
    calib_17b_rep = evaluate_calibration_methods(
        train_data["X"],
        train_data["y17"],
        val_data["X"],
        val_data["y17"],
        test_data["X"],
        test_data["y17"],
        seed=seed,
    )
    calib_8b_rep = evaluate_calibration_methods(
        train_data["X"],
        train_data["y8"],
        val_data["X"],
        val_data["y8"],
        test_data["X"],
        test_data["y8"],
        seed=seed,
    )
    # Remove non-JSON-serializable fitted estimator objects from saved reports
    calib_17b_json = {k: v for k, v in calib_17b_rep.items() if k != "fitted_estimators"}
    calib_8b_json = {k: v for k, v in calib_8b_rep.items() if k != "fitted_estimators"}

    router = CostAwareRouter(tps_8b=tps_8b, calibration_method="auto", seed=seed).fit(
        train_data["X"],
        train_data["y17"],
        train_data["y8"],
        train_data["tok8"],
        X_val=val_data["X"],
        y17_val=val_data["y17"],
        y8_val=val_data["y8"],
    )

    lambda_selection = router.select_lambda_on_val(
        val_data["X"],
        val_data["y17"],
        val_data["y8"],
        val_data["tok17"],
        val_data["tok8"],
        tps_17b,
        tps_8b,
        truncated_val=val_data["trunc17"],
        lambda_grid=(0.0, 0.0005, 0.0010, 0.0015, 0.0020),
    )
    best_lambda = float(lambda_selection["best_lambda"])

    dev_stability = run_dev_split_stability_check(
        np.vstack([train_data["X"], val_data["X"]]),
        np.concatenate([train_data["y17"], val_data["y17"]]),
        np.concatenate([train_data["y8"], val_data["y8"]]),
        np.concatenate([train_data["tok17"], val_data["tok17"]]),
        np.concatenate([train_data["tok8"], val_data["tok8"]]),
        np.concatenate([train_data["levels"], val_data["levels"]]),
        np.concatenate([train_data["trunc17"], val_data["trunc17"]]),
        tps_17b,
        tps_8b,
        n_splits=30,
        seed=seed,
    )

    val_scores = router.compute_routing_scores(
        val_data["X"], lam=best_lambda, truncated_mask=val_data["trunc17"]
    )
    test_scores = router.compute_routing_scores(
        test_data["X"], lam=best_lambda, truncated_mask=test_data["trunc17"]
    )
    test_comp = router.predict_components(
        test_data["X"], truncated_mask=test_data["trunc17"]
    )

    target_val_rates = (0.12, 0.20, 0.25, 0.30, 0.40, 0.50)
    locked_thresholds: dict[str, dict[str, float]] = {}
    val_locked_points: dict[str, dict[str, dict[str, Any]]] = {}

    for policy_name, sc_val in val_scores.items():
        if policy_name == "truncation_only":
            continue
        mode = "direct" if policy_name == "pre_gen_qlen" else "cascade"
        taus = lock_val_thresholds(sc_val, target_val_rates=target_val_rates)
        locked_thresholds[policy_name] = taus
        val_locked_points[policy_name] = {
            q_key: evaluate_val_locked_operating_point(
                test_scores[policy_name], tau, test_data, tps_17b, tps_8b, dispatch_mode=mode
            )
            for q_key, tau in taus.items()
        }

    # Explicit Truncation-Only Fallback baseline (escalate iff Atom 1 truncated)
    val_locked_points["truncation_only"] = {
        "fixed": evaluate_val_locked_mask(
            test_data["trunc17"], test_data, tps_17b, tps_8b, dispatch_mode="cascade"
        )
    }

    # Matched-k ConfGate point matching Router @Val 20% escalation count
    r_v20 = val_locked_points["tier1_trunc_cost_utility"]["0.20"]
    k_val20 = int(r_v20["escalation_count"])
    conf_order = np.argsort(-test_scores["conf_gate"], kind="mergesort")
    conf_matched_mask = np.zeros(len(test_data["y17"]), dtype=bool)
    conf_matched_mask[conf_order[:k_val20]] = True
    val_locked_points["conf_gate"]["matched_k_v20"] = evaluate_val_locked_mask(
        conf_matched_mask, test_data, tps_17b, tps_8b, dispatch_mode="cascade"
    )

    # 3. Phase 3: Baselines, Curves (with fair Direct Dispatch for pre_gen_qlen), Pareto, Stress Test, Stats
    baselines = evaluate_atomic_baselines(test_data, tps_17b, tps_8b, tps_think)

    curves: dict[str, dict[str, Any]] = {}
    curve_summary: dict[str, dict[str, Any]] = {}
    for policy_name, sc_test in test_scores.items():
        if policy_name == "truncation_only":
            continue
        mode = "direct" if policy_name == "pre_gen_qlen" else "cascade"
        c = trace_routing_curve(
            sc_test,
            test_data["y17"],
            test_data["y8"],
            test_data["tok17"],
            test_data["tok8"],
            test_data["bat17"],
            test_data["bat8"],
            tps_17b,
            tps_8b,
            dispatch_mode=mode,
        )
        curves[policy_name] = c
        curve_summary[policy_name] = {
            "dispatch_mode": mode,
            "lat_auac": c["lat_auac"],
            "lat_excess_over_endpoints_line": c["lat_excess_over_endpoints_line"],
            "tok_auac": c["tok_auac"],
            "tok_excess_over_endpoints_line": c["tok_excess_over_endpoints_line"],
        }

    # Non-dominated Pareto envelope across all evaluated strategies
    a1 = baselines["atom1_1.7b_nonthink"]
    a2 = baselines["atom2_1.7b_self_refine"]
    a3a = baselines["atom3a_1.7b_adaptive_sc"]
    a3b = baselines["atom3b_1.7b_majority_sc3"]
    a4 = baselines["atom4_8b_direct"]
    a5 = baselines["atom5_1.7b_thinking"]
    trunc_pt = val_locked_points["truncation_only"]["fixed"]

    candidate_pts = [
        {"name": "Atom 1: 1.7B 1-Pass", "accuracy": a1["accuracy"], "mean_bs1_latency_s": a1["mean_bs1_latency_s"], "mean_tokens": a1["mean_tokens"]},
        {"name": "Trunc-Only Fallback", "accuracy": trunc_pt["accuracy"], "mean_bs1_latency_s": trunc_pt["mean_bs1_latency_s"], "mean_tokens": trunc_pt["mean_tokens"]},
        {"name": "Atom 2: Self-Refine", "accuracy": a2["accuracy"], "mean_bs1_latency_s": a2["mean_bs1_latency_s"], "mean_tokens": a2["mean_tokens"]},
        {"name": "Atom 3a: Adaptive SC", "accuracy": a3a["accuracy"], "mean_bs1_latency_s": a3a["mean_bs1_latency_s"], "mean_tokens": a3a["mean_tokens"]},
        {"name": "Atom 3b: 3-Path SC", "accuracy": a3b["accuracy"], "mean_bs1_latency_s": a3b["mean_bs1_latency_s"], "mean_tokens": a3b["mean_tokens"]},
        {"name": "Atom 4: 8B Direct", "accuracy": a4["accuracy"], "mean_bs1_latency_s": a4["mean_bs1_latency_s"], "mean_tokens": a4["mean_tokens"]},
        {"name": "Atom 5: Thinking 4K", "accuracy": a5["accuracy"], "mean_bs1_latency_s": a5["mean_bs1_latency_s"], "mean_tokens": a5["mean_tokens"]},
    ]
    for q_key in ("0.12", "0.20", "0.25", "0.30", "0.40", "0.50"):
        pt_r = val_locked_points["tier1_trunc_cost_utility"][q_key]
        pt_p = val_locked_points["pre_gen_qlen"][q_key]
        pt_c = val_locked_points["conf_gate"][q_key]
        candidate_pts.append({"name": f"Router @Val {int(round(float(q_key)*100))}%", "accuracy": pt_r["accuracy"], "mean_bs1_latency_s": pt_r["mean_bs1_latency_s"], "mean_tokens": pt_r["mean_tokens"]})
        candidate_pts.append({"name": f"Pre-Gen Direct @Val {int(round(float(q_key)*100))}%", "accuracy": pt_p["accuracy"], "mean_bs1_latency_s": pt_p["mean_bs1_latency_s"], "mean_tokens": pt_p["mean_tokens"]})
        candidate_pts.append({"name": f"ConfGate @Val {int(round(float(q_key)*100))}%", "accuracy": pt_c["accuracy"], "mean_bs1_latency_s": pt_c["mean_bs1_latency_s"], "mean_tokens": pt_c["mean_tokens"]})

    pareto_latency = compute_pareto_envelope(candidate_pts, cost_key="mean_bs1_latency_s")
    pareto_tokens = compute_pareto_envelope(candidate_pts, cost_key="mean_tokens")

    # Throughput speed ratio stress test
    obs_ratio = float(tps_17b / tps_8b)
    stress_test = compute_throughput_stress_test(
        r_v20, a1, a3a, a4, tps_8b=tps_8b, observed_speed_ratio=obs_ratio
    )

    # Statistical comparisons (accuracy + estimated latency paired CIs + raw & Holm McNemar p)
    r_v12 = val_locked_points["tier1_trunc_cost_utility"]["0.12"]
    r_v30 = val_locked_points["tier1_trunc_cost_utility"]["0.30"]
    cg_v20 = val_locked_points["conf_gate"]["0.20"]
    cg_k = val_locked_points["conf_gate"]["matched_k_v20"]
    pre_v20 = val_locked_points["pre_gen_qlen"]["0.20"]

    completed_mask = np.array([r["status"] != "truncated" for r in test_data["atom1"]], dtype=bool)
    lat17_arr = test_data["tok17"] / float(tps_17b)
    lat8_arr = test_data["tok8"] / float(tps_8b)

    binary_pairs = [
        ("Router @Val 20% vs 1.7B 1-Pass", r_v20["correct_arr"], r_v20["bs1_arr"], a1["correct_arr"], a1["bs1_arr"], True, True),
        ("Router @Val 20% vs Trunc-Only", r_v20["correct_arr"], r_v20["bs1_arr"], trunc_pt["correct_arr"], trunc_pt["bs1_arr"], True, True),
        ("Router @Val 20% vs Self-Refine", r_v20["correct_arr"], r_v20["bs1_arr"], a2["correct_arr"], a2["bs1_arr"], True, True),
        ("Router @Val 20% vs Thinking 4K", r_v20["correct_arr"], r_v20["bs1_arr"], a5["correct_arr"], a5["bs1_arr"], True, True),
        ("Router @Val 20% vs Pre-Gen @Val 20%", r_v20["correct_arr"], r_v20["bs1_arr"], pre_v20["correct_arr"], pre_v20["bs1_arr"], True, True),
        ("Router @Val 20% vs Adaptive SC", r_v20["correct_arr"], r_v20["bs1_arr"], a3a["correct_arr"], a3a["bs1_arr"], True, True),
        ("Router @Val 20% vs ConfGate (@Val 20%)", r_v20["correct_arr"], r_v20["bs1_arr"], cg_v20["correct_arr"], cg_v20["bs1_arr"], True, True),
        (f"Router @Val 20% vs ConfGate (@k={k_val20})", r_v20["correct_arr"], r_v20["bs1_arr"], cg_k["correct_arr"], cg_k["bs1_arr"], True, True),
        ("Router @Val 20% vs 8B Direct", r_v20["correct_arr"], r_v20["bs1_arr"], a4["correct_arr"], a4["bs1_arr"], True, True),
        ("Router @Val 12% vs 1.7B 1-Pass", r_v12["correct_arr"], r_v12["bs1_arr"], a1["correct_arr"], a1["bs1_arr"], True, False),
        ("Router @Val 30% vs 3-Path Maj SC", r_v30["correct_arr"], r_v30["bs1_arr"], a3b["correct_arr"], a3b["bs1_arr"], True, False),
        ("Self-Refine vs 1.7B (All N=500)", a2["correct_arr"], a2["bs1_arr"], a1["correct_arr"], a1["bs1_arr"], True, False),
        ("Self-Refine vs 1.7B (Completed N=447)", a2["correct_arr"][completed_mask], a2["bs1_arr"][completed_mask], a1["correct_arr"][completed_mask], a1["bs1_arr"][completed_mask], True, False),
    ]

    stat_tests: list[dict[str, Any]] = []
    for label, ya, la, yb, lb, run_mc, show_f in binary_pairs:
        boot_acc = paired_bootstrap_test(ya, yb, n_boot=n_boot, seed=seed)
        boot_lat = paired_bootstrap_test(la, lb, n_boot=n_boot, seed=seed)
        mc = mcnemar_test(ya, yb) if run_mc else None
        stat_tests.append(
            {
                "comparison": label,
                "n": int(len(ya)),
                "acc_a": boot_acc["mean_a"],
                "acc_b": boot_acc["mean_b"],
                "diff": boot_acc["diff"],
                "ci_low": boot_acc["ci_low"],
                "ci_high": boot_acc["ci_high"],
                "boot_pos_share": boot_acc["boot_pos_share"],
                "lat_diff_s": boot_lat["diff"],
                "lat_ci_low_s": boot_lat["ci_low"],
                "lat_ci_high_s": boot_lat["ci_high"],
                "n_a_only": mc["n_a_only"] if mc else "",
                "n_b_only": mc["n_b_only"] if mc else "",
                "mcnemar_p": mc["p_value"] if mc else None,
                "holm_p": None,
                "show_in_forest": bool(show_f),
            }
        )

    holm_adjusted = apply_holm_bonferroni([t["mcnemar_p"] for t in stat_tests])
    for t, hp in zip(stat_tests, holm_adjusted):
        t["holm_p"] = hp

    # Add matched-latency Rao-Blackwellized random baselines (Direct Dispatch & Cascade)
    for rand_label, rand_mode in [
        ("Router @Val 20% vs Random Direct (@Same Lat)", "direct"),
        ("Router @Val 20% vs Random Cascade (@Same Lat)", "cascade"),
    ]:
        rb = paired_bootstrap_matched_random_dispatch(
            r_v20["correct_arr"],
            r_v20["bs1_arr"],
            test_data["y17"],
            test_data["y8"],
            lat17_arr,
            lat8_arr,
            mode=rand_mode,
            n_boot=n_boot,
            seed=seed,
        )
        stat_tests.append(
            {
                "comparison": rand_label,
                "n": int(len(test_data["y17"])),
                "acc_a": rb["mean_a"],
                "acc_b": rb["mean_b"],
                "diff": rb["diff"],
                "ci_low": rb["ci_low"],
                "ci_high": rb["ci_high"],
                "boot_pos_share": rb["boot_pos_share"],
                "lat_diff_s": 0.0,
                "lat_ci_low_s": 0.0,
                "lat_ci_high_s": 0.0,
                "n_a_only": "",
                "n_b_only": "",
                "mcnemar_p": None,
                "holm_p": None,
                "p_matched_random": rb["p_matched"],
                "show_in_forest": True,
            }
        )

    deciles = compute_decile_analysis(test_comp["p17"], test_comp["p8"], test_data, n_deciles=10)
    case_studies = extract_case_studies(
        test_data, test_comp, r_v20["mask"], conf_matched_mask
    )
    sensitivity_audit = audit_grading_formatting_sensitivity(bundle_dir)

    # 4. Write Tables & JSON Summaries
    main_rows: list[dict[str, Any]] = []
    ordered_entries = [
        ("Atom 1: 1.7B Non-Think (1-Pass)", 0.0, a1, "single"),
        ("Truncation-Only Fallback (1.7B -> 8B)", trunc_pt["escalation_rate"], trunc_pt, "cascade"),
        ("Atom 2: 1.7B Self-Refine", 0.0, a2, "single"),
        ("Atom 3a: 1.7B Adaptive SC (2->3 paths)", 0.0, a3a, "single"),
        ("Atom 3b: 1.7B 3-Path Majority SC", 0.0, a3b, "single"),
        ("Atom 5: 1.7B Thinking (max=4096)", 0.0, a5, "single"),
        ("Pre-Gen Direct Router (q_len, Val 20%)", pre_v20["escalation_rate"], pre_v20, "direct"),
        ("Pre-Gen Direct Router (q_len, Val 30%)", val_locked_points["pre_gen_qlen"]["0.30"]["escalation_rate"], val_locked_points["pre_gen_qlen"]["0.30"], "direct"),
        ("Naive ConfGate (Val 20%)", cg_v20["escalation_rate"], cg_v20, "cascade"),
        (f"Naive ConfGate (Matched k={k_val20})", cg_k["escalation_rate"], cg_k, "cascade"),
        ("Hurdle + Trunc Priority (Val 20%)", val_locked_points["hurdle_trunc_cost_utility"]["0.20"]["escalation_rate"], val_locked_points["hurdle_trunc_cost_utility"]["0.20"], "cascade"),
        ("Cost-Aware Router (Ours, Val 12%)", r_v12["escalation_rate"], r_v12, "cascade"),
        ("Cost-Aware Router (Ours, Val 20%)", r_v20["escalation_rate"], r_v20, "cascade"),
        ("Cost-Aware Router (Ours, Val 25%)", val_locked_points["tier1_trunc_cost_utility"]["0.25"]["escalation_rate"], val_locked_points["tier1_trunc_cost_utility"]["0.25"], "cascade"),
        ("Cost-Aware Router (Ours, Val 30%)", r_v30["escalation_rate"], r_v30, "cascade"),
        ("Cost-Aware Router (Ours, Val 40%)", val_locked_points["tier1_trunc_cost_utility"]["0.40"]["escalation_rate"], val_locked_points["tier1_trunc_cost_utility"]["0.40"], "cascade"),
        ("Cost-Aware Router (Ours, Val 50%)", val_locked_points["tier1_trunc_cost_utility"]["0.50"]["escalation_rate"], val_locked_points["tier1_trunc_cost_utility"]["0.50"], "cascade"),
        ("Atom 4: 8B Non-Think (Direct)", 1.0, a4, "direct"),
        ("100% Cascade (1.7B -> 8B)", 1.0, baselines["cascade_100pct"], "cascade"),
    ]
    for method_name, esc_rate, info, d_mode in ordered_entries:
        main_rows.append(
            {
                "method": method_name,
                "dispatch_mode": d_mode,
                "escalation_rate_pct": f"{100.0 * esc_rate:.1f}",
                "accuracy_pct": f"{100.0 * info['accuracy']:.2f}",
                "correct_count": info["correct_count"],
                "truncated_pct": f"{100.0 * info['truncated_rate']:.1f}",
                "mean_tokens": f"{info['mean_tokens']:.1f}",
                "est_serial_latency_s": f"{info['mean_bs1_latency_s']:.2f}",
                "amortized_batch_s": f"{info['mean_bat_latency_s']:.3f}",
            }
        )

    main_cols = [
        "method",
        "dispatch_mode",
        "escalation_rate_pct",
        "accuracy_pct",
        "correct_count",
        "truncated_pct",
        "mean_tokens",
        "est_serial_latency_s",
        "amortized_batch_s",
    ]
    _write_csv(output_dir / "main_comparison_table.csv", main_rows, main_cols)
    (output_dir / "main_comparison_table.md").write_text(
        _rows_to_markdown(main_rows, main_cols), encoding="utf-8"
    )

    stat_rows_fmt = [
        {
            "comparison": t["comparison"],
            "n": t["n"],
            "acc_a_pct": f"{100.0 * t['acc_a']:.2f}",
            "acc_b_pct": f"{100.0 * t['acc_b']:.2f}",
            "diff_pp": f"{100.0 * t['diff']:+.2f}",
            "ci95_low_pp": f"{100.0 * t['ci_low']:+.2f}",
            "ci95_high_pp": f"{100.0 * t['ci_high']:+.2f}",
            "lat_diff_s": f"{t['lat_diff_s']:+.2f}",
            "lat_ci95_s": f"[{t['lat_ci_low_s']:+.2f}, {t['lat_ci_high_s']:+.2f}]",
            "n_a_only": t["n_a_only"],
            "n_b_only": t["n_b_only"],
            "mcnemar_p": f"{t['mcnemar_p']:.3e}" if t["mcnemar_p"] is not None else "N/A",
            "holm_p": f"{t['holm_p']:.3e}" if t["holm_p"] is not None else "N/A",
        }
        for t in stat_tests
    ]
    stat_cols = [
        "comparison",
        "n",
        "acc_a_pct",
        "acc_b_pct",
        "diff_pp",
        "ci95_low_pp",
        "ci95_high_pp",
        "lat_diff_s",
        "lat_ci95_s",
        "n_a_only",
        "n_b_only",
        "mcnemar_p",
        "holm_p",
    ]
    _write_csv(output_dir / "statistical_tests.csv", stat_rows_fmt, stat_cols)
    (output_dir / "statistical_tests.md").write_text(
        _rows_to_markdown(stat_rows_fmt, stat_cols), encoding="utf-8"
    )

    dec_cols = list(deciles[0].keys())
    _write_csv(output_dir / "decile_analysis.csv", deciles, dec_cols)

    phase2_diagnostics = {
        "seed": seed,
        "regraded_under_fixed_grader": bool(regrade),
        "features": list(ONLINE_FEATURE_NAMES),
        "selected_calibration": {
            "clf_17b": router.selected_calibration_17b_,
            "clf_8b": router.selected_calibration_8b_,
        },
        "calibration_17b": calib_17b_json,
        "calibration_8b": calib_8b_json,
        "lambda_selection_val": lambda_selection,
        "dev_split_stability_30x": dev_stability,
        "locked_val_thresholds": locked_thresholds,
        "standardized_coefficients": router.get_standardized_coefficients(),
        "test_roc_auc": {
            "post_1.7b_6feat": float(roc_auc_score(test_data["y17"], test_comp["p17"])),
            "post_8b_6feat": float(roc_auc_score(test_data["y8"], test_comp["p8"])),
            "pre_gen_qlen_only": float(roc_auc_score(test_data["y17"], test_comp["p17_pre_qlen"])),
            "min_token_logprob_only": float(roc_auc_score(test_data["y17"], test_data["X"][:, ONLINE_FEATURE_NAMES.index("min_token_logprob")])),
            "output_length_only": float(roc_auc_score(test_data["y17"], -test_data["X"][:, ONLINE_FEATURE_NAMES.index("output_length")])),
        },
        "pred_tok8_test_corr": float(np.corrcoef(test_comp["pred_tok8"], test_data["tok8"])[0, 1]),
    }
    (output_dir / "phase2_router_diagnostics.json").write_text(
        json.dumps(phase2_diagnostics, indent=2), encoding="utf-8"
    )

    def _strip_arrays(d: dict[str, Any]) -> dict[str, Any]:
        skip = {"correct_arr", "statuses_list", "tokens_arr", "bs1_arr", "bat_arr", "mask"}
        return {k: v for k, v in d.items() if k not in skip}

    phase3_summary = {
        "regraded_under_fixed_grader": bool(regrade),
        "regraded_completion_diff_count": len(all_regrade_diffs),
        "latency_calibration": lat_cal,
        "baselines": {k: _strip_arrays(v) for k, v in baselines.items()},
        "oracle_union_acc": float(np.mean((test_data["y17"] == 1) | (test_data["y8"] == 1))),
        "curve_auac_summary": curve_summary,
        "pareto_envelopes": {
            "estimated_serial_latency": pareto_latency,
            "mean_raw_tokens": pareto_tokens,
        },
        "throughput_stress_test": stress_test,
        "val_locked_operating_points": {
            pol: {q: _strip_arrays(pts) for q, pts in qmap.items()}
            for pol, qmap in val_locked_points.items()
        },
        "statistical_tests": stat_tests,
    }
    (output_dir / "phase3_summary_metrics.json").write_text(
        json.dumps(phase3_summary, indent=2), encoding="utf-8"
    )
    (output_dir / "grading_regrade_diff.json").write_text(
        json.dumps(
            {"total_changed_completions": len(all_regrade_diffs), "changes": all_regrade_diffs},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "grading_sensitivity_audit.json").write_text(
        json.dumps(sensitivity_audit, indent=2), encoding="utf-8"
    )
    (output_dir / "case_studies.json").write_text(
        json.dumps(case_studies, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 5. Generate 4 Figures
    fig_paths = generate_all_figures(
        figures_dir=figures_dir,
        baselines=baselines,
        curves=curves,
        val_locked_points=val_locked_points,
        pareto_latency=pareto_latency,
        pareto_tokens=pareto_tokens,
        test_data=test_data,
        calib_17b_report=calib_17b_json,
        calib_8b_report=calib_8b_json,
        router=router,
        deciles=deciles,
        stat_tests=stat_tests,
    )

    return {
        "phase2_diagnostics": phase2_diagnostics,
        "phase3_summary": phase3_summary,
        "figure_paths": fig_paths,
        "output_dir": str(output_dir),
    }
