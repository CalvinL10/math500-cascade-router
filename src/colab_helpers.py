"""Pure-Python helpers shared with and tested for the Step 1B Colab collection notebook.

Contains prompt construction, token log-probability feature extraction, boxed-span token
alignment, JSONL atomic checkpoint resume/deduplication, and Pilot diagnostics—without
requiring `torch` or `transformers` to be loaded locally.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from .grading import classify_response, get_last_boxed_with_span


SOLVE_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer inside \\boxed{{...}} at the end.\n\n"
    "Problem:\n{problem}"
)

CRITIQUE_PROMPT_TEMPLATE = (
    "You are a careful mathematical reviewer. Check the draft solution step by step "
    "for any logical gaps, algebraic mistakes, or arithmetic errors. "
    "If the draft solution and its final answer are completely correct, explicitly state "
    "'No errors found; the draft is correct.' Do not invent errors and do not use \\boxed{{...}} in your review.\n\n"
    "Problem:\n{problem}\n\n"
    "Draft solution:\n{draft}\n\n"
    "Provide a concise review (under 150 words)."
)

REFINE_PROMPT_TEMPLATE = (
    "Review the problem, the draft solution, and the reviewer's critique below. "
    "If the critique identifies a real error, fix it and provide the corrected solution. "
    "If the draft solution is already correct, confirm and keep the original answer. "
    "Put only the final mathematical answer inside \\boxed{{...}} at the end "
    "(never put words like 'correct' or 'no errors found' inside \\boxed{{...}}).\n\n"
    "Problem:\n{problem}\n\n"
    "Draft solution:\n{draft}\n\n"
    "Critique:\n{critique}"
)


def build_solve_messages(problem: str) -> list[dict[str, str]]:
    """Build chat messages for initial or escalated math solving."""

    return [{"role": "user", "content": SOLVE_PROMPT_TEMPLATE.format(problem=problem)}]


def build_critique_messages(problem: str, draft: str) -> list[dict[str, str]]:
    """Build strictly neutral critique messages that permit confirming correct drafts."""

    return [
        {
            "role": "user",
            "content": CRITIQUE_PROMPT_TEMPLATE.format(problem=problem, draft=draft),
        }
    ]


def build_refine_messages(problem: str, draft: str, critique: str) -> list[dict[str, str]]:
    """Build self-refine messages allowing the model to keep or revise the draft."""

    return [
        {
            "role": "user",
            "content": REFINE_PROMPT_TEMPLATE.format(
                problem=problem, draft=draft, critique=critique
            ),
        }
    ]


def find_boxed_content_token_indices(
    token_ids: Sequence[int],
    decode_tokens: Callable[[Sequence[int]], str],
) -> list[int]:
    """Locate generated token indices overlapping the inner content of the last `\\boxed{...}`.

    Uses cumulative decoding on the tail of `token_ids` so token boundaries match the exact
    generation sequence without re-tokenization drift.
    """

    n = len(token_ids)
    if n == 0:
        return []
    full_text = decode_tokens(token_ids)
    found = get_last_boxed_with_span(full_text)
    if found is None:
        return []
    _, char_start, char_end = found
    if char_end <= char_start:
        return []

    # Find exact character offset intervals for each token via prefix lengths.
    # For efficiency on long outputs, start searching near the end where `\boxed` appears.
    prefix_start_idx = 0
    for probe in range(max(0, n - 128), -1, -1):
        suffix_text = decode_tokens(token_ids[probe:])
        if r"\boxed" in suffix_text:
            prefix_start_idx = probe
            break

    base_prefix_len = len(decode_tokens(token_ids[:prefix_start_idx])) if prefix_start_idx else 0
    prev_len = base_prefix_len
    overlapping: list[int] = []
    for idx in range(prefix_start_idx, n):
        cur_len = len(decode_tokens(token_ids[: idx + 1]))
        tok_start, tok_end = prev_len, cur_len
        if tok_end > char_start and tok_start < char_end:
            overlapping.append(idx)
        prev_len = cur_len
        if tok_start >= char_end:
            break
    return overlapping


def extract_router_features(
    token_logprobs: Sequence[float],
    token_ids: Sequence[int],
    decode_tokens: Callable[[Sequence[int]], str],
    *,
    input_tokens: int,
    truncated: bool,
    response_text: str | None = None,
) -> dict[str, float | int | bool]:
    """Compute the 6 strictly online first-pass features for the logistic router.

    Features:
      1. `mean_token_logprob`
      2. `min_token_logprob`
      3. `answer_logprob`
      4. `output_length`
      5. `format_valid` (`not truncated and has_boxed`)
      6. `question_length` (`input_tokens`)
    """

    if len(token_logprobs) != len(token_ids):
        raise ValueError("token_logprobs and token_ids must have identical length")

    clean_lps = [float(lp) if math.isfinite(float(lp)) else -20.0 for lp in token_logprobs]
    out_len = len(clean_lps)
    if out_len == 0:
        mean_lp = -20.0
        min_lp = -20.0
    else:
        mean_lp = sum(clean_lps) / out_len
        min_lp = min(clean_lps)

    text = response_text if response_text is not None else decode_tokens(token_ids)
    has_boxed = get_last_boxed_with_span(text) is not None
    answer_indices = find_boxed_content_token_indices(token_ids, decode_tokens)
    if answer_indices:
        answer_lp = sum(clean_lps[i] for i in answer_indices) / len(answer_indices)
        answer_span_found = True
    else:
        answer_lp = mean_lp
        answer_span_found = False

    format_valid = float((not truncated) and has_boxed)
    return {
        "mean_token_logprob": float(mean_lp),
        "min_token_logprob": float(min_lp),
        "answer_logprob": float(answer_lp),
        "output_length": int(out_len),
        "format_valid": float(format_valid),
        "question_length": int(input_tokens),
        "has_boxed": bool(has_boxed),
        "answer_span_found": bool(answer_span_found),
    }


def load_completed_ids(jsonl_path: Path) -> set[str]:
    """Read already-checkpointed `unique_id` values from a JSONL cache file."""

    completed: set[str] = set()
    if not jsonl_path.is_file():
        return completed
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = rec.get("unique_id")
            if uid is not None:
                completed.add(str(uid))
    return completed


def append_jsonl_records(jsonl_path: Path, records: Sequence[dict[str, Any]]) -> None:
    """Append records to a JSONL checkpoint file safely even after a truncated crash."""

    if not records:
        return
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    needs_leading_newline = False
    if jsonl_path.is_file() and jsonl_path.stat().st_size > 0:
        with jsonl_path.open("rb") as rb:
            rb.seek(-1, 2)
            needs_leading_newline = rb.read(1) != b"\n"
    with jsonl_path.open("a", encoding="utf-8", newline="\n") as handle:
        if needs_leading_newline:
            handle.write("\n")
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        handle.flush()


def read_jsonl_records(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load JSONL records deduplicated by `unique_id` (latest row wins)."""

    by_id: dict[str, dict[str, Any]] = {}
    if not jsonl_path.is_file():
        return []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = rec.get("unique_id")
            if uid is not None:
                by_id[str(uid)] = rec
    return list(by_id.values())


def summarize_pilot_diagnostics(
    atom1_records: Sequence[dict[str, Any]],
    atom2_records: Sequence[dict[str, Any]],
    atom3_records: Sequence[dict[str, Any]],
    atom4_records: Sequence[dict[str, Any]],
    atom5_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compute Step 1B Pilot sanity metrics across Atom 1-5 caches."""

    def _by_uid(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {str(r["unique_id"]): r for r in rows}

    a1 = _by_uid(atom1_records)
    a2 = _by_uid(atom2_records)
    a3 = _by_uid(atom3_records)
    a4 = _by_uid(atom4_records)
    a5 = _by_uid(atom5_records)

    def _atom_stats(rows: Sequence[dict[str, Any]], prefix: str = "") -> dict[str, Any]:
        if not rows:
            return {"count": 0}
        n = len(rows)
        statuses = {s: 0 for s in ("correct", "wrong", "invalid", "truncated")}
        for r in rows:
            st = str(r.get(f"{prefix}status", r.get("status", "wrong")))
            statuses[st] = statuses.get(st, 0) + 1
        out_toks = [float(r.get(f"{prefix}out_tokens", r.get("out_tokens", 0))) for r in rows]
        lats = [float(r.get(f"{prefix}latency_s", r.get("latency_s", 0.0))) for r in rows]
        return {
            "count": n,
            "accuracy": statuses["correct"] / n,
            "statuses": statuses,
            "truncated_rate": statuses["truncated"] / n,
            "invalid_rate": statuses["invalid"] / n,
            "mean_out_tokens": sum(out_toks) / n,
            "mean_latency_s": sum(lats) / n,
        }

    # Flip matrix on paired Atom 1 -> Atom 2
    common_12 = sorted(set(a1) & set(a2))
    flips = {"right_to_right": 0, "right_to_wrong": 0, "wrong_to_right": 0, "wrong_to_wrong": 0}
    for uid in common_12:
        c1 = bool(a1[uid]["correct"])
        c2 = bool(a2[uid]["refine_correct"])
        if c1 and c2:
            flips["right_to_right"] += 1
        elif c1 and not c2:
            flips["right_to_wrong"] += 1
        elif not c1 and c2:
            flips["wrong_to_right"] += 1
        else:
            flips["wrong_to_wrong"] += 1

    # Level breakdown for 1.7B vs 8B
    level_acc: dict[str, dict[str, float]] = {}
    for lv in range(1, 6):
        lv_1 = [r for r in atom1_records if int(r.get("level", 0)) == lv]
        lv_4 = [r for r in atom4_records if int(r.get("level", 0)) == lv]
        level_acc[f"level_{lv}"] = {
            "qwen3_1.7b_acc": (sum(bool(r["correct"]) for r in lv_1) / len(lv_1)) if lv_1 else 0.0,
            "qwen3_8b_acc": (sum(bool(r["correct"]) for r in lv_4) / len(lv_4)) if lv_4 else 0.0,
        }

    # Feature finite check on Atom 1
    feat_keys = (
        "mean_token_logprob",
        "min_token_logprob",
        "answer_logprob",
        "output_length",
        "format_valid",
        "question_length",
    )
    finite_ok = all(
        all(math.isfinite(float(r["features"][k])) for k in feat_keys) for r in atom1_records
    )

    # Atom 3 sample-0 accuracy sanity
    a3_s0_acc = (
        sum(bool(r["samples"][0]["correct"]) for r in atom3_records if r.get("samples"))
        / len(atom3_records)
        if atom3_records
        else 0.0
    )

    return {
        "atom1_1.7b_nonthink": _atom_stats(atom1_records),
        "atom2_1.7b_self_refine": _atom_stats(atom2_records, prefix="refine_"),
        "atom3_1.7b_sample0_acc": a3_s0_acc,
        "atom3_count": len(atom3_records),
        "atom4_8b_nonthink": _atom_stats(atom4_records),
        "atom5_1.7b_thinking": _atom_stats(atom5_records),
        "self_refine_flips": flips,
        "level_accuracy": level_acc,
        "features_all_finite": finite_ok,
        "common_12_count": len(common_12),
        "common_14_count": len(set(a1) & set(a4)),
        "common_15_count": len(set(a1) & set(a5)),
    }


def grade_and_check_sample(
    text: str,
    ground_truth: str,
    *,
    truncated: bool,
    strip_thinking: bool,
    think_closed: bool | None = None,
) -> dict[str, Any]:
    """Thin dict-returning wrapper around `classify_response` for JSONL serialization."""

    return classify_response(
        text,
        ground_truth,
        truncated=truncated,
        strip_thinking=strip_thinking,
        think_closed=think_closed,
    ).to_dict()


def split_stratified_halves(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Partition `rows` into two level-stratified halves (`core`, `extra`) deterministically."""

    by_level: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        lv = int(row.get("level", 0))
        by_level.setdefault(lv, []).append(row)

    core_uids: set[str] = set()
    for lv in sorted(by_level):
        grp = by_level[lv]
        half = len(grp) // 2
        core_uids.update(str(r["unique_id"]) for r in grp[:half])

    core = [r for r in rows if str(r["unique_id"]) in core_uids]
    extra = [r for r in rows if str(r["unique_id"]) not in core_uids]
    return core, extra
