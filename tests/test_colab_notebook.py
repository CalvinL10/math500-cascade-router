"""Tests for Step 1B pure-Python logic and Colab notebook structural/AST integrity.

Strictly does NOT import or execute HuggingFace models or local GPU/LLM inference.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import nbformat as nbf
import pytest

from src.colab_helpers import (
    build_critique_messages,
    build_refine_messages,
    build_solve_messages,
    extract_router_features,
    find_boxed_content_token_indices,
    load_completed_ids,
    append_jsonl_records,
    read_jsonl_records,
    split_stratified_halves,
    summarize_pilot_diagnostics,
)
from scripts.build_colab_notebook import build_notebook
from scripts.build_vllm_probe import build_vllm_probe_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "phase1_colab_data_collection.ipynb"
PROBE_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "track_a_vllm_probe.ipynb"


def test_boxed_span_token_alignment_and_router_features() -> None:
    # Synthetic token stream: ["Solve", " -> ", "\\boxed{", "4", "2", "}."]
    vocab = {0: "Solve", 1: " -> ", 2: r"\boxed{", 3: "4", 4: "2", 5: "}."}
    token_ids = [0, 1, 2, 3, 4, 5]
    logprobs = [-0.5, -0.4, -0.2, -0.1, -0.3, -0.1]
    decode_fn = lambda ids: "".join(vocab[i] for i in ids)

    span_indices = find_boxed_content_token_indices(token_ids, decode_fn)
    assert span_indices == [3, 4]

    feats = extract_router_features(
        logprobs,
        token_ids,
        decode_fn,
        input_tokens=42,
        truncated=False,
    )
    assert feats["has_boxed"] is True
    assert feats["answer_span_found"] is True
    assert feats["format_valid"] == 1.0
    assert feats["question_length"] == 42
    assert feats["output_length"] == 6
    assert feats["min_token_logprob"] == pytest.approx(-0.5)
    assert feats["mean_token_logprob"] == pytest.approx(sum(logprobs) / 6)
    assert feats["answer_logprob"] == pytest.approx((-0.1 - 0.3) / 2)


def test_router_features_when_missing_boxed_or_truncated() -> None:
    vocab = {0: "Still", 1: " working"}
    token_ids = [0, 1]
    logprobs = [-1.2, -0.8]
    decode_fn = lambda ids: "".join(vocab[i] for i in ids)

    feats = extract_router_features(
        logprobs,
        token_ids,
        decode_fn,
        input_tokens=30,
        truncated=True,
    )
    assert feats["has_boxed"] is False
    assert feats["answer_span_found"] is False
    assert feats["format_valid"] == 0.0
    assert feats["answer_logprob"] == pytest.approx(-1.0)


def test_prompt_builders_remain_neutral_and_handle_latex_braces() -> None:
    latex_prob = r"Compute $\frac{1}{2} + \frac{1}{3}$ for $x \in \{1, 2\}$."
    latex_draft = r"We compute $\frac{5}{6}$, so \boxed{\frac{5}{6}}."
    s = build_solve_messages(latex_prob)[0]["content"]
    c = build_critique_messages(latex_prob, latex_draft)[0]["content"]
    r = build_refine_messages(latex_prob, latex_draft, "No errors found.")[0]["content"]
    assert r"\boxed{...}" in s
    assert latex_prob in s
    assert "No errors found; the draft is correct." in c
    assert latex_draft in c
    assert "confirm and keep the original answer" in r
    assert r"\boxed{...}" in r


def test_jsonl_checkpoint_resume_and_dedup(tmp_path: Path) -> None:
    p = tmp_path / "atom1.jsonl"
    append_jsonl_records(p, [{"unique_id": "a", "v": 1}, {"unique_id": "b", "v": 2}])
    # Simulate crash mid-line
    with p.open("a", encoding="utf-8") as fh:
        fh.write('{"unique_id": "broken')
    done = load_completed_ids(p)
    assert done == {"a", "b"}
    append_jsonl_records(p, [{"unique_id": "b", "v": 9}, {"unique_id": "c", "v": 3}])
    rows = read_jsonl_records(p)
    assert {r["unique_id"]: r["v"] for r in rows} == {"a": 1, "b": 9, "c": 3}


def test_pilot_diagnostics_summary_math() -> None:
    a1 = [
        {
            "unique_id": "1",
            "level": 1,
            "status": "correct",
            "correct": True,
            "out_tokens": 50,
            "latency_s": 0.2,
            "features": {
                "mean_token_logprob": -0.1,
                "min_token_logprob": -0.4,
                "answer_logprob": -0.05,
                "output_length": 50,
                "format_valid": 1.0,
                "question_length": 20,
            },
        },
        {
            "unique_id": "2",
            "level": 5,
            "status": "wrong",
            "correct": False,
            "out_tokens": 120,
            "latency_s": 0.4,
            "features": {
                "mean_token_logprob": -0.3,
                "min_token_logprob": -1.5,
                "answer_logprob": -0.6,
                "output_length": 120,
                "format_valid": 1.0,
                "question_length": 30,
            },
        },
    ]
    a2 = [
        {"unique_id": "1", "refine_status": "wrong", "refine_correct": False, "refine_out_tokens": 60, "refine_latency_s": 0.3},
        {"unique_id": "2", "refine_status": "correct", "refine_correct": True, "refine_out_tokens": 90, "refine_latency_s": 0.3},
    ]
    a3 = [
        {"unique_id": "1", "samples": [{"correct": True}, {"correct": True}, {"correct": False}]},
        {"unique_id": "2", "samples": [{"correct": False}, {"correct": False}, {"correct": False}]},
    ]
    a4 = [
        {"unique_id": "1", "level": 1, "status": "correct", "correct": True, "out_tokens": 45, "latency_s": 0.5},
        {"unique_id": "2", "level": 5, "status": "correct", "correct": True, "out_tokens": 110, "latency_s": 0.8},
    ]
    a5 = [
        {"unique_id": "1", "status": "correct", "correct": True, "out_tokens": 300, "latency_s": 1.1},
        {"unique_id": "2", "status": "truncated", "correct": False, "out_tokens": 4096, "latency_s": 8.0},
    ]
    summary = summarize_pilot_diagnostics(a1, a2, a3, a4, a5)
    assert summary["self_refine_flips"] == {
        "right_to_right": 0,
        "right_to_wrong": 1,
        "wrong_to_right": 1,
        "wrong_to_wrong": 0,
    }
    assert summary["features_all_finite"] is True
    assert summary["atom1_1.7b_nonthink"]["accuracy"] == pytest.approx(0.5)
    assert summary["atom4_8b_nonthink"]["accuracy"] == pytest.approx(1.0)
    assert summary["atom5_1.7b_thinking"]["truncated_rate"] == pytest.approx(0.5)


def test_generated_colab_notebook_is_valid_and_parses_cleanly() -> None:
    generated = build_notebook(NOTEBOOK_PATH)
    assert generated.is_file()
    nb = nbf.read(generated, as_version=4)
    nbf.validate(nb)
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 7

    for idx, cell in enumerate(code_cells):
        src = cell["source"]
        # Strip any IPython line magics before AST parsing
        cleaned = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith(("!", "%"))
        )
        ast.parse(cleaned, filename=f"colab_cell_{idx}")

    full_source = "\n".join(c["source"] for c in code_cells)
    assert 'RUN_MODE = "pilot"' in full_source
    assert 'RUN_TAG = "v2"' in full_source
    assert 'ENGINE_NAME = "vllm"' in full_source
    assert "ALLOW_NON_COLAB_OVERRIDE = False" in full_source
    assert "MAX_NEW_TOKENS_NON_THINK = 2048" in full_source
    assert "MAX_NEW_TOKENS_CRITIQUE = 512" in full_source
    assert "MAX_NEW_TOKENS_THINK = 4096" in full_source
    assert '"out_dir": "router_val_200", "atoms": (1, 4)' in full_source
    assert '"label": "router_train_core_300"' in full_source
    assert '"label": "router_train_extra_300"' in full_source
    assert "SamplingParams" in full_source
    assert '"logprobs": 1' in full_source
    assert "Qwen/Qwen3-1.7B" in full_source
    assert "Qwen/Qwen3-8B" in full_source
    assert "</think>" in full_source
    for fn_name in (
        "def grade_answer(",
        "def classify_response(",
        "def build_splits(",
        "def extract_router_features(",
        "def split_stratified_halves(",
        "def summarize_pilot_diagnostics(",
    ):
        assert fn_name in code_cells[2]["source"]

    manifest = json.loads((PROJECT_ROOT / "data" / "splits" / "splits_manifest.json").read_text(encoding="utf-8"))
    for name, info in manifest["files"].items():
        if "sha256" in info:
            assert info["sha256"] in full_source


def test_split_stratified_halves_exact_balance_and_disjointness() -> None:
    train_path = PROJECT_ROOT / "data" / "splits" / "router_train_600.json"
    if not train_path.is_file():
        pytest.skip("run scripts/prepare_data.py first")
    rows = json.loads(train_path.read_text(encoding="utf-8"))
    core, extra = split_stratified_halves(rows)
    assert len(core) == 300
    assert len(extra) == 300
    core_ids = {r["unique_id"] for r in core}
    extra_ids = {r["unique_id"] for r in extra}
    assert not (core_ids & extra_ids)
    assert core_ids | extra_ids == {r["unique_id"] for r in rows}
    from collections import Counter
    assert dict(Counter(r["level"] for r in core)) == {1: 26, 2: 54, 3: 63, 4: 77, 5: 80}
    assert dict(Counter(r["level"] for r in extra)) == {1: 26, 2: 54, 3: 63, 4: 77, 5: 80}


def test_generated_vllm_probe_notebook_is_valid_and_parses_cleanly() -> None:
    generated = build_vllm_probe_notebook(PROBE_NOTEBOOK_PATH)
    assert generated.is_file()
    nb = nbf.read(generated, as_version=4)
    nbf.validate(nb)
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 3

    for idx, cell in enumerate(code_cells):
        ast.parse(cell["source"], filename=f"vllm_probe_cell_{idx}")

    full_source = "\n".join(c["source"] for c in code_cells)
    assert "ALLOW_NON_COLAB_OVERRIDE = False" in full_source
    assert "Qwen/Qwen3-1.7B" in full_source
    assert "Qwen/Qwen3-8B" in full_source
    assert "logprobs=1" in full_source
    assert "enforce_eager" in full_source
