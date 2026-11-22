"""Unit tests for the ported MATH-500 grader and 4-state reliability classification."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.grading import (
    classify_response,
    extract_final_candidate,
    get_last_boxed,
    grade_answer,
    split_thinking_response,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# All 16 official tests from course notebook demo_base_model_evaluation.ipynb (cell 28)
PROFESSOR_CH03_TESTS = [
    ("check_1", "3/4", r"\frac{3}{4}", True),
    ("check_2", "(3)/(4)", r"3/4", True),
    ("check_3", r"\frac{\sqrt{8}}{2}", "sqrt(2)", True),
    ("check_4", r"\( \frac{1}{2} + \frac{1}{6} \)", "2/3", True),
    ("check_5", "(1, 2)", r"(1,2)", True),
    ("check_6", "(2, 1)", "(1, 2)", False),
    ("check_7", "(1, 2, 3)", "(1, 2)", False),
    ("check_8", "0.5", "1/2", True),
    ("check_9", "0.3333333333", "1/3", False),
    ("check_10", "1,234/2", "617", True),
    ("check_11", r"\text{2/3}", "2/3", True),
    ("check_12", "50%", "1/2", False),
    ("check_13", r"2\cdot 3/4", "3/2", True),
    ("check_14", r"90^\circ", "90", True),
    ("check_15", r"\left(\frac{3}{4}\right)", "3/4", True),
    ("check_16", "2²", "2**2", True),
]


@pytest.mark.parametrize(("name", "pred", "truth", "expected"), PROFESSOR_CH03_TESTS)
def test_professor_ch03_equivalence_suite(name: str, pred: str, truth: str, expected: bool) -> None:
    assert grade_answer(pred, truth) is expected, name


@pytest.mark.parametrize(
    ("pred", "truth", "expected"),
    [
        (r"\frac{1}{2}", r"\frac12", True),
        (r"-\frac{3}{2}", r"-\frac 32", True),
        (r"\frac{9}{19}", r"\frac9{19}", True),
        (r"\frac{50}{3}", r"\frac{50}3", True),
        (r"6 \frac{1}{5}", r"6\frac15", True),
        (r"6/5", r"6\frac15", False),
        (r"11\sqrt{2}", r"11\sqrt2", True),
        (r"\sqrt{2} + \sqrt{5}", r"\sqrt 2+\sqrt 5", True),
        (r"\left(2, \frac{7\pi}{4}\right)", r"\left( 2, \frac{7 \pi}{4} \right)", True),
        (r"\frac{11}{15}", r"\frac{11}{15}\text{ days}", True),
        (r"864", r"864 \mbox{ inches}^2", True),
        (r"k = \frac{12}{7}", r"\frac{12}{7}", True),
        (r"5", r"x=5", True),
        (r"2x - y + 3 = 0", r"y = 2x + 3", True),
        (r"y = 2x + 4", r"y = 2x + 3", False),
        (r"0.67", r"\$\dfrac23 \approx \$0.67", True),
        (r"\mathbf{M} = \begin{pmatrix} 4 & 2 \\ -1 & 1 \end{pmatrix}", r"\begin{pmatrix} 4 & 2 \\ -1 & 1 \end{pmatrix}", True),
        (r"\begin{pmatrix} 4 & 2 \\ -1 & 2 \end{pmatrix}", r"\begin{pmatrix} 4 & 2 \\ -1 & 1 \end{pmatrix}", False),
        (r"(-oo, 0]", r"(-\infty, 0]", True),
        (r"[-oo, 0]", r"(-\infty, 0]", False),
        (r"[-2, 7]", r"x \in [-2,7]", True),
        (r"1 + \sqrt{19}, 1 - \sqrt{19}", r"1 \pm \sqrt{19}", True),
        (r"49", "049", True),
        (r"(2, 12) \cup (12, 102)", r"(2,12) \cup (12,102)", True),
        (r"(1,234)", r"(1234)", False),
        (r"11111111100", r"11,\! 111,\! 111,\! 100", True),
        (r"58,500", "58500", True),
        (r"1, 100, 200", "1100200", False),
        (r"\begin{pmatrix} -\frac{1}{3} \\ \frac{2}{3} \\ \frac{5}{3} \end{pmatrix}", r"\begin{pmatrix} -1/3 \\ 2/3 \\ 5/3 \end{pmatrix}", True),
        (r"\begin{pmatrix} \dfrac{16}{49} \\ \dfrac{48}{49} \\ \dfrac{24}{49} \end{pmatrix}", r"\begin{pmatrix} 16/49 \\ 48/49 \\ 24/49 \end{pmatrix}", True),
        ("1,234.56", "1234.56", True),
        (r"\$1,234.56", "1234.56", True),
        ("0.1,234", "0.1234", False),
        (r"\{1,234\}", r"\{1, 234\}", True),
        (r"\{1,234\}", "1234", False),
        (r"\{100,200,300\}", "100200300", False),
        ("14916", "14{,}916", True),
        (r"30p\sqrt{p}", r"30p \sqrt{p}", True),
        (r"36x\sqrt{35x}", r"36x \sqrt{35x}", True),
    ],
)
def test_extended_math500_latex_equivalence(pred: str, truth: str, expected: bool) -> None:
    assert grade_answer(pred, truth) is expected


def test_nested_boxed_and_fallback_behavior() -> None:
    text = r"First try \boxed{1}. Final answer: \[\boxed{\dfrac{14}{3}}\]"
    assert get_last_boxed(text) == r"\dfrac{14}{3}"
    assert extract_final_candidate(text) == r"\dfrac{14}{3}"
    assert grade_answer(extract_final_candidate(text), r"\frac{14}{3}") is True

    prose_only = "I am not sure how to solve this problem."
    assert extract_final_candidate(prose_only, fallback="number_only") == ""


def test_four_state_classification_contract() -> None:
    c = classify_response(r"Therefore \boxed{\frac{1}{2}}.", "0.5", truncated=False)
    assert c.status == "correct"
    assert c.correct is True
    assert c.has_boxed is True

    w = classify_response(r"Therefore \boxed{7}.", "8", truncated=False)
    assert w.status == "wrong"
    assert w.correct is False
    assert w.has_boxed is True

    inv = classify_response("Sorry, I cannot solve this.", "8", truncated=False)
    assert inv.status == "invalid"
    assert inv.correct is False
    assert inv.has_boxed is False

    # Truncated responses always count as truncated / incorrect under primary contract,
    # even if a boxed string happened to appear right before hitting max_new_tokens.
    tr = classify_response(r"So \boxed{8}", "8", truncated=True)
    assert tr.status == "truncated"
    assert tr.correct is False
    assert tr.extracted_correct_ignoring_truncation is True


def test_thinking_splitter_and_unclosed_think() -> None:
    body, closed = split_thinking_response("lots of math</think>\n\nAnswer: \\boxed{42}", think_in_prompt=True)
    assert closed is True
    assert r"\boxed{42}" in body
    res = classify_response(body, "42", truncated=False, strip_thinking=True, think_closed=closed)
    assert res.status == "correct"

    unclosed_body, closed2 = split_thinking_response("still thinking without closing", think_in_prompt=True)
    assert closed2 is False
    assert unclosed_body == ""
    res2 = classify_response(
        "still thinking without closing \\boxed{42}",
        "42",
        truncated=True,
        strip_thinking=True,
        think_closed=False,
    )
    assert res2.status == "truncated"
    assert res2.correct is False


def test_all_split_ground_truths_grade_against_themselves() -> None:
    splits_dir = PROJECT_ROOT / "data" / "splits"
    for split_name in ("pilot_50", "router_train_600", "router_val_200", "math500_test_500"):
        path = splits_dir / f"{split_name}.json"
        if not path.is_file():
            pytest.skip("run scripts/prepare_data.py first")
        rows = json.loads(path.read_text(encoding="utf-8"))
        failures = [r["unique_id"] for r in rows if not grade_answer(r["answer"], r["answer"])]
        assert failures == [], f"self-grading failed in {split_name}: {failures}"
