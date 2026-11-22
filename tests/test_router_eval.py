"""Unit and integration tests for audit-hardened Phase 2 (router) and Phase 3 (replay/eval)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import (
    _SymmetricEquivalenceCache,
    _majority_vote_cluster,
    _read_jsonl_strict,
    apply_holm_bonferroni,
    evaluate_val_locked_mask,
    load_aligned_split,
    load_latency_calibration,
    mcnemar_test,
    paired_bootstrap_test,
    run_phase2_phase3_pipeline,
    trace_routing_curve,
)
from src.router import (
    CostAwareRouter,
    ONLINE_FEATURE_NAMES,
    compute_curve_auac,
    compute_ece,
    evaluate_calibration_methods,
    extract_feature_matrix,
    lock_val_thresholds,
    run_dev_split_stability_check,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = PROJECT_ROOT / "outputs" / "math550_full_v2_bundle"
PHASE23_DIR = PROJECT_ROOT / "outputs" / "phase2_3"


def _sample_record(uid: str = "x1", **feature_overrides: float) -> dict:
    feats = {
        "mean_token_logprob": -0.08,
        "min_token_logprob": -1.10,
        "answer_logprob": -0.02,
        "output_length": 400.0,
        "format_valid": 1.0,
        "question_length": 80.0,
    }
    feats.update(feature_overrides)
    return {"unique_id": uid, "level": 5, "correct": True, "features": feats}


def test_feature_extraction_forbids_leakage_and_nan() -> None:
    recs = [_sample_record("q1"), _sample_record("q2", output_length=600.0)]
    X = extract_feature_matrix(recs, ONLINE_FEATURE_NAMES)
    assert X.shape == (2, 6)
    assert np.all(np.isfinite(X))

    for bad_col in ("level", "subject", "correct", "ground_truth"):
        with pytest.raises(ValueError, match="leakage"):
            extract_feature_matrix(recs, ("mean_token_logprob", bad_col))

    nan_recs = [_sample_record("q_nan", min_token_logprob=float("nan"))]
    with pytest.raises(ValueError, match="Non-finite"):
        extract_feature_matrix(nan_recs)


def test_strict_jsonl_reader_fails_fast(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.jsonl"
    bad_json.write_text('{"unique_id": "a"}\n{corrupt\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSON"):
        _read_jsonl_strict(bad_json)

    dup_json = tmp_path / "dup.jsonl"
    dup_json.write_text('{"unique_id": "a"}\n{"unique_id": "a"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate unique_id"):
        _read_jsonl_strict(dup_json)


def test_representative_first_majority_vote_ignores_stale_sample_correct() -> None:
    eq_cache = _SymmetricEquivalenceCache()
    # Even if `s["correct"]` were misleadingly True on wrong answers, cluster winning on
    # extracted_answer ("5") must be graded once against ground_truth ("42") -> False.
    samples = [
        {"truncated": False, "extracted_answer": "5", "correct": True},
        {"truncated": False, "extracted_answer": "5.0", "correct": True},
        {"truncated": False, "extracted_answer": "42", "correct": False},
    ]
    is_corr, status, rep = _majority_vote_cluster(samples, "42", eq_cache)
    assert rep in {"5", "5.0"}
    assert is_corr is False
    assert status == "wrong"


def test_direct_dispatch_cost_never_double_counts_small_model() -> None:
    # 2 items: item 0 escalated, item 1 kept on 1.7B
    scores = np.array([0.9, 0.1])
    y17 = np.array([0, 1])
    y8 = np.array([1, 1])
    tok17 = np.array([1000.0, 200.0])
    tok8 = np.array([600.0, 300.0])
    bat17 = np.array([1.0, 0.2])
    bat8 = np.array([3.0, 1.5])
    curve_dir = trace_routing_curve(
        scores, y17, y8, tok17, tok8, bat17, bat8, 100.0, 20.0, dispatch_mode="direct"
    )
    curve_cas = trace_routing_curve(
        scores, y17, y8, tok17, tok8, bat17, bat8, 100.0, 20.0, dispatch_mode="cascade"
    )
    # At k=1 (item 0 escalated): direct pays (600 + 200)/2 = 400 tokens; cascade pays (1000 + 600 + 200)/2 = 900.
    assert curve_dir["mean_tokens"][1] == pytest.approx(400.0)
    assert curve_cas["mean_tokens"][1] == pytest.approx(900.0)
    # At k=2 (100% direct 8B): direct pays (600+300)/2 = 450; cascade pays (1000+600+200+300)/2 = 1050.
    assert curve_dir["mean_tokens"][2] == pytest.approx(450.0)
    assert curve_cas["mean_tokens"][2] == pytest.approx(1050.0)
    # Non-monotone direct token cost (600 -> 400 -> 450) must return None for tok_auac.
    assert curve_dir["tok_auac"] is None
    assert curve_dir["tok_excess_over_endpoints_line"] is None
    assert isinstance(curve_dir["lat_auac"], float)
    assert compute_curve_auac(np.array([600.0, 400.0, 450.0]), np.array([0.5, 1.0, 1.0]), allow_non_monotone=False) == (None, None)


def test_dev_split_stability_check_uses_auto_calibration_and_tracks_8b() -> None:
    rng = np.random.default_rng(550)
    n = 300
    X = rng.normal(size=(n, len(ONLINE_FEATURE_NAMES)))
    X[:, ONLINE_FEATURE_NAMES.index("format_valid")] = 1.0
    y17 = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    y8 = (X[:, 0] + 0.3 + 0.3 * rng.normal(size=n) > 0).astype(np.int64)
    tok17 = rng.uniform(300.0, 900.0, size=n)
    tok8 = rng.uniform(300.0, 900.0, size=n)
    levels = np.tile(np.arange(1, 6), n // 5)
    trunc = np.zeros(n, dtype=bool)
    rep = run_dev_split_stability_check(
        X, y17, y8, tok17, tok8, levels, trunc, 67.0, 16.5, n_splits=2, seed=550
    )
    assert rep["calibration_method"] == "auto"
    assert sum(rep["lambda_winner_counts"].values()) == 2
    assert sum(rep["ece_winner_counts"].values()) == 2
    assert sum(rep["ece_winner_counts_8b"].values()) == 2
    assert sum(rep["brier_winner_counts_8b"].values()) == 2


def test_calibration_wiring_and_val_selection() -> None:
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0], dtype=np.int64)
    p = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.85, 0.15], dtype=np.float64)
    assert 0.0 <= compute_ece(y, p, n_bins=10) <= 1.0

    train = load_aligned_split(BUNDLE_DIR, "router_train_600", regrade=True)
    val = load_aligned_split(BUNDLE_DIR, "router_val_200", regrade=True)
    test = load_aligned_split(BUNDLE_DIR, "math500_test_500", regrade=True)
    lat_cal = load_latency_calibration(BUNDLE_DIR / "latency_calibration_v2.jsonl")
    tps8 = lat_cal["qwen3_8b_non_think"]["tokens_per_sec"]

    rep17 = evaluate_calibration_methods(
        train["X"], train["y17"], val["X"], val["y17"], test["X"], test["y17"], seed=550
    )
    rep8 = evaluate_calibration_methods(
        train["X"], train["y8"], val["X"], val["y8"], test["X"], test["y8"], seed=550
    )
    router = CostAwareRouter(tps_8b=tps8, calibration_method="auto", seed=550).fit(
        train["X"],
        train["y17"],
        train["y8"],
        train["tok8"],
        X_val=val["X"],
        y17_val=val["y17"],
        y8_val=val["y8"],
    )
    assert router.selected_calibration_17b_ == rep17["best_method"]
    assert router.selected_calibration_8b_ == rep8["best_method"]
    comp = router.predict_components(test["X"], truncated_mask=test["trunc17"])
    np.testing.assert_allclose(
        comp["p8"],
        rep8["fitted_estimators"][rep8["best_method"]].predict_proba(test["X"])[:, 1],
        rtol=1e-10,
        atol=1e-10,
    )


def test_val_threshold_locking_zero_test_leakage() -> None:
    val_scores = np.linspace(0.0, 1.0, 201)
    taus = lock_val_thresholds(val_scores, target_val_rates=(0.12, 0.20, 0.30))
    assert taus["0.12"] > taus["0.20"] > taus["0.30"]
    assert math.isclose(taus["0.20"], 0.80, rel_tol=1e-6)


def test_paired_bootstrap_mcnemar_and_holm() -> None:
    y_a = np.array([1] * 12 + [0] * 2 + [1] * 10 + [0] * 10)
    y_b = np.array([0] * 12 + [1] * 2 + [1] * 10 + [0] * 10)
    mc = mcnemar_test(y_a, y_b)
    assert mc["n_a_only"] == 12
    assert mc["n_b_only"] == 2
    assert 0.0 < float(mc["p_value"]) < 0.02

    boot = paired_bootstrap_test(y_a, y_b, n_boot=1000, seed=550)
    assert boot["diff"] == pytest.approx((12 - 2) / len(y_a))
    assert boot["ci_low"] > 0.0

    adj = apply_holm_bonferroni([0.01, None, 0.04, 0.03])
    assert adj[1] is None
    assert adj[0] == pytest.approx(0.03)
    assert adj[3] == pytest.approx(0.06)
    assert adj[2] == pytest.approx(0.06)


def test_end_to_end_repaired_phase2_phase3_outputs(tmp_path: Path) -> None:
    out_dir = tmp_path / "phase2_3_out"
    res = run_phase2_phase3_pipeline(BUNDLE_DIR, out_dir, seed=550, n_boot=5000, regrade=True)

    expected_files = [
        "phase2_router_diagnostics.json",
        "phase3_summary_metrics.json",
        "grading_regrade_diff.json",
        "grading_sensitivity_audit.json",
        "main_comparison_table.csv",
        "main_comparison_table.md",
        "statistical_tests.csv",
        "statistical_tests.md",
        "decile_analysis.csv",
        "case_studies.json",
        "figures/fig1_pareto_frontier.png",
        "figures/fig1_pareto_frontier.pdf",
        "figures/fig2_mechanism_decomposition.png",
        "figures/fig2_mechanism_decomposition.pdf",
        "figures/fig3_router_diagnostics.png",
        "figures/fig3_router_diagnostics.pdf",
        "figures/fig4_matched_budget_stat_tests.png",
        "figures/fig4_matched_budget_stat_tests.pdf",
    ]
    for rel in expected_files:
        p = out_dir / rel
        assert p.is_file(), f"Missing generated output: {p}"
        assert p.stat().st_size > 0, f"Empty generated output: {p}"

    p3 = res["phase3_summary"]
    assert p3["regraded_completion_diff_count"] == 25
    assert p3["baselines"]["atom1_1.7b_nonthink"]["accuracy"] == pytest.approx(0.690)
    assert p3["baselines"]["atom4_8b_direct"]["accuracy"] == pytest.approx(0.784)
    assert p3["baselines"]["atom5_1.7b_thinking"]["output_completed_count"] == 289
    assert p3["baselines"]["atom5_1.7b_thinking"]["think_closed_count"] == 326
    assert p3["baselines"]["atom5_1.7b_thinking"]["think_closed_but_body_truncated_count"] == 37
    assert p3["baselines"]["atom5_1.7b_thinking"]["think_unclosed_truncated_count"] == 174
    assert p3["val_locked_operating_points"]["truncation_only"]["fixed"]["accuracy"] == pytest.approx(0.730)
    assert p3["val_locked_operating_points"]["tier1_trunc_cost_utility"]["0.20"]["accuracy"] == pytest.approx(0.754)
    # Pre-gen direct dispatch at Val 20% must be faster than cascade (no double-charged 1.7B latency)
    pre20 = p3["val_locked_operating_points"]["pre_gen_qlen"]["0.20"]
    assert pre20["dispatch_mode"] == "direct"
    assert pre20["mean_tokens"] < 777.35

    if PHASE23_DIR.is_dir():
        summary_disk = json.loads(
            (PHASE23_DIR / "phase3_summary_metrics.json").read_text(encoding="utf-8")
        )
        assert summary_disk["regraded_completion_diff_count"] == 25
        assert summary_disk["baselines"]["atom1_1.7b_nonthink"]["accuracy"] == pytest.approx(0.690)
        assert (
            summary_disk["val_locked_operating_points"]["tier1_trunc_cost_utility"]["0.20"]["accuracy"]
            == pytest.approx(0.754)
        )
