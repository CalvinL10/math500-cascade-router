"""Phase 2: Zero-leakage router training, probability calibration, and Val threshold locking.

Fits lightweight tabular models on `router_train_600` using strictly online,
post-generation features from the base `Qwen3-1.7B` pass (`Atom 1`), selects
calibration and hyperparameters exclusively on `router_val_200`, wires the
selected calibration estimator directly into the deployed router, and assesses
selection stability across 30 stratified dev-only 600/200 re-splits.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


ONLINE_FEATURE_NAMES: tuple[str, ...] = (
    "mean_token_logprob",
    "min_token_logprob",
    "answer_logprob",
    "output_length",
    "format_valid",
    "question_length",
)

FORBIDDEN_LEAKAGE_KEYS: frozenset[str] = frozenset(
    {"level", "subject", "ground_truth", "correct", "status", "extracted_answer", "unique_id"}
)

DEFAULT_TPS_17B = 67.23592473
DEFAULT_TPS_8B = 16.57216817
TIER1_PRECEDENCE_OFFSET = 10.0


def trapz_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Compute trapezoidal integral compatibly across NumPy 1.x and 2.x."""

    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def extract_feature_matrix(
    records: Sequence[dict[str, Any]],
    feature_names: Sequence[str] = ONLINE_FEATURE_NAMES,
) -> np.ndarray:
    """Extract a finite float64 feature matrix while enforcing zero leakage."""

    forbidden = [name for name in feature_names if name in FORBIDDEN_LEAKAGE_KEYS]
    if forbidden:
        raise ValueError(f"Forbidden leakage feature(s) requested: {forbidden}")

    n_cols = len(feature_names)
    if not records:
        return np.empty((0, n_cols), dtype=np.float64)

    rows: list[list[float]] = []
    for idx, rec in enumerate(records):
        feat_dict = rec.get("features")
        if not isinstance(feat_dict, dict):
            raise ValueError(f"Record at index {idx} is missing a 'features' dict.")
        row: list[float] = []
        for name in feature_names:
            if name not in feat_dict:
                raise ValueError(
                    f"Record {rec.get('unique_id', idx)} is missing feature '{name}'."
                )
            val = float(feat_dict[name])
            if not math.isfinite(val):
                raise ValueError(
                    f"Non-finite value for feature '{name}' in record {rec.get('unique_id', idx)}: {val}"
                )
            row.append(val)
        rows.append(row)

    return np.asarray(rows, dtype=np.float64)


def compute_reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Compute non-empty reliability diagram bins with counts and binomial SEs."""

    y_arr = np.asarray(y_true, dtype=np.float64)
    p_arr = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    if y_arr.shape != p_arr.shape:
        raise ValueError(f"Shape mismatch: y_true {y_arr.shape} vs y_prob {p_arr.shape}")
    if y_arr.size == 0:
        raise ValueError("Cannot compute reliability bins on empty arrays.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict[str, float]] = []
    for i in range(n_bins):
        low, high = float(edges[i]), float(edges[i + 1])
        if i < n_bins - 1:
            mask = (p_arr >= low) & (p_arr < high)
        else:
            mask = (p_arr >= low) & (p_arr <= high)
        count = int(mask.sum())
        if count > 0:
            acc = float(y_arr[mask].mean())
            se = math.sqrt(max(acc * (1.0 - acc), 0.0) / count)
            bins.append(
                {
                    "bin_idx": float(i),
                    "bin_low": low,
                    "bin_high": high,
                    "count": float(count),
                    "mean_pred": float(p_arr[mask].mean()),
                    "empirical_acc": acc,
                    "se": float(se),
                }
            )
    return bins


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) over `n_bins` probability bins."""

    y_arr = np.asarray(y_true, dtype=np.float64)
    bins = compute_reliability_bins(y_arr, y_prob, n_bins=n_bins)
    n_total = float(len(y_arr))
    return float(
        sum((b["count"] / n_total) * abs(b["empirical_acc"] - b["mean_pred"]) for b in bins)
    )


def _make_lr_pipeline(seed: int = 550) -> Pipeline:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
    )


def build_calibration_candidates(seed: int = 550) -> dict[str, Any]:
    """Create candidate probability estimators trained exclusively on `Train`."""

    return {
        "uncalibrated": _make_lr_pipeline(seed=seed),
        "platt_cv5": CalibratedClassifierCV(
            _make_lr_pipeline(seed=seed), cv=5, method="sigmoid"
        ),
        "isotonic_cv5": CalibratedClassifierCV(
            _make_lr_pipeline(seed=seed), cv=5, method="isotonic"
        ),
    }


def evaluate_calibration_methods(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    seed: int = 550,
) -> dict[str, Any]:
    """Compare uncalibrated, Platt (`sigmoid`), and Isotonic CV5 probability calibration.

    All models and calibrators are fit strictly on `(X_train, y_train)` via 5-fold CV.
    Selection uses ONLY `Val` (`min_val_ece`). `X_test`/`y_test` are optional and
    purely for post-hoc diagnostic reporting.
    """

    estimators = build_calibration_candidates(seed=seed)
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
    }
    if X_test is not None and y_test is not None:
        splits["test"] = (X_test, y_test)

    methods_report: dict[str, dict[str, Any]] = {}
    fitted_estimators: dict[str, Any] = {}
    for method_name, est in estimators.items():
        est.fit(X_train, y_train)
        fitted_estimators[method_name] = est
        split_metrics: dict[str, Any] = {}
        for split_name, (X_s, y_s) in splits.items():
            probs = est.predict_proba(X_s)[:, 1]
            split_metrics[split_name] = {
                "n": int(len(y_s)),
                "roc_auc": float(roc_auc_score(y_s, probs)),
                "brier": float(brier_score_loss(y_s, probs)),
                "ece": compute_ece(y_s, probs, n_bins=10),
                "reliability_bins": compute_reliability_bins(y_s, probs, n_bins=10),
            }
        methods_report[method_name] = split_metrics

    best_method_ece = min(
        methods_report.keys(),
        key=lambda m: methods_report[m]["val"]["ece"],
    )
    best_method_brier = min(
        methods_report.keys(),
        key=lambda m: methods_report[m]["val"]["brier"],
    )

    return {
        "selection_criterion": "min_val_ece",
        "best_method": best_method_ece,
        "best_method_brier": best_method_brier,
        "methods": methods_report,
        "fitted_estimators": fitted_estimators,
    }


def compute_curve_auac(
    costs: np.ndarray,
    accs: np.ndarray,
    *,
    allow_non_monotone: bool = True,
) -> tuple[float | None, float | None]:
    """Compute normalized Area Under Accuracy-Cost Curve and excess over endpoints line."""

    c_arr = np.asarray(costs, dtype=np.float64)
    a_arr = np.asarray(accs, dtype=np.float64)
    if (
        not allow_non_monotone
        and c_arr.size >= 2
        and not np.all(np.diff(c_arr) >= -1e-12)
    ):
        return None, None
    if c_arr.size < 2 or float(c_arr.max() - c_arr.min()) <= 0.0:
        return float(a_arr.mean()), 0.0

    acc_start = float(a_arr[0])
    acc_end = float(a_arr[-1])
    rand_auac = 0.5 * (acc_start + acc_end)

    order = np.argsort(c_arr, kind="mergesort")
    c_sorted = c_arr[order]
    a_sorted = a_arr[order]
    area = trapz_integral(a_sorted, c_sorted) / float(c_sorted[-1] - c_sorted[0])
    return float(area), float(area - rand_auac)


def lock_val_thresholds(
    val_scores: np.ndarray,
    target_val_rates: Sequence[float] = (0.12, 0.20, 0.25, 0.30, 0.40, 0.50),
) -> dict[str, float]:
    """Freeze escalation thresholds `tau_q` using ONLY `router_val_200` score quantiles."""

    scores = np.asarray(val_scores, dtype=np.float64)
    if scores.size == 0:
        raise ValueError("Cannot lock thresholds on empty validation scores.")
    thresholds: dict[str, float] = {}
    for q in target_val_rates:
        if not (0.0 < q < 1.0):
            raise ValueError(f"target_val_rate must be in (0, 1), got {q}")
        thresholds[f"{q:.2f}"] = float(np.quantile(scores, 1.0 - float(q)))
    return thresholds


class CostAwareRouter:
    """Two-Tier Truncation-Fallback & Cost-Aware Expected Net-Utility Cascade Router.

    Fits lightweight tabular models exclusively on `router_train_600`:
      - `clf_17b`: P(1.7B correct | z_x) (using the Val-selected calibration family)
      - `clf_17b_hurdle_valid`: P(1.7B correct | z_x, format_valid=1) for structural Hurdle ablation
      - `clf_8b`: P(8B correct | z_x) (using the Val-selected calibration family)
      - `reg_tok8b`: E[8B out_tokens | z_x] clipped to [100, 2048]
      - `clf_pre_qlen`: Pre-generation baseline P(1.7B correct | question_length)
    """

    def __init__(
        self,
        *,
        tps_8b: float = DEFAULT_TPS_8B,
        calibration_method: str = "auto",
        seed: int = 550,
    ) -> None:
        self.tps_8b = float(tps_8b)
        self.calibration_method = str(calibration_method)
        self.seed = int(seed)

        self.base_lr_17b: Pipeline = _make_lr_pipeline(seed=self.seed)
        self.base_lr_8b: Pipeline = _make_lr_pipeline(seed=self.seed)
        self.clf_17b: Any = self.base_lr_17b
        self.clf_8b: Any = self.base_lr_8b
        self.clf_17b_hurdle_valid: Pipeline = _make_lr_pipeline(seed=self.seed)
        self.reg_tok8b: Pipeline = make_pipeline(
            StandardScaler(), Ridge(alpha=10.0, random_state=self.seed)
        )
        self.clf_pre_qlen: Pipeline = _make_lr_pipeline(seed=self.seed)

        self.selected_calibration_17b_: str = "uncalibrated"
        self.selected_calibration_8b_: str = "uncalibrated"
        self._is_fitted = False

        self._idx_min_lp = ONLINE_FEATURE_NAMES.index("min_token_logprob")
        self._idx_out_len = ONLINE_FEATURE_NAMES.index("output_length")
        self._idx_fmt_valid = ONLINE_FEATURE_NAMES.index("format_valid")
        self._idx_q_len = ONLINE_FEATURE_NAMES.index("question_length")

    def fit(
        self,
        X_train: np.ndarray,
        y17_train: np.ndarray,
        y8_train: np.ndarray,
        tok8_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y17_val: np.ndarray | None = None,
        y8_val: np.ndarray | None = None,
    ) -> "CostAwareRouter":
        """Fit estimators on Train 600 and wire the Val-selected calibrator directly."""

        X_arr = np.asarray(X_train, dtype=np.float64)
        y17_arr = np.asarray(y17_train, dtype=np.int64)
        y8_arr = np.asarray(y8_train, dtype=np.int64)
        tok8_arr = np.asarray(tok8_train, dtype=np.float64)

        self.base_lr_17b.fit(X_arr, y17_arr)
        self.base_lr_8b.fit(X_arr, y8_arr)

        if self.calibration_method == "auto":
            if X_val is None or y17_val is None:
                raise ValueError(
                    "X_val and y17_val are required when calibration_method='auto'."
                )
            rep17 = evaluate_calibration_methods(
                X_arr, y17_arr, X_val, np.asarray(y17_val, dtype=np.int64), seed=self.seed
            )
            self.selected_calibration_17b_ = rep17["best_method"]
            self.clf_17b = rep17["fitted_estimators"][self.selected_calibration_17b_]

            if y8_val is not None:
                rep8 = evaluate_calibration_methods(
                    X_arr, y8_arr, X_val, np.asarray(y8_val, dtype=np.int64), seed=self.seed
                )
                self.selected_calibration_8b_ = rep8["best_method"]
                self.clf_8b = rep8["fitted_estimators"][self.selected_calibration_8b_]
            else:
                self.selected_calibration_8b_ = "uncalibrated"
                self.clf_8b = self.base_lr_8b
        else:
            cands17 = build_calibration_candidates(seed=self.seed)
            cands8 = build_calibration_candidates(seed=self.seed)
            if self.calibration_method not in cands17:
                raise ValueError(f"Unknown calibration_method: {self.calibration_method}")
            self.clf_17b = cands17[self.calibration_method].fit(X_arr, y17_arr)
            self.clf_8b = cands8[self.calibration_method].fit(X_arr, y8_arr)
            self.selected_calibration_17b_ = self.calibration_method
            self.selected_calibration_8b_ = self.calibration_method

        valid_mask = X_arr[:, self._idx_fmt_valid] == 1.0
        self.clf_17b_hurdle_valid.fit(X_arr[valid_mask], y17_arr[valid_mask])
        self.reg_tok8b.fit(X_arr, tok8_arr)
        self.clf_pre_qlen.fit(X_arr[:, [self._idx_q_len]], y17_arr)
        self._is_fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("CostAwareRouter must be fit before calling predict.")

    def predict_components(
        self,
        X: np.ndarray,
        *,
        truncated_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Predict probability, latency, format-invalid, and truncation components."""

        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        p17 = self.clf_17b.predict_proba(X_arr)[:, 1]
        p8 = self.clf_8b.predict_proba(X_arr)[:, 1]
        format_valid = X_arr[:, self._idx_fmt_valid] == 1.0
        p17_hurdle = np.where(
            format_valid, self.clf_17b_hurdle_valid.predict_proba(X_arr)[:, 1], 0.0
        )
        pred_tok8 = np.clip(self.reg_tok8b.predict(X_arr), 100.0, 2048.0)
        pred_lat8 = pred_tok8 / self.tps_8b
        p17_pre_qlen = self.clf_pre_qlen.predict_proba(X_arr[:, [self._idx_q_len]])[:, 1]

        format_invalid = (~format_valid).astype(np.float64)
        if truncated_mask is not None:
            is_truncated = np.asarray(truncated_mask, dtype=bool).astype(np.float64)
        else:
            is_truncated = format_invalid

        return {
            "p17": p17,
            "p17_hurdle": p17_hurdle,
            "p8": p8,
            "pred_tok8": pred_tok8,
            "pred_lat8": pred_lat8,
            "p17_pre_qlen": p17_pre_qlen,
            "format_invalid": format_invalid,
            "is_truncated": is_truncated,
        }

    def compute_routing_scores(
        self,
        X: np.ndarray,
        lam: float = 0.0010,
        *,
        truncated_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute escalation priority scores across policies (higher => escalate).

        Note: `tier1_trunc_*` scores encode strict lexicographic Tier-1 escalation for
        truncated completions (`+ TIER1_PRECEDENCE_OFFSET * is_truncated`), followed by
        Tier-2 continuous expected net accuracy gain / cost-aware utility ranking among
        completed completions. They are explicit two-tier policies, not pure probabilities.
        """

        X_arr = np.asarray(X, dtype=np.float64)
        comp = self.predict_components(X_arr, truncated_mask=truncated_mask)
        p17 = comp["p17"]
        p17_hurdle = comp["p17_hurdle"]
        p8 = comp["p8"]
        pred_lat8 = comp["pred_lat8"]
        is_trunc = comp["is_truncated"]

        raw_net_gain = p8 - p17
        raw_cost_utility = raw_net_gain - float(lam) * pred_lat8
        hurdle_cost_utility = (p8 - p17_hurdle) - float(lam) * pred_lat8

        return {
            "conf_gate": 1.0 - p17,
            "net_gain": raw_net_gain,
            "raw_cost_utility": raw_cost_utility,
            "tier1_trunc_net_gain": TIER1_PRECEDENCE_OFFSET * is_trunc + raw_net_gain,
            "tier1_trunc_cost_utility": TIER1_PRECEDENCE_OFFSET * is_trunc + raw_cost_utility,
            "hurdle_trunc_cost_utility": TIER1_PRECEDENCE_OFFSET * is_trunc + hurdle_cost_utility,
            "truncation_only": is_trunc,
            "pre_gen_qlen": 1.0 - comp["p17_pre_qlen"],
            "min_logprob_only": -X_arr[:, self._idx_min_lp],
            "output_length_only": X_arr[:, self._idx_out_len],
        }

    def select_lambda_on_val(
        self,
        X_val: np.ndarray,
        y17_val: np.ndarray,
        y8_val: np.ndarray,
        tok17_val: np.ndarray,
        tok8_val: np.ndarray,
        tps_17b: float,
        tps_8b: float,
        *,
        truncated_val: np.ndarray | None = None,
        lambda_grid: Sequence[float] = (0.0, 0.0005, 0.0010, 0.0015, 0.0020),
    ) -> dict[str, Any]:
        """Select latency penalty `lambda` solely on `router_val_200` by maximizing LatAUAC."""

        self._check_fitted()
        lat17 = np.asarray(tok17_val, dtype=np.float64) / float(tps_17b)
        lat8 = np.asarray(tok8_val, dtype=np.float64) / float(tps_8b)
        y17_arr = np.asarray(y17_val, dtype=np.float64)
        y8_arr = np.asarray(y8_val, dtype=np.float64)
        n = len(y17_arr)

        sweep_records: list[dict[str, float]] = []
        for lam in lambda_grid:
            scores = self.compute_routing_scores(
                X_val, lam=float(lam), truncated_mask=truncated_val
            )["tier1_trunc_cost_utility"]
            order = np.argsort(-scores, kind="mergesort")
            lats = np.empty(n + 1, dtype=np.float64)
            accs = np.empty(n + 1, dtype=np.float64)

            cur_lat = lat17.copy()
            cur_acc = y17_arr.copy()
            lats[0] = float(cur_lat.mean())
            accs[0] = float(cur_acc.mean())

            for k in range(1, n + 1):
                idx = int(order[k - 1])
                cur_lat[idx] += lat8[idx]
                cur_acc[idx] = y8_arr[idx]
                lats[k] = float(cur_lat.mean())
                accs[k] = float(cur_acc.mean())

            lat_auac, lat_excess = compute_curve_auac(lats, accs)
            assert lat_auac is not None and lat_excess is not None
            sweep_records.append(
                {
                    "lambda": float(lam),
                    "val_lat_auac": float(lat_auac),
                    "val_lat_excess_over_random": float(lat_excess),
                }
            )

        best = max(sweep_records, key=lambda row: row["val_lat_auac"])
        return {
            "selection_criterion": "max_val_lat_auac",
            "best_lambda": float(best["lambda"]),
            "best_val_lat_auac": float(best["val_lat_auac"]),
            "sweep": sweep_records,
        }

    def get_standardized_coefficients(self) -> dict[str, dict[str, Any]]:
        """Return associative standardized LR/Ridge weights (non-causal)."""

        self._check_fitted()
        lr17 = self.base_lr_17b.named_steps["logisticregression"]
        lr8 = self.base_lr_8b.named_steps["logisticregression"]
        ridge8 = self.reg_tok8b.named_steps["ridge"]
        return {
            "clf_17b": {
                "intercept": float(lr17.intercept_[0]),
                "weights": {
                    name: float(w)
                    for name, w in zip(ONLINE_FEATURE_NAMES, lr17.coef_[0])
                },
            },
            "clf_8b": {
                "intercept": float(lr8.intercept_[0]),
                "weights": {
                    name: float(w)
                    for name, w in zip(ONLINE_FEATURE_NAMES, lr8.coef_[0])
                },
            },
            "reg_tok8b": {
                "intercept": float(ridge8.intercept_),
                "weights": {
                    name: float(w)
                    for name, w in zip(ONLINE_FEATURE_NAMES, ridge8.coef_)
                },
            },
        }


def run_dev_split_stability_check(
    X_dev: np.ndarray,
    y17_dev: np.ndarray,
    y8_dev: np.ndarray,
    tok17_dev: np.ndarray,
    tok8_dev: np.ndarray,
    levels_dev: np.ndarray,
    truncated_dev: np.ndarray,
    tps_17b: float,
    tps_8b: float,
    *,
    n_splits: int = 30,
    seed: int = 550,
    lambda_grid: Sequence[float] = (0.0, 0.0005, 0.0010, 0.0015, 0.0020),
) -> dict[str, Any]:
    """Evaluate lambda and calibration selection stability via 30 stratified dev 600/200 re-splits.

    Strictly uses only the 800 dev items (`Train 600 + Val 200`); `Test 500` is never touched.
    Uses `calibration_method='auto'` to match the deployed router pipeline and tracks both
    `p17` and `p8` calibration selection counts.
    """

    splitter = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=200, random_state=seed
    )
    lambda_counts: dict[str, int] = {str(lam): 0 for lam in lambda_grid}
    lambda_counts_uncal: dict[str, int] = {str(lam): 0 for lam in lambda_grid}
    ece_counts: dict[str, int] = {"uncalibrated": 0, "platt_cv5": 0, "isotonic_cv5": 0}
    brier_counts: dict[str, int] = {"uncalibrated": 0, "platt_cv5": 0, "isotonic_cv5": 0}
    ece_counts_8b: dict[str, int] = {"uncalibrated": 0, "platt_cv5": 0, "isotonic_cv5": 0}
    brier_counts_8b: dict[str, int] = {"uncalibrated": 0, "platt_cv5": 0, "isotonic_cv5": 0}
    auac_spreads: list[float] = []

    for tr_idx, va_idx in splitter.split(X_dev, levels_dev):
        cal_rep = evaluate_calibration_methods(
            X_dev[tr_idx],
            y17_dev[tr_idx],
            X_dev[va_idx],
            y17_dev[va_idx],
            seed=seed,
        )
        ece_counts[cal_rep["best_method"]] += 1
        brier_counts[cal_rep["best_method_brier"]] += 1

        cal_rep_8b = evaluate_calibration_methods(
            X_dev[tr_idx],
            y8_dev[tr_idx],
            X_dev[va_idx],
            y8_dev[va_idx],
            seed=seed,
        )
        ece_counts_8b[cal_rep_8b["best_method"]] += 1
        brier_counts_8b[cal_rep_8b["best_method_brier"]] += 1

        r_sub = CostAwareRouter(
            tps_8b=tps_8b, calibration_method="auto", seed=seed
        ).fit(
            X_dev[tr_idx],
            y17_dev[tr_idx],
            y8_dev[tr_idx],
            tok8_dev[tr_idx],
            X_val=X_dev[va_idx],
            y17_val=y17_dev[va_idx],
            y8_val=y8_dev[va_idx],
        )
        lam_rep = r_sub.select_lambda_on_val(
            X_dev[va_idx],
            y17_dev[va_idx],
            y8_dev[va_idx],
            tok17_dev[va_idx],
            tok8_dev[va_idx],
            tps_17b,
            tps_8b,
            truncated_val=truncated_dev[va_idx],
            lambda_grid=lambda_grid,
        )
        lambda_counts[str(lam_rep["best_lambda"])] += 1
        vals = [row["val_lat_auac"] for row in lam_rep["sweep"]]
        auac_spreads.append(float(max(vals) - min(vals)))

        r_uncal = CostAwareRouter(
            tps_8b=tps_8b, calibration_method="uncalibrated", seed=seed
        ).fit(
            X_dev[tr_idx],
            y17_dev[tr_idx],
            y8_dev[tr_idx],
            tok8_dev[tr_idx],
        )
        lam_rep_uncal = r_uncal.select_lambda_on_val(
            X_dev[va_idx],
            y17_dev[va_idx],
            y8_dev[va_idx],
            tok17_dev[va_idx],
            tok8_dev[va_idx],
            tps_17b,
            tps_8b,
            truncated_val=truncated_dev[va_idx],
            lambda_grid=lambda_grid,
        )
        lambda_counts_uncal[str(lam_rep_uncal["best_lambda"])] += 1

    return {
        "n_splits": int(n_splits),
        "dev_size": int(len(X_dev)),
        "test_touched": False,
        "calibration_method": "auto",
        "lambda_winner_counts": lambda_counts,
        "lambda_winner_counts_uncalibrated": lambda_counts_uncal,
        "ece_winner_counts": ece_counts,
        "brier_winner_counts": brier_counts,
        "ece_winner_counts_8b": ece_counts_8b,
        "brier_winner_counts_8b": brier_counts_8b,
        "mean_auac_range_across_lambda_grid": float(np.mean(auac_spreads)),
        "max_auac_range_across_lambda_grid": float(np.max(auac_spreads)),
    }
