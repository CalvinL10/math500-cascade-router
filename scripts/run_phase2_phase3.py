"""CLI entrypoint for Phase 2 (Router Training & Calibration) and Phase 3 (Blind Replay & Plotting)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import run_phase2_phase3_pipeline


DEFAULT_BUNDLE_DIR = PROJECT_ROOT / "outputs" / "math550_full_v2_bundle"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase2_3"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local Phase 2 (router calibration) & Phase 3 (MATH-500 replay & figures)."
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE_DIR,
        help="Path to math550_full_v2_bundle directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for Phase 2/3 tables, JSON summaries, and figures.",
    )
    parser.add_argument("--seed", type=int, default=550, help="Random seed (default: 550).")
    parser.add_argument(
        "--n-boot",
        type=int,
        default=5000,
        help="Number of paired bootstrap resamples (default: 5000).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_phase2_phase3_pipeline(
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        n_boot=args.n_boot,
    )
    p2 = result["phase2_diagnostics"]
    p3 = result["phase3_summary"]
    print(f"Output directory: {result['output_dir']}")
    c17_best = p2["selected_calibration"]["clf_17b"]
    c8_best = p2["selected_calibration"]["clf_8b"]
    print(
        f"Selected calibration on Val 200: 1.7B={c17_best} "
        f"(Val ECE={p2['calibration_17b']['methods'][c17_best]['val']['ece']:.4f}), "
        f"8B={c8_best} "
        f"(Val ECE={p2['calibration_8b']['methods'][c8_best]['val']['ece']:.4f}) "
        f"| Test AUC={p2['test_roc_auc']['post_1.7b_6feat']:.4f}"
    )
    print(
        f"Selected lambda on Val 200: {p2['lambda_selection_val']['best_lambda']:.4f} "
        f"(Val LatAUAC={p2['lambda_selection_val']['best_val_lat_auac']:.4f}) "
        f"| Regraded completions changed: {p3['regraded_completion_diff_count']}"
    )
    r20 = p3["val_locked_operating_points"]["tier1_trunc_cost_utility"]["0.20"]
    print(
        f"Cost-Aware Router @Val 20%: esc={r20['escalation_rate']:.1%} "
        f"(k={r20['escalation_count']}) | Acc={r20['accuracy']:.4f} "
        f"| MeanTok={r20['mean_tokens']:.1f} | EstSerialLat={r20['mean_bs1_latency_s']:.2f}s"
    )
    for name, fpaths in result["figure_paths"].items():
        print(f"Figure {name}: {fpaths['png']}")


if __name__ == "__main__":
    main()
