"""Build self-contained slide figures and auxiliary sensitivity tables/charts.

Generates `outputs/phase2_3/supplementary/S1_lambda_budget_sensitivity.*`,
`S2_throughput_ratio_sensitivity.*`, and `S3_val_vs_test_locked_points.csv`
directly from the local bundle and `outputs/phase2_3/` (no external paths),
then crops publication figures and renders Fig 3D into `slides/assets/`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (
    evaluate_val_locked_mask,
    evaluate_val_locked_operating_point,
    load_aligned_split,
    load_latency_calibration,
)
from src.router import CostAwareRouter, lock_val_thresholds


def _bootstrap_speed_ratio_ci(
    cal_jsonl_path: Path, n_boot: int = 5000, seed: int = 550
) -> tuple[float, float]:
    """Compute paired bootstrap 95% CI on `TPS_1.7B / TPS_8B` from calibration prompts."""

    rows = [
        json.loads(line)
        for line in cal_jsonl_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    p17 = {str(r["unique_id"]): r for r in rows if r["mode_label"] == "qwen3_1.7b_non_think"}
    p8 = {str(r["unique_id"]): r for r in rows if r["mode_label"] == "qwen3_8b_non_think"}
    common_ids = sorted(set(p17) & set(p8))
    tok17 = np.array([float(p17[u]["out_tokens"]) for u in common_ids], dtype=np.float64)
    lat17 = np.array([float(p17[u]["bs1_latency_s"]) for u in common_ids], dtype=np.float64)
    tok8 = np.array([float(p8[u]["out_tokens"]) for u in common_ids], dtype=np.float64)
    lat8 = np.array([float(p8[u]["bs1_latency_s"]) for u in common_ids], dtype=np.float64)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(common_ids), size=(n_boot, len(common_ids)))
    tps17_b = tok17[idx].sum(axis=1) / lat17[idx].sum(axis=1)
    tps8_b = tok8[idx].sum(axis=1) / lat8[idx].sum(axis=1)
    ratios = tps17_b / tps8_b
    return float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))


def generate_auxiliary_artifacts(
    bundle_dir: Path, phase23_dir: Path, aux_dir: Path, seed: int = 550
) -> None:
    """Generate S1, S2, and S3 sensitivity artifacts strictly inside `aux_dir`."""

    aux_dir.mkdir(parents=True, exist_ok=True)

    p2 = json.loads((phase23_dir / "phase2_router_diagnostics.json").read_text(encoding="utf-8"))
    p3 = json.loads((phase23_dir / "phase3_summary_metrics.json").read_text(encoding="utf-8"))
    deployed_lam = float(p2["lambda_selection_val"]["best_lambda"])

    tr = load_aligned_split(bundle_dir, "router_train_600", regrade=True)
    va = load_aligned_split(bundle_dir, "router_val_200", regrade=True)
    te = load_aligned_split(bundle_dir, "math500_test_500", regrade=True)
    cal_path = bundle_dir / "latency_calibration_v2.jsonl"
    lat_cal = load_latency_calibration(cal_path)
    tps_17b = float(lat_cal["qwen3_1.7b_non_think"]["tokens_per_sec"])
    tps_8b = float(lat_cal["qwen3_8b_non_think"]["tokens_per_sec"])

    router = CostAwareRouter(tps_8b=tps_8b, calibration_method="auto", seed=seed).fit(
        tr["X"],
        tr["y17"],
        tr["y8"],
        tr["tok8"],
        X_val=va["X"],
        y17_val=va["y17"],
        y8_val=va["y8"],
    )

    # ------------------------------------------------------------------
    # S1: lambda x Val-target sensitivity (Val acc | Test acc | Test lat)
    # ------------------------------------------------------------------
    lams = [0.0, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030, 0.0050]
    n_in = 5
    rates = [0.12, 0.20, 0.25, 0.30, 0.40, 0.50]
    deployed_i = lams.index(deployed_lam) if deployed_lam in lams else 3
    headline_j = rates.index(0.20)

    c17_te = te["tok17"] / tps_17b
    c8_te = te["tok8"] / tps_8b
    va_mat = np.zeros((len(lams), len(rates)), dtype=np.float64)
    ta_mat = np.zeros_like(va_mat)
    tl_mat = np.zeros_like(va_mat)
    tk_mat = np.zeros_like(va_mat, dtype=int)
    s1_rows: list[dict[str, object]] = []

    for i, lam in enumerate(lams):
        sv = router.compute_routing_scores(va["X"], lam=lam, truncated_mask=va["trunc17"])[
            "tier1_trunc_cost_utility"
        ]
        st = router.compute_routing_scores(te["X"], lam=lam, truncated_mask=te["trunc17"])[
            "tier1_trunc_cost_utility"
        ]
        for j, q in enumerate(rates):
            tau = float(np.quantile(sv, 1.0 - q))
            mv = sv >= tau
            mt = st >= tau
            va_mat[i, j] = 100.0 * float(np.where(mv, va["y8"], va["y17"]).mean())
            ta_mat[i, j] = 100.0 * float(np.where(mt, te["y8"], te["y17"]).mean())
            tl_mat[i, j] = float((c17_te + np.where(mt, c8_te, 0.0)).mean())
            tk_mat[i, j] = int(mt.sum())
            s1_rows.append(
                {
                    "lam": lam,
                    "in_original_grid": i < n_in,
                    "val_target_rate": q,
                    "val_acc_pct": round(va_mat[i, j], 2),
                    "test_k": tk_mat[i, j],
                    "test_acc_pct": round(ta_mat[i, j], 2),
                    "test_est_latency_s": round(tl_mat[i, j], 3),
                }
            )

    with (aux_dir / "S1_lambda_budget_sensitivity.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(s1_rows[0].keys()))
        writer.writeheader()
        writer.writerows(s1_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.9))
    specs = [
        (va_mat, "Val-200 accuracy (%) at the locked threshold\n(the only panel selection is allowed to see)", "Blues", "{:.1f}"),
        (ta_mat, "Test-500 accuracy (%) — report only\n(never used to choose λ or budget)", "Blues", "{:.1f}"),
        (tl_mat, "Test-500 estimated serial latency (s)\n(out_tokens / calibrated L4 TPS)", "Greys", "{:.1f}"),
    ]
    ylab = [f"λ = {l:.4f}" + ("" if i < n_in else "  (outside grid)") for i, l in enumerate(lams)]
    for ax, (mat, title, cmap, fmt) in zip(axes, specs):
        lo, hi = float(mat.min()), float(mat.max())
        span = max(hi - lo, 1e-6)
        ax.imshow(mat, cmap=cmap, aspect="auto", vmin=lo - 0.35 * span, vmax=hi + 0.15 * span)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                frac = (mat[i, j] - (lo - 0.35 * span)) / (1.50 * span)
                ax.text(
                    j,
                    i,
                    fmt.format(mat[i, j]),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if frac > 0.62 else "#1F2328",
                    fontweight="bold" if (i, j) == (deployed_i, headline_j) else "normal",
                )
        ax.add_patch(
            Rectangle(
                (headline_j - 0.5, deployed_i - 0.5),
                1,
                1,
                fill=False,
                edgecolor="#CF222E",
                linewidth=2.6,
            )
        )
        ax.axhline(n_in - 0.5, color="#57606A", linewidth=1.4, linestyle="--")
        ax.set_xticks(range(len(rates)))
        ax.set_xticklabels([f"{int(100 * q)}%" for q in rates])
        ax.set_xlabel("Val-locked target escalation rate")
        ax.set_yticks(range(len(lams)))
        ax.set_yticklabels(ylab if ax is axes[0] else [""] * len(lams))
        ax.set_title(title, pad=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)

    axes[1].set_xticklabels([f"{int(100 * q)}%\n(k={tk_mat[deployed_i, j]})" for j, q in enumerate(rates)])
    axes[1].set_xlabel("Val-locked target escalation rate  (realised Test k shown for the deployed λ)")
    fig.suptitle(
        f"Supplementary S1. λ × budget sensitivity of the deployed two-tier router "
        f"(p17 {router.selected_calibration_17b_}, p8 {router.selected_calibration_8b_})",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.012,
        f"Red box = deployed configuration (λ = {deployed_lam:.4f} chosen by Val LatAUAC inside the original 5-value grid; headline budget = Val 20%).  "
        "Dashed line separates the original grid (above) from out-of-grid values (below).\n"
        f"Reading: across the 5 in-grid λ values, Test accuracy varies by {np.ptp(ta_mat[:n_in, headline_j]):.1f} pp at the 20% headline budget "
        f"and by at most {max(np.ptp(ta_mat[:n_in, j]) for j in range(len(rates))):.1f} pp at any budget; "
        f"Val accuracy at the 20% point is {va_mat[:n_in, headline_j].min():.1f}–{va_mat[:n_in, headline_j].max():.1f}% for every in-grid λ (Trunc-Only on Val = 73.0%).",
        ha="center",
        va="bottom",
        fontsize=9.2,
        color="#24292F",
    )
    fig.tight_layout(rect=[0, 0.085, 1, 0.955])
    fig.savefig(aux_dir / "S1_lambda_budget_sensitivity.png", dpi=220, bbox_inches="tight")
    fig.savefig(aux_dir / "S1_lambda_budget_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # S2: throughput speed-ratio sensitivity
    # ------------------------------------------------------------------
    r20 = p3["val_locked_operating_points"]["tier1_trunc_cost_utility"]["0.20"]
    a1 = p3["baselines"]["atom1_1.7b_nonthink"]
    a3a = p3["baselines"]["atom3a_1.7b_adaptive_sc"]
    a4 = p3["baselines"]["atom4_8b_direct"]
    tok17 = float(a1["mean_tokens"])
    tok8_added = float(r20["mean_tokens"]) - tok17
    tok8_direct = float(a4["mean_tokens"])
    tok_sc = float(a3a["mean_tokens"])
    r_obs = tps_17b / tps_8b
    r_ci_lo, r_ci_hi = _bootstrap_speed_ratio_ci(cal_path, n_boot=5000, seed=seed)
    rc8 = tok17 / (tok8_direct - tok8_added)
    rcsc = (tok_sc - tok17) / tok8_added

    r_grid = np.linspace(1.0, 8.0, 500)
    lat_router = tok17 / (r_grid * tps_8b) + tok8_added / tps_8b
    lat_sc = tok_sc / (r_grid * tps_8b)
    lat_8 = np.full_like(r_grid, tok8_direct / tps_8b)

    with (aux_dir / "S2_throughput_ratio_sensitivity.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["speed_ratio_r", "router_val20_est_latency_s", "adaptive_sc_est_latency_s", "direct_8b_est_latency_s"])
        for a, b, c, d in zip(r_grid, lat_router, lat_sc, lat_8):
            writer.writerow([round(float(a), 4), round(float(b), 4), round(float(c), 4), round(float(d), 4)])

    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    ax.axvspan(rc8, rcsc, color="#CF222E", alpha=0.07, zorder=0)
    ax.plot(r_grid, lat_8, color="#0550AE", linewidth=2.4, linestyle="-", zorder=3)
    ax.plot(r_grid, lat_sc, color="#1A7F37", linewidth=2.4, linestyle="--", zorder=3)
    ax.plot(r_grid, lat_router, color="#CF222E", linewidth=2.8, linestyle="-", zorder=4)
    for x, lab, yy in [
        (rc8, f"r = {rc8:.2f}\nRouter = 8B Direct", 26.0),
        (rcsc, f"r = {rcsc:.2f}\nRouter = Adaptive SC", 60.5),
    ]:
        ax.axvline(x, color="#57606A", linewidth=1.1, linestyle=":", zorder=2)
        ax.text(x + 0.06, yy, lab, ha="left", va="top", fontsize=9.2, color="#57606A")

    ax.axvline(r_obs, color="#24292F", linewidth=1.6, zorder=2)
    ax.text(
        r_obs - 0.07,
        60.5,
        f"observed on L4\nr = {r_obs:.2f}\n(bootstrap 95% CI\n{r_ci_lo:.2f}–{r_ci_hi:.2f})",
        ha="right",
        va="top",
        fontsize=9.2,
        color="#24292F",
        fontweight="semibold",
    )
    ax.scatter(
        [r_obs] * 3,
        [tok17 / (r_obs * tps_8b) + tok8_added / tps_8b, tok_sc / (r_obs * tps_8b), tok8_direct / tps_8b],
        color=["#CF222E", "#1A7F37", "#0550AE"],
        s=46,
        zorder=6,
        edgecolor="white",
        linewidth=0.9,
    )
    ax.text(
        7.95,
        float(lat_8[-1]) + 1.0,
        f"8B Direct ({100*a4['accuracy']:.1f}%)  {tok8_direct / tps_8b:.1f} s at any r",
        color="#0550AE",
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="semibold",
    )
    ax.text(
        7.95,
        float(lat_router[-1]) + 2.9,
        f"Router @Val 20% ({100*r20['accuracy']:.1f}%)",
        color="#A40E26",
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="semibold",
    )
    ax.text(
        7.95,
        float(lat_sc[-1]) - 1.6,
        f"Adaptive SC 2→3 ({100*a3a['accuracy']:.1f}%)",
        color="#116329",
        ha="right",
        va="top",
        fontsize=10,
        fontweight="semibold",
    )
    ax.text(
        (rc8 + rcsc) / 2.0,
        5.2,
        "Router is the fastest of the three\nonly inside this band",
        ha="center",
        va="bottom",
        fontsize=9.6,
        color="#A40E26",
    )
    ax.set_xlim(1.0, 8.0)
    ax.set_ylim(0, 62)
    ax.set_xlabel(f"Speed ratio  r = TPS(1.7B) / TPS(8B)   (8B throughput held at the calibrated {tps_8b:.2f} tok/s)")
    ax.set_ylabel("Estimated serial latency per query (s)")
    ax.set_title("Supplementary S2. How the latency ranking depends on the small/large throughput ratio", fontsize=12.5, fontweight="bold", pad=12)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.text(
        0.5,
        0.005,
        "Latency = out_tokens / TPS (no prefill, queueing or model-switch cost). Token counts are the MATH-500 means of each strategy; accuracy does not depend on r.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#57606A",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    fig.savefig(aux_dir / "S2_throughput_ratio_sensitivity.png", dpi=220, bbox_inches="tight")
    fig.savefig(aux_dir / "S2_throughput_ratio_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # S3: Val vs Test Val-locked operating points table
    # ------------------------------------------------------------------
    val_sc = router.compute_routing_scores(va["X"], lam=deployed_lam, truncated_mask=va["trunc17"])
    te_sc = router.compute_routing_scores(te["X"], lam=deployed_lam, truncated_mask=te["trunc17"])
    s3_rows: list[dict[str, object]] = []

    for split_lbl, d in [("Val 200", va), ("Test 500", te)]:
        pt = evaluate_val_locked_mask(d["trunc17"], d, tps_17b, tps_8b, dispatch_mode="cascade")
        s3_rows.append(
            {
                "policy": "Truncation-Only Fallback",
                "val_target": "n/a",
                "split": split_lbl,
                "k": pt["escalation_count"],
                "esc_rate_pct": round(100.0 * pt["escalation_rate"], 1),
                "acc_pct": round(100.0 * pt["accuracy"], 1),
                "est_latency_s": round(pt["mean_bs1_latency_s"], 2),
            }
        )

    for pol_key, pol_label, mode in [
        ("tier1_trunc_cost_utility", "Cost-Aware Router (deployed)", "cascade"),
        ("conf_gate", "Naive ConfGate (1 - p17)", "cascade"),
        ("pre_gen_qlen", "Pre-Gen Direct Router (q_len)", "direct"),
    ]:
        taus = lock_val_thresholds(val_sc[pol_key], target_val_rates=rates)
        for q_key, tau in taus.items():
            for split_lbl, sc, d in [("Val 200", val_sc[pol_key], va), ("Test 500", te_sc[pol_key], te)]:
                pt = evaluate_val_locked_operating_point(sc, tau, d, tps_17b, tps_8b, dispatch_mode=mode)
                s3_rows.append(
                    {
                        "policy": pol_label,
                        "val_target": q_key,
                        "split": split_lbl,
                        "k": pt["escalation_count"],
                        "esc_rate_pct": round(100.0 * pt["escalation_rate"], 1),
                        "acc_pct": round(100.0 * pt["accuracy"], 1),
                        "est_latency_s": round(pt["mean_bs1_latency_s"], 2),
                    }
                )

    for split_lbl, d in [("Val 200", va), ("Test 500", te)]:
        n_split = len(d["y17"])
        s3_rows.append(
            {
                "policy": "1.7B 1-Pass",
                "val_target": "n/a",
                "split": split_lbl,
                "k": 0,
                "esc_rate_pct": 0.0,
                "acc_pct": round(100.0 * float(d["y17"].mean()), 1),
                "est_latency_s": round(float((d["tok17"] / tps_17b).mean()), 2),
            }
        )
        s3_rows.append(
            {
                "policy": "8B Direct",
                "val_target": "n/a",
                "split": split_lbl,
                "k": n_split,
                "esc_rate_pct": 100.0,
                "acc_pct": round(100.0 * float(d["y8"].mean()), 1),
                "est_latency_s": round(float((d["tok8"] / tps_8b).mean()), 2),
            }
        )

    with (aux_dir / "S3_val_vs_test_locked_points.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(s3_rows[0].keys()))
        writer.writeheader()
        writer.writerows(s3_rows)


def main() -> None:
    slides_dir = Path(__file__).resolve().parent
    out_dir = slides_dir / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle_dir = PROJECT_ROOT / "outputs" / "math550_full_v2_bundle"
    phase23_dir = PROJECT_ROOT / "outputs" / "phase2_3"
    fig_dir = phase23_dir / "figures"
    aux_dir = phase23_dir / "supplementary"

    generate_auxiliary_artifacts(bundle_dir, phase23_dir, aux_dir, seed=550)

    # Inspect fig2 gap between B and C (y>140)
    im2 = Image.open(fig_dir / "fig2_mechanism_decomposition.png").convert("RGB")
    arr2 = np.asarray(im2)
    ink2 = np.any(arr2[140:, :] < 248, axis=2)
    col_counts = ink2[:, 3350:3560].sum(axis=0)
    empty_cols = [3350 + i for i, c in enumerate(col_counts) if c == 0]
    col_counts_ab = ink2[:, 1650:1770].sum(axis=0)
    empty_ab = [1650 + i for i, c in enumerate(col_counts_ab) if c == 0]

    im3 = Image.open(fig_dir / "fig3_router_diagnostics.png").convert("RGB")
    arr3 = np.asarray(im3)
    ink3 = np.any(arr3 < 248, axis=2)
    er_mid = [y for y in range(1580, 1690) if not ink3[y, :].any()]

    # Crop clean assets
    im1 = Image.open(fig_dir / "fig1_pareto_frontier.png")
    im1.crop((0, 220, im1.width, im1.height)).save(out_dir / "fig1_pareto_slide.png", dpi=(300, 300))

    gap_ab = int(np.median(empty_ab)) if empty_ab else 1694
    gap_bc = 3480
    if empty_cols:
        best_run: list[int] = []
        cur: list[int] = []
        for c in empty_cols:
            if not cur or c == cur[-1] + 1:
                cur.append(c)
            else:
                if len(cur) > len(best_run):
                    best_run = cur
                cur = [c]
        if len(cur) > len(best_run):
            best_run = cur
        gap_bc = int(np.mean(best_run))

    im2_clean = im2.copy()
    arr2_clean = np.array(im2_clean)
    arr2_clean[340:, 3403:] = 255
    im2_clean = Image.fromarray(arr2_clean)

    im2_clean.crop((0, 210, gap_bc, im2.height)).save(out_dir / "fig2ab_slide.png", dpi=(300, 300))
    im2_clean.crop((0, 210, gap_ab, im2.height)).save(out_dir / "fig2a_slide.png", dpi=(300, 300))
    im2_clean.crop((gap_ab, 210, gap_bc, im2.height)).save(out_dir / "fig2b_slide.png", dpi=(300, 300))

    ec_top = [x for x in range(2100, 2260) if not ink3[150:1635, x].any()]
    ec_bot = [x for x in range(2100, 2260) if not ink3[1635:3043, x].any()]
    split_x_top = int(np.median(ec_top)) if ec_top else 2175
    split_x_bot = int(np.median(ec_bot)) if ec_bot else 2175
    split_y = int(np.median(er_mid)) if er_mid else 1638

    im3.crop((split_x_top, 210, im3.width, split_y)).save(out_dir / "fig3b_slide.png", dpi=(300, 300))
    im3.crop((0, split_y, split_x_bot, im3.height)).save(out_dir / "fig3c_slide.png", dpi=(300, 300))

    im4 = Image.open(fig_dir / "fig4_matched_budget_stat_tests.png")
    im4.crop((0, 225, im4.width, im4.height)).save(out_dir / "fig4_stat_slide.png", dpi=(300, 300))

    ims1 = Image.open(aux_dir / "S1_lambda_budget_sensitivity.png")
    ims1.crop((0, 140, ims1.width, ims1.height)).save(out_dir / "S1_lambda_budget_slide.png", dpi=(220, 220))

    ims2 = Image.open(aux_dir / "S2_throughput_ratio_sensitivity.png")
    ims2.crop((0, 68, ims2.width, ims2.height)).save(out_dir / "S2_throughput_ratio_slide.png", dpi=(220, 220))

    # Render standalone Fig 3D from outputs/phase2_3/decile_analysis.csv
    dec_path = phase23_dir / "decile_analysis.csv"
    deciles = []
    with dec_path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            deciles.append(
                {
                    "decile": int(r["decile"]),
                    "count": int(r["count"]),
                    "actual_17b_acc": float(r["actual_17b_acc"]),
                    "actual_8b_acc": float(r["actual_8b_acc"]),
                    "mean_tok8": float(r["mean_tok8"]),
                }
            )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 11.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.edgecolor": "#D0D7DE",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax_dec = plt.subplots(figsize=(7.6, 5.7))
    dec_ids = np.array([d["decile"] for d in deciles])
    dec_17 = np.array([100.0 * d["actual_17b_acc"] for d in deciles])
    dec_8 = np.array([100.0 * d["actual_8b_acc"] for d in deciles])
    dec_tok8 = np.array([d["mean_tok8"] for d in deciles])
    dec_counts = sorted({d["count"] for d in deciles})
    bin_lbl = str(dec_counts[0]) if len(dec_counts) == 1 else f"{dec_counts[0]}-{dec_counts[-1]}"
    dec1_n = deciles[0]["count"]

    ax_dec2 = ax_dec.twinx()
    ax_dec2.spines["top"].set_visible(False)
    ax_dec.set_zorder(ax_dec2.get_zorder() + 2)
    ax_dec.patch.set_visible(False)

    ax_dec2.bar(dec_ids, dec_tok8, color="#D8DEE4", alpha=0.68, width=0.62, label=f"Mean 8B Tokens (n={bin_lbl}/bin)", zorder=1)
    ax_dec.plot(dec_ids, dec_17, color="#1F2328", marker="o", linewidth=2.3, markersize=6.0, label="1.7B Empirical Acc (%)", zorder=5)
    ax_dec.plot(dec_ids, dec_8, color="#CF222E", marker="D", linewidth=2.5, markersize=6.0, label="8B Empirical Acc (%)", zorder=6)
    ax_dec.fill_between(dec_ids, dec_17, dec_8, where=(dec_8 >= dec_17), color="#CF222E", alpha=0.14, label="8B Rescue Gain", zorder=3)

    ax_dec.annotate(
        f"Decile 1: Truncated\n(+{dec_8[0]-dec_17[0]:.0f} pp, n={dec1_n})",
        xy=(1, dec_8[0]),
        xytext=(1.15, 86.0),
        fontsize=9.0,
        fontweight="semibold",
        color="#A40E26",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CF222E", alpha=0.96),
        arrowprops=dict(arrowstyle="->", color="#A40E26", lw=1.2),
        zorder=8,
    )
    ax_dec.annotate(
        f"Decile 2: Low Net Gain\n(+{dec_8[1]-dec_17[1]:.0f} pp, {int(round(dec_tok8[1]))} tok)",
        xy=(2, dec_8[1]),
        xytext=(3.2, 13.0),
        fontsize=9.0,
        fontweight="semibold",
        color="#1F2328",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#6E7781", alpha=0.96),
        arrowprops=dict(arrowstyle="->", color="#57606A", lw=1.2),
        zorder=8,
    )
    ax_dec.set_xticks(dec_ids)
    ax_dec.set_xlabel(f"Predicted 1.7B Confidence Decile (n = {bin_lbl} per bin)")
    ax_dec.set_ylabel("Empirical Accuracy (%)")
    ax_dec2.set_ylabel("Mean 8B Output Tokens", color="#57606A")
    ax_dec.set_ylim(0, 108)
    ax_dec2.set_ylim(0, 2150)
    ax_dec.set_title(f"(D) MATH-500 Confidence Decile Anatomy ({bin_lbl} Questions/Decile)", pad=10)

    h1, l1 = ax_dec.get_legend_handles_labels()
    h2, l2 = ax_dec2.get_legend_handles_labels()
    leg = ax_dec.legend(
        h1 + h2,
        l1 + l2,
        loc="center right",
        bbox_to_anchor=(0.985, 0.56),
        fontsize=8.6,
        facecolor="white",
        framealpha=0.96,
    )
    leg.set_zorder(10)

    fig.tight_layout()
    fig.savefig(out_dir / "fig3D_decile_slide.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved self-contained supplementary artifacts to", aux_dir)
    print("Saved all slide assets to", out_dir)


if __name__ == "__main__":
    main()
