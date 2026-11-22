# Cost-Aware Two-Tier LLM Cascade Routing on MATH-500

Two-tier cascade router (`Qwen3-1.7B -> Qwen3-8B`, 2,048-token cap) evaluated on `MATH-500` (`n = 500`) using strictly disjoint `600 / 200` Dev Train/Val splits, an audited 4-state symbolic math grader (`correct`, `wrong`, `invalid`, `truncated`), and mechanism decomposition into Tier-1 deterministic truncation rescue plus Tier-2 calibrated expected-utility ranking. Under L4 calibration (`r = TPS_1.7B / TPS_8B = 4.06`), escalating `24.6%` (`k = 123`) of queries achieves **`75.40%`** accuracy at **`27.28 s`** estimated serial latency (`40.5%` faster than `8B Direct` at `78.40%` / `45.82 s`, while using `+36.7%` more raw tokens).

---

## Key Results on MATH-500 (`n = 500`)

Selected from [outputs/phase2_3/main_comparison_table.md](outputs/phase2_3/main_comparison_table.md):

| Method | Mode | Esc (%) | Acc (%) | Trunc (%) | Mean Tok | Est. Serial Lat (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Atom 1: `Qwen3-1.7B` Non-Think (1-Pass) | single | 0.0 | 69.00 | 10.6 | 777.4 | 11.56 |
| Atom 2: `1.7B` Self-Refine | single | 0.0 | 70.60 | 0.4 | 1173.1 | 17.45 |
| Truncation-Only Fallback (`1.7B -> 8B`) | cascade | 10.6 | 73.00 | 5.4 | 955.5 | 22.31 |
| Pre-Gen Direct Router (`q_len`, Val 20%) | direct | 21.8 | 72.60 | 10.6 | 757.2 | 22.95 |
| Atom 3a: `1.7B` Adaptive SC (`2->3` paths) | single | 0.0 | 73.80 | 5.0 | 1950.6 | 29.01 |
| Atom 5: `1.7B` Thinking (`max=4096`) | single | 0.0 | 56.80 | 42.2 | 3151.1 | 49.02 |
| Naive ConfGate (`Val 20%`) | cascade | 18.4 | 73.20 | 8.2 | 1064.7 | 28.90 |
| Naive ConfGate (`Matched k=123`) | cascade | 24.6 | 74.00 | 8.8 | 1130.6 | 32.88 |
| **Cost-Aware Router (Ours, Val 20%)** | cascade | **24.6** | **75.40** | **5.6** | **1037.9** | **27.28** |
| Atom 4: `Qwen3-8B` Non-Think (Direct) | direct | 100.0 | 78.40 | 9.6 | 759.4 | 45.82 |

### Mechanism & Statistical Takeaways
- **Router specification (selected strictly on Dev Val `n = 200`)**: Uses 6 post-1.7B observable features (`mean_logprob`, `min_logprob`, `boxed_mean_logprob`, `out_tokens`, `is_complete_and_valid`, `prompt_tokens`) with `clf_17b = uncalibrated` (Val/Test ECE `0.0379 / 0.0237`, Test AUC `0.8931`), `clf_8b = isotonic_cv5` (Val/Test ECE `0.0858 / 0.0592`, Test AUC `0.8760`), Ridge `8B` token predictor (Test Pearson `r = 0.8453`), and Val-selected penalty `lambda = 0.0015` (Val LatAUAC `0.7680`, cutoff `tau = 0.0809`).
- **Two-tier breakdown (`k = 123` on Test-500)**: `53` Tier-1 deterministic truncated escalations (`+20` net rescue) + `70` Tier-2 completed picks (`+14 / -2 = +12` net rescue).
- **Paired hypothesis tests (Holm-Bonferroni across 13 comparisons, [outputs/phase2_3/statistical_tests.md](outputs/phase2_3/statistical_tests.md))**: Significantly beats `1.7B 1-Pass` (`+6.40 pp`, `p_H = 2.33e-7`), `Self-Refine` (`+4.80 pp`, `p_H = 1.82e-3`), `Thinking 4K` (`+18.60 pp`, `p_H = 9.60e-17`), and `Trunc-Only` (`+2.40 pp`, `p_H = 0.0376`). At matched `k = 123`, saves **`5.59 s/query`** (`95% CI [-8.53, -2.58] s`) vs. `Naive ConfGate` by skipping slow, low-utility problems. Differences vs. `ConfGate`, `Adaptive SC`, and `8B Direct` (`-3.00 pp`, unadjusted McNemar `p = 0.0674`, 95% CI `[-6.00, +0.00] pp`) are not statistically separated.

---

## Repository Structure

```text
project/
├── src/                        # Core library: splits, 4-state LaTeX/SymPy grader, router, eval & plots
│   ├── data_prep.py            # Deterministic stratified splits & zero-leakage checks
│   ├── grading.py              # Audited answer extractor & equivalence checker
│   ├── router.py               # CostAwareRouter, calibration selection, LatAUAC, stability checks
│   ├── evaluation.py           # Phase 2/3 replay, paired bootstrap/McNemar-Holm tests, figures
│   └── colab_helpers.py        # L4 inference/logprob helpers bundled into Colab notebook
├── scripts/                    # CLI entrypoints
│   ├── prepare_data.py         # Regenerate data/splits/ + manifest
│   ├── run_phase2_phase3.py    # Run full Phase 2/3 evaluation -> outputs/phase2_3/
│   ├── build_colab_notebook.py # Bundle src/ into notebooks/phase1_colab_data_collection.ipynb
│   └── build_vllm_probe.py     # Build notebooks/track_a_vllm_probe.ipynb
├── data/
│   ├── raw/                    # MATH train-minus-500 & MATH-500 reference JSONs
│   └── splits/                 # Disjoint pilot_50, router_train_600, router_val_200, math500_test_500
├── outputs/
│   ├── math550_full_v2_bundle/ # Cached Colab L4 JSONL generations & single-stream latency calibration
│   └── phase2_3/               # Regenerated tables, JSON diagnostics, main figs (1-4), supplementary (S1-S3)
├── slides/                     # 15-page presentation deck (HTML/CSS -> PDF), speaker cue cards & builder
├── tests/                      # 78 pytest unit & end-to-end regression tests
├── requirements.txt            # Lightweight local dependencies (no torch/GPU required offline)
└── pytest.ini / conftest.py
```

---

## Quick Reproduction & Verification

Requires Python 3.12+ with dependencies in `requirements.txt` (plus `playwright` and `pymupdf` if rebuilding slide PDFs):

```powershell
# 1. Install local dependencies
pip install -r requirements.txt

# 2. Verify data splits & run full Phase 2/3 router training, replay, and plotting
python scripts/prepare_data.py
python scripts/run_phase2_phase3.py

# 3. Rebuild slide assets (S1/S2/S3 + cropped PNGs) and compile 15p Deck + 3p Cue Cards
python slides/build_assets.py
python slides/build_pdf.py

# 4. Run full regression & end-to-end test suite (78 tests)
python -m pytest -q
```

See [slides/README.md](slides/README.md) for presentation and speaker cue-card build details.

---

## Scope & Known Limitations

1. **Offline Serial Latency Model**: Latency is estimated deterministically as `out_tokens / calibrated_L4_TPS` (`67.24 tok/s` for `1.7B` non-think with OLS $R^2 = 0.999978$; `16.57 tok/s` for `8B` non-think with OLS $R^2 = 0.999993$ from `latency_calibration_v2.jsonl`), omitting prompt prefill, request queueing, and dual-model residency/switching overhead. Analytical throughput-ratio sensitivity ([outputs/phase2_3/supplementary/S2_throughput_ratio_sensitivity.csv](outputs/phase2_3/supplementary/S2_throughput_ratio_sensitivity.csv)) shows `Router @Val 20%` remains strictly faster than both `8B Direct` and `Adaptive SC` inside **`1.5583 < r < 4.5031`** (observed L4 ratio `r = 4.06`, paired bootstrap 95% CI `[4.04, 4.09]`).
2. **Val vs. Test Tier-2 Variance**: While Tier-1 truncation rescue generalizes reliably (`+9` net on `Val 200`; `+20` net on `Test 500`), Tier-2 completed-answer routing yielded `+1 / -1 = 0` net gain on `Val 200` (`18` picks) versus `+14 / -2 = +12` on `Test 500` (`70` picks). Accordingly, Tier-2 learned lift is treated as Test-observed rather than Val-confirmed, and `Router @Val 20%` is reported as trading `-3.00 pp` accuracy against `8B Direct` rather than claiming equivalence.
3. **Static Benchmark Replay**: All Phase 2/3 metrics replay cached `Qwen3-1.7B` / `Qwen3-8B` outputs under the repaired 4-state grader (`25` label changes `wrong -> correct` across `5,100` cached completions: `4` Dev, `21` Test).
