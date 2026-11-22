"""Generate the self-contained vLLM-accelerated Colab notebook for Step 1B (Pilot 50 v2) & Step 1C (Full Cache)."""

from __future__ import annotations

import json
from pathlib import Path
import re
import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
MANIFEST_PATH = PROJECT_ROOT / "data" / "splits" / "splits_manifest.json"
DEFAULT_NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "phase1_colab_data_collection.ipynb"


def _strip_package_imports(source: str) -> str:
    """Remove relative intra-package imports and future imports when inlining into a single cell."""

    lines: list[str] = []
    skipping_parens = False
    for line in source.splitlines():
        if skipping_parens:
            if ")" in line:
                skipping_parens = False
            continue
        if re.match(r"^\s*from\s+\.[A-Za-z0-9_]*\s+import\s+", line):
            if "(" in line and ")" not in line:
                skipping_parens = True
            continue
        if re.match(r"^\s*from\s+__future__\s+import\s+", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _load_expected_hashes() -> dict[str, str]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {MANIFEST_PATH}. Run `python -m scripts.prepare_data` first."
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        name: str(meta["sha256"])
        for name, meta in manifest["files"].items()
        if "sha256" in meta
    }


def build_notebook(out_path: Path = DEFAULT_NOTEBOOK_PATH) -> Path:
    expected_hashes = _load_expected_hashes()

    grading_src = _strip_package_imports((SRC_DIR / "grading.py").read_text(encoding="utf-8"))
    data_prep_src = _strip_package_imports((SRC_DIR / "data_prep.py").read_text(encoding="utf-8"))
    helpers_src = _strip_package_imports((SRC_DIR / "colab_helpers.py").read_text(encoding="utf-8"))

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# MATH 550 Final Project — Phase 1 (Track A vLLM: Step 1B Pilot v2 & Step 1C Full Cache)\n\n"
            "**Research Question:** *When should an LLM self-refine, sample more, think longer, or escalate to a larger model?*\n\n"
            "### How to Run (Colab GPU Only: L4 22.5GB / A100 / G4)\n"
            "1. Connect this notebook to a **Google Colab GPU runtime** (`L4`, `A100`, or `G4`; `T4` cannot hold unquantized `Qwen3-8B`).\n"
            "2. **Step 1B (Pilot 50 v2, ~3–5 min):** Keep `RUN_MODE = \"pilot\"` in Cell 1 and click **Run All**. "
            "Verify low Non-Thinking truncation (`max_new_tokens=2048`), clean `vLLM` log-probs, and download `math550_pilot_v2_bundle.zip`.\n"
            "3. **Step 1C (Lean Full Cache, 5,600 gens):** Change `RUN_MODE = \"full\"` in Cell 1 and click **Run All**. "
            "Prioritizes `math500_test_500 (Atom 1-3)` -> `router_val_200` & stratified `router_train_600` (`core 300` then `extra 300`) -> "
            "`Atom 4 (8B)` -> `Atom 5 (1.7B Thinking)` last, with continuous-batch chunks saved to Google Drive.\n\n"
            "> **Note:** Cell 1 & Cell 2 intentionally run `vLLM` setup *before* importing `torch`."
        )
    )

    cell_1_config = '''# === Cell 1: Pre-Import Colab Guard & Experiment Config (Do NOT import torch yet!) ===
import importlib.util
import os
import subprocess
from pathlib import Path

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["VLLM_WORKER_MULTIPROCESS_METHOD"] = "spawn"

ALLOW_NON_COLAB_OVERRIDE = False
HAS_COLAB_PKG = importlib.util.find_spec("google.colab") is not None
IN_COLAB = HAS_COLAB_PKG or Path("/content").exists()
if not IN_COLAB and not ALLOW_NON_COLAB_OVERRIDE:
    raise RuntimeError("Non-Colab runtime detected. Per project rules, run this notebook on Google Colab.")

gpu_query = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True,
    text=True,
)
if gpu_query.returncode != 0:
    raise RuntimeError(f"CUDA GPU unavailable via nvidia-smi: {gpu_query.stderr}")
print("Detected GPU(s):", gpu_query.stdout.strip())

# --- Experiment Mode & Version ---
# Keep "pilot" for Step 1B (pilot_50_v2), then switch to "full" for Step 1C.
RUN_MODE = "pilot"  # choices: "pilot", "full"
RUN_TAG = "v2"
ENGINE_NAME = "vllm"
TRY_MOUNT_GOOGLE_DRIVE = True
RUN_BS1_LATENCY_CALIBRATION = True
CALIBRATION_SAMPLE_COUNT = 10
CALIBRATION_THINK_COUNT = 5

# --- Models, Limits & Decoding Contract ---
MODEL_SMALL = "Qwen/Qwen3-1.7B"
MODEL_LARGE = "Qwen/Qwen3-8B"
MAX_NEW_TOKENS_NON_THINK = 2048
MAX_NEW_TOKENS_CRITIQUE = 512
MAX_NEW_TOKENS_THINK = 4096
N_SAMPLING_PATHS = 3
SAMPLING_PARAMS_NON_THINK = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
SAMPLING_PARAMS_THINK = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
CHUNK_SIZE_SMALL = 250
CHUNK_SIZE_LARGE = 200
SEED = 550
'''
    cells.append(nbf.v4.new_code_cell(cell_1_config))

    cell_2_deps = '''# === Cell 2: Verify/Install vLLM & Dependencies (Pre-Torch) + Mount Google Drive ===
import importlib.metadata
import subprocess
import sys

def _ver_tuple(pkg: str) -> tuple[int, ...]:
    try:
        raw = importlib.metadata.version(pkg).split("+")[0]
        parts = []
        for seg in raw.split(".")[:3]:
            digits = "".join(ch for ch in seg if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        return tuple(parts)
    except importlib.metadata.PackageNotFoundError:
        return (0, 0, 0)

def _check_vllm_subprocess() -> tuple[str, str, str]:
    code = "\\n".join([
        "import importlib.metadata as md, importlib.util, torch",
        "vllm_ver = 'MISSING'",
        "if importlib.util.find_spec('vllm') is not None:",
        "    import vllm",
        "    vllm_ver = getattr(vllm, '__version__', md.version('vllm'))",
        "print(f'{torch.__version__}|{torch.version.cuda}|{vllm_ver}')",
    ])
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if res.returncode != 0:
        return ("ERROR", res.stderr.strip(), res.stdout.strip())
    parts = res.stdout.strip().splitlines()[-1].split("|")
    return (parts[0], parts[1], parts[2])

torch_before, cuda_before, vllm_before = _check_vllm_subprocess()
print(f"Environment check -> PyTorch={torch_before} | CUDA={cuda_before} | vLLM={vllm_before}")

need_install = []
if vllm_before in ("MISSING", "ERROR"):
    need_install.append("vllm")
if _ver_tuple("transformers") < (4, 51, 0):
    need_install.append("transformers>=4.51.0")
if _ver_tuple("sympy") < (1, 12, 0):
    need_install.append("sympy>=1.12")

if need_install:
    print("Installing required packages:", need_install)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need_install])
    torch_after, cuda_after, vllm_after = _check_vllm_subprocess()
    print(f"After install     -> PyTorch={torch_after} | CUDA={cuda_after} | vLLM={vllm_after}")
    if torch_after == "ERROR":
        raise RuntimeError(f"vLLM import check failed after install: {cuda_after}")
    if torch_after != torch_before:
        print(
            "NOTE: PyTorch changed during install. If Cell 5 hits any shared-library error, "
            "click Runtime -> Restart session once and re-run from Cell 1."
        )
else:
    print("All required packages already satisfy minimum versions.")

import torch
import transformers
import vllm

if not torch.cuda.is_available() and not ALLOW_NON_COLAB_OVERRIDE:
    raise RuntimeError("torch.cuda.is_available() is False.")

gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
major_cc = torch.cuda.get_device_capability(0)[0]
DTYPE_STR = "bfloat16" if (major_cc >= 8 and torch.cuda.is_bf16_supported()) else "float16"

print(
    f"Active GPU={gpu_name} | VRAM={vram_gb:.1f} GB | dtype={DTYPE_STR} | "
    f"transformers={transformers.__version__} | vllm={getattr(vllm, '__version__', 'ok')}"
)
if vram_gb < 18.5:
    raise RuntimeError(
        f"Active GPU only has {vram_gb:.1f} GB VRAM (likely T4). "
        "Unquantized Qwen3-8B bfloat16 requires >=22 GB (switch Colab GPU to L4, A100, or G4)."
    )

OUTPUT_ROOT = Path("/content/math550_outputs") if Path("/content").exists() else Path("./outputs")
if TRY_MOUNT_GOOGLE_DRIVE and HAS_COLAB_PKG:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        OUTPUT_ROOT = Path("/content/drive/MyDrive/math550_project_outputs")
        print(f"Google Drive mounted. Persistent OUTPUT_ROOT = {OUTPUT_ROOT}")
    except Exception as exc:
        print(f"Drive mount skipped/unavailable ({exc}); using local VM path {OUTPUT_ROOT}")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
'''
    cells.append(nbf.v4.new_code_cell(cell_2_deps))

    cell_3_modules = (
        "# === Cell 3: Verified Project Utilities (Grading, Splits, Feature Math, Resume I/O) ===\n"
        "from __future__ import annotations\n\n"
        + grading_src
        + "\n\n"
        + data_prep_src
        + "\n\n"
        + helpers_src
        + "\n"
    )
    cells.append(nbf.v4.new_code_cell(cell_3_modules))

    expected_hashes_literal = json.dumps(expected_hashes, indent=2)
    cell_4_data = f'''# === Cell 4: Load or Deterministically Rebuild Zero-Leakage Splits + Verify SHA-256 ===
EXPECTED_SPLIT_HASHES = {expected_hashes_literal}

SPLITS_DIR = OUTPUT_ROOT / "splits"
RAW_DIR = OUTPUT_ROOT / "raw"
BUNDLE_CANDIDATES = [
    SPLITS_DIR / "splits_bundle.json",
    Path("/content/splits_bundle.json"),
    Path("./data/splits/splits_bundle.json"),
]

bundle_loaded_from = None
for cand in BUNDLE_CANDIDATES:
    if cand.is_file() and sha256_file(cand) == EXPECTED_SPLIT_HASHES["splits_bundle.json"]:
        splits = json.loads(cand.read_text(encoding="utf-8"))
        bundle_loaded_from = cand
        print(f"Loaded verified splits_bundle.json from {{cand}}")
        break

if bundle_loaded_from is None:
    print("Downloading immutable official MATH sources & rebuilding deterministic splits...")
    raw_meta = ensure_raw_datasets(RAW_DIR, course_math500_path=None, force=False)
    raw_train = json.loads((RAW_DIR / "math_full_minus_math500.json").read_text(encoding="utf-8"))
    raw_test = json.loads((RAW_DIR / "math500_test.json").read_text(encoding="utf-8"))
    splits, manifest = build_splits(raw_train, raw_test, seed=SEED)
    write_splits(splits, manifest, SPLITS_DIR, raw_metadata=raw_meta)

    for fname, expected_sha in EXPECTED_SPLIT_HASHES.items():
        observed_sha = sha256_file(SPLITS_DIR / fname)
        assert observed_sha == expected_sha, (
            f"SHA-256 mismatch for {{fname}}: expected {{expected_sha}}, got {{observed_sha}}"
        )
    print("All split SHA-256 digests verified bit-for-bit against Step 1A manifest!")

for split_name, rows in splits.items():
    print(f"  - {{split_name}}: {{len(rows)}} problems")
'''
    cells.append(nbf.v4.new_code_cell(cell_4_data))

    cell_5_engine = '''# === Cell 5: vLLM Continuous-Batch Engine, Exact Log-Probs & Single-GPU Calibration ===
import gc
import time
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
except ImportError:
    destroy_distributed_environment = None
    destroy_model_parallel = None


def load_vllm_model(
    model_id: str,
    *,
    candidates: list[tuple[bool, int, int, float]],
):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False, token=False)
    stop_ids = set()
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    for tok in ("<|im_end|>", "<|endoftext|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0:
            stop_ids.add(tid)

    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    if not isinstance(think_end_id, int) or think_end_id < 0:
        think_end_id = 151668

    llm = None
    active_max_len = 0
    last_err = None
    for eager_flag, max_len, max_seqs, gpu_util in candidates:
        try:
            t0 = time.perf_counter()
            llm = LLM(
                model=model_id,
                dtype=DTYPE_STR,
                max_model_len=max_len,
                max_num_seqs=max_seqs,
                gpu_memory_utilization=gpu_util,
                enforce_eager=eager_flag,
                seed=SEED,
                trust_remote_code=False,
            )
            active_max_len = max_len
            print(
                f"Loaded {model_id} with vLLM (enforce_eager={eager_flag}, max_len={max_len}, "
                f"max_seqs={max_seqs}, util={gpu_util}) in {time.perf_counter() - t0:.1f}s"
            )
            break
        except Exception as exc:
            last_err = repr(exc)
            print(f"Load attempt (eager={eager_flag}, max_len={max_len}) failed: {last_err[:300]}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if llm is None:
        raise RuntimeError(f"Failed to load {model_id} in vLLM: {last_err}")

    return llm, tokenizer, stop_ids, think_end_id, active_max_len


def unload_vllm(llm_obj, tokenizer=None) -> None:
    try:
        engine = getattr(llm_obj, "llm_engine", None)
        core = getattr(engine, "engine_core", None)
        if core is not None and hasattr(core, "shutdown"):
            core.shutdown()
    except Exception:
        pass
    try:
        if destroy_model_parallel is not None:
            destroy_model_parallel()
        if destroy_distributed_environment is not None:
            destroy_distributed_environment()
    except Exception:
        pass
    del llm_obj
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2.0)


def render_qwen3_prompt(tokenizer, messages: list[dict[str, str]], *, enable_thinking: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def _decode_completion_output(
    comp,
    tokenizer,
    stop_ids: set[int],
    think_end_id: int,
    *,
    allowed_max_tokens: int,
    collect_logprobs: bool,
    split_think_tokens: bool,
) -> dict[str, Any]:
    tids = list(comp.token_ids)
    ends_with_stop = bool(tids and tids[-1] in stop_ids)
    valid_ids = tids[:-1] if ends_with_stop else tids
    truncated = (comp.finish_reason == "length") or (not ends_with_stop and len(valid_ids) >= allowed_max_tokens)

    tok_logprobs: list[float] | None = None
    if collect_logprobs:
        raw_lps = comp.logprobs or []
        tok_logprobs = []
        for t_idx, tid in enumerate(valid_ids):
            if t_idx < len(raw_lps) and raw_lps[t_idx] is not None and tid in raw_lps[t_idx]:
                entry = raw_lps[t_idx][tid]
                val = float(getattr(entry, "logprob", entry))
                tok_logprobs.append(val if math.isfinite(val) else -20.0)
            else:
                tok_logprobs.append(-20.0)

    think_closed: bool | None = None
    think_tokens = 0
    body_text = ""
    full_text = tokenizer.decode(valid_ids, skip_special_tokens=True)

    if split_think_tokens:
        rev_pos = None
        for idx in range(len(valid_ids) - 1, -1, -1):
            if valid_ids[idx] == think_end_id:
                rev_pos = idx
                break
        if rev_pos is not None:
            think_closed = True
            think_tokens = rev_pos + 1
            body_text = tokenizer.decode(valid_ids[rev_pos + 1 :], skip_special_tokens=True).strip()
        else:
            think_closed = False
            think_tokens = len(valid_ids)
            body_text = ""

    return {
        "full_text": full_text,
        "body_text": body_text if split_think_tokens else full_text,
        "valid_ids": valid_ids,
        "out_tokens": len(valid_ids),
        "truncated": truncated,
        "token_logprobs": tok_logprobs,
        "think_closed": think_closed,
        "think_tokens": think_tokens,
    }


def generate_vllm_batch(
    llm,
    tokenizer,
    stop_ids: set[int],
    think_end_id: int,
    active_max_model_len: int,
    prompts: list[str],
    *,
    max_new_tokens: int,
    do_sample: bool = False,
    sampling_params: dict | None = None,
    n: int = 1,
    collect_logprobs: bool = False,
    split_think_tokens: bool = False,
    seed: int | None = None,
    use_tqdm: bool = True,
) -> list[Any]:
    prompt_id_lists = tokenizer(prompts, add_special_tokens=False)["input_ids"]
    allowed_list = [
        min(max_new_tokens, max(128, active_max_model_len - len(pids) - 16))
        for pids in prompt_id_lists
    ]

    base_kwargs: dict[str, Any] = {
        "n": n,
        "logprobs": 1 if collect_logprobs else None,
        "seed": seed if seed is not None else SEED,
    }
    if do_sample and sampling_params:
        base_kwargs.update(sampling_params)
    else:
        base_kwargs["temperature"] = 0.0

    if len(set(allowed_list)) == 1:
        sp_arg: Any = SamplingParams(max_tokens=allowed_list[0], **base_kwargs)
    else:
        sp_arg = [SamplingParams(max_tokens=amt, **base_kwargs) for amt in allowed_list]

    t0 = time.perf_counter()
    raw_outs = llm.generate(prompts, sp_arg, use_tqdm=use_tqdm)
    batch_elapsed = time.perf_counter() - t0
    per_item_latency = batch_elapsed / max(len(prompts), 1)
    per_path_latency = batch_elapsed / max(len(prompts) * n, 1)

    decoded: list[Any] = []
    for idx, out in enumerate(raw_outs):
        in_toks = len(out.prompt_token_ids) if getattr(out, "prompt_token_ids", None) else len(prompt_id_lists[idx])
        amt = allowed_list[idx]
        if n == 1:
            item = _decode_completion_output(
                out.outputs[0],
                tokenizer,
                stop_ids,
                think_end_id,
                allowed_max_tokens=amt,
                collect_logprobs=collect_logprobs,
                split_think_tokens=split_think_tokens,
            )
            item["in_tokens"] = in_toks
            item["batch_latency_s"] = per_item_latency
            decoded.append(item)
        else:
            paths = []
            for comp in out.outputs:
                item = _decode_completion_output(
                    comp,
                    tokenizer,
                    stop_ids,
                    think_end_id,
                    allowed_max_tokens=amt,
                    collect_logprobs=collect_logprobs,
                    split_think_tokens=split_think_tokens,
                )
                item["in_tokens"] = in_toks
                item["batch_latency_s"] = per_path_latency
                paths.append(item)
            decoded.append(paths)
    return decoded


def calibrate_bs1_latency(
    llm,
    tokenizer,
    stop_ids: set[int],
    think_end_id: int,
    active_max_model_len: int,
    rows: list[dict[str, Any]],
    *,
    mode_label: str,
    enable_thinking: bool,
    max_new_tokens: int,
    do_sample: bool,
    sampling_params: dict | None,
    out_jsonl: Path,
    limit: int = 10,
) -> dict[str, Any]:
    """Measure true single-request (bs=1) end-to-end latency on the active GPU and engine."""

    done_keys = set()
    if out_jsonl.is_file():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    if rec.get("gpu_name") == gpu_name and rec.get("engine") == ENGINE_NAME:
                        done_keys.add((rec["mode_label"], str(rec["unique_id"])))
                except Exception:
                    pass

    sample_rows = rows[:limit]
    pending_cal = [
        (idx, row)
        for idx, row in enumerate(sample_rows)
        if (mode_label, str(row["unique_id"])) not in done_keys
    ]
    print(
        f"[Latency Calibration | {mode_label} | {gpu_name} | {ENGINE_NAME}] "
        f"pending={len(pending_cal)}/{len(sample_rows)} (bs=1)...",
        flush=True,
    )

    if pending_cal and not any(k[0] == mode_label for k in done_keys):
        warm_p = render_qwen3_prompt(
            tokenizer, build_solve_messages(sample_rows[0]["problem"]), enable_thinking=enable_thinking
        )
        _ = generate_vllm_batch(
            llm, tokenizer, stop_ids, think_end_id, active_max_model_len, [warm_p],
            max_new_tokens=min(64, max_new_tokens),
            do_sample=do_sample, sampling_params=sampling_params,
            collect_logprobs=False, split_think_tokens=enable_thinking,
            seed=SEED, use_tqdm=False,
        )

    for idx, row in tqdm(pending_cal, desc=f"Calibrate bs=1 {mode_label}"):
        prompt = render_qwen3_prompt(
            tokenizer, build_solve_messages(row["problem"]), enable_thinking=enable_thinking
        )
        res = generate_vllm_batch(
            llm, tokenizer, stop_ids, think_end_id, active_max_model_len, [prompt],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample, sampling_params=sampling_params,
            collect_logprobs=False, split_think_tokens=enable_thinking,
            seed=SEED + idx, use_tqdm=False,
        )[0]
        rec = {
            "mode_label": mode_label,
            "unique_id": row["unique_id"],
            "level": row["level"],
            "in_tokens": res["in_tokens"],
            "out_tokens": res["out_tokens"],
            "bs1_latency_s": res["batch_latency_s"],
            "truncated": res["truncated"],
            "gpu_name": gpu_name,
            "engine": ENGINE_NAME,
            "max_new_tokens": max_new_tokens,
        }
        append_jsonl_records(out_jsonl, [rec])

    all_mode = []
    if out_jsonl.is_file():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if (
                    rec.get("mode_label") == mode_label
                    and rec.get("gpu_name") == gpu_name
                    and rec.get("engine") == ENGINE_NAME
                ):
                    all_mode.append(rec)
    total_tok = sum(r["out_tokens"] for r in all_mode)
    total_sec = sum(r["bs1_latency_s"] for r in all_mode)
    tok_per_s = total_tok / total_sec if total_sec > 0 else 0.0
    print(f"[Latency Calibration | {mode_label} | {gpu_name} | {ENGINE_NAME}] n={len(all_mode)} | {tok_per_s:.1f} tok/s (bs=1)")
    return {"mode_label": mode_label, "count": len(all_mode), "bs1_tok_per_s": tok_per_s}
'''
    cells.append(nbf.v4.new_code_cell(cell_5_engine))

    cell_6_runner = '''# === Cell 6: Resumable Priority Queue with Stratified Incremental Checkpoints ===
from tqdm.auto import tqdm


def _chunked(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run_atom1_first_pass(
    llm, tokenizer, stop_ids, think_end_id, max_len, rows, out_path: Path, *, chunk_size: int, label: str = ""
) -> list[dict[str, Any]]:
    done = load_completed_ids(out_path)
    pending = [r for r in rows if str(r["unique_id"]) not in done]
    tag = label or out_path.parent.name
    print(f"[Atom 1 | 1.7B Non-Think] {tag}: pending={len(pending)}/{len(rows)}")
    decode_fn = lambda ids: tokenizer.decode(ids, skip_special_tokens=True)

    for chunk in _chunked(pending, chunk_size):
        prompts = [
            render_qwen3_prompt(tokenizer, build_solve_messages(r["problem"]), enable_thinking=False)
            for r in chunk
        ]
        outs = generate_vllm_batch(
            llm, tokenizer, stop_ids, think_end_id, max_len, prompts,
            max_new_tokens=MAX_NEW_TOKENS_NON_THINK,
            do_sample=False, collect_logprobs=True, split_think_tokens=False,
        )
        records = []
        for r, out in zip(chunk, outs):
            grade = grade_and_check_sample(
                out["body_text"], r["answer"], truncated=out["truncated"], strip_thinking=False
            )
            feats = extract_router_features(
                out["token_logprobs"], out["valid_ids"], decode_fn,
                input_tokens=out["in_tokens"], truncated=out["truncated"],
                response_text=out["body_text"],
            )
            records.append(
                {
                    "unique_id": r["unique_id"],
                    "split_name": r["split_name"],
                    "level": r["level"],
                    "subject": r["subject"],
                    "problem": r["problem"],
                    "ground_truth": r["answer"],
                    "response_text": out["body_text"],
                    "in_tokens": out["in_tokens"],
                    "out_tokens": out["out_tokens"],
                    "latency_s": out["batch_latency_s"],
                    "features": feats,
                    **grade,
                }
            )
        append_jsonl_records(out_path, records)
    return read_jsonl_records(out_path)


def run_atom2_self_refine(
    llm, tokenizer, stop_ids, think_end_id, max_len, atom1_rows: list[dict[str, Any]], out_path: Path, *, chunk_size: int
) -> list[dict[str, Any]]:
    done = load_completed_ids(out_path)
    pending = [r for r in atom1_rows if str(r["unique_id"]) not in done]
    print(f"[Atom 2 | 1.7B Self-Refine] {out_path.parent.name}: pending={len(pending)}/{len(atom1_rows)}")

    for chunk in _chunked(pending, chunk_size):
        crit_prompts = [
            render_qwen3_prompt(
                tokenizer,
                build_critique_messages(r["problem"], r["response_text"]),
                enable_thinking=False,
            )
            for r in chunk
        ]
        crit_outs = generate_vllm_batch(
            llm, tokenizer, stop_ids, think_end_id, max_len, crit_prompts,
            max_new_tokens=MAX_NEW_TOKENS_CRITIQUE,
            do_sample=False, collect_logprobs=False, split_think_tokens=False,
        )
        ref_prompts = [
            render_qwen3_prompt(
                tokenizer,
                build_refine_messages(r["problem"], r["response_text"], c["body_text"]),
                enable_thinking=False,
            )
            for r, c in zip(chunk, crit_outs)
        ]
        ref_outs = generate_vllm_batch(
            llm, tokenizer, stop_ids, think_end_id, max_len, ref_prompts,
            max_new_tokens=MAX_NEW_TOKENS_NON_THINK,
            do_sample=False, collect_logprobs=False, split_think_tokens=False,
        )
        records = []
        for r, c, rf in zip(chunk, crit_outs, ref_outs):
            grade = grade_and_check_sample(
                rf["body_text"], r["ground_truth"], truncated=rf["truncated"], strip_thinking=False
            )
            records.append(
                {
                    "unique_id": r["unique_id"],
                    "split_name": r["split_name"],
                    "level": r["level"],
                    "critique_text": c["body_text"],
                    "critique_in_tokens": c["in_tokens"],
                    "critique_out_tokens": c["out_tokens"],
                    "critique_truncated": c["truncated"],
                    "critique_latency_s": c["batch_latency_s"],
                    "refine_text": rf["body_text"],
                    "refine_in_tokens": rf["in_tokens"],
                    "refine_out_tokens": rf["out_tokens"],
                    "refine_latency_s": rf["batch_latency_s"],
                    "refine_status": grade["status"],
                    "refine_correct": grade["correct"],
                    "refine_extracted_answer": grade["extracted_answer"],
                    "refine_has_boxed": grade["has_boxed"],
                    "refine_truncated": grade["truncated"],
                }
            )
        append_jsonl_records(out_path, records)
    return read_jsonl_records(out_path)


def run_atom3_sampling(
    llm, tokenizer, stop_ids, think_end_id, max_len, rows, out_path: Path, *, chunk_size: int
) -> list[dict[str, Any]]:
    done = load_completed_ids(out_path)
    pending = [r for r in rows if str(r["unique_id"]) not in done]
    print(f"[Atom 3 | 1.7B {N_SAMPLING_PATHS}-Path Shared-Prefix] {out_path.parent.name}: pending={len(pending)}/{len(rows)}")

    for chunk_idx, chunk in enumerate(_chunked(pending, chunk_size)):
        prompts = [
            render_qwen3_prompt(tokenizer, build_solve_messages(r["problem"]), enable_thinking=False)
            for r in chunk
        ]
        outs_per_prompt = generate_vllm_batch(
            llm, tokenizer, stop_ids, think_end_id, max_len, prompts,
            max_new_tokens=MAX_NEW_TOKENS_NON_THINK,
            do_sample=True, sampling_params=SAMPLING_PARAMS_NON_THINK,
            n=N_SAMPLING_PATHS, collect_logprobs=False, split_think_tokens=False,
            seed=SEED + chunk_idx,
        )
        records = []
        for r, path_outs in zip(chunk, outs_per_prompt):
            samples = []
            for path_idx, out in enumerate(path_outs):
                grade = grade_and_check_sample(
                    out["body_text"], r["answer"], truncated=out["truncated"], strip_thinking=False
                )
                samples.append(
                    {
                        "sample_idx": path_idx,
                        "response_text": out["body_text"],
                        "in_tokens": out["in_tokens"],
                        "out_tokens": out["out_tokens"],
                        "latency_s": out["batch_latency_s"],
                        **grade,
                    }
                )
            records.append(
                {
                    "unique_id": r["unique_id"],
                    "split_name": r["split_name"],
                    "level": r["level"],
                    "ground_truth": r["answer"],
                    "samples": samples,
                }
            )
        append_jsonl_records(out_path, records)
    return read_jsonl_records(out_path)


def run_atom4_large_model(
    llm, tokenizer, stop_ids, think_end_id, max_len, rows, out_path: Path, *, chunk_size: int, label: str = ""
) -> list[dict[str, Any]]:
    done = load_completed_ids(out_path)
    pending = [r for r in rows if str(r["unique_id"]) not in done]
    tag = label or out_path.parent.name
    print(f"[Atom 4 | 8B Non-Think] {tag}: pending={len(pending)}/{len(rows)}")

    for chunk in _chunked(pending, chunk_size):
        prompts = [
            render_qwen3_prompt(tokenizer, build_solve_messages(r["problem"]), enable_thinking=False)
            for r in chunk
        ]
        outs = generate_vllm_batch(
            llm, tokenizer, stop_ids, think_end_id, max_len, prompts,
            max_new_tokens=MAX_NEW_TOKENS_NON_THINK,
            do_sample=False, collect_logprobs=False, split_think_tokens=False,
        )
        records = []
        for r, out in zip(chunk, outs):
            grade = grade_and_check_sample(
                out["body_text"], r["answer"], truncated=out["truncated"], strip_thinking=False
            )
            records.append(
                {
                    "unique_id": r["unique_id"],
                    "split_name": r["split_name"],
                    "level": r["level"],
                    "ground_truth": r["answer"],
                    "response_text": out["body_text"],
                    "in_tokens": out["in_tokens"],
                    "out_tokens": out["out_tokens"],
                    "latency_s": out["batch_latency_s"],
                    **grade,
                }
            )
        append_jsonl_records(out_path, records)
    return read_jsonl_records(out_path)


def run_atom5_thinking(
    llm, tokenizer, stop_ids, think_end_id, max_len, rows, out_path: Path, *, chunk_size: int
) -> list[dict[str, Any]]:
    done = load_completed_ids(out_path)
    pending = [r for r in rows if str(r["unique_id"]) not in done]
    print(f"[Atom 5 | 1.7B Thinking] {out_path.parent.name}: pending={len(pending)}/{len(rows)}")

    for chunk_idx, chunk in enumerate(_chunked(pending, chunk_size)):
        prompts = [
            render_qwen3_prompt(tokenizer, build_solve_messages(r["problem"]), enable_thinking=True)
            for r in chunk
        ]
        outs = generate_vllm_batch(
            llm, tokenizer, stop_ids, think_end_id, max_len, prompts,
            max_new_tokens=MAX_NEW_TOKENS_THINK,
            do_sample=True, sampling_params=SAMPLING_PARAMS_THINK,
            collect_logprobs=False, split_think_tokens=True, seed=SEED + chunk_idx,
        )
        records = []
        for r, out in zip(chunk, outs):
            grade = grade_and_check_sample(
                out["body_text"],
                r["answer"],
                truncated=out["truncated"],
                strip_thinking=True,
                think_closed=out["think_closed"],
            )
            records.append(
                {
                    "unique_id": r["unique_id"],
                    "split_name": r["split_name"],
                    "level": r["level"],
                    "ground_truth": r["answer"],
                    "full_text": out["full_text"],
                    "final_body": out["body_text"],
                    "in_tokens": out["in_tokens"],
                    "out_tokens": out["out_tokens"],
                    "think_tokens": out["think_tokens"],
                    "latency_s": out["batch_latency_s"],
                    **grade,
                }
            )
        append_jsonl_records(out_path, records)
    return read_jsonl_records(out_path)


# --- Build Stratified Execution Queue ---
if RUN_MODE == "pilot":
    EXEC_QUEUE = [
        {"label": "pilot_50_v2", "rows": splits["pilot_50"], "out_dir": f"pilot_50_{RUN_TAG}", "atoms": (1, 2, 3, 4, 5)},
    ]
    SUMMARY_SPECS = [
        {"out_dir": f"pilot_50_{RUN_TAG}", "atoms": (1, 2, 3, 4, 5)},
    ]
elif RUN_MODE == "full":
    train_core_300, train_extra_300 = split_stratified_halves(splits["router_train_600"])
    EXEC_QUEUE = [
        {"label": "math500_test_500", "rows": splits["math500_test_500"], "out_dir": "math500_test_500", "atoms": (1, 2, 3, 4, 5)},
        {"label": "router_val_200", "rows": splits["router_val_200"], "out_dir": "router_val_200", "atoms": (1, 4)},
        {"label": "router_train_core_300", "rows": train_core_300, "out_dir": "router_train_600", "atoms": (1, 4)},
        {"label": "router_train_extra_300", "rows": train_extra_300, "out_dir": "router_train_600", "atoms": (1, 4)},
    ]
    SUMMARY_SPECS = [
        {"out_dir": "math500_test_500", "atoms": (1, 2, 3, 4, 5)},
        {"out_dir": "router_val_200", "atoms": (1, 4)},
        {"out_dir": "router_train_600", "atoms": (1, 4)},
    ]
else:
    raise ValueError(f"Unknown RUN_MODE={RUN_MODE!r}")

run_started = time.perf_counter()
cal_path = OUTPUT_ROOT / f"latency_calibration_{RUN_TAG}.jsonl"

# --- Stage A: Qwen3-1.7B Non-Thinking (Atoms 1, 2, 3) ---
print(f"\\nLoading {MODEL_SMALL} (Stage A: Non-Thinking Atoms 1-3)...")
llm_small, tok_small, stop_small, think_end_small, max_len_small = load_vllm_model(
    MODEL_SMALL,
    candidates=[(False, 8192, 64, 0.85), (False, 6144, 32, 0.85), (True, 6144, 32, 0.85)],
)

if RUN_BS1_LATENCY_CALIBRATION:
    calibrate_bs1_latency(
        llm_small, tok_small, stop_small, think_end_small, max_len_small, splits["pilot_50"],
        mode_label="qwen3_1.7b_non_think", enable_thinking=False,
        max_new_tokens=MAX_NEW_TOKENS_NON_THINK, do_sample=False, sampling_params=None,
        out_jsonl=cal_path, limit=CALIBRATION_SAMPLE_COUNT,
    )

for job in EXEC_QUEUE:
    sdir = OUTPUT_ROOT / job["out_dir"]
    rows = job["rows"]
    atoms = set(job["atoms"])
    a1 = []
    if 1 in atoms:
        a1 = run_atom1_first_pass(
            llm_small, tok_small, stop_small, think_end_small, max_len_small,
            rows, sdir / "atom1_1.7b_first.jsonl", chunk_size=CHUNK_SIZE_SMALL, label=job["label"],
        )
    if 2 in atoms:
        a1_by_uid = {str(r["unique_id"]): r for r in a1}
        ordered_a1 = [a1_by_uid[str(r["unique_id"])] for r in rows if str(r["unique_id"]) in a1_by_uid]
        run_atom2_self_refine(
            llm_small, tok_small, stop_small, think_end_small, max_len_small,
            ordered_a1, sdir / "atom2_1.7b_self_refine.jsonl", chunk_size=CHUNK_SIZE_SMALL,
        )
    if 3 in atoms:
        run_atom3_sampling(
            llm_small, tok_small, stop_small, think_end_small, max_len_small,
            rows, sdir / "atom3_1.7b_samples3.jsonl", chunk_size=CHUNK_SIZE_SMALL,
        )

unload_vllm(llm_small, tok_small)

# --- Stage B: Qwen3-8B Non-Thinking (Atom 4, runs BEFORE slow Thinking) ---
print(f"\\nLoading {MODEL_LARGE} (Stage B: 8B Atom 4)...")
llm_large, tok_large, stop_large, think_end_large, max_len_large = load_vllm_model(
    MODEL_LARGE,
    candidates=[(False, 4096, 32, 0.92), (False, 3072, 32, 0.90), (True, 3072, 32, 0.92), (True, 2560, 16, 0.92)],
)

if RUN_BS1_LATENCY_CALIBRATION:
    calibrate_bs1_latency(
        llm_large, tok_large, stop_large, think_end_large, max_len_large, splits["pilot_50"],
        mode_label="qwen3_8b_non_think", enable_thinking=False,
        max_new_tokens=MAX_NEW_TOKENS_NON_THINK, do_sample=False, sampling_params=None,
        out_jsonl=cal_path, limit=CALIBRATION_SAMPLE_COUNT,
    )

for job in EXEC_QUEUE:
    if 4 in job["atoms"]:
        sdir = OUTPUT_ROOT / job["out_dir"]
        run_atom4_large_model(
            llm_large, tok_large, stop_large, think_end_large, max_len_large,
            job["rows"], sdir / "atom4_8b_first.jsonl", chunk_size=CHUNK_SIZE_LARGE, label=job["label"],
        )

unload_vllm(llm_large, tok_large)

# --- Stage C: Reload cached Qwen3-1.7B for Slow Thinking (Atom 5) LAST ---
if any(5 in job["atoms"] for job in EXEC_QUEUE):
    print(f"\\nLoading {MODEL_SMALL} (Stage C: Slow Thinking Atom 5)...")
    llm_small, tok_small, stop_small, think_end_small, max_len_small = load_vllm_model(
        MODEL_SMALL,
        candidates=[(False, 6144, 64, 0.85), (True, 6144, 32, 0.85)],
    )
    if RUN_BS1_LATENCY_CALIBRATION:
        calibrate_bs1_latency(
            llm_small, tok_small, stop_small, think_end_small, max_len_small, splits["pilot_50"],
            mode_label="qwen3_1.7b_think", enable_thinking=True,
            max_new_tokens=MAX_NEW_TOKENS_THINK, do_sample=True, sampling_params=SAMPLING_PARAMS_THINK,
            out_jsonl=cal_path, limit=CALIBRATION_THINK_COUNT,
        )
    for job in EXEC_QUEUE:
        if 5 in job["atoms"]:
            sdir = OUTPUT_ROOT / job["out_dir"]
            run_atom5_thinking(
                llm_small, tok_small, stop_small, think_end_small, max_len_small,
                job["rows"], sdir / "atom5_1.7b_thinking.jsonl", chunk_size=CHUNK_SIZE_SMALL,
            )
    unload_vllm(llm_small, tok_small)

total_elapsed_min = (time.perf_counter() - run_started) / 60.0
print(f"\\nFinished RUN_MODE={RUN_MODE!r} ({RUN_TAG}, engine={ENGINE_NAME}) in {total_elapsed_min:.2f} min.")
'''
    cells.append(nbf.v4.new_code_cell(cell_6_runner))

    cell_7_summary = '''# === Cell 7: Diagnostics Summary & Zip Bundle Export ===
import zipfile

all_summaries = {}
for spec in SUMMARY_SPECS:
    sdir = OUTPUT_ROOT / spec["out_dir"]
    atoms = set(spec["atoms"])
    a1 = read_jsonl_records(sdir / "atom1_1.7b_first.jsonl") if 1 in atoms else []
    a2 = read_jsonl_records(sdir / "atom2_1.7b_self_refine.jsonl") if 2 in atoms else []
    a3 = read_jsonl_records(sdir / "atom3_1.7b_samples3.jsonl") if 3 in atoms else []
    a4 = read_jsonl_records(sdir / "atom4_8b_first.jsonl") if 4 in atoms else []
    a5 = read_jsonl_records(sdir / "atom5_1.7b_thinking.jsonl") if 5 in atoms else []
    diag = summarize_pilot_diagnostics(a1, a2, a3, a4, a5)
    if a3:
        distinct_counts = [len({s["response_text"] for s in r.get("samples", [])}) for r in a3]
        diag["atom3_all_3_distinct"] = sum(c == N_SAMPLING_PATHS for c in distinct_counts)
    diag["elapsed_minutes"] = round(total_elapsed_min, 2)
    diag["gpu_name"] = gpu_name
    diag["engine"] = ENGINE_NAME
    diag["max_new_tokens_non_think"] = MAX_NEW_TOKENS_NON_THINK
    diag["max_new_tokens_critique"] = MAX_NEW_TOKENS_CRITIQUE
    all_summaries[spec["out_dir"]] = diag

    summary_path = sdir / "summary.json"
    summary_path.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\\n===== SUMMARY: {spec['out_dir']} =====")
    print(json.dumps(diag, ensure_ascii=False, indent=2))

zip_name = f"math550_{RUN_MODE}_{RUN_TAG}_bundle.zip"
zip_path = OUTPUT_ROOT / zip_name
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for spec in SUMMARY_SPECS:
        sdir = OUTPUT_ROOT / spec["out_dir"]
        for p in sorted(sdir.glob("*")):
            if p.is_file():
                zf.write(p, arcname=f"{spec['out_dir']}/{p.name}")
    if cal_path.is_file():
        zf.write(cal_path, arcname=cal_path.name)

print(f"\\nPackaged bundle ready: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
if HAS_COLAB_PKG:
    try:
        from google.colab import files
        files.download(str(zip_path))
    except Exception as exc:
        print(f"Auto-download skipped ({exc}); download {zip_path} manually from sidebar or Drive.")
'''
    cells.append(nbf.v4.new_code_cell(cell_7_summary))

    for idx, cell in enumerate(cells):
        cell["id"] = f"cell-{idx:02d}"
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = nbf.writes(nb).encode("utf-8")
    out_path.write_bytes(raw)
    return out_path


def main() -> None:
    path = build_notebook()
    print(f"Generated vLLM Colab notebook: {path}")


if __name__ == "__main__":
    main()
