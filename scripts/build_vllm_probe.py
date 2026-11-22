"""Generate a fast (~3-5 min) standalone Colab probe notebook for Track A (vLLM compatibility & speed)."""

from __future__ import annotations

import json
from pathlib import Path
import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_PATH = PROJECT_ROOT / "data" / "splits" / "pilot_50.json"
DEFAULT_PROBE_PATH = PROJECT_ROOT / "notebooks" / "track_a_vllm_probe.ipynb"

# Exact `transformers` reference values from pilot_50 (Atom 1, first 3 non-truncated problems)
TF_REF_ATOM1 = {
    "math_train:1943": {"out_tokens": 243, "mean_token_logprob": -0.03235},
    "math_train:9354": {"out_tokens": 316, "mean_token_logprob": -0.05573},
    "math_train:8634": {"out_tokens": 368, "mean_token_logprob": -0.07978},
}


def build_vllm_probe_notebook(out_path: Path = DEFAULT_PROBE_PATH) -> Path:
    if not PILOT_PATH.is_file():
        raise FileNotFoundError(f"Missing {PILOT_PATH}; run `python -m scripts.prepare_data` first.")
    pilot_rows = json.loads(PILOT_PATH.read_text(encoding="utf-8"))[:10]
    sample_problems_literal = json.dumps(
        [
            {
                "unique_id": r["unique_id"],
                "level": r["level"],
                "problem": r["problem"],
                "answer": r["answer"],
            }
            for r in pilot_rows
        ],
        ensure_ascii=False,
        indent=2,
    )
    tf_ref_literal = json.dumps(TF_REF_ATOM1, ensure_ascii=False, indent=2)

    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# Track A Probe — vLLM Compatibility, Log-Prob Alignment & Speed Check (Colab Only)\n\n"
            "**Purpose (3–5 minutes total):** Test whether `vLLM` works cleanly on your active Colab GPU/PyTorch "
            "runtime before switching the full Phase 1 pipeline over.\n\n"
            "**What this probe verifies:**\n"
            "1. **Install & ABI Check:** Runs `pip install --dry-run vllm` first to see if `torch` would be overwritten, "
            "installs `vllm` *before* importing `torch` in this kernel, and verifies `import torch, vllm` in a fresh subprocess.\n"
            "2. **1.7B Speed + Log-Prob Alignment:** Loads `Qwen/Qwen3-1.7B`, checks `logprobs=1` & stop-token alignment "
            "against your saved `transformers` `pilot_50` values, and times 10 Non-Think (`max_tokens=2048`) + 10 Thinking (`max_tokens=4096`) problems.\n"
            "3. **8B L4 Memory Fit & Speed:** Cleanly unloads `1.7B` and tests whether `Qwen/Qwen3-8B` (`bfloat16`) fits on "
            "your active GPU (`L4 22.5GB` / `A100` / `G4`), trying CUDA graphs first and `enforce_eager=True` as fallback.\n"
            "4. **Go / No-Go Report:** Prints projected total runtime for the lean 5,600-generation Full Plan."
        )
    )

    cell_1_env = '''# === Cell 1: Colab Guard & Pre-Import Environment Probe (Do NOT import torch yet!) ===
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

ALLOW_NON_COLAB_OVERRIDE = False
IN_COLAB = (importlib.util.find_spec("google.colab") is not None) or Path("/content").exists()
if not IN_COLAB and not ALLOW_NON_COLAB_OVERRIDE:
    raise RuntimeError("Non-Colab runtime detected. Connect this notebook to Google Colab GPU first.")

gpu_query = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True,
    text=True,
)
if gpu_query.returncode != 0:
    raise RuntimeError(f"No CUDA GPU detected via nvidia-smi: {gpu_query.stderr}")
print("GPU:", gpu_query.stdout.strip())

def _check_env_subprocess() -> tuple[str, str, str]:
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

torch_before, cuda_before, vllm_before = _check_env_subprocess()
print(f"Before install -> PyTorch={torch_before} | CUDA={cuda_before} | vLLM={vllm_before}")

if vllm_before in ("MISSING", "ERROR"):
    dry = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--dry-run", "vllm"],
        capture_output=True,
        text=True,
    )
    dry_lines = [ln for ln in dry.stdout.splitlines() if "torch" in ln.lower() or "would install" in ln.lower()]
    print("Dry-run check (torch-related lines):", dry_lines[:6] if dry_lines else "none")
    if dry.returncode != 0:
        print("Dry-run failed:", dry.stderr[-1500:])

    print("Installing vllm...")
    pip_res = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "vllm"],
        capture_output=True,
        text=True,
    )
    if pip_res.returncode != 0:
        print("pip install vllm FAILED! Tail of stderr:")
        print(pip_res.stderr[-2500:])
        raise RuntimeError("Track A install failed -> fall back to Track B.")

    torch_after, cuda_after, vllm_after = _check_env_subprocess()
    print(f"After install  -> PyTorch={torch_after} | CUDA={cuda_after} | vLLM={vllm_after}")
    if torch_after == "ERROR":
        print("Subprocess import FAILED after install:")
        print(cuda_after)
        raise RuntimeError("Track A import failed -> fall back to Track B.")
    if torch_after != torch_before:
        print(
            "WARNING: PyTorch version changed during vLLM install. "
            "If Cell 2 hits any shared-library error, click Runtime -> Restart session and re-run from Cell 1."
        )
else:
    print("vLLM already installed and importable in subprocess!")
'''
    cells.append(nbf.v4.new_code_cell(cell_1_env))

    cell_2_17b = f'''# === Cell 2: Probe Qwen3-1.7B (Log-Probs Alignment + Non-Think & Thinking Speed) ===
import gc
import math
import time
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel
except ImportError:
    destroy_distributed_environment = None
    destroy_model_parallel = None

PROBE_ROWS = {sample_problems_literal}
TF_REF = {tf_ref_literal}

def render_solve_prompt(tokenizer, problem: str, *, enable_thinking: bool) -> str:
    msg = (
        "Solve the following math problem step by step. "
        "Put your final answer inside \\\\boxed{{...}} at the end.\\n\\n"
        f"Problem:\\n{{problem}}"
    )
    return tokenizer.apply_chat_template(
        [{{"role": "user", "content": msg}}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

def unload_vllm(llm_obj) -> None:
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2.0)

vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
major_cc = torch.cuda.get_device_capability(0)[0]
dtype_str = "bfloat16" if (major_cc >= 8 and torch.cuda.is_bf16_supported()) else "float16"
print(f"Active GPU={{torch.cuda.get_device_name(0)}} | VRAM={{vram_gb:.1f}} GB | dtype={{dtype_str}}")

MODEL_SMALL = "Qwen/Qwen3-1.7B"
tok_small = AutoTokenizer.from_pretrained(MODEL_SMALL, trust_remote_code=False, token=False)
stop_ids_small = {{int(tok_small.eos_token_id)}} if tok_small.eos_token_id is not None else set()
for s in ("<|im_end|>", "<|endoftext|>"):
    tid = tok_small.convert_tokens_to_ids(s)
    if isinstance(tid, int) and tid >= 0:
        stop_ids_small.add(tid)

t0 = time.perf_counter()
llm_small = LLM(
    model=MODEL_SMALL,
    dtype=dtype_str,
    max_model_len=4608,
    gpu_memory_utilization=0.85,
    seed=550,
    trust_remote_code=False,
)
load_17b_s = time.perf_counter() - t0
print(f"Loaded {{MODEL_SMALL}} in {{load_17b_s:.1f}}s")

# 1) Warm-up + 10-problem Non-Think batch with logprobs=1
prompts_nt = [render_solve_prompt(tok_small, r["problem"], enable_thinking=False) for r in PROBE_ROWS]
sp_nt = SamplingParams(temperature=0.0, max_tokens=2048, logprobs=1)
_ = llm_small.generate([prompts_nt[0]], SamplingParams(temperature=0.0, max_tokens=32, logprobs=1), use_tqdm=False)

t0 = time.perf_counter()
outs_nt = llm_small.generate(prompts_nt, sp_nt, use_tqdm=True)
nt_10_s = time.perf_counter() - t0

# Audit token_ids, stop-token presence, and logprobs extraction vs transformers reference
nt_tokens = 0
nt_trunc = 0
lp_ok = True
for idx, out in enumerate(outs_nt):
    comp = out.outputs[0]
    tids = list(comp.token_ids)
    lps = comp.logprobs or []
    ends_with_stop = bool(tids and tids[-1] in stop_ids_small)
    clean_tids = tids[:-1] if ends_with_stop else tids
    clean_lps = []
    for t_idx, tid in enumerate(clean_tids):
        if t_idx >= len(lps) or lps[t_idx] is None or tid not in lps[t_idx]:
            lp_ok = False
            break
        entry = lps[t_idx][tid]
        val = float(getattr(entry, "logprob", entry))
        if not math.isfinite(val):
            lp_ok = False
        clean_lps.append(val)
    nt_tokens += len(clean_tids)
    if comp.finish_reason == "length":
        nt_trunc += 1
    if idx < 3:
        uid = PROBE_ROWS[idx]["unique_id"]
        mean_lp = sum(clean_lps) / max(len(clean_lps), 1)
        ref = TF_REF.get(uid)
        ref_str = f" | tf_ref=(out={{ref['out_tokens']}}, mean_lp={{ref['mean_token_logprob']:.5f}})" if ref else ""
        print(
            f"  [NT #{{idx}}] {{uid}} | out_toks={{len(clean_tids)}} | "
            f"finish={{comp.finish_reason}} | ends_with_stop={{ends_with_stop}} | mean_lp={{mean_lp:.5f}}{{ref_str}}"
        )

# 2) Quick bs=1 timing on first 3 Non-Think prompts
t0 = time.perf_counter()
bs1_17b_toks = 0
for p in prompts_nt[:3]:
    o1 = llm_small.generate([p], sp_nt, use_tqdm=False)[0]
    bs1_17b_toks += len(o1.outputs[0].token_ids)
bs1_17b_tok_s = bs1_17b_toks / max(time.perf_counter() - t0, 1e-6)

# 3) 10-problem Thinking batch (max_tokens=4096)
prompts_thk = [render_solve_prompt(tok_small, r["problem"], enable_thinking=True) for r in PROBE_ROWS]
sp_thk = SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096, seed=550)
t0 = time.perf_counter()
outs_thk = llm_small.generate(prompts_thk, sp_thk, use_tqdm=True)
thk_10_s = time.perf_counter() - t0
thk_tokens = sum(len(o.outputs[0].token_ids) for o in outs_thk)
thk_trunc = sum(o.outputs[0].finish_reason == "length" for o in outs_thk)

print(
    f"\\n[1.7B Summary] NT_10={{nt_10_s:.1f}}s ({{nt_tokens}} toks, {{nt_tokens/nt_10_s:.1f}} tok/s, trunc={{nt_trunc}}/10, lp_ok={{lp_ok}}) | "
    f"bs1={{bs1_17b_tok_s:.1f}} tok/s"
)
print(
    f"[1.7B Think Summary] THK_10={{thk_10_s:.1f}}s ({{thk_tokens}} toks, {{thk_tokens/thk_10_s:.1f}} tok/s, trunc={{thk_trunc}}/10)"
)

unload_vllm(llm_small)
'''
    cells.append(nbf.v4.new_code_cell(cell_2_17b))

    cell_3_8b = '''# === Cell 3: Probe Qwen3-8B Fit on Active GPU (L4 22.5GB Check) + Speed Projection ===
if vram_gb < 18.5:
    raise RuntimeError(
        f"Active GPU only has {vram_gb:.1f} GB VRAM (likely T4); unquantized Qwen3-8B bfloat16 needs >=22 GB (L4/A100/G4)."
    )

MODEL_LARGE = "Qwen/Qwen3-8B"
tok_large = AutoTokenizer.from_pretrained(MODEL_LARGE, trust_remote_code=False, token=False)
prompts_8b = [render_solve_prompt(tok_large, r["problem"], enable_thinking=False) for r in PROBE_ROWS]

llm_large = None
used_eager = False
load_err = None
for eager_flag, max_len, max_seqs, util in [
    (False, 3072, 32, 0.90),
    (True, 3072, 32, 0.92),
    (True, 2560, 16, 0.92),
]:
    try:
        t0 = time.perf_counter()
        llm_large = LLM(
            model=MODEL_LARGE,
            dtype=dtype_str,
            max_model_len=max_len,
            max_num_seqs=max_seqs,
            gpu_memory_utilization=util,
            enforce_eager=eager_flag,
            seed=550,
            trust_remote_code=False,
        )
        load_8b_s = time.perf_counter() - t0
        used_eager = eager_flag
        print(f"Loaded {MODEL_LARGE} (enforce_eager={eager_flag}, max_len={max_len}, max_seqs={max_seqs}) in {load_8b_s:.1f}s")
        break
    except Exception as exc:
        load_err = repr(exc)
        print(f"Attempt (eager={eager_flag}, max_len={max_len}) failed: {load_err[:300]}")
        gc.collect()
        torch.cuda.empty_cache()

if llm_large is None:
    raise RuntimeError(f"Qwen3-8B bfloat16 failed to load under vLLM on this GPU: {load_err}")

sp_8b = SamplingParams(temperature=0.0, max_tokens=2048)
_ = llm_large.generate([prompts_8b[0]], SamplingParams(temperature=0.0, max_tokens=32), use_tqdm=False)

t0 = time.perf_counter()
outs_8b = llm_large.generate(prompts_8b, sp_8b, use_tqdm=True)
nt8b_10_s = time.perf_counter() - t0
toks_8b = sum(len(o.outputs[0].token_ids) for o in outs_8b)
trunc_8b = sum(o.outputs[0].finish_reason == "length" for o in outs_8b)

t0 = time.perf_counter()
bs1_8b_toks = 0
for p in prompts_8b[:3]:
    o1 = llm_large.generate([p], sp_8b, use_tqdm=False)[0]
    bs1_8b_toks += len(o1.outputs[0].token_ids)
bs1_8b_tok_s = bs1_8b_toks / max(time.perf_counter() - t0, 1e-6)

unload_vllm(llm_large)

# Linear projection from 10-problem batch to the lean 5,600-generation Full Plan:
# - 3,800 x 1.7B Non-Think (Test 500x6 + Val 200 + Train 600)
# - 1,300 x 8B Non-Think (Test 500 + Val 200 + Train 600)
# -   500 x 1.7B Thinking (Test 500)
est_17b_nt_min = (3800 / 10.0) * nt_10_s / 60.0
est_8b_nt_min = (1300 / 10.0) * nt8b_10_s / 60.0
est_17b_thk_min = (500 / 10.0) * thk_10_s / 60.0
est_total_min = est_17b_nt_min + est_8b_nt_min + est_17b_thk_min

print("\\n================ TRACK A PROBE REPORT ================")
print(f"GPU: {torch.cuda.get_device_name(0)} | 8B enforce_eager={used_eager} | logprobs_ok={lp_ok}")
print(f"1.7B NT  10 probs: {nt_10_s:.1f}s ({nt_tokens/nt_10_s:.1f} tok/s, trunc={nt_trunc}/10, bs1={bs1_17b_tok_s:.1f} tok/s)")
print(f"8B   NT  10 probs: {nt8b_10_s:.1f}s ({toks_8b/nt8b_10_s:.1f} tok/s, trunc={trunc_8b}/10, bs1={bs1_8b_tok_s:.1f} tok/s)")
print(f"1.7B THK 10 probs: {thk_10_s:.1f}s ({thk_tokens/thk_10_s:.1f} tok/s, trunc={thk_trunc}/10)")
print(
    f"Projected Lean Full (5,600 gens, conservative 10-item batch rate): "
    f"1.7B_NT~{est_17b_nt_min:.1f}m + 8B~{est_8b_nt_min:.1f}m + 1.7B_THK~{est_17b_thk_min:.1f}m "
    f"≈ {est_total_min:.1f} min total (full 200-500 item queues usually saturate GPU even better)."
)
'''
    cells.append(nbf.v4.new_code_cell(cell_3_8b))

    for idx, cell in enumerate(cells):
        cell["id"] = f"cell-{idx:02d}"
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(nbf.writes(nb).encode("utf-8"))
    return out_path


def main() -> None:
    p = build_vllm_probe_notebook()
    print(f"Generated vLLM probe notebook: {p}")


if __name__ == "__main__":
    main()
