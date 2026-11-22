"""Math answer extraction, SymPy equivalence grading, and 4-state reliability auditing.

Adapted cleanly from the course evaluation utilities (`reasoning_from_scratch.ch03` /
`demo_base_model_evaluation.ipynb`) with explicit 4-state classification
(`correct`, `wrong`, `invalid`, `truncated`) and truncation awareness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Literal

from sympy import simplify
from sympy.core.sympify import SympifyError
from sympy.parsing import sympy_parser as spp
from sympy.polys.polyerrors import PolynomialError
from tokenize import TokenError


OutcomeStatus = Literal["correct", "wrong", "invalid", "truncated"]

RE_NUMBER = re.compile(r"-?(?:\d+/\d+|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
RE_SPECIAL = re.compile(r"<\|[^>]+?\|>")

LATEX_FIXES = [
    (r"\\left\s*", ""),
    (r"\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"~|\\\s", " "),
    (r"\\cdot", "*"),
    ("\u00b7|\u00d7", "*"),
    (r"\\\^\\circ", ""),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    ("°", ""),
]

UNIT_TEXT_RE = re.compile(
    r"\\(?:text|mbox)\s*\{([\sA-Za-z.,/\-]*[A-Za-z][\sA-Za-z.,/\-]*)\}(?:\s*\^\s*\{?\d+\}?)?"
)

STYLE_WRAPPER_RE = re.compile(r"\\(?:mathbf|mathrm|textbf|boldsymbol|mathit)\s*\{([^{}]*)\}")

MATRIX_RE = re.compile(
    r"\\begin\{(?:p|b|v|V)?matrix\}(.*?)\\end\{(?:p|b|v|V)?matrix\}",
    re.DOTALL,
)

TRIG_LOG_RE = re.compile(
    r"\\(arcsin|arccos|arctan|sin|cos|tan|cot|sec|csc|log|ln|exp)(?![A-Za-z])"
)

SINGLE_VAR_ASSIGN_RE = re.compile(
    r"^(?:\\mathbf\s*\{)?([A-Za-z])(?:\})?\s*=\s*(.+)$"
)

SUPERSCRIPT_MAP = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁺": "+",
    "⁻": "-",
    "⁽": "(",
    "⁾": ")",
}


@dataclass(frozen=True)
class GradeResult:
    """Structured outcome for a single model completion."""

    status: OutcomeStatus
    correct: bool
    extracted_correct_ignoring_truncation: bool
    has_boxed: bool
    extracted_answer: str | None
    truncated: bool
    think_closed: bool | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def split_thinking_response(text: str, *, think_in_prompt: bool = False) -> tuple[str, bool]:
    """Split Qwen3 `<think>...</think>` reasoning from the final response body.

    Returns `(final_body, think_closed)`. When `think_in_prompt=True` (the Qwen3 chat
    template appends `<think>` to the prompt), a missing `</think>` means the thinking
    block was never closed (`think_closed=False`).
    """

    if not text:
        return "", not think_in_prompt
    close_tag = "</think>"
    if close_tag in text:
        idx = text.rfind(close_tag)
        return text[idx + len(close_tag) :].strip(), True
    if think_in_prompt or "<think>" in text:
        return "", False
    return text, True


def get_last_boxed_with_span(text: str) -> tuple[str, int, int] | None:
    """Return `(content, content_start, content_end)` for the last `\\boxed{...}`."""

    if not text:
        return None
    boxed_start_idx = text.rfind(r"\boxed")
    if boxed_start_idx == -1:
        return None

    current_idx = boxed_start_idx + len(r"\boxed")
    while current_idx < len(text) and text[current_idx].isspace():
        current_idx += 1

    if current_idx >= len(text) or text[current_idx] != "{":
        return None

    current_idx += 1
    brace_depth = 1
    content_start_idx = current_idx

    while current_idx < len(text) and brace_depth > 0:
        char = text[current_idx]
        if char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        current_idx += 1

    if brace_depth != 0:
        return None

    content_end_idx = current_idx - 1
    return text[content_start_idx:content_end_idx], content_start_idx, content_end_idx


def get_last_boxed(text: str) -> str | None:
    """Return the inner content of the last balanced `\\boxed{...}` expression."""

    found = get_last_boxed_with_span(text)
    return found[0] if found is not None else None


def extract_final_candidate(text: str, fallback: str = "number_only") -> str:
    """Extract the final candidate expression from a completion.

    Prefers the last `\\boxed{...}` expression. When `fallback` is `'number_only'`,
    falls back to the last numeric token and returns `""` if none exists so unparseable
    prose can be classified as `invalid`.
    """

    if not text:
        return ""
    boxed = get_last_boxed(text.strip())
    if boxed is not None and boxed.strip():
        return boxed.strip().strip("$ ")

    if fallback in ("number_then_full", "number_only"):
        matches = RE_NUMBER.findall(text)
        if matches:
            return matches[-1]
        if fallback == "number_then_full":
            return text.strip()
    return ""


def _normalize_latex_fractions_and_roots(text: str) -> str:
    """Canonicalize LaTeX fraction/root shorthands and mixed numbers before SymPy parsing."""

    # Separate `\sqrt` from any immediately preceding alphanumeric/bracket token (e.g. `30p\sqrt{p}`).
    text = re.sub(r"(?<=[0-9A-Za-z\)\]\}])(?=\\sqrt(?![A-Za-z]))", " ", text)

    # Normalize `\sqrt` variants (`\sqrt{x}`, `\sqrt 2`, `\sqrt2`).
    for _ in range(4):
        text = re.sub(r"\\sqrt\s*\{([^{}]*)\}", lambda m: f"sqrt({m.group(1)})", text)
    text = re.sub(r"\\sqrt\s+([0-9A-Za-z]+)", lambda m: f"sqrt({m.group(1)})", text)
    text = re.sub(r"\\sqrt([0-9A-Za-z])", lambda m: f"sqrt({m.group(1)})", text)

    # Canonicalize shorthand `\frac` variants (`\frac12`, `\frac 32`, `\frac9{19}`, `\frac{50}3`).
    for _ in range(4):
        text = re.sub(
            r"\\frac\s*\{([^{}]+)\}\s*([0-9A-Za-z])(?![0-9A-Za-z{])",
            lambda m: f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}",
            text,
        )
        text = re.sub(
            r"\\frac\s*([0-9A-Za-z])\s*\{([^{}]+)\}",
            lambda m: f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}",
            text,
        )
        text = re.sub(
            r"\\frac\s*([0-9A-Za-z])\s*([0-9A-Za-z])",
            lambda m: f"\\frac{{{m.group(1)}}}{{{m.group(2)}}}",
            text,
        )

    # Convert mixed numbers like `6\frac{1}{5}` or `-3\frac{1}{2}` to `(6+(1)/(5))` / `(-(3+(1)/(2)))`.
    text = re.sub(
        r"(?<![A-Za-z0-9._^])(-?)(\d+)\s*\\frac\s*\{(\d+)\}\s*\{(\d+)\}",
        lambda m: (
            f"(-({m.group(2)}+({m.group(3)})/({m.group(4)})))"
            if m.group(1) == "-"
            else f"({m.group(2)}+({m.group(3)})/({m.group(4)}))"
        ),
        text,
    )

    # Expand nested `\frac{num}{den}` inner-to-outer.
    for _ in range(5):
        updated = re.sub(
            r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
            lambda m: f"({m.group(1)})/({m.group(2)})",
            text,
        )
        if updated == text:
            break
        text = updated

    text = re.sub(
        r"\\frac\s+([0-9A-Za-z]+)\s+([0-9A-Za-z]+)",
        lambda m: f"({m.group(1)})/({m.group(2)})",
        text,
    )
    return text


def _strip_unit_and_style_wrappers(text: str) -> str:
    """Remove styling macros and trailing alphabetic units without harming standalone words."""

    for _ in range(3):
        updated = STYLE_WRAPPER_RE.sub(lambda m: m.group(1), text)
        if updated == text:
            break
        text = updated

    match = re.match(r"^\\(?:text|mbox)\s*\{(?P<x>.+?)\}$", text)
    if match:
        text = match.group("x").strip()

    # Strip unit annotations like `\text{ days}` or `\mbox{ inches}^2` when math remains.
    stripped_units = UNIT_TEXT_RE.sub("", text).strip()
    if stripped_units and re.search(r"[0-9]", stripped_units):
        text = stripped_units

    # Unwrap any remaining `\text{...}` / `\mbox{...}` (e.g., `(\text{E})` -> `(E)`).
    text = re.sub(r"\\(?:text|mbox)\s*\{([^{}]*)\}", lambda m: m.group(1), text)
    return text


def _normalize_matrices(text: str) -> str:
    """Canonicalize LaTeX matrix environments to `[[row1];[row2]]` representation."""

    def _repl(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        rows = [r.strip() for r in re.split(r"\\\\", body) if r.strip()]
        norm_rows = []
        for row in rows:
            cols = [normalize_text(c.strip()) for c in row.split("&")]
            norm_rows.append("[" + ",".join(cols) + "]")
        return "matrix(" + ";".join(norm_rows) + ")"

    return MATRIX_RE.sub(_repl, text)


def _strip_plain_thousand_separators(text: str) -> str:
    """Remove ASCII thousand-separator commas (e.g. `1,234/2`, `1,234.56`) at bracket depth 0.

    Leaves commas inside parentheses, brackets, or braces (`(12,102)`, `(1,234)`,
    `[100,200]`, `{1,234}`) untouched so interval, tuple, and set elements are never merged.
    """

    if "," not in text:
        return text

    depths: list[int] = []
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
            depths.append(depth)
        elif ch in ")]}":
            depths.append(depth)
            depth = max(0, depth - 1)
        else:
            depths.append(depth)

    remove_indices: set[int] = set()
    for match in re.finditer(r"(?<![0-9.,])\d{1,3}(?:,\d{3})+(?=(?:\.\d+)?(?![0-9.,]))", text):
        start, end = match.span()
        if all(depths[idx] == 0 for idx in range(start, end)):
            for idx in range(start, end):
                if text[idx] == ",":
                    remove_indices.add(idx)

    if not remove_indices:
        return text
    return "".join(ch for idx, ch in enumerate(text) if idx not in remove_indices)


def normalize_text(text: str | None) -> str:
    """Normalize LaTeX/text answer representations before symbolic comparison."""

    if not text:
        return ""
    text = RE_SPECIAL.sub("", str(text)).strip()

    match = re.match(r"^[A-Za-z]\s*[.:]\s*(.+)$", text)
    if match:
        text = match.group(1)

    text = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", text)
    text = re.sub(r"\^\s*\\circ", "", text)
    text = text.replace("°", "")

    text = re.sub(r"\\\(|\\\)|\\\[|\\\]", "", text)

    # Parse matrix environments BEFORE LATEX_FIXES so `\\ ` row separators stay intact.
    if MATRIX_RE.search(text):
        text = _normalize_matrices(text)

    # Strip explicit LaTeX digit-grouping commas (`11,\! 111,\! 111,\! 100`, `14{,}916`) before `\,`/`\!` removal.
    text = re.sub(r"(?<=\d)(?:,(?:\\[,!;:]\s*)+|\{,\})(?=\d{3}(?!\d))", "", text)

    for pat, rep in LATEX_FIXES:
        text = re.sub(pat, rep, text)

    text = _strip_unit_and_style_wrappers(text)

    def convert_superscripts(s: str, base: str | None = None) -> str:
        converted = "".join(SUPERSCRIPT_MAP.get(ch, ch) for ch in s)
        if base is None:
            return converted
        return f"{base}**{converted}"

    text = re.sub(
        r"([0-9A-Za-z\)\]\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)",
        lambda m: convert_superscripts(m.group(2), base=m.group(1)),
        text,
    )
    text = convert_superscripts(text)

    text = text.replace("\\%", "%").replace("\\$", "").replace("$", "").replace("%", "")
    # Preserve escaped set braces as `{` / `}` until after `_strip_plain_thousand_separators`.
    text = text.replace(r"\{", "{").replace(r"\}", "}")
    text = _normalize_latex_fractions_and_roots(text)

    # Canonicalize common symbolic constants and functions for SymPy parsing.
    text = re.sub(r"\\infty\b", " oo ", text)
    text = re.sub(r"\\pi(?![A-Za-z])", " pi ", text)
    text = TRIG_LOG_RE.sub(lambda m: f" {m.group(1)} ", text)

    text = text.replace("^", "**")
    text = re.sub(r"(?<=\d)\s+(\d+/\d+)", lambda m: "+" + m.group(1), text)
    text = _strip_plain_thousand_separators(text)
    # Remove leading zeros on multi-digit integers (`049` -> `49`) while preserving `0.5`.
    text = re.sub(r"(?<![0-9.])0+(?=\d+(?!\d*\.))", "", text)

    # Strip spaces around operators slightly while keeping multiplication tokens clear.
    text = re.sub(r"\s+", " ", text)
    return text.replace("{", "").replace("}", "").strip().lower()


def sympy_parser(expr: str | None):
    """Parse normalized math text into a SymPy expression, returning None on failure."""

    if expr is None or not expr or len(expr) > 2000:
        return None
    try:
        return spp.parse_expr(
            expr,
            transformations=(
                *spp.standard_transformations,
                spp.implicit_multiplication_application,
            ),
            evaluate=True,
        )
    except (
        SympifyError,
        SyntaxError,
        TypeError,
        AttributeError,
        IndexError,
        TokenError,
        ValueError,
        PolynomialError,
        RecursionError,
    ):
        return None


def _parse_matrix_rows(norm_text: str) -> list[list[str]] | None:
    match = re.fullmatch(r"matrix\((.+)\)", norm_text.strip())
    if not match:
        return None
    rows = []
    for row_str in match.group(1).split(";"):
        row_str = row_str.strip()
        if not (row_str.startswith("[") and row_str.endswith("]")):
            return None
        rows.append([cell.strip() for cell in row_str[1:-1].split(",")])
    return rows


def _equations_equivalent(expr_gtruth: str, expr_pred: str) -> bool:
    if "=" not in expr_gtruth or "=" not in expr_pred:
        return False
    gt_sides = [s.strip() for s in expr_gtruth.split("=")]
    pr_sides = [s.strip() for s in expr_pred.split("=")]
    if len(gt_sides) != 2 or len(pr_sides) != 2:
        return False
    gt_l, gt_r = sympy_parser(gt_sides[0]), sympy_parser(gt_sides[1])
    pr_l, pr_r = sympy_parser(pr_sides[0]), sympy_parser(pr_sides[1])
    if any(x is None for x in (gt_l, gt_r, pr_l, pr_r)):
        return False
    try:
        diff_g = simplify(gt_l - gt_r)
        diff_p = simplify(pr_l - pr_r)
        if diff_g == 0 or diff_p == 0:
            return diff_g == diff_p
        ratio = simplify(diff_g / diff_p)
        return ratio.is_Number and ratio != 0
    except (
        SympifyError,
        TypeError,
        ValueError,
        ZeroDivisionError,
        RecursionError,
        NotImplementedError,
    ):
        return False


def equality_check(expr_gtruth: str, expr_pred: str) -> bool:
    """Return True if two normalized answer expressions are identical or symbolically equal."""

    if expr_gtruth == expr_pred:
        return True
    gt_mat = _parse_matrix_rows(expr_gtruth)
    pr_mat = _parse_matrix_rows(expr_pred)
    if gt_mat is not None and pr_mat is not None:
        if len(gt_mat) != len(pr_mat):
            return False
        return all(
            len(rg) == len(rp) and all(equality_check(cg, cp) for cg, cp in zip(rg, rp))
            for rg, rp in zip(gt_mat, pr_mat)
        )
    if _equations_equivalent(expr_gtruth, expr_pred):
        return True
    gtruth = sympy_parser(expr_gtruth)
    pred = sympy_parser(expr_pred)
    if gtruth is not None and pred is not None:
        if gtruth == pred:
            return True
        try:
            return bool(simplify(gtruth - pred) == 0)
        except (
            SympifyError,
            TypeError,
            ValueError,
            ZeroDivisionError,
            RecursionError,
            NotImplementedError,
        ):
            pass
    return False


def _outer_brackets_wrap_entire(text: str) -> bool:
    if len(text) < 2 or text[0] not in "([" or text[-1] not in ")]":
        return False
    depth = 0
    for idx, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth == 0 and idx < len(text) - 1:
                return False
            if depth < 0:
                return False
    return depth == 0


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    parts.append(text[start:].strip())
    return parts


def _expand_pm(parts: list[str]) -> list[str]:
    expanded: list[str] = []
    for part in parts:
        if r"\pm" in part and part.count(r"\pm") == 1:
            lhs, rhs = part.split(r"\pm", 1)
            expanded.append(f"{lhs.strip()} + {rhs.strip()}".strip())
            expanded.append(f"{lhs.strip()} - {rhs.strip()}".strip())
        else:
            expanded.append(part)
    return expanded


def split_into_parts(text: str) -> list[str]:
    """Split union/tuple/list-like answers into ordered components."""

    if not text:
        return []
    if r"\cup" in text:
        union_parts = [p.strip() for p in text.split(r"\cup") if p.strip()]
        if len(union_parts) > 1:
            expanded: list[str] = []
            for part in union_parts:
                expanded.extend(split_into_parts(part))
            return _expand_pm(expanded)
    if _outer_brackets_wrap_entire(text) and "," in text[1:-1]:
        items = _split_top_level_commas(text[1:-1])
        if all(items):
            return _expand_pm(items)
    if "," in text:
        top_items = _split_top_level_commas(text)
        if len(top_items) > 1 and all(top_items):
            return _expand_pm(top_items)
    return _expand_pm([text])


def _strip_single_var_lhs(raw_text: str) -> tuple[str | None, str]:
    stripped = str(raw_text).strip()
    match = SINGLE_VAR_ASSIGN_RE.match(stripped)
    if match:
        return match.group(1).lower(), match.group(2).strip()
    return None, stripped


def _strip_var_membership(raw_text: str) -> tuple[str | None, str]:
    stripped = str(raw_text).strip()
    match = re.match(r"^(?:\\mathbf\s*\{)?([A-Za-z])(?:\})?\s*\\in\s*(.+)$", stripped)
    if match:
        return match.group(1).lower(), match.group(2).strip()
    return None, stripped


def _interval_bracket_signature(text: str) -> tuple[str, ...] | None:
    if r"\cup" in text:
        segs = [s.strip() for s in text.split(r"\cup") if s.strip()]
        sigs = [_interval_bracket_signature(s) for s in segs]
        return tuple(sig[0] for sig in sigs if sig) if all(sigs) else None
    if _outer_brackets_wrap_entire(text) and "," in text[1:-1]:
        return (text[0] + text[-1],)
    return None


def _grade_single(pred_text: str, gt_text: str) -> bool:
    gt_norm = normalize_text(gt_text)
    pr_norm = normalize_text(pred_text)
    gt_sig = _interval_bracket_signature(gt_norm)
    pr_sig = _interval_bracket_signature(pr_norm)
    if gt_sig is not None and pr_sig is not None and gt_sig != pr_sig:
        return False
    gt_parts = split_into_parts(gt_norm)
    pred_parts = split_into_parts(pr_norm)
    if not gt_parts or not pred_parts or len(gt_parts) != len(pred_parts):
        return False
    return all(equality_check(gt, pred) for gt, pred in zip(gt_parts, pred_parts))


def grade_answer(pred_text: str | None, gt_text: str | None) -> bool:
    """Grade predicted candidate text against reference answer text."""

    if pred_text is None or gt_text is None:
        return False
    if _grade_single(pred_text, gt_text):
        return True

    # Handle reference answers formatted with `\approx`, e.g. `\$\dfrac23 \approx \$0.67`.
    if r"\approx" in gt_text:
        for alt in gt_text.split(r"\approx"):
            if alt.strip() and grade_answer(pred_text, alt.strip()):
                return True

    # Handle set membership forms like `x \in [-2, 7]` vs `[-2, 7]`.
    gt_mem_var, gt_mem_rhs = _strip_var_membership(gt_text)
    pr_mem_var, pr_mem_rhs = _strip_var_membership(pred_text)
    if gt_mem_var is not None and pr_mem_var is None:
        if _grade_single(pred_text, gt_mem_rhs):
            return True
    elif pr_mem_var is not None and gt_mem_var is None:
        if _grade_single(pr_mem_rhs, gt_text):
            return True
    elif gt_mem_var is not None and pr_mem_var == gt_mem_var:
        if _grade_single(pr_mem_rhs, gt_mem_rhs):
            return True

    # Handle single-variable assignment forms like `k = \frac{12}{7}` vs `\frac{12}{7}`.
    gt_var, gt_rhs = _strip_single_var_lhs(gt_text)
    pr_var, pr_rhs = _strip_single_var_lhs(pred_text)
    if gt_var is not None and pr_var is None:
        return _grade_single(pred_text, gt_rhs)
    if pr_var is not None and gt_var is None:
        return _grade_single(pr_rhs, gt_text)
    if gt_var is not None and pr_var == gt_var:
        return _grade_single(pr_rhs, gt_rhs)

    return False


def classify_response(
    text: str,
    ground_truth: str,
    *,
    truncated: bool = False,
    strip_thinking: bool = False,
    think_closed: bool | None = None,
) -> GradeResult:
    """Classify a completion into `{correct, wrong, invalid, truncated}`.

    Truncated responses always receive `status="truncated"` and `correct=False` under
    the primary evaluation contract, while preserving `extracted_correct_ignoring_truncation`
    for diagnostic auditing.
    """

    eval_text = text or ""
    if strip_thinking and think_closed is None:
        eval_text, think_closed = split_thinking_response(eval_text, think_in_prompt=True)
    elif strip_thinking and think_closed is False:
        eval_text = ""

    has_boxed = get_last_boxed(eval_text) is not None
    candidate = extract_final_candidate(eval_text, fallback="number_only")
    extracted = candidate if candidate else None
    extracted_correct = bool(extracted is not None and grade_answer(extracted, ground_truth))

    if truncated or (strip_thinking and think_closed is False):
        return GradeResult(
            status="truncated",
            correct=False,
            extracted_correct_ignoring_truncation=extracted_correct,
            has_boxed=has_boxed,
            extracted_answer=extracted,
            truncated=True,
            think_closed=think_closed,
        )
    if extracted is None:
        return GradeResult(
            status="invalid",
            correct=False,
            extracted_correct_ignoring_truncation=False,
            has_boxed=has_boxed,
            extracted_answer=None,
            truncated=False,
            think_closed=think_closed,
        )
    if extracted_correct:
        return GradeResult(
            status="correct",
            correct=True,
            extracted_correct_ignoring_truncation=True,
            has_boxed=has_boxed,
            extracted_answer=extracted,
            truncated=False,
            think_closed=think_closed,
        )
    return GradeResult(
        status="wrong",
        correct=False,
        extracted_correct_ignoring_truncation=False,
        has_boxed=has_boxed,
        extracted_answer=extracted,
        truncated=False,
        think_closed=think_closed,
    )
