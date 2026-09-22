"""Virtual analyser (ВАК) formulas: parse -> normalise -> validate -> evaluate.

The sheet mixes two notations: some formulas use '*' and '.', others use 'x'
as the multiplication sign and ',' as the decimal separator.  Some reference
laboratory values ('LIMS:24-2000.Pipeline.D15').  One formula has unbalanced
parentheses; it is marked ambiguous and never evaluated by a guess.
"""

from __future__ import annotations

import ast
import json
import operator
import pathlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

TAG_RE = re.compile(r"\b([TPFWDLQ]\d{1,2})\b")
LIMS_RE = re.compile(r"LIMS:([A-Za-z0-9_\-\.%]+)")
DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")
X_MULT_RE = re.compile(r"(?<=[\d\)])\s*[xх]\s*(?=[\(\dTPFWDLQ])")

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Pow: operator.pow,
}


@dataclass
class VakFormula:
    raw_name: str
    block: str
    raw_formula: str
    normalised: str
    tags: Set[str] = field(default_factory=set)
    lims_refs: Set[str] = field(default_factory=set)
    status: str = "OK"
    issues: List[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        return self.status == "OK"


def normalise(expr: str) -> str:
    s = str(expr).strip()
    s = s.replace("\u00a0", " ").replace("–", "-").replace("−", "-")
    s = s.replace("×", "*").replace("LIMS.95%.T", "LIMS:24-2000.Pipeline.95%.T")
    s = s.replace("LIMS.D15", "LIMS:24-2000.Pipeline.D15")
    s = DECIMAL_COMMA_RE.sub(".", s)
    s = X_MULT_RE.sub("*", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _balanced(expr: str) -> bool:
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def parse_formula(raw_name: str, block: Optional[str], raw_formula: str) -> VakFormula:
    norm = normalise(raw_formula)
    issues: List[str] = []
    status = "OK"

    lims_refs = set(LIMS_RE.findall(norm))
    placeholder = {}
    for i, ref in enumerate(sorted(lims_refs)):
        key = f"LIMSREF{i}"
        placeholder[key] = ref
        norm = norm.replace(f"LIMS:{ref}", key)

    if not _balanced(norm):
        status = "AMBIGUOUS"
        issues.append(
            "unbalanced parentheses - the intended grouping cannot be "
            "recovered from the provided material; formula marked "
            "AMBIGUOUS and never evaluated"
        )
    tags = set(TAG_RE.findall(norm))
    if status == "OK":
        try:
            ast.parse(norm, mode="eval")
        except SyntaxError as exc:
            status = "UNPARSEABLE"
            issues.append(f"syntax error: {exc}")
    return VakFormula(
        raw_name=str(raw_name).strip(),
        block=str(block),
        raw_formula=str(raw_formula),
        normalised=norm,
        tags=tags,
        lims_refs={placeholder[k] for k in placeholder},
        status=status,
        issues=issues,
    )


def _eval_node(node: ast.AST, env: Dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval_node(node.left, env), _eval_node(node.right, env))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval_node(node.operand, env))
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise KeyError(node.id)
        return float(env[node.id])
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def evaluate(formula: VakFormula, env: Dict[str, float]) -> Optional[float]:
    """Evaluate a formula; returns None if it is not available or an input is missing."""
    if not formula.available:
        return None
    missing = [t for t in formula.tags if t not in env or env[t] is None or not np.isfinite(env[t])]
    if missing:
        return None
    if any(k.startswith("LIMSREF") for k in re.findall(r"\bLIMSREF\d+\b", formula.normalised)):
        needed = set(re.findall(r"\bLIMSREF\d+\b", formula.normalised))
        if not needed.issubset(env.keys()):
            return None
    try:
        value = _eval_node(ast.parse(formula.normalised, mode="eval"), env)
    except (KeyError, ZeroDivisionError, ValueError):
        return None
    return float(value) if np.isfinite(value) else None


VAK_CACHE = "vak_formulas.json"


def load_formulas() -> List[VakFormula]:
    """Load parsed soft-sensor formulas, preferring the canonical layer.

    At decision time the system must not depend on the organisers' workbooks
    being present: everything it needs is written into ``data/interim`` by
    ``scripts/prepare_data``.  Parsing the workbook is the fallback for the
    first run, before the canonical layer exists.
    """
    from src.config import load_config, project_root

    cfg = load_config()
    cached = project_root() / cfg.main["paths"]["interim"] / VAK_CACHE
    if cached.is_file():
        rows = json.loads(cached.read_text(encoding="utf-8"))
        return [
            VakFormula(
                raw_name=r["raw_name"],
                block=r["block"],
                raw_formula=r["raw_formula"],
                normalised=r["normalised"],
                tags=set(r["tags"]),
                lims_refs=set(r["lims_refs"]),
                status=r["status"],
                issues=list(r["issues"]),
            )
            for r in rows
        ]
    return parse_reference_formulas()


def parse_reference_formulas() -> List[VakFormula]:
    """Parse the organisers' formula workbook (R-MEET-13: the set may grow)."""
    from src.data.tag_reference import load_vak_reference

    ref = load_vak_reference()
    return [parse_formula(r.raw_name, r.block, r.raw_formula) for r in ref.itertuples()]


def save_formulas(formulas: List[VakFormula], path) -> None:
    """Write parsed formulas into the canonical layer as plain JSON."""
    rows = [
        {
            "raw_name": f.raw_name,
            "block": f.block,
            "raw_formula": f.raw_formula,
            "normalised": f.normalised,
            "tags": sorted(f.tags),
            "lims_refs": sorted(f.lims_refs),
            "status": f.status,
            "issues": f.issues,
        }
        for f in formulas
    ]
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def formula_env(
    telemetry_row: pd.Series, unit_prefix: str, lims_values: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """Build the evaluation environment from one telemetry row of a given unit."""
    env: Dict[str, float] = {}
    for col, val in telemetry_row.items():
        if not isinstance(col, str) or ":" not in col:
            continue
        prefix, tag = col.split(":", 1)
        if prefix == unit_prefix and np.isfinite(val):
            env[tag] = float(val)
    for i, (_, v) in enumerate(sorted((lims_values or {}).items())):
        env[f"LIMSREF{i}"] = v
    return env
