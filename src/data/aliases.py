"""Canonical alias registry (raw metric names -> canonical metric)."""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

from src.config import load_config


@lru_cache(maxsize=1)
def _alias_index() -> Dict[str, str]:
    idx: Dict[str, str] = {}
    for canonical, spec in load_config().aliases["metrics"].items():
        for alias in spec["aliases"]:
            idx[_norm(alias)] = canonical
        idx[_norm(canonical)] = canonical
    return idx


def _norm(name: str) -> str:
    return str(name).strip().lower().replace(" ", "")


def resolve_metric(raw_name: str) -> Optional[str]:
    """Resolve a raw metric name to its canonical form.

    Handles the '24-2000:Mg.Sulfur' vs '24-2000:Mg.Sulfur.Q' mismatch and
    strips the unit/point prefix when present.
    """
    if raw_name is None:
        return None
    key = _norm(raw_name)
    idx = _alias_index()
    if key in idx:
        return idx[key]
    if ":" in key:
        tail = key.split(":")[-1]
        if tail in idx:
            return idx[tail]
        tail2 = key.split(":", 1)[-1]
        if tail2 in idx:
            return idx[tail2]
    return None


def canonical_unit(canonical_metric: str) -> Optional[str]:
    spec = load_config().aliases["metrics"].get(canonical_metric)
    return spec["canonical_unit"] if spec else None
