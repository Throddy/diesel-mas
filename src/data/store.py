"""Tiny persistence helper for the canonical interim/processed layer.

Frames are stored as Parquet: a versioned, language-neutral columnar format
that round-trips dtypes and the index reliably and does not execute code on
load (pickle does both badly - see K-20).  ``save_df``/``load_df`` accept any
suffix and normalise it to ``.parquet`` so call sites stay unchanged;
``load_df`` still reads a legacy ``.pkl`` twin when only that one exists, so an
interim directory produced by an older run keeps working until regenerated.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PARQUET = ".parquet"


def _canonical(path: Path) -> Path:
    return Path(path).with_suffix(PARQUET)


def save_df(df: pd.DataFrame, path: Path, csv_twin: bool = False) -> Path:
    """Write ``df`` to ``path`` as Parquet, optionally with an inspectable CSV twin."""
    target = _canonical(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    out.to_parquet(target, index=True)
    if csv_twin:
        df.to_csv(target.with_suffix(".csv"), index=True, encoding="utf-8")
    return target


def load_df(path: Path) -> pd.DataFrame:
    """Read a frame written by :func:`save_df`; index and dtypes are restored."""
    target = _canonical(path)
    if target.exists():
        return pd.read_parquet(target)
    legacy = Path(path).with_suffix(".pkl")
    if legacy.exists():
        return pd.read_pickle(legacy)
    raise FileNotFoundError(f"{target} not found (run scripts/prepare_data.py first)")
