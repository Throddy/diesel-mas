"""Telemetry ingestion for avt_tags.csv / 242000_tags.csv.

Nothing about the file layout is assumed: separator, encoding, decimal
separator and the time format are sniffed from the file itself.  'Unnamed:*'
columns are dropped only after they are proven to be row indices.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import load_config


@dataclass
class CsvDialect:
    encoding: str
    sep: str
    decimal: str
    has_header: bool = True
    notes: List[str] = field(default_factory=list)


def sniff_csv(path: Path, sample_bytes: int = 200_000) -> CsvDialect:
    raw = path.open("rb").read(sample_bytes)
    encoding = "utf-8"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        encoding = "cp1251"
        text = raw.decode("cp1251", errors="replace")
    head = text.splitlines()[:50]
    body = "\n".join(head)
    try:
        sep = csv.Sniffer().sniff(body, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = max([",", ";", "\t"], key=lambda s: body.count(s))
    decimal = "," if (sep == ";" and body.count(",") > 0) else "."
    notes = [f"encoding={encoding}", f"sep={sep!r}", f"decimal={decimal!r}"]
    return CsvDialect(encoding=encoding, sep=sep, decimal=decimal, notes=notes)


def _index_like(series: pd.Series) -> bool:
    """True if the column is a plain 0..n-1 technical index."""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    s = series.dropna()
    if s.empty:
        return False
    return bool(np.array_equal(s.to_numpy(), np.arange(len(series))[: len(s)]))


def load_telemetry(path: Path, prefix: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Load one telemetry CSV into a canonical wide frame indexed by timestamp.

    Column names become ``{prefix}:{tag}`` (e.g. ``ht:T5``, ``avt:T55``) so that
    identical short tags from the two units never collide.
    """
    cfg = load_config()
    dialect = sniff_csv(path)
    time_col = cfg.main["telemetry"]["time_column"]
    df = pd.read_csv(
        path, sep=dialect.sep, decimal=dialect.decimal, encoding=dialect.encoding, nrows=nrows
    )
    dropped = []
    for c in list(df.columns):
        if str(c).lower().startswith("unnamed"):
            if _index_like(df[c]):
                dropped.append(c)
                df = df.drop(columns=[c])
            else:
                df = df.rename(columns={c: f"{prefix}:{c}"})
    if time_col not in df.columns:
        raise ValueError(
            f"{path.name}: time key {time_col!r} not found; columns={list(df.columns)[:8]}"
        )
    ts = pd.to_datetime(df[time_col], errors="coerce")
    df = df.drop(columns=[time_col])
    df.columns = [f"{prefix}:{str(c).strip()}" for c in df.columns]
    df.insert(0, "timestamp", ts)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.set_index("timestamp")
    df = df.apply(pd.to_numeric, errors="coerce").astype("float32")
    df.attrs["dialect"] = dialect.notes
    df.attrs["dropped_index_columns"] = dropped
    return df


def grid_report(df: pd.DataFrame, expected_minutes: int = 10) -> Dict[str, object]:
    """Verify the 10-minute grid claim instead of trusting it."""
    idx = df.index
    deltas = pd.Series(idx).diff().dropna().dt.total_seconds() / 60.0
    counts = deltas.value_counts().sort_values(ascending=False)
    return {
        "n_rows": int(len(df)),
        "t_min": str(idx.min()),
        "t_max": str(idx.max()),
        "duplicate_timestamps": int(idx.duplicated().sum()),
        "monotonic": bool(idx.is_monotonic_increasing),
        "step_minutes_mode": float(counts.index[0]) if len(counts) else float("nan"),
        "share_expected_step": (
            float((deltas == expected_minutes).mean()) if len(deltas) else float("nan")
        ),
        "irregular_steps": {float(k): int(v) for k, v in counts.head(6).items()},
    }
