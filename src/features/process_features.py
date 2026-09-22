"""Trailing-only feature engineering.

Every feature is computed from a window that ends at or before the decision /
label timestamp.  Centred windows are never used.  The optional ``lag_minutes``
shifts the whole window further into the past, which is how the process
response lag between an actuator move and the laboratory result is modelled.

The implementation works one tag at a time and immediately samples the rolling
result at the requested timestamps, so the full rolling matrix is never
materialised (the telemetry grid has ~189 000 rows).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.config import load_config

STD_WINDOW = 180
EXTREMA_WINDOW = 720
DELTA_STEPS = 18
SLOPE_STEPS = 36


def selected_tags(tel: pd.DataFrame) -> List[str]:
    """Tags kept for modelling: the whole hydrotreater plus the AVT tags that
    are referenced by the verified controls or by the VAK formulas."""
    cfg = load_config()
    return [c for c in cfg.main["quality"].get("feature_tags", tel.columns) if c in tel.columns]


def feature_names_for(tags: Sequence[str], windows_min: Sequence[int]) -> List[str]:
    names: List[str] = []
    for tag in tags:
        names.append(f"{tag}|value")
        for w in windows_min:
            names.append(f"{tag}|mean{w}")
        names.append(f"{tag}|std{STD_WINDOW}")
        names.append(f"{tag}|min{EXTREMA_WINDOW}")
        names.append(f"{tag}|max{EXTREMA_WINDOW}")
        names.append(f"{tag}|delta180")
        names.append(f"{tag}|slope360")
    return names


def _positions(
    index: pd.DatetimeIndex, timestamps: pd.DatetimeIndex, lag_minutes: int
) -> np.ndarray:
    lookup = pd.DatetimeIndex(timestamps) - pd.Timedelta(minutes=int(lag_minutes))
    return index.searchsorted(lookup, side="right") - 1


def build_trailing_features(
    tel: pd.DataFrame,
    tags: Sequence[str],
    windows_min: Sequence[int],
    timestamps: pd.DatetimeIndex,
    lag_minutes: int = 0,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (features, feature_timestamp) sampled backward-only at ``timestamps``."""
    ts = pd.DatetimeIndex(timestamps)
    pos = _positions(tel.index, ts, lag_minutes)
    valid = pos >= 0
    safe_pos = np.where(valid, pos, 0)

    columns: Dict[str, np.ndarray] = {}

    def take(series: pd.Series, name: str) -> None:
        arr = np.asarray(series.to_numpy(), dtype="float32")
        vals = arr[safe_pos].copy()
        vals[~valid] = np.nan
        columns[name] = vals

    for tag in tags:
        s = tel[tag]
        take(s, f"{tag}|value")
        for w in windows_min:
            take(s.rolling(f"{int(w)}min", min_periods=2).mean(), f"{tag}|mean{w}")
        take(s.rolling(f"{STD_WINDOW}min", min_periods=2).std(), f"{tag}|std{STD_WINDOW}")
        take(s.rolling(f"{EXTREMA_WINDOW}min", min_periods=2).min(), f"{tag}|min{EXTREMA_WINDOW}")
        take(s.rolling(f"{EXTREMA_WINDOW}min", min_periods=2).max(), f"{tag}|max{EXTREMA_WINDOW}")
        take(s - s.shift(DELTA_STEPS), f"{tag}|delta180")
        take((s - s.shift(SLOPE_STEPS)) / float(SLOPE_STEPS), f"{tag}|slope360")

    feat_ts = pd.Series(pd.NaT, index=ts, dtype="datetime64[ns]")
    feat_ts.loc[valid] = tel.index[pos[valid]]
    suffixes = load_config().main["quality"].get("feature_suffixes")
    if suffixes:
        columns = {k: v for k, v in columns.items() if k.split("|")[-1] in suffixes}
    out = pd.DataFrame(columns, index=ts).astype("float32")
    return out, feat_ts


def online_features(
    tel: pd.DataFrame,
    tags: Sequence[str],
    windows_min: Sequence[int],
    t: pd.Timestamp,
    lag_minutes: int = 0,
    history_hours: int = 48,
) -> Tuple[pd.Series, Optional[pd.Timestamp]]:
    """Single-timestamp feature vector using only data up to ``t``."""
    t = pd.Timestamp(t)
    window = tel.loc[t - pd.Timedelta(hours=history_hours) : t]
    if window.empty:
        return pd.Series(dtype="float32"), None
    feats, feat_ts = build_trailing_features(
        window, tags, windows_min, pd.DatetimeIndex([t]), lag_minutes
    )
    return feats.iloc[0], (None if pd.isna(feat_ts.iloc[0]) else pd.Timestamp(feat_ts.iloc[0]))
