"""Backward-only (as-of) time alignment.

The single rule enforced here: at decision/label time ``t`` only measurements
with ``timestamp <= t`` may be used.  Every aligned value carries its age.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def asof_value(
    series: pd.Series, t: pd.Timestamp
) -> Tuple[Optional[float], Optional[pd.Timestamp], Optional[float]]:
    """Latest value at or before ``t``.

    Returns (value, measurement_timestamp, age_minutes).  Never looks forward.
    """
    if series is None or len(series) == 0:
        return None, None, None
    idx = series.index.searchsorted(pd.Timestamp(t), side="right") - 1
    if idx < 0:
        return None, None, None
    ts = series.index[idx]
    age = (pd.Timestamp(t) - ts).total_seconds() / 60.0
    return float(series.iloc[idx]), ts, float(age)


def asof_frame(values: pd.Series, at: pd.DatetimeIndex, name: str) -> pd.DataFrame:
    """Vectorised backward as-of join of ``values`` onto the timestamps ``at``."""
    left = pd.DataFrame({"timestamp": pd.DatetimeIndex(at)}).sort_values("timestamp")
    right = pd.DataFrame(
        {"timestamp": values.index, name: values.to_numpy(), f"{name}_ts": values.index}
    ).sort_values("timestamp")
    out = pd.merge_asof(left, right, on="timestamp", direction="backward")
    out[f"{name}_age_min"] = (out["timestamp"] - out[f"{name}_ts"]).dt.total_seconds() / 60.0
    return out.set_index("timestamp")


def trailing_window(df: pd.DataFrame, t: pd.Timestamp, minutes: int) -> pd.DataFrame:
    """Rows in (t - minutes, t]: strictly trailing, never centred."""
    t = pd.Timestamp(t)
    return df.loc[(df.index > t - pd.Timedelta(minutes=minutes)) & (df.index <= t)]


def assert_no_future(feature_timestamps: pd.Series, cutoff: pd.Series) -> None:
    """Hard guard used by the pipeline and by the tests."""
    bad = feature_timestamps > cutoff
    if bool(np.any(bad)):
        raise AssertionError(
            f"future leakage: {int(np.sum(bad))} feature timestamps are after the cutoff"
        )
