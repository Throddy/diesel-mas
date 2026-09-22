"""Signal-level data-quality primitives used by the DataQualityAgent.

They are deliberately generic (they work on any series) and every threshold is
injected from config, never hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class FlatlineSegment:
    start: pd.Timestamp
    end: pd.Timestamp
    value: float
    n_points: int

    @property
    def duration_hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0


def find_flatlines(
    series: pd.Series, min_points: int, tolerance: float = 1e-9
) -> List[FlatlineSegment]:
    """Exact (within tolerance) constant runs of at least ``min_points`` samples."""
    if series.empty:
        return []
    v = series.to_numpy(dtype=float)
    changed = np.abs(np.diff(v)) > tolerance
    group = np.concatenate([[0], np.cumsum(changed)])
    out: List[FlatlineSegment] = []
    df = pd.DataFrame({"g": group, "v": v}, index=series.index)
    for _g, sub in df.groupby("g"):
        if len(sub) >= min_points:
            out.append(
                FlatlineSegment(sub.index[0], sub.index[-1], float(sub["v"].iloc[0]), len(sub))
            )
    return out


def flatline_mask(series: pd.Series, min_points: int, tolerance: float = 1e-9) -> pd.Series:
    mask = pd.Series(False, index=series.index)
    for seg in find_flatlines(series, min_points, tolerance):
        mask.loc[seg.start : seg.end] = True
    return mask


def robust_z(values: pd.Series) -> pd.Series:
    med = values.median()
    mad = (values - med).abs().median()
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        std = values.std()
        scale = std if np.isfinite(std) and std > 0 else 1.0
    return (values - med) / scale


def stuck_sensor(series: pd.Series, window: int, min_std: float) -> bool:
    if len(series) < window:
        return False
    return bool(series.tail(window).std(ddof=0) < min_std)


def inter_arrival_stats(index: pd.DatetimeIndex) -> Dict[str, float]:
    if len(index) < 3:
        return {"median_min": float("nan"), "q95_min": float("nan"), "n": len(index)}
    d = pd.Series(index).diff().dropna().dt.total_seconds() / 60.0
    return {
        "median_min": float(d.median()),
        "q95_min": float(d.quantile(0.95)),
        "n": int(len(index)),
    }


def staleness_ratio(age_minutes: Optional[float], stats: Dict[str, float]) -> Optional[float]:
    """Age relative to the historical q95 inter-arrival time of this analysis."""
    if (
        age_minutes is None
        or not np.isfinite(stats.get("q95_min", np.nan))
        or stats["q95_min"] <= 0
    ):
        return None
    return float(age_minutes / stats["q95_min"])


def saturation_hypothesis(series: pd.Series, top_n: int = 5) -> Dict[str, float]:
    """Check whether a series piles up at its maximum (clipping candidate)."""
    if series.empty:
        return {}
    mx = float(series.max())
    near = float((series > mx * 0.999).mean())
    return {"max": mx, "share_within_0_1pct_of_max": near}


def gap_report(index: pd.DatetimeIndex, expected_minutes: int) -> Dict[str, object]:
    if len(index) < 2:
        return {"n_gaps": 0, "max_gap_min": 0.0}
    d = pd.Series(index).diff().dropna().dt.total_seconds() / 60.0
    gaps = d[d > expected_minutes * 1.5]
    return {
        "n_gaps": int(len(gaps)),
        "max_gap_min": float(gaps.max()) if len(gaps) else 0.0,
        "total_gap_hours": float(gaps.sum() / 60.0) if len(gaps) else 0.0,
    }


def causal_run_length(series: pd.Series, tolerance: float = 1e-9) -> pd.Series:
    """Length of the constant run ENDING at each sample.

    Strictly causal: the value at ``t`` depends only on samples <= t, so it can
    be used both as a model feature and as an online health signal.
    """
    if series.empty:
        return pd.Series(dtype=float)
    v = series.to_numpy(dtype=float)
    same = np.abs(np.diff(v)) <= tolerance
    run = np.ones(len(v), dtype=np.int64)
    for i in range(1, len(v)):
        if same[i - 1]:
            run[i] = run[i - 1] + 1
    return pd.Series(run, index=series.index)
