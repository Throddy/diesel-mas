"""Features describing how hard the controller is working.

The natural-experiment search found no held steps: the median range of reactor
temperature inside a 12-hour window is 9.2 C, because the controller keeps
adjusting.  That movement was treated as noise to be averaged away.  It need
not be: a controller works harder when the unit is having trouble holding
quality, and its effort is visible immediately, while the laboratory result is
six hours behind.

Each window yields four numbers per control: how far the value travelled, how
often it changed direction, the total distance covered, and the slope of the
level.  Range and slope describe where the regime is going; reversals and
travelled distance describe how much work it took to keep it there.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

CONTROLS = ("ht:T5", "ht:F9", "ht:P3")
WINDOWS_HOURS = (6, 12, 24, 72)


def _reversals(values: np.ndarray) -> float:
    """How many times the direction of change flipped inside the window."""
    if len(values) < 3:
        return 0.0
    signs = np.sign(np.diff(values))
    signs = signs[signs != 0]
    if len(signs) < 2:
        return 0.0
    return float(np.sum(signs[1:] != signs[:-1]))


def _slope_per_hour(values: np.ndarray, step_minutes: float) -> float:
    """Least-squares drift of the level, in units per hour."""
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values), dtype=float) * (step_minutes / 60.0)
    x = x - x.mean()
    denominator = float(np.sum(x**2))
    if denominator <= 0:
        return 0.0
    return float(np.sum(x * (values - values.mean())) / denominator)


def activity_at(
    telemetry: pd.DataFrame,
    moment: pd.Timestamp,
    controls: Sequence[str] = CONTROLS,
    windows_hours: Iterable[int] = WINDOWS_HOURS,
    step_minutes: float = 10.0,
) -> dict:
    """Controller activity in the windows ending at ``moment``.

    Backward only: the window closes at the moment of interest, so nothing
    after it can enter.
    """
    out: dict[str, float] = {}
    history = telemetry.loc[:moment]
    for hours in windows_hours:
        window = history.loc[moment - pd.Timedelta(hours=hours) : moment]
        for tag in controls:
            if tag not in window.columns:
                continue
            values = window[tag].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            prefix = f"{tag}|act{hours}h"
            if len(values) < 3:
                out[f"{prefix}_range"] = np.nan
                out[f"{prefix}_reversals"] = np.nan
                out[f"{prefix}_travel"] = np.nan
                out[f"{prefix}_slope"] = np.nan
                continue
            out[f"{prefix}_range"] = float(values.max() - values.min())
            out[f"{prefix}_reversals"] = _reversals(values)
            out[f"{prefix}_travel"] = float(np.sum(np.abs(np.diff(values))))
            out[f"{prefix}_slope"] = _slope_per_hour(values, step_minutes)
    return out


def activity_frame(
    telemetry: pd.DataFrame, moments: pd.DatetimeIndex, lag_minutes: float = 0.0, **kwargs
) -> pd.DataFrame:
    """Activity features for a series of decision moments.

    ``lag_minutes`` shifts the window back, matching the lag the rest of the
    feature set uses, so the block can be compared like for like.
    """
    shift = pd.Timedelta(minutes=float(lag_minutes))
    rows = [activity_at(telemetry, moment - shift, **kwargs) for moment in moments]
    return pd.DataFrame(rows, index=moments)
