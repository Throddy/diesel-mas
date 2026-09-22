"""The two feature blocks added after the first submission must not see ahead.

Controller activity reads a moving window of the levers, and the catalyst index
is built from laboratory analyses, so both are places where a future value can
slip into a feature.  Each test changes only the future and asserts the feature
does not move.
"""

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.catalyst_index import CatalystIndexFeature
from src.features.controller_activity import CONTROLS, WINDOWS_HOURS, activity_frame


def _telemetry(periods: int = 800) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq="10min")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {tag: 340.0 + np.cumsum(rng.normal(0, 0.1, periods)) for tag in CONTROLS}, index=idx
    )


def test_controller_activity_ignores_the_future():
    tel = _telemetry()
    moment = tel.index[600]
    before = activity_frame(tel, pd.DatetimeIndex([moment]))
    tampered = tel.copy()
    tampered.iloc[601:] += 50.0
    after = activity_frame(tampered, pd.DatetimeIndex([moment]))
    pd.testing.assert_frame_equal(before, after)


def test_controller_activity_covers_every_window_and_statistic():
    frame = activity_frame(_telemetry(), pd.DatetimeIndex([_telemetry().index[700]]))
    expected = {
        f"{tag}|act{h}h_{stat}"
        for tag in CONTROLS
        for h in WINDOWS_HOURS
        for stat in ("range", "reversals", "travel", "slope")
    }
    assert set(frame.columns) == expected


def test_controller_activity_sees_a_quiet_period_as_quiet():
    idx = pd.date_range("2024-01-01", periods=600, freq="10min")
    flat = pd.DataFrame(dict.fromkeys(CONTROLS, 340.0), index=idx)
    row = activity_frame(flat, pd.DatetimeIndex([idx[-1]])).iloc[0]
    for tag in CONTROLS:
        assert row[f"{tag}|act24h_range"] == 0.0
        assert row[f"{tag}|act24h_travel"] == 0.0
        assert row[f"{tag}|act24h_reversals"] == 0.0


def test_catalyst_index_waits_for_the_laboratory_delay():
    """The analysis a row is labelled with cannot appear in that row's feature."""
    cfg = load_config()
    delay = float(cfg.main["data_quality"]["lims_delay_minutes"])
    stamps = pd.date_range("2024-01-01", periods=40, freq="3D")
    feature = CatalystIndexFeature.__new__(CatalystIndexFeature)
    feature.name = "catalyst_index_median90d"
    feature.known = pd.Series(np.arange(40, dtype=float), index=stamps)
    feature.known.index = feature.known.index + pd.Timedelta(minutes=delay)

    moment = stamps[20]
    value = feature.at(pd.DatetimeIndex([moment])).iloc[0, 0]
    assert value == 19.0
