"""Properties of the Kalman fusion module.

The module is not wired into the decision path: on the test block it scores
worse than the current forecast (reports/fusion_eval.json).  These tests fix
the behaviour it does have, so the comparison stays honest and a later attempt
starts from a known state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.models.nowcast_fusion import FusionParams, NowcastFusion, build_observations, nowcast_at

CFG = load_config()


@pytest.fixture
def analyser() -> pd.Series:
    """Two days of hourly analyser readings, slowly drifting upward."""
    index = pd.date_range("2026-01-01", periods=48, freq="1h")
    return pd.Series(np.linspace(9.0, 9.4, 48), index=index)


def test_delayed_analysis_lands_at_its_sampling_time(analyser):
    """A late result must give the same estimate as a timely one.

    The laboratory number refers to when the sample was taken.  If the filter
    applied it at the reporting moment instead, the two cases would diverge.
    """
    lab = pd.Series([7.5], index=[pd.Timestamp("2026-01-01 12:00")])
    t = pd.Timestamp("2026-01-02 00:00")
    on_time = nowcast_at(lab, analyser, t, lab_delay_minutes=0)
    delayed = nowcast_at(lab, analyser, t, lab_delay_minutes=120)
    assert on_time["sulfur"] == pytest.approx(delayed["sulfur"])
    assert on_time["drift"] == pytest.approx(delayed["drift"])


def test_analysis_not_yet_reported_is_not_used(analyser):
    """An analysis whose result has not come back yet cannot inform the estimate."""
    lab = pd.Series([7.5], index=[pd.Timestamp("2026-01-01 23:30")])
    t = pd.Timestamp("2026-01-02 00:00")
    observations = build_observations(lab, analyser, t, lab_delay_minutes=120)
    assert all(o["source"] != "LIMS" for o in observations)


def test_failed_analyser_is_ignored_and_uncertainty_grows(analyser):
    """A failed sensor contributes nothing, and not knowing is expressed as variance."""
    t = pd.Timestamp("2026-01-02 00:00")
    lab = pd.Series([8.0], index=[pd.Timestamp("2026-01-01 00:00")])
    healthy = nowcast_at(lab, analyser, t, 0, pak_health="OK")
    failed = nowcast_at(lab, analyser, t, 0, pak_health="FAILED")
    assert failed["variance"] > healthy["variance"]
    assert failed["sulfur"] == pytest.approx(8.0, rel=0.02)


def test_infinite_variance_leaves_the_state_untouched():
    fusion = NowcastFusion()
    state = fusion.initial(8.0, pd.Timestamp("2026-01-01"))
    after = fusion.update(state, 12.0, float("inf"), observes_drift=True)
    assert np.allclose(after.x, state.x)
    assert np.allclose(after.P, state.P)


def test_uncertainty_grows_with_time_since_the_last_measurement():
    fusion = NowcastFusion()
    state = fusion.initial(8.0, pd.Timestamp("2026-01-01"))
    near = fusion.predict(state, pd.Timestamp("2026-01-01 06:00"))
    far = fusion.predict(state, pd.Timestamp("2026-01-05 00:00"))
    assert far.P[0, 0] > near.P[0, 0] > state.P[0, 0]


def test_estimate_converges_to_the_laboratory_when_data_are_healthy():
    """Repeated lab results pull the estimate onto the laboratory value."""
    index = pd.date_range("2026-01-01", periods=20, freq="1D")
    lab = pd.Series([6.0] * 20, index=index)
    analyser = pd.Series(
        np.full(20 * 24, 9.0), index=pd.date_range("2026-01-01", periods=20 * 24, freq="1h")
    )
    estimate = nowcast_at(lab, analyser, index[-1], 0, lookback_hours=24 * 25)
    assert estimate["sulfur"] == pytest.approx(6.0, rel=0.05)
    assert estimate["drift"] == pytest.approx(np.log(9.0 / 6.0), rel=0.2)


def test_laboratory_outweighs_the_analyser_by_variance_not_by_a_rule():
    """Source priority follows from the noise parameters, not from an if."""
    params = FusionParams()
    assert params.r_lims < params.r_pak
    fusion = NowcastFusion(params)
    state = fusion.initial(8.0, pd.Timestamp("2026-01-01"))
    from_lab = fusion.update(state, 6.0, params.r_lims, observes_drift=False)
    from_pak = fusion.update(state, 6.0, params.r_pak, observes_drift=True)
    assert abs(from_lab.x[0] - np.log(6.0)) < abs(from_pak.x[0] - np.log(6.0))


def test_two_identical_runs_give_identical_estimates(analyser):
    lab = pd.Series([7.5], index=[pd.Timestamp("2026-01-01 12:00")])
    t = pd.Timestamp("2026-01-02 00:00")
    first = nowcast_at(lab, analyser, t, 120)
    second = nowcast_at(lab, analyser, t, 120)
    assert first == second


def test_parameters_come_from_config():
    params = FusionParams.from_config(CFG)
    assert params.r_lims == CFG.main["fusion"]["r_lims"]
    assert params.q_state == CFG.main["fusion"]["q_state"]
