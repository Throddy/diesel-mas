"""Availability of data at the decision moment, across fitting and inference."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features.dataset import available_until, training_blocks
from src.models.quality import SulfurModel, calibrate_model, make_model
from src.models.uncertainty import (
    STATUS_INSUFFICIENT,
    STATUS_NO_TIMESTAMPS,
    STATUS_OK,
    ConformalCalibration,
)

CFG = load_config()


def calibration_over(start: str, periods: int = 20, window: int = 90) -> ConformalCalibration:
    stamps = pd.date_range(start, periods=periods, freq="D")
    return ConformalCalibration(
        residuals=np.linspace(0.5, 2.0, periods),
        alpha=0.1,
        timestamps=stamps.values,
        window=window,
        minimum=10,
    )


def test_as_of_returns_no_residual_from_the_future():
    """A query before every residual must select nothing at all."""
    calibration = calibration_over("2024-01-10")
    selected = calibration.as_of(pd.Timestamp("2024-01-01"))
    assert selected.n == 0
    assert selected.status == STATUS_INSUFFICIENT
    assert np.isnan(selected.quantile())
    assert np.isnan(selected.upper_quantile())


def test_as_of_reports_insufficient_history_instead_of_widening():
    """Too few past residuals give a status, not a fallback to the full set."""
    calibration = calibration_over("2024-01-10")
    selected = calibration.as_of(pd.Timestamp("2024-01-14"))
    assert 0 < selected.n < calibration.minimum
    assert selected.status == STATUS_INSUFFICIENT
    assert np.isnan(selected.quantile())


def test_as_of_uses_past_residuals_when_there_are_enough():
    calibration = calibration_over("2024-01-10")
    selected = calibration.as_of(pd.Timestamp("2024-01-30"))
    assert selected.n == 20
    assert selected.status == STATUS_OK
    assert np.isfinite(selected.quantile())


def test_exceedance_risk_is_undefined_without_calibration():
    calibration = calibration_over("2024-01-10").as_of(pd.Timestamp("2024-01-01"))
    assert np.isnan(calibration.exceedance_risk(np.array([np.log(9.0)]), 10.0)).all()


def test_calibration_residuals_are_stamped_by_result_availability():
    """A residual is known only after the laboratory reports it."""
    index = pd.date_range("2025-01-01", periods=40, freq="D")
    X = pd.DataFrame({"a": np.linspace(0, 1, 40)}, index=index)
    y = pd.Series(np.linspace(5, 9, 40), index=index)
    model = SulfurModel("ridge", 0, ["a"], make_model("ridge", CFG.seed), limit=CFG.sulfur_limit)
    model.fit(X, y)
    calibrate_model(model, X, y, CFG)
    delay = pd.Timedelta(minutes=float(CFG.main["data_quality"]["lims_delay_minutes"]))
    assert delay > pd.Timedelta(0)
    expected = (pd.DatetimeIndex(index) + delay).values
    assert np.array_equal(model.calibration.timestamps, expected)


def test_available_until_excludes_results_reported_after_the_cutoff():
    stamps = pd.date_range("2025-01-01", periods=5, freq="h")
    mask = available_until(stamps, pd.Timestamp("2025-01-01 03:00"), 120)
    assert mask.tolist() == [True, True, False, False, False]


@pytest.fixture(scope="module")
def dataset():
    from src.data.loaders import clean_telemetry, feed_sulfur, pak_sulfur, target_series
    from src.data.store import load_df
    from src.features.dataset import build_dataset

    interim = CFG.path("interim")
    if not (interim / "telemetry.parquet").exists():
        pytest.skip("canonical layer not built")
    telemetry = clean_telemetry(load_df(interim / "telemetry.pkl"), CFG)
    lims, pak = load_df(interim / "lims_long.pkl"), load_df(interim / "pak_long.pkl")
    return build_dataset(
        telemetry,
        target_series(lims, CFG),
        pak_sulfur(pak),
        {"feed_sulfur": feed_sulfur(lims)},
        cfg=CFG,
    )


def test_expanding_mode_grows_the_fitting_sample(dataset):
    fixed = training_blocks(dataset, CFG, "fixed")
    expanding = training_blocks(dataset, CFG, "expanding")
    assert len(expanding.train.y) > len(fixed.train.y)
    assert expanding.training_cutoff > fixed.training_cutoff


def test_training_blocks_do_not_overlap(dataset):
    for mode in ("fixed", "expanding"):
        blocks = training_blocks(dataset, CFG, mode)
        assert blocks.train.y.index.max() < blocks.selection.y.index.min()
        assert blocks.selection.y.index.max() < blocks.calibration.y.index.min()


def test_quality_agent_passes_the_decision_time_to_the_forecast():
    """The calibration must see the decision moment, not a positional index."""
    from src.pipeline import build_system

    interim = CFG.path("interim")
    if not (interim / "telemetry.parquet").exists():
        pytest.skip("canonical layer not built")
    system = build_system(CFG)
    agent = system.orchestrator.qa
    moment = pd.Timestamp(CFG.main["demo"]["stable"])
    features = agent.build_features(moment)

    seen = []
    original = agent.model.predict_with_interval

    def record(frame, alpha=None):
        seen.append(pd.DatetimeIndex(frame.index))
        return original(frame, alpha)

    agent.model.predict_with_interval = record
    try:
        agent.predict_features(features, moment)
    finally:
        agent.model.predict_with_interval = original
    assert seen and seen[0][0] == moment


def test_forecast_before_any_calibration_reports_no_interval():
    """A decision earlier than every residual gets a status, not a wide guess."""
    index = pd.date_range("2025-01-01", periods=40, freq="D")
    X = pd.DataFrame({"a": np.linspace(0, 1, 40)}, index=index)
    y = pd.Series(np.linspace(5, 9, 40), index=index)
    model = SulfurModel("ridge", 0, ["a"], make_model("ridge", CFG.seed), limit=CFG.sulfur_limit)
    model.fit(X, y)
    calibrate_model(model, X, y, CFG)
    early = pd.DataFrame({"a": [0.5]}, index=pd.DatetimeIndex(["2024-06-01"]))
    point, lower, upper, risk = model.predict_with_interval(early)
    assert np.isfinite(point[0])
    assert np.isnan(lower[0]) and np.isnan(upper[0]) and np.isnan(risk[0])
    assert model.interval_status(early) == [STATUS_INSUFFICIENT]


def test_as_of_filters_by_time_even_without_a_window():
    calibration = calibration_over("2024-01-10", window=0)
    assert calibration.window == 0
    assert calibration.as_of(pd.Timestamp("2024-01-01")).n == 0
    assert calibration.as_of(pd.Timestamp("2024-01-30")).n == 20


def test_as_of_keeps_timestamps_so_a_repeated_call_stays_safe():
    calibration = calibration_over("2024-01-10")
    once = calibration.as_of(pd.Timestamp("2024-01-30"))
    assert once.timestamps is not None and once.window == calibration.window
    twice = once.as_of(pd.Timestamp("2024-01-01"))
    assert twice.n == 0
    assert twice.status == STATUS_INSUFFICIENT


@pytest.mark.parametrize("available", [0, 1, 5, 9])
def test_fewer_residuals_than_the_minimum_give_no_quantile(available):
    calibration = calibration_over("2024-01-10")
    moment = pd.Timestamp("2024-01-10") + pd.Timedelta(days=available)
    selected = calibration.as_of(moment)
    assert selected.n == available
    assert selected.status == STATUS_INSUFFICIENT
    assert np.isnan(selected.quantile())
    assert np.isnan(selected.upper_quantile())


def test_calibration_without_timestamps_cannot_be_selected_as_of():
    calibration = ConformalCalibration(residuals=np.linspace(0.1, 1.0, 40), alpha=0.1)
    selected = calibration.as_of(pd.Timestamp("2024-01-01"))
    assert selected.n == 0
    assert selected.status == STATUS_NO_TIMESTAMPS
    assert np.isnan(selected.quantile())


def test_dataset_meta_carries_the_three_moments(dataset):
    for column in ("decision_time", "sample_time", "result_available_time"):
        assert column in dataset.meta.columns
    delay = pd.Timedelta(minutes=float(CFG.main["data_quality"]["lims_delay_minutes"]))
    sample = pd.DatetimeIndex(dataset.meta["sample_time"])
    available = pd.DatetimeIndex(dataset.meta["result_available_time"])
    assert (available == sample + delay).all()


def test_every_feature_timestamp_precedes_the_decision(dataset):
    decision = pd.DatetimeIndex(dataset.meta["decision_time"])
    delay = pd.Timedelta(minutes=float(CFG.main["data_quality"]["lims_delay_minutes"]))
    for column in dataset.meta.columns:
        if not column.endswith("_ts"):
            continue
        used = pd.DatetimeIndex(dataset.meta[column])
        known = used.notna()
        if not known.any():
            continue
        offset = delay if column.startswith("lims_") else pd.Timedelta(0)
        assert (used[known] + offset <= decision[known]).all(), column


def test_feature_timestamp_never_exceeds_the_decision(dataset):
    decision = pd.DatetimeIndex(dataset.meta["decision_time"])
    features = pd.DatetimeIndex(dataset.meta["feature_timestamp"])
    known = features.notna()
    assert (features[known] <= decision[known]).all()
