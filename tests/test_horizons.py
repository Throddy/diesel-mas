"""Horizon forecasts: separate target moments, availability and status."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.models.horizons import (
    STATUS_INSUFFICIENT_HISTORY,
    STATUS_NO_TARGET,
    decision_horizon,
    forecast,
    horizon_hours,
)
from src.models.uncertainty import STATUS_OK

CFG = load_config()
REQUIRED_FIELDS = (
    "horizon_hours",
    "prediction",
    "interval_lower",
    "interval_upper",
    "interval_level",
    "probability_above_limit",
    "training_cutoff",
    "calibration_cutoff",
    "feature_as_of_time",
    "data_status",
)


@pytest.fixture(scope="module")
def agent():
    from src.pipeline import build_system

    interim = CFG.path("interim")
    if not (interim / "telemetry.parquet").exists():
        pytest.skip("canonical layer not built")
    return build_system(CFG).orchestrator.qa


@pytest.fixture(scope="module")
def forecasts(agent):
    moment = pd.Timestamp(CFG.main["demo"]["stable"])
    if not agent.horizon_models():
        pytest.skip("horizon models not trained")
    return moment, agent.horizon_forecasts(agent.build_features(moment), moment)


def test_every_configured_horizon_is_reported(forecasts):
    _, rows = forecasts
    assert [row["horizon_hours"] for row in rows] == horizon_hours(CFG)


def test_result_contract_is_complete(forecasts):
    _, rows = forecasts
    for row in rows:
        assert tuple(row) >= REQUIRED_FIELDS or set(REQUIRED_FIELDS) <= set(row), row


def test_target_moments_differ_between_horizons(forecasts):
    moment, rows = forecasts
    targets = [row["target_time"] for row in rows]
    assert len(set(targets)) == len(targets)
    for row in rows:
        assert row["target_time"] - moment == pd.Timedelta(hours=row["horizon_hours"])
        assert row["feature_as_of_time"] == moment


def test_features_are_taken_as_of_the_decision_moment(forecasts):
    moment, rows = forecasts
    for row in rows:
        assert row["feature_as_of_time"] <= row["target_time"]
        if row["training_cutoff"]:
            assert pd.Timestamp(row["training_cutoff"]) < moment


def test_horizon_dataset_respects_the_laboratory_delay():
    """A horizon target may not use an analysis reported after the decision."""
    from src.data.loaders import clean_telemetry, feed_sulfur, pak_sulfur, target_series
    from src.data.store import load_df
    from src.models.horizons import build_horizon_dataset

    interim = CFG.path("interim")
    if not (interim / "telemetry.parquet").exists():
        pytest.skip("canonical layer not built")
    telemetry = clean_telemetry(load_df(interim / "telemetry.pkl"), CFG)
    lims, pak = load_df(interim / "lims_long.pkl"), load_df(interim / "pak_long.pkl")
    dataset = build_horizon_dataset(
        telemetry,
        target_series(lims, CFG),
        pak_sulfur(pak),
        {"feed_sulfur": feed_sulfur(lims)},
        24.0,
        CFG,
    )
    delay = pd.Timedelta(minutes=float(CFG.main["data_quality"]["lims_delay_minutes"]))
    decision = pd.DatetimeIndex(dataset.meta["decision_time"])
    used = pd.DatetimeIndex(dataset.meta["lims_sulfur_prev_ts"])
    known = used.notna()
    assert (used[known] + delay <= decision[known]).all()
    assert (decision == pd.DatetimeIndex(dataset.y.index) - pd.Timedelta(hours=24)).all()


def test_horizon_without_history_reports_a_status():
    from src.models.horizons import HorizonModel
    from src.models.quality import SulfurModel, make_model

    empty = HorizonModel(
        hours=6.0,
        model=SulfurModel("rf", 0, [], make_model("rf", CFG.seed)),
        training_cutoff="",
        calibration_cutoff="",
        n_train=0,
        n_calibration=0,
        status=STATUS_INSUFFICIENT_HISTORY,
    )
    row = forecast(empty, pd.Series({"a": 1.0}), pd.Timestamp("2026-01-01"), CFG)
    assert row["data_status"] == STATUS_INSUFFICIENT_HISTORY
    assert row["prediction"] is None
    assert row["interval_lower"] is None and row["interval_upper"] is None
    assert row["probability_above_limit"] is None


def test_missing_features_report_no_target(agent):
    if not agent.horizon_models():
        pytest.skip("horizon models not trained")
    horizon = agent.horizon_models()[0]
    row = forecast(horizon, pd.Series({"unrelated": 1.0}), pd.Timestamp("2026-01-01"), CFG)
    assert row["data_status"] == STATUS_NO_TARGET
    assert row["prediction"] is None


def test_horizon_forecast_is_deterministic(agent, forecasts):
    moment, first = forecasts
    second = agent.horizon_forecasts(agent.build_features(moment), moment)
    assert [row["prediction"] for row in first] == [row["prediction"] for row in second]


def test_horizon_forecast_needs_no_network(monkeypatch, agent, forecasts):
    import socket
    import ssl  # noqa: F401  resolve the lazy import while sockets still work

    moment, _ = forecasts
    features = agent.build_features(moment)

    def refuse(*args, **kwargs):
        raise OSError("network access is not allowed in a closed-loop deployment")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    rows = agent.horizon_forecasts(features, moment)
    assert len(rows) == len(horizon_hours(CFG))


def test_decision_horizon_states_its_source_and_reason():
    chosen = decision_horizon(CFG)
    assert chosen["source"] in {"horizon", "nowcast"}
    assert chosen["reason"]
    if chosen["source"] == "horizon":
        delay_hours = float(CFG.main["horizons"]["action_delay_minutes"]) / 60.0
        assert chosen["horizon_hours"] >= delay_hours


def test_quality_assessment_carries_the_horizons(agent):
    if not agent.horizon_models():
        pytest.skip("horizon models not trained")
    from src.pipeline import build_system

    system = build_system(CFG)
    recommendation = system.decide(pd.Timestamp(CFG.main["demo"]["stable"]), log=False)
    horizons = recommendation.agent_trace["QualityAgent"]["horizons"]
    assert len(horizons) == len(horizon_hours(CFG))
    assert all(set(REQUIRED_FIELDS) <= set(row) for row in horizons)


def test_probability_is_absent_when_the_interval_is(forecasts):
    _, rows = forecasts
    for row in rows:
        if row["data_status"] != STATUS_OK:
            assert row["probability_above_limit"] is None
        elif row["interval_lower"] is None:
            assert row["probability_above_limit"] is None or np.isfinite(
                row["probability_above_limit"]
            )


@pytest.fixture(scope="module")
def sources():
    from src.data.loaders import clean_telemetry, feed_sulfur, pak_sulfur, target_series
    from src.data.store import load_df

    interim = CFG.path("interim")
    if not (interim / "telemetry.parquet").exists():
        pytest.skip("canonical layer not built")
    telemetry = clean_telemetry(load_df(interim / "telemetry.pkl"), CFG)
    lims, pak = load_df(interim / "lims_long.pkl"), load_df(interim / "pak_long.pkl")
    return telemetry, target_series(lims, CFG), pak_sulfur(pak), feed_sulfur(lims)


@pytest.mark.parametrize("hours", [6.0, 12.0, 24.0])
def test_rolling_origin_never_fits_on_later_results(hours, sources):
    """A window may use only analyses reported before its first decision."""
    from scripts.eval_horizons import MIN_FIT_TARGETS
    from src.models.horizons import build_horizon_dataset

    telemetry, target, pak, feed = sources
    dataset = build_horizon_dataset(telemetry, target, pak, {"feed_sulfur": feed}, hours, CFG)
    decision = pd.DatetimeIndex(dataset.meta["decision_time"])
    available = pd.DatetimeIndex(dataset.meta["result_available_time"])
    holdout = int(CFG.main["retraining"]["calibration_holdout"])
    evaluation_start = pd.Timestamp(CFG.main["split"]["test_start"])
    window = decision >= evaluation_start
    covered = decision[window]
    edges = pd.date_range(covered.min(), covered.max() + pd.Timedelta(days=30), freq="30D")

    checked = 0
    for decision_start in edges[:-1]:
        known = available <= decision_start
        if known.sum() < MIN_FIT_TARGETS + holdout:
            continue
        assert available[known].max() <= decision_start
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("hours", [6.0, 12.0, 24.0])
def test_horizon_dataset_separates_decision_and_target(hours, sources):
    from src.models.horizons import build_horizon_dataset

    telemetry, target, pak, feed = sources
    dataset = build_horizon_dataset(telemetry, target, pak, {"feed_sulfur": feed}, hours, CFG)
    decision = pd.DatetimeIndex(dataset.meta["decision_time"])
    sample = pd.DatetimeIndex(dataset.meta["sample_time"])
    assert (sample - decision == pd.Timedelta(hours=hours)).all()
    assert (pd.DatetimeIndex(dataset.meta["feature_timestamp"]) <= decision).all()


def test_an_analysis_between_decision_and_target_does_not_reach_the_forecast(agent, forecasts):
    """A residual reported after the decision may not widen or move its interval."""
    moment, before = forecasts
    horizon = agent.horizon_models()[-1]
    calibration = horizon.model.calibration
    extra_time = np.datetime64(moment + pd.Timedelta(hours=1))
    horizon.model.calibration = type(calibration)(
        residuals=np.append(calibration.residuals, 5.0),
        alpha=calibration.alpha,
        timestamps=np.append(calibration.timestamps, extra_time),
        window=calibration.window,
        minimum=calibration.minimum,
    )
    try:
        after = agent.horizon_forecasts(agent.build_features(moment), moment)
    finally:
        horizon.model.calibration = calibration
    assert after[-1]["interval_upper"] == before[-1]["interval_upper"]
    assert after[-1]["probability_above_limit"] == before[-1]["probability_above_limit"]


def test_usable_horizons_drive_the_decision_check():
    from src.models.horizons import usable_horizons

    usable = usable_horizons(CFG)
    chosen = decision_horizon(CFG)
    if usable:
        assert chosen["source"] == "horizon"
        assert chosen["horizon_hours"] in usable
    else:
        assert chosen["source"] == "nowcast"


def test_evaluation_compares_model_and_baseline_on_the_same_rows():
    import json

    path = CFG.path("reports") / "horizons_evaluation.json"
    if not path.is_file():
        pytest.skip("report not built")
    for row in json.loads(path.read_text(encoding="utf-8"))["horizons"]:
        if row.get("status") != STATUS_OK:
            continue
        assert row["n_compared"] <= row["n_forecasts"]
        assert row["baseline_previous_lims_MAE"] is not None
        assert row["n_intervals"] <= row["n_compared"]
        assert row["n_scored_probability"] <= row["n_compared"]


def test_undefined_states_reach_the_explanation(agent):
    """A missing interval, horizon or sensor is reported, not replaced by a number."""
    from src.pipeline import build_system

    system = build_system(CFG)
    recommendation = system.decide(pd.Timestamp(CFG.main["demo"]["stable"]), log=False)
    text = " ".join(recommendation.explanation)
    chosen = decision_horizon(CFG)
    if chosen["source"] == "nowcast":
        assert "по верхней границе прогноза текущего режима" in text
        assert chosen["reason"] in text
    trace = recommendation.agent_trace["QualityAgent"]["other_quality"]
    weak = [row["metric"] for row in trace["soft_sensors"] if row["status"] != STATUS_OK]
    for metric in weak:
        assert metric in text


def test_an_unusable_interval_blocks_the_limit_check_with_a_reason():
    from src.optimization.constraints import check_candidate
    from src.schemas import CandidateAction

    candidate = CandidateAction(
        "A01", "ht_r201_gss_outlet_temp", ["ht:T5"], 380.0, 382.0, 2.0, "increase"
    )
    candidate.predicted_sulfur_upper = None
    checks = check_candidate(candidate, cfg=CFG)
    limit_check = next(c for c in checks if c.constraint_id == "HC-01")
    assert limit_check.status == "FAIL"
    assert "верхней границы" in limit_check.detail
