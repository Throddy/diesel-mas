"""Fitted soft sensors: eligibility, availability and status reporting."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.models import soft_sensors
from src.models.soft_sensors import (
    PREVIOUS_FEATURE,
    STATUS_INSUFFICIENT,
    STATUS_NO_SKILL,
    SoftSensor,
    SoftSensorTarget,
)
from src.models.uncertainty import STATUS_OK

CFG = load_config()


@pytest.fixture(scope="module")
def fitted():
    sensors = soft_sensors.load(CFG.path("models") / "soft_sensors.joblib")
    if not sensors:
        pytest.skip("soft sensors not trained")
    return sensors


def test_every_declared_target_states_its_units_and_sample_point():
    for target in soft_sensors.declared_targets(CFG):
        assert target.metric and target.unit and target.sample_point
        assert target.process_unit
        if target.limit is not None:
            assert target.limit_source


def test_a_sensor_without_skill_returns_no_value(fitted):
    """A model that loses to the previous laboratory value produces no estimate."""
    weak = [s for s in fitted if s.status != STATUS_OK]
    if not weak:
        pytest.skip("every sensor has skill")
    row = weak[0].predict(pd.Series(dict.fromkeys(weak[0].feature_names, 0.0)))
    assert row["value"] is None
    assert row["status"] in {STATUS_NO_SKILL, STATUS_INSUFFICIENT}


def test_a_usable_sensor_reports_value_unit_and_interval(fitted):
    usable = [s for s in fitted if s.status == STATUS_OK]
    if not usable:
        pytest.skip("no usable sensor")
    sensor = usable[0]
    row = sensor.predict(pd.Series(dict.fromkeys(sensor.feature_names, 1.0)))
    assert np.isfinite(row["value"])
    assert row["lower"] < row["value"] < row["upper"]
    assert row["unit"] and row["kind"] == "soft_sensor"


def test_missing_features_are_reported_rather_than_imputed(fitted):
    sensor = fitted[0]
    row = sensor.predict(pd.Series({"unrelated": 1.0}))
    assert row["value"] is None
    assert row["status"] == STATUS_INSUFFICIENT


def test_previous_value_feature_respects_the_laboratory_delay():
    history = pd.Series(
        [1.0, 2.0, 3.0], index=pd.DatetimeIndex(["2025-01-01", "2025-01-02", "2025-01-03"])
    )
    telemetry = pd.DataFrame(
        {"ht:T5": np.linspace(350, 360, 600)},
        index=pd.date_range("2024-12-30", periods=600, freq="10min"),
    )
    moments = pd.DatetimeIndex(["2025-01-03 01:00"])
    frame = soft_sensors.build_features(telemetry, moments, history, CFG)
    delay = float(CFG.main["data_quality"]["lims_delay_minutes"])
    assert delay >= 60
    assert frame[PREVIOUS_FEATURE].iloc[0] == 2.0


def test_sensor_without_history_reports_insufficient():
    target = SoftSensorTarget("Metric", "Гидроочистка", "2", "°С", None, None)
    sensor = SoftSensor(
        target, None, [], None, STATUS_INSUFFICIENT, {"n_observations": 0}, pd.Series(dtype=float)
    )
    telemetry = pd.DataFrame({"ht:T5": [350.0]}, index=pd.DatetimeIndex(["2025-01-01"]))
    row = sensor.estimate(telemetry, pd.Timestamp("2025-01-01"), CFG)
    assert row["status"] == STATUS_INSUFFICIENT
    assert row["value"] is None


def test_report_lists_units_limits_and_status():
    import json

    path = CFG.path("reports") / "soft_sensors.json"
    if not path.is_file():
        pytest.skip("report not built")
    report = json.loads(path.read_text(encoding="utf-8"))
    for row in report["targets"]:
        assert row["unit"] and row["source"] == "ЛИМС"
        assert row["result_delay_minutes"] > 0
        assert row["limit_source"]
        assert row["status"] in {STATUS_OK, STATUS_NO_SKILL, STATUS_INSUFFICIENT}


def test_sensor_without_confirmed_limit_carries_no_limit():
    for target in soft_sensors.declared_targets(CFG):
        if target.limit is None:
            assert target.limit_source is None


def test_estimates_are_deterministic(fitted):
    telemetry_path = CFG.path("interim") / "telemetry.parquet"
    if not telemetry_path.exists():
        pytest.skip("canonical layer not built")
    from src.data.loaders import clean_telemetry
    from src.data.store import load_df

    telemetry = clean_telemetry(load_df(CFG.path("interim") / "telemetry.pkl"), CFG)
    moment = pd.Timestamp(CFG.main["demo"]["stable"])
    first = [s.estimate(telemetry, moment, CFG)["value"] for s in fitted]
    second = [s.estimate(telemetry, moment, CFG)["value"] for s in fitted]
    assert first == second


def test_model_and_baseline_use_the_same_rows(fitted):
    for sensor in fitted:
        metrics = sensor.metrics
        if "n_compared" not in metrics or metrics.get("MAE") is None:
            continue
        assert metrics["n_compared"] <= metrics["n_evaluation"]
        assert metrics["baseline_previous_value_MAE"] is not None
        assert metrics["beats_previous_value"] == (
            metrics["MAE"] < metrics["baseline_previous_value_MAE"]
        )


def test_blocks_are_split_by_result_availability(fitted):
    for sensor in fitted:
        counts = [sensor.metrics.get(key) for key in ("n_train", "n_calibration", "n_evaluation")]
        if any(value is None for value in counts):
            continue
        assert sum(counts) <= sensor.metrics["n_observations"]
        assert all(value > 0 for value in counts)


def test_calibration_residuals_carry_availability_times(fitted):
    for sensor in fitted:
        if sensor.calibration is None or sensor.calibration.n == 0:
            continue
        assert sensor.calibration.timestamps is not None
        assert len(sensor.calibration.timestamps) == sensor.calibration.n


def test_no_future_analysis_enters_the_previous_value_feature():
    history = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.DatetimeIndex(["2025-01-01 10:00", "2025-01-02 10:00", "2025-01-03 10:00"]),
    )
    telemetry = pd.DataFrame(
        {"ht:T5": np.linspace(350, 360, 600)},
        index=pd.date_range("2024-12-30", periods=600, freq="10min"),
    )
    delay = float(CFG.main["data_quality"]["lims_delay_minutes"])
    moment = pd.Timestamp("2025-01-03 10:00") + pd.Timedelta(minutes=delay - 1)
    frame = soft_sensors.build_features(telemetry, pd.DatetimeIndex([moment]), history, CFG)
    assert frame[PREVIOUS_FEATURE].iloc[0] == 2.0


def test_a_sensor_without_skill_is_absent_from_the_usable_list():
    import json

    path = CFG.path("reports") / "soft_sensors.json"
    if not path.is_file():
        pytest.skip("report not built")
    report = json.loads(path.read_text(encoding="utf-8"))
    for row in report["targets"]:
        if row["status"] != STATUS_OK:
            assert row["name"] not in report["usable"]


def test_t95_is_not_claimed_as_a_fitted_sensor_without_skill():
    import json

    path = CFG.path("reports") / "soft_sensors.json"
    if not path.is_file():
        pytest.skip("report not built")
    report = json.loads(path.read_text(encoding="utf-8"))
    row = next(r for r in report["targets"] if r["metric"] == "95%.T")
    if row["status"] == STATUS_OK:
        assert row["MAE"] < row["baseline_previous_value_MAE"]
    else:
        assert row["name"] not in report["usable"]


def test_every_target_reports_the_full_metric_set():
    import json

    path = CFG.path("reports") / "soft_sensors.json"
    if not path.is_file():
        pytest.skip("report not built")
    required = (
        "n_observations",
        "n_train",
        "n_calibration",
        "n_evaluation",
        "n_compared",
        "MAE",
        "MedianAE",
        "RMSE",
        "bias",
        "interval_coverage",
        "interval_mean_width",
        "status",
        "unit",
        "limit",
        "limit_source",
    )
    for row in json.loads(path.read_text(encoding="utf-8"))["targets"]:
        missing = [key for key in required if key not in row]
        assert not missing, (row["name"], missing)
