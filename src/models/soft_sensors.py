"""Fitted soft sensors for laboratory properties of the hydrotreated product."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.process_features import build_trailing_features, selected_tags
from src.features.quality_features import available_feature
from src.models.quality import make_model, regression_metrics
from src.models.uncertainty import STATUS_OK, ConformalCalibration

STATUS_INSUFFICIENT = "INSUFFICIENT_HISTORY"
STATUS_NO_SKILL = "NO_SKILL"
PREVIOUS_FEATURE = "previous_value"


@dataclass
class SoftSensorTarget:
    """Declared target with the fields needed to judge whether it can be fitted."""

    metric: str
    process_unit: str
    sample_point: str
    unit: str
    limit: Optional[float]
    limit_source: Optional[str]

    @property
    def name(self) -> str:
        return f"{self.process_unit}|{self.sample_point}|{self.metric}"


@dataclass
class SoftSensor:
    """Fitted estimator with the history that builds its previous-value feature."""

    target: SoftSensorTarget
    pipeline: object
    feature_names: List[str]
    calibration: Optional[ConformalCalibration]
    status: str
    metrics: Dict[str, object]
    history: pd.Series = None

    def estimate(
        self,
        telemetry: pd.DataFrame,
        moment: pd.Timestamp,
        cfg,
        process: Optional[pd.DataFrame] = None,
    ) -> Dict[str, object]:
        """Estimate the property at ``moment`` from data available by then.

        ``process`` lets several sensors share one telemetry feature frame, which
        is the same for all of them at a given moment.
        """
        history = self.history if self.history is not None else pd.Series(dtype=float)
        moments = pd.DatetimeIndex([moment])
        if process is None:
            process = process_features(telemetry, moments, cfg)
        features = attach_previous_value(process, moments, history, cfg)
        return self.predict(features.iloc[0])

    def predict(self, features: pd.Series) -> Dict[str, object]:
        base = {
            "name": self.target.name,
            "metric": self.target.metric,
            "unit": self.target.unit,
            "sample_point": self.target.sample_point,
            "limit": self.target.limit,
            "limit_source": self.target.limit_source,
            "kind": "soft_sensor",
        }
        missing = set(self.feature_names) - set(features.index)
        if self.status != STATUS_OK or missing:
            return {
                **base,
                "value": None,
                "lower": None,
                "upper": None,
                "status": STATUS_INSUFFICIENT if missing else self.status,
            }
        row = features[self.feature_names].to_numpy(dtype=float).reshape(1, -1)
        value = float(self.pipeline.predict(row)[0])
        if self.calibration is None or not self.calibration.usable:
            return {
                **base,
                "value": value,
                "lower": None,
                "upper": None,
                "status": STATUS_INSUFFICIENT,
            }
        spread = self.calibration.quantile()
        return {
            **base,
            "value": value,
            "lower": value - spread,
            "upper": value + spread,
            "status": STATUS_OK,
        }


def declared_targets(cfg) -> List[SoftSensorTarget]:
    return [
        SoftSensorTarget(
            metric=row["metric"],
            process_unit=row["process_unit"],
            sample_point=str(row["sample_point"]),
            unit=row["unit"],
            limit=row.get("limit_c"),
            limit_source=row.get("limit_source"),
        )
        for row in cfg.main["soft_sensors"]["targets"]
    ]


def target_series(lims: pd.DataFrame, target: SoftSensorTarget) -> pd.Series:
    rows = lims[
        (lims.process_unit == target.process_unit)
        & (lims.sample_point.astype(str) == target.sample_point)
        & (lims.canonical_metric == target.metric)
    ]
    values = pd.to_numeric(rows.value, errors="coerce")
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(rows.timestamp))
    series = series[np.isfinite(series.to_numpy())].sort_index()
    return series[~series.index.duplicated(keep="last")]


def process_features(telemetry: pd.DataFrame, moments: pd.DatetimeIndex, cfg) -> pd.DataFrame:
    """Telemetry statistics at each moment, shared by every sensor."""
    windows = cfg.main["soft_sensors"]["trailing_windows_minutes"]
    process, _ = build_trailing_features(telemetry, selected_tags(telemetry), windows, moments)
    process.index = moments
    return process


def attach_previous_value(
    process: pd.DataFrame, moments: pd.DatetimeIndex, history: pd.Series, cfg
) -> pd.DataFrame:
    """Add the last laboratory value of this property reported by each moment."""
    delay = float(cfg.main["data_quality"]["lims_delay_minutes"])
    if len(history) and isinstance(history.index, pd.DatetimeIndex):
        previous = available_feature(moments, history, PREVIOUS_FEATURE, delay)
        column = previous[[PREVIOUS_FEATURE]].set_axis(moments)
    else:
        column = pd.DataFrame({PREVIOUS_FEATURE: np.nan}, index=moments)
    frame = pd.concat([process, column], axis=1)
    return frame.replace([np.inf, -np.inf], np.nan)


def build_features(
    telemetry: pd.DataFrame, moments: pd.DatetimeIndex, history: pd.Series, cfg
) -> pd.DataFrame:
    """Telemetry statistics at the moment plus the last reported laboratory value."""
    return attach_previous_value(process_features(telemetry, moments, cfg), moments, history, cfg)


def fit(telemetry: pd.DataFrame, lims: pd.DataFrame, target: SoftSensorTarget, cfg) -> SoftSensor:
    """Fit on the training block, calibrate on validation, score on the later block.

    Blocks are separated by the moment the laboratory result becomes available,
    and the model is compared with the previous reported value of the same
    property on exactly the rows where that value exists.
    """
    history = target_series(lims, target)
    minimum = int(cfg.main["soft_sensors"]["min_observations"])
    empty = SoftSensor(
        target, None, [], None, STATUS_INSUFFICIENT, {"n_observations": int(len(history))}, history
    )
    if len(history) < minimum:
        return empty

    moments = pd.DatetimeIndex(history.index)
    features = build_features(telemetry, moments, history, cfg)
    usable = features.notna().any(axis=1).to_numpy()
    features, history = features[usable], history[usable]
    columns = features.columns[features.notna().any()].tolist()
    if len(history) < minimum:
        return empty

    delay = pd.Timedelta(minutes=float(cfg.main["data_quality"]["lims_delay_minutes"]))
    available = pd.DatetimeIndex(history.index) + delay
    split = cfg.main["split"]
    train_end = pd.Timestamp(split["train_end"])
    valid_end = pd.Timestamp(split["valid_end"])
    train = np.asarray(available <= train_end)
    calibration = np.asarray((available > train_end) & (available <= valid_end))
    evaluation = np.asarray(available > valid_end)
    if train.sum() < minimum // 2 or calibration.sum() < 10 or evaluation.sum() < 10:
        return empty

    pipeline = make_model("rf", cfg.seed)
    pipeline.fit(
        features[columns][train].to_numpy(dtype=float), history[train].to_numpy(dtype=float)
    )

    residuals = history[calibration].to_numpy(dtype=float) - pipeline.predict(
        features[columns][calibration].to_numpy(dtype=float)
    )
    conformal = ConformalCalibration(
        residuals=residuals,
        alpha=float(cfg.main["quality"]["conformal_alpha"]),
        timestamps=np.asarray(available[calibration].values),
        minimum=int(cfg.main["quality"].get("min_calibration_residuals", 10)),
    )

    predicted = pipeline.predict(features[columns][evaluation].to_numpy(dtype=float))
    truth = history[evaluation].to_numpy(dtype=float)
    baseline = features[PREVIOUS_FEATURE][evaluation].to_numpy(dtype=float)
    compared = np.isfinite(baseline) & np.isfinite(predicted)
    metrics: Dict[str, object] = {
        "n_observations": int(len(history)),
        "n_train": int(train.sum()),
        "n_calibration": int(calibration.sum()),
        "n_evaluation": int(evaluation.sum()),
        "n_compared": int(compared.sum()),
    }
    if not compared.any():
        metrics["reason"] = "нет строк с доступной базовой оценкой"
        return SoftSensor(target, pipeline, columns, conformal, STATUS_NO_SKILL, metrics, history)

    truth_c, predicted_c, baseline_c = truth[compared], predicted[compared], baseline[compared]
    metrics.update(regression_metrics(truth_c, predicted_c))
    metrics["bias"] = float(np.mean(predicted_c - truth_c))
    metrics["baseline_previous_value_MAE"] = float(np.mean(np.abs(truth_c - baseline_c)))
    metrics["baseline_previous_value_MedianAE"] = float(np.median(np.abs(truth_c - baseline_c)))
    metrics["baseline_previous_value_RMSE"] = float(np.sqrt(np.mean((truth_c - baseline_c) ** 2)))
    metrics["baseline_previous_value_bias"] = float(np.mean(baseline_c - truth_c))
    if np.var(truth_c) < 1e-12:
        metrics["R2"] = None

    spread = conformal.quantile() if conformal.usable else float("nan")
    if np.isfinite(spread):
        lower, upper = predicted_c - spread, predicted_c + spread
        metrics["interval_half_width"] = float(spread)
        metrics["interval_mean_width"] = float(2 * spread)
        metrics["interval_coverage"] = float(np.mean((truth_c >= lower) & (truth_c <= upper)))
    else:
        metrics["interval_half_width"] = None
        metrics["interval_mean_width"] = None
        metrics["interval_coverage"] = None

    has_skill = metrics["MAE"] < metrics["baseline_previous_value_MAE"]
    metrics["beats_previous_value"] = bool(has_skill)
    status = STATUS_OK if has_skill and conformal.usable else STATUS_NO_SKILL
    return SoftSensor(target, pipeline, columns, conformal, status, metrics, history)


def save(sensors: List[SoftSensor], path: Path) -> Path:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(sensors, path)
    return path


def load(path: Path) -> List[SoftSensor]:
    import joblib

    path = Path(path)
    return joblib.load(path) if path.is_file() else []
