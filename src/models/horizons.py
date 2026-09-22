"""Forecast models for fixed horizons ahead of the decision moment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.dataset import Dataset, build_dataset, training_blocks
from src.models.quality import SulfurModel, calibrate_model, make_model
from src.models.uncertainty import STATUS_OK

STATUS_NO_TARGET = "NO_TARGET"
STATUS_INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"


def horizon_hours(cfg) -> List[float]:
    return [float(h) for h in cfg.main["horizons"]["hours"]]


def artifact_name(hours: float) -> str:
    return f"sulfur_model_h{int(hours)}.joblib"


@dataclass
class HorizonModel:
    """A sulfur model tied to one horizon and to the moments that trained it."""

    hours: float
    model: SulfurModel
    training_cutoff: str
    calibration_cutoff: str
    n_train: int
    n_calibration: int
    status: str

    @property
    def minutes(self) -> float:
        return self.hours * 60.0

    def save(self, directory: Path) -> Path:
        path = Path(directory) / artifact_name(self.hours)
        self.model.metrics = dict(self.model.metrics or {})
        self.model.metrics.update(
            {
                "horizon_hours": self.hours,
                "training_cutoff": self.training_cutoff,
                "calibration_cutoff": self.calibration_cutoff,
                "n_train": self.n_train,
                "n_calibration": self.n_calibration,
                "status": self.status,
            }
        )
        self.model.save(path)
        return path

    @staticmethod
    def load(directory: Path, hours: float) -> "HorizonModel":
        model = SulfurModel.load(Path(directory) / artifact_name(hours))
        metrics = model.metrics or {}
        return HorizonModel(
            hours=float(metrics.get("horizon_hours", hours)),
            model=model,
            training_cutoff=metrics.get("training_cutoff", ""),
            calibration_cutoff=metrics.get("calibration_cutoff", ""),
            n_train=int(metrics.get("n_train", 0)),
            n_calibration=int(metrics.get("n_calibration", 0)),
            status=metrics.get("status", STATUS_OK),
        )


STATUS_NO_SKILL = "NO_SKILL"


def usable_horizons(cfg) -> List[float]:
    """Horizons whose measured skill passes the published criterion."""
    path = Path(cfg.main["paths"]["reports"]) / "horizons_evaluation.json"
    if not path.is_absolute():
        from src.config import project_root

        path = project_root() / path
    if not path.is_file():
        return []
    import json

    report = json.loads(path.read_text(encoding="utf-8"))
    return [float(h) for h in report.get("usable_horizons", [])]


def decision_horizon(cfg) -> Dict[str, object]:
    """Horizon used to check an action, with the reason for the choice.

    The action takes effect after the transport delay, so the horizon covering
    that delay is preferred. A horizon without measured skill is not used, and
    the check stays on the nowcast upper bound.
    """
    delay_hours = float(cfg.main["horizons"]["action_delay_minutes"]) / 60.0
    covering = [h for h in sorted(horizon_hours(cfg)) if h >= delay_hours]
    usable = set(usable_horizons(cfg))
    for hours in covering:
        if hours in usable:
            return {
                "horizon_hours": hours,
                "source": "horizon",
                "reason": "горизонт покрывает запаздывание действия и прошёл " "проверку навыка",
            }
    return {
        "horizon_hours": 0.0,
        "source": "nowcast",
        "reason": "ни один горизонт, покрывающий запаздывание действия, не прошёл "
        "проверку навыка",
    }


def build_horizon_dataset(
    telemetry: pd.DataFrame,
    target: pd.Series,
    pak_sulfur: pd.Series,
    lims_other: Dict[str, pd.Series],
    hours: float,
    cfg,
) -> Dataset:
    return build_dataset(
        telemetry, target, pak_sulfur, lims_other, cfg=cfg, horizon_minutes=hours * 60.0
    )


def fit_horizon(dataset: Dataset, hours: float, cfg, kind: str = "rf") -> HorizonModel:
    """Fit and calibrate one horizon on disjoint chronological blocks."""
    minimum = int(cfg.main["horizons"]["min_training_targets"])
    if len(dataset.y) < minimum:
        empty = SulfurModel(
            kind,
            0,
            list(dataset.X.columns),
            make_model(kind, cfg.seed),
            limit=cfg.sulfur_limit,
            model_version=cfg.model_version,
        )
        return HorizonModel(hours, empty, "", "", len(dataset.y), 0, STATUS_INSUFFICIENT_HISTORY)

    blocks = training_blocks(dataset, cfg)
    columns = blocks.train.X.columns[blocks.train.X.notna().any()].tolist()
    model = SulfurModel(
        kind,
        0,
        [],
        make_model(kind, cfg.seed),
        limit=cfg.sulfur_limit,
        model_version=cfg.model_version,
    )
    model.fit(blocks.train.X[columns], blocks.train.y)
    calibrate_model(model, blocks.calibration.X[columns], blocks.calibration.y, cfg)
    status = STATUS_OK if model.calibration.usable else STATUS_INSUFFICIENT_HISTORY
    return HorizonModel(
        hours=hours,
        model=model,
        training_cutoff=str(blocks.training_cutoff),
        calibration_cutoff=str(blocks.calibration_cutoff),
        n_train=len(blocks.train.y),
        n_calibration=len(blocks.calibration.y),
        status=status,
    )


def forecast(
    horizon: HorizonModel, features: pd.Series, moment: pd.Timestamp, cfg
) -> Dict[str, object]:
    """Forecast for ``moment`` plus the horizon, with the status of its inputs."""
    level = 1.0 - float(cfg.main["quality"]["conformal_alpha"])
    target_time = pd.Timestamp(moment) + pd.Timedelta(hours=horizon.hours)
    base = {
        "horizon_hours": horizon.hours,
        "interval_level": level,
        "training_cutoff": horizon.training_cutoff or None,
        "calibration_cutoff": horizon.calibration_cutoff or None,
        "feature_as_of_time": pd.Timestamp(moment),
        "target_time": target_time,
        "model_version": horizon.model.model_version,
    }
    if horizon.status != STATUS_OK:
        return {
            **base,
            "prediction": None,
            "interval_lower": None,
            "interval_upper": None,
            "probability_above_limit": None,
            "data_status": horizon.status,
        }

    missing = set(horizon.model.feature_names) - set(features.index)
    if missing:
        return {
            **base,
            "prediction": None,
            "interval_lower": None,
            "interval_upper": None,
            "probability_above_limit": None,
            "data_status": STATUS_NO_TARGET,
        }

    frame = pd.DataFrame([features.to_dict()], index=pd.DatetimeIndex([moment]))
    frame = frame[horizon.model.feature_names]
    point, lower, upper, risk = horizon.model.predict_with_interval(frame)
    status = horizon.model.interval_status(frame)[0]
    interval_defined = np.isfinite(lower[0]) and np.isfinite(upper[0])
    return {
        **base,
        "prediction": float(point[0]),
        "interval_lower": float(lower[0]) if interval_defined else None,
        "interval_upper": float(upper[0]) if interval_defined else None,
        "probability_above_limit": float(risk[0]) if np.isfinite(risk[0]) else None,
        "data_status": status,
    }
