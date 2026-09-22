"""Product-sulfur forecasting model.

Design choices (all justified in docs/evaluation.md):
  * target is ``log(sulfur)`` - the distribution spans 0.1 .. 2120 mg/kg;
  * a small, reproducible tabular pipeline (median imputation +
    HistGradientBoosting / Ridge) rather than anything exotic;
  * selection strictly on the chronological validation block;
  * uncertainty through split-conformal calibration on the same block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.uncertainty import (
    STATUS_INSUFFICIENT,
    ConformalCalibration,
    coverage,
    mean_interval_width,
)

EPS = 1e-6


def make_model(kind: str, seed: int = 42):
    if kind == "hgb":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.06,
                        max_depth=None,
                        max_leaf_nodes=31,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if kind == "rf":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=250, min_samples_leaf=3, n_jobs=1, random_state=seed
                    ),
                ),
            ]
        )
    if kind == "ridge":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0, random_state=seed)),
            ]
        )
    raise ValueError(f"unknown model kind {kind!r}")


@dataclass
class SulfurModel:
    kind: str
    lag_minutes: int
    feature_names: List[str]
    pipeline: object
    calibration: Optional[ConformalCalibration] = None
    limit: float = 10.0
    model_version: str = "quality-sulfur-v1"
    metrics: Dict[str, object] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SulfurModel":
        self.feature_names = list(X.columns)
        self.pipeline.fit(
            X[self.feature_names].to_numpy(dtype=float), np.log(np.maximum(y.to_numpy(float), EPS))
        )
        return self

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing model features: {sorted(missing)}")
        Xa = X[self.feature_names]
        return self.pipeline.predict(Xa.to_numpy(dtype=float))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self.predict_log(X))

    def calibrate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        alpha: float = 0.1,
        window: int = 0,
        result_delay_minutes: float = 0.0,
        minimum: int = 10,
    ) -> None:
        """Store residuals stamped by the moment the laboratory result is known."""
        residuals = np.log(np.maximum(y.to_numpy(float), EPS)) - self.predict_log(X)
        available = pd.DatetimeIndex(y.index) + pd.Timedelta(minutes=float(result_delay_minutes))
        self.calibration = ConformalCalibration(
            residuals=np.asarray(residuals, dtype=float),
            alpha=alpha,
            timestamps=np.asarray(available.values),
            window=int(window),
            minimum=int(minimum),
        )

    def predict_with_interval(self, X: pd.DataFrame, alpha: Optional[float] = None):
        """Return point, lower bound, upper bound and exceedance risk.

        The index of ``X`` carries the decision moment of each row. Calibration
        is selected as of that moment through one code path, so replay, fitting,
        back-test and online inference obey the same availability rule. A row
        without enough available residuals gets NaN bounds and NaN risk.
        """
        p_log = self.predict_log(X)
        if self.calibration is None:
            nan = np.full_like(p_log, np.nan, dtype=float)
            return np.exp(p_log), nan, nan, nan
        lo_log, hi_log, risk = self._as_of_bounds(X, p_log, alpha)
        return np.exp(p_log), np.exp(lo_log), np.exp(hi_log), risk

    def interval_status(self, X: pd.DataFrame) -> List[str]:
        """Calibration status per row, so an unusable interval stays visible."""
        if self.calibration is None:
            return [STATUS_INSUFFICIENT] * len(X)
        return [self.calibration.as_of(moment).status for moment in X.index]

    def _as_of_bounds(self, X: pd.DataFrame, p_log: np.ndarray, alpha: Optional[float]):
        """Per-row bounds from the residuals reported before that row's moment."""
        lo = np.empty_like(p_log)
        hi = np.empty_like(p_log)
        risk = np.empty_like(p_log)
        for position, (moment, point) in enumerate(zip(X.index, p_log)):
            calibration = self.calibration.as_of(moment)
            one = np.asarray([point])
            low, high = calibration.interval_log(one, alpha)
            lo[position], hi[position] = low[0], high[0]
            risk[position] = calibration.exceedance_risk(one, self.limit)[0]
        return lo, hi, risk

    def save(self, path: Path) -> Path:
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "kind": self.kind,
                "lag_minutes": self.lag_minutes,
                "feature_names": self.feature_names,
                "pipeline": self.pipeline,
                "residuals": None if self.calibration is None else self.calibration.residuals,
                "alpha": None if self.calibration is None else self.calibration.alpha,
                "cal_timestamps": None if self.calibration is None else self.calibration.timestamps,
                "cal_window": 0 if self.calibration is None else self.calibration.window,
                "cal_minimum": 10 if self.calibration is None else self.calibration.minimum,
                "limit": self.limit,
                "model_version": self.model_version,
                "metrics": self.metrics,
            },
            path,
        )
        return path

    @staticmethod
    def load(path: Path) -> "SulfurModel":
        import joblib

        blob = joblib.load(Path(path))
        cal = None
        if blob["residuals"] is not None:
            cal = ConformalCalibration(
                residuals=blob["residuals"],
                alpha=blob["alpha"],
                timestamps=blob.get("cal_timestamps"),
                window=blob.get("cal_window", 0),
                minimum=blob.get("cal_minimum", 10),
            )
        return SulfurModel(
            kind=blob["kind"],
            lag_minutes=blob["lag_minutes"],
            feature_names=blob["feature_names"],
            pipeline=blob["pipeline"],
            calibration=cal,
            limit=blob["limit"],
            model_version=blob["model_version"],
            metrics=blob.get("metrics", {}),
        )


def calibrate_model(model: "SulfurModel", X: pd.DataFrame, y: pd.Series, cfg) -> None:
    """Calibrate with the project settings for delay, window and minimum size."""
    quality = cfg.main["quality"]
    model.calibrate(
        X,
        y,
        float(quality["conformal_alpha"]),
        window=int(quality.get("conformal_window", 0)),
        result_delay_minutes=float(cfg.main["data_quality"]["lims_delay_minutes"]),
        minimum=int(quality.get("min_calibration_residuals", 10)),
    )


def baseline_previous_lims(X: pd.DataFrame) -> np.ndarray:
    return X["lims_sulfur_prev"].to_numpy(dtype=float)


def baseline_pak(X: pd.DataFrame) -> np.ndarray:
    col = "pak_sulfur_healthy" if "pak_sulfur_healthy" in X.columns else "pak_sulfur"
    return X[col].to_numpy(dtype=float)


def baseline_train_median(X: pd.DataFrame, train_median: float) -> np.ndarray:
    return np.full(len(X), float(train_median))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return {"n": 0}
    yt, yp = np.asarray(y_true)[m], np.asarray(y_pred)[m]
    return {
        "n": int(m.sum()),
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(np.sqrt(np.mean((yt - yp) ** 2))),
        "MedianAE": float(median_absolute_error(yt, yp)),
        "R2": float(r2_score(yt, yp)) if len(yt) > 2 else float("nan"),
        "MAE_log": float(
            mean_absolute_error(np.log(np.maximum(yt, EPS)), np.log(np.maximum(yp, EPS)))
        ),
    }


def violation_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, limit: float = 10.0, score: Optional[np.ndarray] = None
) -> Dict[str, object]:
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = (np.asarray(y_true)[m] > limit).astype(int)
    yp = (np.asarray(y_pred)[m] > limit).astype(int)
    out: Dict[str, object] = {
        "n": int(m.sum()),
        "n_positive": int(yt.sum()),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "confusion": {
            "tn": int(((yt == 0) & (yp == 0)).sum()),
            "fp": int(((yt == 0) & (yp == 1)).sum()),
            "fn": int(((yt == 1) & (yp == 0)).sum()),
            "tp": int(((yt == 1) & (yp == 1)).sum()),
        },
    }
    if score is not None and yt.sum() > 0:
        from sklearn.metrics import average_precision_score

        sc = np.asarray(score)[m]
        if np.isfinite(sc).all():
            out["pr_auc"] = float(average_precision_score(yt, sc))
    return out


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(y_true) & np.isfinite(lower) & np.isfinite(upper)
    if m.sum() == 0:
        return {}
    return {
        "coverage": coverage(np.asarray(y_true)[m], np.asarray(lower)[m], np.asarray(upper)[m]),
        "mean_width": mean_interval_width(np.asarray(lower)[m], np.asarray(upper)[m]),
    }


def permutation_importance_df(
    model: SulfurModel,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 5,
    seed: int = 42,
    top: int = 25,
) -> pd.DataFrame:
    """Permutation importance in log space (deterministic given the seed)."""
    rng = np.random.default_rng(seed)
    base = mean_absolute_error(np.log(np.maximum(y.to_numpy(float), EPS)), model.predict_log(X))
    rows = []
    Xv = X.reindex(columns=model.feature_names).copy()
    for col in model.feature_names:
        if Xv[col].notna().sum() == 0:
            continue
        losses = []
        original = Xv[col].to_numpy().copy()
        for _ in range(n_repeats):
            Xv[col] = rng.permutation(original)
            losses.append(
                mean_absolute_error(
                    np.log(np.maximum(y.to_numpy(float), EPS)), model.predict_log(Xv)
                )
            )
        Xv[col] = original
        rows.append({"feature": col, "importance": float(np.mean(losses) - base)})
    df = pd.DataFrame(rows).sort_values("importance", ascending=False)
    return df.head(top).reset_index(drop=True)
