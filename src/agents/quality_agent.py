"""QualityAgent: current and forecast product quality with uncertainty."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import load_config
from src.features.controller_activity import activity_frame
from src.features.process_features import online_features, selected_tags
from src.features.quality_features import pak_health_series, quality_source_features
from src.models.horizons import HorizonModel, decision_horizon
from src.models.quality import SulfurModel
from src.models.uncertainty import STATUS_OK
from src.models.vak import evaluate, formula_env, load_formulas
from src.schemas import ProcessState, QualityAssessment


class QualityAgent:
    name = "QualityAgent"

    def __init__(
        self,
        telemetry: pd.DataFrame,
        lims_target: pd.Series,
        pak_sulfur: pd.Series,
        pak_d15: Optional[pd.Series],
        model: SulfurModel,
        cfg=None,
        importance: Optional[pd.DataFrame] = None,
    ):
        self.cfg = cfg or load_config()
        self.tel = telemetry
        self.lims = lims_target
        self.pak = pak_sulfur
        self.pak_d15 = pak_d15
        self.model = model
        self.tags = selected_tags(telemetry)
        self.windows = self.cfg.main["quality"]["trailing_windows_minutes"]
        self.formulas = load_formulas()
        self.importance = importance
        self._pak_health = pak_health_series(pak_sulfur, self.cfg)
        self.lims_other = {}
        self._catalyst = None
        self._horizons = None
        self._soft_sensors = None

    def build_features(self, t: pd.Timestamp) -> pd.Series:
        proc, _ = online_features(
            self.tel, self.tags, self.windows, t, lag_minutes=self.model.lag_minutes
        )
        qual = quality_source_features(
            pd.DatetimeIndex([t]),
            self.lims,
            self.pak,
            self.lims_other,
            cfg=self.cfg,
            pak_health=self._pak_health,
        )
        qual = qual.drop(columns=[c for c in qual.columns if c.endswith("_ts")])
        parts = [proc, qual.iloc[0]]
        for block in self._optional_blocks(t):
            parts.append(block.iloc[0])
        row = pd.concat(parts)
        missing = set(self.model.feature_names) - set(row.index)
        if missing:
            raise ValueError(f"Missing online features: {sorted(missing)}")
        return row[self.model.feature_names]

    def soft_sensor_estimates(self, t: pd.Timestamp) -> List[Dict[str, object]]:
        """Fitted laboratory-property estimates, each with its own quality status."""
        from src.models import soft_sensors

        if self._soft_sensors is None:
            self._soft_sensors = soft_sensors.load(self.cfg.path("models") / "soft_sensors.joblib")
        if not self._soft_sensors:
            return []
        process = soft_sensors.process_features(self.tel, pd.DatetimeIndex([t]), self.cfg)
        return [sensor.estimate(self.tel, t, self.cfg, process) for sensor in self._soft_sensors]

    def horizon_models(self) -> List[HorizonModel]:
        """Horizon models loaded once; a missing artifact leaves the list empty."""
        from src.models.horizons import HorizonModel, horizon_hours

        if self._horizons is None:
            directory = self.cfg.path("models")
            loaded = []
            for hours in horizon_hours(self.cfg):
                try:
                    loaded.append(HorizonModel.load(directory, hours))
                except FileNotFoundError:
                    continue
            self._horizons = loaded
        return self._horizons

    def horizon_forecasts(
        self, features: pd.Series, moment: pd.Timestamp
    ) -> List[Dict[str, object]]:
        from src.models.horizons import forecast

        return [forecast(horizon, features, moment, self.cfg) for horizon in self.horizon_models()]

    def _optional_blocks(self, t: pd.Timestamp) -> List[pd.DataFrame]:
        """The post-submission feature blocks, in the form the model expects.

        Kept behind the same flags as the offline builder so the two paths
        cannot drift apart: a block switched on for training is switched on
        here, and a model trained without it never sees it.
        """
        flags = self.cfg.main.get("features", {})
        moments = pd.DatetimeIndex([t])
        blocks: List[pd.DataFrame] = []
        if flags.get("controller_activity", False):
            blocks.append(activity_frame(self.tel, moments, lag_minutes=self.model.lag_minutes))
        if flags.get("catalyst_index", False):
            blocks.append(self._catalyst_feature(flags).at(moments))
        return blocks

    def _catalyst_feature(self, flags):
        from src.features.catalyst_index import CatalystIndexFeature

        if self._catalyst is None:
            self._catalyst = CatalystIndexFeature(
                self.tel,
                self.lims,
                self.lims_other.get("feed_sulfur", pd.Series(dtype=float)),
                self.cfg,
                int(flags.get("catalyst_index_window_days", 90)),
            )
        return self._catalyst

    def predict_features(self, features: pd.Series, moment: pd.Timestamp):
        """Predict at ``moment``; the index carries the decision time forward.

        Conformal calibration selects residuals known before the decision, so
        an integer index would hide that moment and silently widen the sample.
        """
        frame = pd.DataFrame([features.to_dict()], index=pd.DatetimeIndex([moment]))
        frame = frame[self.model.feature_names]
        point, lower, upper, risk = self.model.predict_with_interval(frame)
        status = self.model.interval_status(frame)[0]
        return (float(point[0]), float(lower[0]), float(upper[0]), float(risk[0]), status)

    def _lims_refs(self, t: pd.Timestamp) -> Dict[str, float]:
        """Laboratory values some soft-sensor formulas refer to, as of ``t``.

        Backward-only: a formula never sees an analysis taken after the
        decision moment, and the laboratory delay applies as everywhere else.
        """
        from src.data.alignment import asof_value

        delay = float(self.cfg.dq("lims_delay_minutes"))
        refs: Dict[str, float] = {}
        series = (self.lims_other or {}).get("product_t95")
        if series is not None and len(series):
            ready = series.loc[: t - pd.Timedelta(minutes=delay)]
            value, _, _ = asof_value(ready, t)
            if value is not None:
                refs["24-2000.Pipeline.95%.T"] = float(value)
        return refs

    def vak_predictions(self, t: pd.Timestamp) -> Dict[str, Optional[float]]:
        hist = self.tel.loc[:t]
        if hist.empty:
            return {}
        row = hist.iloc[-1]
        out: Dict[str, Optional[float]] = {}
        for f in self.formulas:
            prefix = "avt" if "AVT" in f.raw_name.upper() else "ht"
            if not f.available:
                out[f.raw_name] = None
                continue
            env = formula_env(row, prefix, self._lims_refs(t))
            out[f.raw_name] = evaluate(f, env)
        return out

    def run(self, state: ProcessState) -> QualityAssessment:
        cfg = self.cfg
        t = pd.Timestamp(state.timestamp)
        flags: List[str] = []
        features = self.build_features(t)
        point, lower, upper, risk, interval_status = self.predict_features(features, t)
        if interval_status != STATUS_OK:
            flags.append(f"INTERVAL_{interval_status}")

        lims = state.quality_sources["lims"]
        pak = state.quality_sources["pak"]
        if lims["value"] is not None and lims["health"] == "OK":
            nowcast_source = "MODEL_WITH_LIMS"
        elif pak["value"] is not None and pak["health"] == "OK":
            nowcast_source = "MODEL_WITH_PAK"
        else:
            nowcast_source = "MODEL"
            flags.append("NO_HEALTHY_MEASURED_SOURCE")

        vak = self.vak_predictions(t)
        ambiguous = [f.raw_name for f in self.formulas if not f.available]
        if ambiguous:
            flags.append(f"VAK_UNAVAILABLE:{len(ambiguous)}")

        conf_terms = {}
        ratio = lims.get("staleness_ratio")
        conf_terms["lims_freshness"] = (
            1.0
            if (ratio is not None and ratio <= 1.0)
            else (0.5 if ratio is not None and ratio <= 2.0 else 0.2)
        )
        conf_terms["pak_health"] = {"OK": 1.0, "SUSPECT": 0.4, "FAILED": 0.1, "MISSING": 0.3}.get(
            pak["health"], 0.3
        )
        width = (upper - lower) if np.isfinite(upper) and np.isfinite(lower) else np.nan
        rel_width = width / max(point, 1e-6) if np.isfinite(width) else np.nan
        conf_terms["model_uncertainty"] = (
            float(np.clip(1.0 - min(rel_width, 3.0) / 3.0, 0.05, 1.0))
            if np.isfinite(rel_width)
            else 0.2
        )
        dis = state.quality_sources.get("disagreement")
        conf_terms["source_agreement"] = 0.2 if (dis and dis["severe"]) else (1.0 if dis else 0.6)
        confidence = float(np.prod(list(conf_terms.values())) ** (1.0 / len(conf_terms)))

        top_factors = self._top_factors(features, t)
        horizons = self.horizon_forecasts(features, t)
        sensors = self.soft_sensor_estimates(t)

        other: Dict[str, object] = {
            "vak": vak,
            "vak_ambiguous": ambiguous,
            "confidence_terms": conf_terms,
            "interval_status": interval_status,
            "decision_horizon": decision_horizon(self.cfg),
            "soft_sensors": sensors,
            "estimate_kinds": {
                "measurement": "лабораторный анализ или показание анализатора",
                "vak_formula": "формула ВАК организаторов",
                "soft_sensor": "обученная модель показателя",
                "proxy": "расчётный прокси без измеряемого аналога",
            },
        }
        other["sparse_metrics_policy"] = (
            "CetaneNumber (42 observations) and PourPoint (33) are reported as "
            "last-known laboratory values with LOW confidence; no model is fitted"
        )
        return QualityAssessment(
            decision_timestamp=t,
            metric=cfg.main["quality"]["target_metric"],
            unit=cfg.main["quality"]["target_unit"],
            point_forecast=point,
            lower=lower,
            upper=upper,
            risk_exceed_limit=risk,
            limit=cfg.sulfur_limit,
            model_version=self.model.model_version,
            baseline_forecast=float(features.get("lims_sulfur_prev", np.nan)),
            vak_forecast=None,
            nowcast_source=nowcast_source,
            other_quality=other,
            horizons=horizons,
            top_risk_factors=top_factors,
            confidence=confidence,
            flags=flags,
        )

    def _top_factors(
        self, features: pd.Series, moment: pd.Timestamp, k: int = 5
    ) -> List[Dict[str, object]]:
        """Local sensitivity around the current point, ranked by global importance."""
        if self.importance is None or self.importance.empty:
            return []
        base_log = float(np.log(max(self.predict_features(features, moment)[0], 1e-9)))
        out = []
        for _, r in self.importance.head(12).iterrows():
            col = r["feature"]
            if col not in features.index or not np.isfinite(features.get(col, np.nan)):
                continue
            probe = features.copy()
            step = abs(float(features[col])) * 0.02 + 1e-6
            probe[col] = float(features[col]) + step
            new_log = float(np.log(max(self.predict_features(probe, moment)[0], 1e-9)))
            out.append(
                {
                    "feature": col,
                    "global_importance": float(r["importance"]),
                    "local_effect_per_+2pct": round(new_log - base_log, 5),
                    "direction": (
                        "не меняет прогноз"
                        if abs(new_log - base_log) < 1e-5
                        else ("повышает прогноз" if new_log > base_log else "снижает прогноз")
                    ),
                    "current_value": float(features[col]),
                }
            )
            if len(out) >= k:
                break
        return out
