"""Operating-severity / reliability PROXY.

The package contains no equipment- or catalyst-failure labels, so nothing here
is a failure probability.  What is computed is a transparent severity proxy:
how far the current operating point sits from the historical operating
envelope learned on the TRAIN block only, plus an out-of-distribution score.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEVERITY_SIGNALS = [
    ("ht:T5", "Температура Р-201"),
    ("ht:P8", "Перепад давления Р-202"),
    ("ht:P3", "Давление С-201"),
    ("ht:F9", "Массовый расход сырья"),
]
DIRECTIONS = {"ht:T5": 1, "ht:P8": 1, "ht:P3": -1, "ht:F9": 1}


@dataclass
class ReliabilityModel:
    columns: List[str]
    q_low: Dict[str, float]
    q_high: Dict[str, float]
    median: Dict[str, float]
    scale: Dict[str, float]
    rate_scale: Dict[str, float]
    iforest: Optional[object] = None
    severity_warn: float = 0.5
    severity_high: float = 0.8
    ood_threshold: float = 0.0
    version: str = "reliability-proxy-v2"
    robust_z_max: float = 8.0

    @staticmethod
    def fit(train_tel: pd.DataFrame, cfg, seed: int = 42) -> "ReliabilityModel":
        cols = [c for c, _ in SEVERITY_SIGNALS if c in train_tel.columns]
        sub = train_tel[cols].astype(float)
        q_low = sub.quantile(0.01).to_dict()
        q_high = sub.quantile(0.99).to_dict()
        med = sub.median().to_dict()
        mad = (sub - sub.median()).abs().median() * 1.4826
        scale = {
            c: float(mad[c]) if np.isfinite(mad[c]) and mad[c] > 0 else float(sub[c].std() or 1.0)
            for c in cols
        }
        rate = sub.diff().abs()
        rate_scale = {c: float(rate[c].quantile(0.99)) or 1.0 for c in cols}
        iso = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "iso",
                    IsolationForest(
                        n_estimators=200,
                        contamination=float(cfg.main["reliability"]["ood_contamination"]),
                        random_state=seed,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        sample = sub.iloc[:: max(1, len(sub) // 60000)]
        iso.fit(sample.to_numpy(dtype=float))
        model = ReliabilityModel(
            columns=cols,
            q_low=q_low,
            q_high=q_high,
            median=med,
            scale=scale,
            rate_scale=rate_scale,
            iforest=iso,
        )
        sev = model.severity_frame(sample)["severity_index"]
        rc = cfg.main["reliability"]
        model.severity_warn = float(sev.quantile(float(rc["severity_warn_quantile"])))
        model.severity_high = float(sev.quantile(float(rc["severity_high_quantile"])))
        scores = iso.named_steps["iso"].score_samples(
            iso.named_steps["scale"].transform(
                iso.named_steps["impute"].transform(sample.to_numpy(dtype=float))
            )
        )
        model.robust_z_max = float(rc["robust_z_max"])
        model.ood_threshold = float(np.quantile(scores, float(rc["ood_contamination"])))
        return model

    def _components(self, row: pd.Series) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for c in self.columns:
            v = row.get(c, np.nan)
            if v is None or not np.isfinite(v):
                out[c] = np.nan
                continue
            out[c] = max(0.0, DIRECTIONS[c] * (float(v) - self.median[c])) / (self.scale[c] or 1.0)
        return out

    def severity_frame(self, tel: pd.DataFrame) -> pd.DataFrame:
        sub = tel[self.columns].astype(float)
        z = ((sub - pd.Series(self.median)) * pd.Series(DIRECTIONS)).clip(lower=0) / pd.Series(
            self.scale
        ).replace(0, np.nan)
        z = z.clip(upper=10.0)
        sev = 1.0 - np.exp(-z.mean(axis=1) / 2.0)
        return pd.DataFrame({"severity_index": sev.astype(float)}, index=tel.index)

    def assess(self, row: pd.Series) -> Dict[str, object]:
        comps = self._components(row)
        vals = [v for v in comps.values() if np.isfinite(v)]
        z_mean = float(np.mean(vals)) if vals else float("nan")
        severity = (
            float(1.0 - np.exp(-min(z_mean, 10.0) / 2.0)) if np.isfinite(z_mean) else float("nan")
        )
        envelope_violations = []
        for c in self.columns:
            v = row.get(c, np.nan)
            if v is None or not np.isfinite(v):
                continue
            if v < self.q_low[c] or v > self.q_high[c]:
                envelope_violations.append(c)
        ood_score = float("nan")
        if self.iforest is not None:
            x = np.array([[row.get(c, np.nan) for c in self.columns]], dtype=float)
            if True:
                ood_score = float(
                    self.iforest.named_steps["iso"].score_samples(
                        self.iforest.named_steps["scale"].transform(
                            self.iforest.named_steps["impute"].transform(x)
                        )
                    )[0]
                )
        cls = "NORMAL"
        if np.isfinite(severity):
            if severity >= self.severity_high:
                cls = "HIGH"
            elif severity >= self.severity_warn:
                cls = "ELEVATED"
        top = sorted([(c, v) for c, v in comps.items() if np.isfinite(v)], key=lambda kv: -kv[1])[
            :3
        ]
        reasons = dict(SEVERITY_SIGNALS)
        return {
            "severity_index": severity,
            "severity_class": cls,
            "ood_score": ood_score,
            "ood_flag": bool(
                not np.isfinite(ood_score)
                or ood_score < self.ood_threshold
                or any(
                    abs((float(row[c]) - self.median[c]) / self.scale[c]) > self.robust_z_max
                    for c in self.columns
                    if pd.notna(row.get(c))
                )
                or any(pd.isna(row.get(c)) for c in self.columns)
            ),
            "envelope_violations": envelope_violations,
            "top_factors": [
                {"signal": c, "robust_z": round(v, 3), "meaning": reasons.get(c, "")}
                for c, v in top
            ],
        }

    def save(self, path: Path) -> Path:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, Path(path))
        return Path(path)

    @staticmethod
    def load(path: Path) -> "ReliabilityModel":
        import joblib

        return joblib.load(Path(path))
