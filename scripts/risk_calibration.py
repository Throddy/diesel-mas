"""Is the exceedance probability calibrated?  Everything downstream depends on it.

A decision rule that compares expected losses needs the probability to mean
what it says: among the moments where the system claims a 30 % chance of
breaching the limit, about 30 % should actually breach it.  This scores that
claim on the untouched test block with a reliability diagram and the Brier
score, against two references - always predicting the base rate, and the
threshold rule's implicit zero-or-one answer.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import load_config, project_root
from src.data.loaders import clean_telemetry
from src.data.store import load_df
from src.features.dataset import build_dataset_for_model
from src.models.quality import SulfurModel


def reliability_bins(
    probability: np.ndarray, outcome: np.ndarray, edges=(0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01)
) -> list[dict]:
    """Observed frequency against claimed probability, bin by bin."""
    rows = []
    for low, high in zip(edges[:-1], edges[1:], strict=False):
        mask = (probability >= low) & (probability < high)
        if not mask.any():
            rows.append({"from": low, "to": high, "n": 0})
            continue
        rows.append(
            {
                "from": float(low),
                "to": float(high),
                "n": int(mask.sum()),
                "mean_predicted": float(probability[mask].mean()),
                "observed_frequency": float(outcome[mask].mean()),
                "gap": float(probability[mask].mean() - outcome[mask].mean()),
            }
        )
    return rows


def expected_calibration_error(rows: list[dict], total: int) -> float:
    """Average gap between claim and reality, weighted by how often each occurs."""
    return float(sum(row["n"] / total * abs(row["gap"]) for row in rows if row["n"]))


def main() -> dict:
    cfg = load_config()
    interim = project_root() / cfg.main["paths"]["interim"]
    tel = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims, pak = load_df(interim / "lims_long.pkl"), load_df(interim / "pak_long.pkl")
    model = SulfurModel.load(project_root() / cfg.main["paths"]["models"] / "sulfur_model.joblib")

    dataset = build_dataset_for_model(tel, lims, pak, model, cfg)
    test = dataset.slice(start=pd.Timestamp(cfg.main["split"]["test_start"]))
    point, lower, upper, risk = model.predict_with_interval(test.X)
    truth = test.y.to_numpy(float)
    limit = cfg.sulfur_limit
    outcome = (truth > limit).astype(float)
    risk = np.asarray(risk, dtype=float)

    base_rate = float(outcome.mean())
    brier = float(np.mean((risk - outcome) ** 2))
    brier_base = float(np.mean((base_rate - outcome) ** 2))
    threshold_call = (upper > limit).astype(float)
    brier_threshold = float(np.mean((threshold_call - outcome) ** 2))

    rows = reliability_bins(risk, outcome)
    report = {
        "period": [str(test.y.index.min()), str(test.y.index.max())],
        "n": int(len(truth)),
        "limit_mgkg": limit,
        "base_rate": base_rate,
        "brier": {
            "model_probability": brier,
            "always_base_rate": brier_base,
            "threshold_rule_as_certainty": brier_threshold,
            "skill_against_base_rate": float(1 - brier / brier_base) if brier_base else None,
        },
        "mean_predicted": float(risk.mean()),
        "expected_calibration_error": expected_calibration_error(rows, len(truth)),
        "reliability": rows,
    }
    verdict = (report["brier"]["skill_against_base_rate"] or 0) > 0
    report["verdict"] = {
        "beats_base_rate": bool(verdict),
        "usable_for_expected_loss": bool(verdict and report["expected_calibration_error"] < 0.15),
        "rule": (
            "вероятность пригодна для сравнения ожидаемых потерь, если она обыгрывает "
            "базовую частоту по Brier и ошибка калибровки ниже 0,15"
        ),
    }
    out = project_root() / cfg.main["paths"]["reports"] / "risk_calibration.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "n",
                    "base_rate",
                    "mean_predicted",
                    "brier",
                    "expected_calibration_error",
                    "verdict",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\nдиаграмма калибровки:")
    for row in rows:
        if not row["n"]:
            continue
        print(
            f"  [{row['from']:.2f}; {row['to']:.2f}): n={row['n']:3d}  "
            f"заявлено {row['mean_predicted']:.3f}  наблюдалось {row['observed_frequency']:.3f}  "
            f"разрыв {row['gap']:+.3f}"
        )
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    argparse.ArgumentParser(description="калибровка вероятности превышения").parse_args()
    main()
