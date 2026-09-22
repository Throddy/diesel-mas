"""Rolling-origin evaluation of the horizon models against a baseline.

Each origin refits on the analyses reported before it, holds the most recent
ones out for calibration and predicts the targets of the following window, so
no figure on the evaluation period comes from a model calibrated on it.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import numpy as np
import pandas as pd

from src.config import load_config, project_root
from src.data.loaders import clean_telemetry, feed_sulfur, pak_sulfur, target_series
from src.data.store import load_df
from src.features.dataset import Dataset
from src.models.horizons import STATUS_NO_TARGET, build_horizon_dataset, horizon_hours
from src.models.quality import SulfurModel, calibrate_model, make_model
from src.models.uncertainty import STATUS_OK

BASELINE_FEATURE = "lims_sulfur_prev"
MIN_FIT_TARGETS = 260


def rolling_origin(dataset: Dataset, cfg, every_days: int) -> Dict[str, np.ndarray]:
    """Predict each window with a model fitted only on analyses reported earlier.

    Windows run over decision moments. A model serving decisions from
    ``decision_start`` is fitted on rows whose laboratory result was reported
    before that moment, so neither the target nor the calibration residual of a
    later analysis can enter it.
    """
    holdout = int(cfg.main["retraining"]["calibration_holdout"])
    evaluation_start = pd.Timestamp(cfg.main["split"]["test_start"])
    decision = pd.DatetimeIndex(dataset.meta["decision_time"])
    available = pd.DatetimeIndex(dataset.meta["result_available_time"])
    window = np.asarray(decision >= evaluation_start)
    if not window.any():
        return {}

    covered = decision[window]
    edges = pd.date_range(
        covered.min(), covered.max() + pd.Timedelta(days=every_days), freq=f"{every_days}D"
    )
    point = np.full(len(decision), np.nan)
    lower = np.full(len(decision), np.nan)
    upper = np.full(len(decision), np.nan)
    risk = np.full(len(decision), np.nan)
    fits = 0
    for decision_start, decision_end in zip(edges[:-1], edges[1:], strict=False):
        block = (
            window & np.asarray(decision >= decision_start) & np.asarray(decision < decision_end)
        )
        if not block.any():
            continue
        known = np.asarray(available <= decision_start)
        if known.sum() < MIN_FIT_TARGETS + holdout:
            continue
        fit_X = dataset.X[known].iloc[:-holdout]
        fit_y = dataset.y[known].iloc[:-holdout]
        calibration_X = dataset.X[known].iloc[-holdout:]
        calibration_y = dataset.y[known].iloc[-holdout:]
        model = SulfurModel("rf", 0, [], make_model("rf", cfg.seed), limit=cfg.sulfur_limit)
        model.fit(fit_X, fit_y)
        calibrate_model(model, calibration_X, calibration_y, cfg)
        block_X = dataset.X[block].set_axis(decision[block])
        predicted, low, high, exceedance = model.predict_with_interval(block_X)
        point[block] = predicted
        lower[block] = low
        upper[block] = high
        risk[block] = exceedance
        fits += 1
    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "risk": risk,
        "mask": window,
        "decision": decision,
        "fits": fits,
    }


def brier(probability: np.ndarray, outcome: np.ndarray) -> float:
    return float(np.mean((probability - outcome.astype(float)) ** 2))


def scores(
    truth: np.ndarray,
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    risk: np.ndarray,
    baseline: np.ndarray,
    limit: float,
) -> dict:
    """Score the model and the baseline on one common set of rows.

    A row enters the comparison only when the model produced a forecast and the
    baseline exists, so every figure below refers to the same ``n_compared``
    observations.
    """
    from src.models.quality import regression_metrics

    predicted = np.isfinite(point)
    compared = predicted & np.isfinite(baseline)
    out: Dict[str, object] = {
        "n_targets": int(len(truth)),
        "n_forecasts": int(predicted.sum()),
        "n_missing": int((~predicted).sum()),
        "n_compared": int(compared.sum()),
    }
    if not compared.any():
        out["status"] = STATUS_NO_TARGET
        return out

    truth_c, point_c, baseline_c = truth[compared], point[compared], baseline[compared]
    out.update(regression_metrics(truth_c, point_c))
    out["variance_of_truth"] = float(np.var(truth_c))
    if out["variance_of_truth"] < 1e-9:
        out["R2"] = None
    out["baseline_previous_lims_MAE"] = float(np.mean(np.abs(truth_c - baseline_c)))
    out["baseline_previous_lims_MedianAE"] = float(np.median(np.abs(truth_c - baseline_c)))
    out["baseline_previous_lims_RMSE"] = float(np.sqrt(np.mean((truth_c - baseline_c) ** 2)))

    interval = compared & np.isfinite(lower) & np.isfinite(upper)
    out["n_intervals"] = int(interval.sum())
    if interval.any():
        inside = (truth[interval] >= lower[interval]) & (truth[interval] <= upper[interval])
        out["interval_coverage"] = float(np.mean(inside))
        out["interval_mean_width_mgkg"] = float(np.mean(upper[interval] - lower[interval]))
    else:
        out["interval_coverage"] = None
        out["interval_mean_width_mgkg"] = None

    over = truth > limit
    scored = compared & np.isfinite(risk)
    out["n_scored_probability"] = int(scored.sum())
    out["n_positive"] = int(over[compared].sum())
    if scored.any():
        out["brier_probability_above_limit"] = brier(risk[scored], over[scored])
        out["brier_base_rate"] = brier(np.full(scored.sum(), over[scored].mean()), over[scored])
    else:
        out["brier_probability_above_limit"] = None
        out["brier_base_rate"] = None

    flagged = interval & (upper > limit)
    actual = over & interval
    true_positive = int(np.sum(flagged & actual))
    out["n_flagged"] = int(flagged.sum())
    out["recall_upper_bound"] = float(true_positive / actual.sum()) if actual.sum() else None
    out["precision_upper_bound"] = float(true_positive / flagged.sum()) if flagged.sum() else None
    out["status"] = STATUS_OK
    out["skill"] = skill_verdict(out)
    return out


def skill_verdict(row: dict) -> dict:
    """A horizon is usable only if it beats the baseline on both criteria."""
    baseline_mae = row.get("baseline_previous_lims_MAE")
    beats_baseline = (
        baseline_mae is not None and row.get("MAE") is not None and row["MAE"] < baseline_mae
    )
    model_brier = row.get("brier_probability_above_limit")
    base_brier = row.get("brier_base_rate")
    beats_base_rate = (
        model_brier is not None and base_brier is not None and model_brier < base_brier
    )
    return {
        "beats_baseline_mae": bool(beats_baseline),
        "beats_base_rate_brier": bool(beats_base_rate),
        "usable_for_decisions": bool(beats_baseline and beats_base_rate),
        "rule": "горизонт участвует в проверке действия, если его MAE ниже базовой "
        "оценки по предыдущему анализу и Brier ниже Brier базовой частоты",
    }


def main(every_days: int = 30) -> dict:
    cfg = load_config()
    interim = cfg.path("interim")
    telemetry = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims, pak = load_df(interim / "lims_long.pkl"), load_df(interim / "pak_long.pkl")
    target, feed = target_series(lims, cfg), feed_sulfur(lims)

    rows: List[dict] = []
    for hours in horizon_hours(cfg):
        dataset = build_horizon_dataset(
            telemetry, target, pak_sulfur(pak), {"feed_sulfur": feed}, hours, cfg
        )
        result = rolling_origin(dataset, cfg, every_days)
        if not result:
            rows.append({"horizon_hours": hours, "status": STATUS_NO_TARGET})
            continue
        mask = result["mask"]
        baseline = (
            dataset.X[BASELINE_FEATURE].to_numpy(float)[mask]
            if BASELINE_FEATURE in dataset.X.columns
            else np.full(int(mask.sum()), np.nan)
        )
        row = {
            "horizon_hours": hours,
            "n_refits": result["fits"],
            "decision_period": [
                str(result["decision"][mask].min()),
                str(result["decision"][mask].max()),
            ],
            **scores(
                dataset.y.to_numpy(float)[mask],
                result["point"][mask],
                result["lower"][mask],
                result["upper"][mask],
                result["risk"][mask],
                baseline,
                cfg.sulfur_limit,
            ),
        }
        rows.append(row)

    report = {
        "scheme": f"rolling-origin, переобучение каждые {every_days} сут",
        "limit_mgkg": cfg.sulfur_limit,
        "interval_level": 1.0 - float(cfg.main["quality"]["conformal_alpha"]),
        "baseline": "предыдущий лабораторный анализ, доступный на момент решения",
        "horizons": rows,
        "usable_horizons": [
            row["horizon_hours"] for row in rows if row.get("skill", {}).get("usable_for_decisions")
        ],
    }
    out = project_root() / cfg.main["paths"]["reports"] / "horizons_evaluation.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    header = (
        f"{'ч':>3s} {'сравн.':>7s} {'проп.':>6s} {'MAE':>7s} {'база':>7s} "
        f"{'MedAE':>7s} {'RMSE':>7s} {'R2':>6s} {'покр.':>6s} {'ширина':>7s} "
        f"{'Brier':>7s} {'база':>7s} {'полнота':>7s} {'точн.':>6s}"
    )
    print(header)
    for row in rows:
        if row.get("status") != STATUS_OK:
            print(f"{row['horizon_hours']:3.0f} {row.get('status', 'нет'):>9s}")
            continue
        print(
            f"{row['horizon_hours']:3.0f} {row['n_compared']:7d} {row['n_missing']:6d} "
            f"{row['MAE']:7.3f} {_num(row['baseline_previous_lims_MAE']):>7s} "
            f"{row['MedianAE']:7.3f} {row['RMSE']:7.3f} {_num(row['R2'], 3):>6s} "
            f"{_num(row['interval_coverage'], 3):>6s} "
            f"{_num(row['interval_mean_width_mgkg'], 2):>7s} "
            f"{_num(row['brier_probability_above_limit'], 3):>7s} "
            f"{_num(row['brier_base_rate'], 3):>7s} "
            f"{_num(row['recall_upper_bound'], 3):>7s} "
            f"{_num(row['precision_upper_bound'], 3):>6s}"
        )
    print(f"\n-> {out}")
    return report


def _num(value, digits: int = 3) -> str:
    return "нет" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="оценка моделей по горизонтам")
    parser.add_argument("--every-days", type=int, default=30)
    main(**vars(parser.parse_args()))
