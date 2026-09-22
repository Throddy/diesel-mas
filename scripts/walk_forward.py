"""Compare one-off training against retraining as time moves forward.

The shipped model is fitted once on data up to February 2025 and then used
through August 2026, a gap of eighteen months during which the unit, the
catalyst and the feed all change.  This measures what periodic retraining
would buy, on the same test block and with the same split rules.

At every step the model sees only analyses that were already reported at that
moment: fitting, calibration and prediction all respect the laboratory delay,
so nothing from the future enters.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import load_config, project_root
from src.data.loaders import clean_telemetry
from src.data.store import load_df
from src.features.dataset import build_dataset_for_model, chronological_split
from src.models.quality import SulfurModel, calibrate_model, make_model, regression_metrics


def interval_scores(truth: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    return {
        "coverage": float(np.mean((truth >= low) & (truth <= high))),
        "mean_width": float(np.mean(high - low)),
    }


def violation_scores(truth: np.ndarray, bound: np.ndarray, limit: float) -> dict:
    """Detection of limit breaches by the upper bound."""
    actual = truth > limit
    flagged = bound > limit
    tp = int(np.sum(actual & flagged))
    fn = int(np.sum(actual & ~flagged))
    fp = int(np.sum(~actual & flagged))
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    return {"recall": recall, "precision": precision, "n_positive": int(actual.sum())}


def fixed_schedule(split, cfg) -> dict:
    """The shipped scheme: fit once on training, calibrate on validation."""
    model = SulfurModel("rf", 0, [], make_model("rf", cfg.seed), limit=cfg.sulfur_limit)
    model.fit(split["train"].X, split["train"].y)
    calibrate_model(model, split["valid"].X, split["valid"].y, cfg)
    test = split["test"]
    point, low, high, _ = model.predict_with_interval(test.X)
    return {
        "point": point,
        "low": low,
        "high": high,
        "truth": test.y.to_numpy(),
        "index": test.y.index,
        "n_fits": 1,
    }


def expanding_schedule(dataset, split, cfg, every_days: int, lab_delay_minutes: float) -> dict:
    """Refit every ``every_days`` on everything reported before that moment."""
    test = split["test"]
    delay = pd.Timedelta(minutes=float(lab_delay_minutes))
    stamps = test.y.index
    boundaries = pd.date_range(
        stamps.min(), stamps.max() + pd.Timedelta(days=every_days), freq=f"{every_days}D"
    )

    point = np.empty(len(stamps))
    low = np.empty(len(stamps))
    high = np.empty(len(stamps))
    fits = 0
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=False):
        window = (stamps >= start) & (stamps < end)
        if not window.any():
            continue
        known = dataset.X.index <= (start - delay)
        known_X, known_y = dataset.X[known], dataset.y[known]
        if len(known_y) < 260:
            continue
        holdout = min(
            len(known_y) // 4, max(90, int(cfg.main["quality"].get("conformal_window", 90)))
        )
        fit_X, fit_y = known_X.iloc[:-holdout], known_y.iloc[:-holdout]
        cal_X, cal_y = known_X.iloc[-holdout:], known_y.iloc[-holdout:]
        model = SulfurModel("rf", 0, [], make_model("rf", cfg.seed), limit=cfg.sulfur_limit)
        model.fit(fit_X, fit_y)
        calibrate_model(model, cal_X, cal_y, cfg)
        p, lo, hi, _ = model.predict_with_interval(test.X[window])
        point[window], low[window], high[window] = p, lo, hi
        fits += 1
    return {
        "point": point,
        "low": low,
        "high": high,
        "truth": test.y.to_numpy(),
        "index": stamps,
        "n_fits": fits,
    }


def score(result: dict, limit: float) -> dict:
    truth, point = result["truth"], result["point"]
    return {
        **regression_metrics(truth, point),
        **interval_scores(truth, result["low"], result["high"]),
        **violation_scores(truth, result["high"], limit),
        "n_fits": result["n_fits"],
    }


def by_period(result: dict, limit: float, periods) -> dict:
    out = {}
    for label, (start, end) in periods.items():
        mask = (result["index"] >= pd.Timestamp(start)) & (result["index"] <= pd.Timestamp(end))
        if mask.sum() < 5:
            continue
        sub = {
            k: (v[mask] if isinstance(v, np.ndarray) else v)
            for k, v in result.items()
            if k != "index"
        }
        sub["index"] = result["index"][mask]
        out[label] = score(sub, limit)
    return out


def main(every_days: int = 30) -> dict:
    cfg = load_config()
    interim = project_root() / cfg.main["paths"]["interim"]
    reference = SulfurModel.load(
        project_root() / cfg.main["paths"]["models"] / "sulfur_model.joblib"
    )
    tel = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims, pak = load_df(interim / "lims_long.pkl"), load_df(interim / "pak_long.pkl")
    dataset = build_dataset_for_model(tel, lims, pak, reference, cfg)
    split = chronological_split(dataset, cfg)
    limit = cfg.sulfur_limit
    delay = float(cfg.main["data_quality"]["lims_delay_minutes"])

    fixed = fixed_schedule(split, cfg)
    expanding = expanding_schedule(dataset, split, cfg, every_days, delay)

    periods = {
        "январь 2026": ("2026-01-01", "2026-02-01"),
        "май 2026": ("2026-05-01", "2026-06-01"),
    }
    report = {
        "every_days": every_days,
        "test_period": [str(split["test"].y.index.min()), str(split["test"].y.index.max())],
        "fixed": score(fixed, limit),
        "expanding": score(expanding, limit),
        "fixed_by_period": by_period(fixed, limit, periods),
        "expanding_by_period": by_period(expanding, limit, periods),
    }
    better_mae = report["expanding"]["MAE"] < report["fixed"]["MAE"]
    coverage_kept = report["expanding"]["coverage"] >= report["fixed"]["coverage"] - 0.02
    report["verdict"] = {
        "mae_improves": bool(better_mae),
        "coverage_not_worse": bool(coverage_kept),
        "adopt": bool(better_mae and coverage_kept),
        "rule": "внедрять, если MAE ниже и покрытие интервала не упало более чем на 0,02",
    }
    out = project_root() / cfg.main["paths"]["reports"] / "walk_forward.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    keys = (
        "MAE",
        "MedianAE",
        "RMSE",
        "R2",
        "coverage",
        "mean_width",
        "recall",
        "precision",
        "n_fits",
    )
    print(f"{'схема':12s} " + " ".join(f"{k:>9s}" for k in keys))
    for name in ("fixed", "expanding"):
        row = report[name]
        print(
            f"{name:12s} "
            + " ".join(
                f"{row[k]:9.3f}" if isinstance(row[k], float) else f"{str(row[k]):>9s}"
                for k in keys
            )
        )
    print("\nвердикт:", report["verdict"])
    print(f"-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="сравнение схем обучения")
    ap.add_argument("--every-days", type=int, default=30)
    main(**vars(ap.parse_args()))
