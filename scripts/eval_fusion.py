"""Score the Kalman fusion against the current forecast and the baselines.

For every laboratory analysis in the test block the estimate is built from
what was knowable when the sample was taken: analyser readings up to that
moment and laboratory results already reported.  The analysis being scored is
never among them.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import load_config, project_root
from src.data.loaders import pak_sulfur, target_series
from src.data.store import load_df
from src.models.nowcast_fusion import FusionParams, nowcast_at


def metrics(truth: np.ndarray, estimate: np.ndarray, lower=None, upper=None) -> dict:
    """MAE, RMSE, R2 and, when an interval is given, coverage and width."""
    err = estimate - truth
    out = {
        "n": int(len(truth)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MedianAE": float(np.median(np.abs(err))),
        "R2": float(1 - np.sum(err**2) / np.sum((truth - truth.mean()) ** 2)),
        "MAE_log": float(
            np.mean(np.abs(np.log(np.maximum(estimate, 1e-6)) - np.log(np.maximum(truth, 1e-6))))
        ),
    }
    if lower is not None:
        inside = (truth >= lower) & (truth <= upper)
        out["coverage"] = float(np.mean(inside))
        out["mean_width"] = float(np.mean(upper - lower))
    return out


def main(z: float = 1.645) -> dict:
    cfg = load_config()
    interim = project_root() / cfg.main["paths"]["interim"]
    lims_long = load_df(interim / "lims_long.pkl")
    pak_long = load_df(interim / "pak_long.pkl")
    target = target_series(lims_long, cfg)
    analyser = pak_sulfur(pak_long)
    params = FusionParams.from_config(cfg)
    fusion_cfg = cfg.main["fusion"]
    delay = float(cfg.main["data_quality"]["lims_delay_minutes"])

    test_start = pd.Timestamp(cfg.main["split"]["test_start"])
    truth = target[(target.index >= test_start) & (target > 0) & (target <= 100)]

    rows = []
    for stamp, value in truth.items():
        history = target[target.index < stamp]
        estimate = nowcast_at(
            history,
            analyser,
            stamp,
            delay,
            params=params,
            lookback_hours=float(fusion_cfg["lookback_hours"]),
            pak_every_minutes=float(fusion_cfg["pak_every_minutes"]),
        )
        if estimate is None:
            continue
        sigma = estimate["sigma_log"]
        rows.append(
            {
                "t": str(stamp),
                "truth": float(value),
                "fusion": estimate["sulfur"],
                "lower": float(np.exp(estimate["log_sulfur"] - z * sigma)),
                "upper": float(np.exp(estimate["log_sulfur"] + z * sigma)),
                "drift": estimate["drift"],
                "sigma_log": sigma,
            }
        )

    frame = pd.DataFrame(rows)
    evaluation = json.loads(
        (project_root() / cfg.main["paths"]["reports"] / "evaluation_report.json").read_text(
            encoding="utf-8"
        )
    )
    current = evaluation["model"]["test"]

    report = {
        "period": [str(truth.index.min()), str(truth.index.max())],
        "n_scored": int(len(frame)),
        "params": vars(params),
        "fusion": metrics(
            frame.truth.to_numpy(),
            frame.fusion.to_numpy(),
            frame.lower.to_numpy(),
            frame.upper.to_numpy(),
        ),
        "current_model": {**current["regression"], **current["interval"]},
        "baselines": current["baselines"],
        "drift": {
            "mean": float(frame.drift.mean()),
            "min": float(frame.drift.min()),
            "max": float(frame.drift.max()),
        },
        "verdict_rule": (
            "внедрять только если MAE не хуже текущей модели и покрытие "
            "не хуже; подгонка параметров под улучшение запрещена"
        ),
    }
    better_mae = report["fusion"]["MAE"] <= report["current_model"]["MAE"]
    better_cov = report["fusion"]["coverage"] >= report["current_model"]["coverage"]
    report["verdict"] = "внедрять" if better_mae and better_cov else "не внедрять"
    report["comparison"] = {
        "MAE_fusion": report["fusion"]["MAE"],
        "MAE_current": report["current_model"]["MAE"],
        "coverage_fusion": report["fusion"]["coverage"],
        "coverage_current": report["current_model"]["coverage"],
        "width_fusion": report["fusion"]["mean_width"],
        "width_current": report["current_model"]["mean_width"],
    }
    out = project_root() / cfg.main["paths"]["reports"] / "fusion_eval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("n_scored", "comparison", "verdict")},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="оценка калмановского слияния")
    ap.add_argument("--z", type=float, default=1.645, help="квантиль для интервала 90 %%")
    main(**vars(ap.parse_args()))
