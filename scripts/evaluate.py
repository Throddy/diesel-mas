"""Stage 3: evaluation of the MODEL and of the DECISION SYSTEM on the
untouched chronological test block."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.config import load_config, project_root
from src.data.loaders import clean_telemetry
from src.data.store import load_df
from src.features.dataset import build_dataset_for_model, chronological_split
from src.models.quality import (
    SulfurModel,
    baseline_pak,
    baseline_previous_lims,
    baseline_train_median,
    interval_metrics,
    regression_metrics,
    violation_metrics,
)
from src.pipeline import build_system


def _status_counts(statuses) -> dict:
    """How many rows received a calibrated interval and how many did not."""
    from collections import Counter

    return dict(Counter(statuses))


def model_evaluation(cfg) -> dict:
    interim = project_root() / cfg.main["paths"]["interim"]
    tel = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims, pak = load_df(interim / "lims_long.pkl"), load_df(interim / "pak_long.pkl")
    model = SulfurModel.load(project_root() / cfg.main["paths"]["models"] / "sulfur_model.joblib")
    ds = build_dataset_for_model(tel, lims, pak, model, cfg)
    sp = chronological_split(ds, cfg)
    out = {}
    for name in ("valid", "test"):
        blk = sp[name]
        p, lo, hi, risk = model.predict_with_interval(blk.X)
        out[name] = {
            "period": [str(blk.y.index.min()), str(blk.y.index.max())],
            "regression": regression_metrics(blk.y.to_numpy(), p),
            "violation_point_forecast": violation_metrics(
                blk.y.to_numpy(), p, cfg.sulfur_limit, risk
            ),
            "violation_upper_bound": violation_metrics(
                blk.y.to_numpy(), hi, cfg.sulfur_limit, risk
            ),
            "interval": interval_metrics(blk.y.to_numpy(), lo, hi),
            "interval_status": _status_counts(model.interval_status(blk.X)),
            "baselines": {
                "previous_lims": regression_metrics(
                    blk.y.to_numpy(), baseline_previous_lims(blk.X)
                ),
                "pak_nowcast": regression_metrics(blk.y.to_numpy(), baseline_pak(blk.X)),
                "train_median": regression_metrics(
                    blk.y.to_numpy(), baseline_train_median(blk.X, float(sp["train"].y.median()))
                ),
            },
        }
    return out


def system_evaluation(cfg, n_cycles: int = 12) -> tuple[dict, dict]:
    system = build_system(cfg)
    start, end = pd.Timestamp(cfg.main["split"]["test_start"]), system.telemetry.index.max()
    stamps = pd.date_range(
        start + pd.Timedelta(days=5), end - pd.Timedelta(days=5), periods=n_cycles
    )
    rows, times = [], []
    for t in stamps:
        t0 = time.time()
        rec = system.decide(t, log=False)
        times.append(time.time() - t0)
        cands = rec.agent_trace["OptimizationAgent"]["candidates"]
        sel = rec.selected_action
        rows.append(
            {
                "t": str(t),
                "abstained": rec.abstained,
                "n_candidates": len(cands),
                "n_feasible": len(rec.agent_trace["SafetyAgent"]["feasible_ids"]),
                "n_unsafe_rejected": sum(
                    1
                    for c in cands
                    if c["predicted_sulfur_upper"] is not None
                    and c["predicted_sulfur_upper"] > cfg.sulfur_limit
                    and not c["feasible"]
                ),
                "n_unsafe_total": sum(
                    1
                    for c in cands
                    if c["predicted_sulfur_upper"] is not None
                    and c["predicted_sulfur_upper"] > cfg.sulfur_limit
                ),
                "selected_is_do_nothing": bool(sel and sel.get("is_do_nothing")),
                "selected_upper": None if not sel else sel.get("predicted_sulfur_upper"),
                "ood": rec.confidence["ood_flag"],
                "severity": rec.confidence["severity_class"],
                "confidence": rec.confidence["overall"],
            }
        )
    df = pd.DataFrame(rows)
    unsafe_selected = int(
        ((~df.abstained) & (df.selected_upper.astype(float) > cfg.sulfur_limit)).sum()
    )
    stable = df[(~df.abstained)]
    summary = {
        "n_cycles": int(len(df)),
        "abstention_rate": float(df.abstained.mean()),
        "recommendation_rate": float(1 - df.abstained.mean()),
        "hard_constraint_violation_rate_of_recommendations": float(
            unsafe_selected / max(len(stable), 1)
        ),
        "unsafe_candidates_rejected_share": float(
            df.n_unsafe_rejected.sum() / max(df.n_unsafe_total.sum(), 1)
        ),
        "unnecessary_action_rate_when_recommending": (
            float(1 - stable.selected_is_do_nothing.mean()) if len(stable) else None
        ),
        "ood_rejection_rate": float(df.ood.mean()),
        "mean_confidence": float(df.confidence.mean()),
    }
    timing = {
        "inference_seconds_mean": float(np.mean(times)),
        "inference_seconds_max": float(np.max(times)),
        "n_cycles": len(times),
    }
    return {"summary": summary, "cycles": rows}, timing


def determinism_check(cfg) -> dict:
    system = build_system(cfg)
    t = pd.Timestamp(cfg.main["demo"]["stable"])
    a = system.decide(t, log=False)
    b = build_system(cfg).decide(t, log=False)
    same = (
        a.headline == b.headline
        and a.abstained == b.abstained
        and json.dumps(a.expected_effect, default=str) == json.dumps(b.expected_effect, default=str)
    )
    return {"identical_recommendation_for_identical_input": bool(same), "headline": a.headline}


def main(cycles: int = 12) -> dict:
    cfg = load_config()
    system, timing = system_evaluation(cfg, cycles)
    report = {
        "model_version": cfg.model_version,
        "model": model_evaluation(cfg),
        "system": system,
        "reproducibility": determinism_check(cfg),
    }
    reports = project_root() / cfg.main["paths"]["reports"]
    out = reports / "evaluation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (reports / "evaluation_timing.json").write_text(
        json.dumps(timing, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "test_regression": report["model"]["test"]["regression"],
                "test_violation_by_upper_bound": report["model"]["test"]["violation_upper_bound"],
                "interval": report["model"]["test"]["interval"],
                "system": report["system"]["summary"],
                "reproducibility": report["reproducibility"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nfull report -> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="оценка на фиксированном holdout; с --start/--end — за произвольный период"
    )
    ap.add_argument("--cycles", type=int, default=12)
    ap.add_argument("--start", default=None, help="начало произвольного периода")
    ap.add_argument("--end", default=None, help="конец произвольного периода")
    args = ap.parse_args()
    if bool(args.start) != bool(args.end):
        raise SystemExit("укажите --start и --end вместе")
    if args.start:
        from scripts.backtest import main as backtest_main

        backtest_main(args.start, args.end, args.cycles)
    else:
        main(args.cycles)
