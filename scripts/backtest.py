"""Evaluate model and decision system over an ARBITRARY period.

The jury picks a period and looks at the result (R-MEET-17), so every figure
the system reports has to be reproducible for any window, not only for the
fixed chronological test block.  Results land in
``reports/periods/<start>_<end>/`` so several periods can coexist.

A period that overlaps the training block is scored and clearly labelled as
optimistic - it is not evidence of generalisation.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.config import load_config, project_root
from src.data.loaders import clean_telemetry
from src.data.store import load_df
from src.features.dataset import build_dataset_for_model
from src.models.quality import (
    SulfurModel,
    baseline_pak,
    baseline_previous_lims,
    interval_metrics,
    regression_metrics,
    violation_metrics,
)
from src.pipeline import build_system


def _blocks_touched(start: pd.Timestamp, end: pd.Timestamp, cfg) -> list[str]:
    """Name every chronological block the period overlaps."""
    split = cfg.main["split"]
    train_end, valid_end = pd.Timestamp(split["train_end"]), pd.Timestamp(split["valid_end"])
    touched = []
    if start <= train_end:
        touched.append("train")
    if start <= valid_end and end > train_end:
        touched.append("valid")
    if end > valid_end:
        touched.append("test")
    return touched


def model_metrics(cfg, start, end) -> dict:
    """Forecast quality against the laboratory analyses inside the period."""
    interim = project_root() / cfg.main["paths"]["interim"]
    tel = clean_telemetry(load_df(interim / "telemetry.pkl"), cfg)
    lims, pak = load_df(interim / "lims_long.pkl"), load_df(interim / "pak_long.pkl")
    model = SulfurModel.load(project_root() / cfg.main["paths"]["models"] / "sulfur_model.joblib")
    dataset = build_dataset_for_model(tel, lims, pak, model, cfg).slice(start, end)
    if not len(dataset.y):
        return {
            "n_analyses": 0,
            "note": "в периоде нет лабораторных анализов, пригодных для оценки",
        }
    point, lower, upper, risk = model.predict_with_interval(dataset.X)
    truth = dataset.y.to_numpy()
    return {
        "n_analyses": int(len(truth)),
        "analyses_period": [str(dataset.y.index.min()), str(dataset.y.index.max())],
        "regression": regression_metrics(truth, point),
        "violation_point_forecast": violation_metrics(truth, point, cfg.sulfur_limit, risk),
        "violation_upper_bound": violation_metrics(truth, upper, cfg.sulfur_limit, risk),
        "interval": interval_metrics(truth, lower, upper),
        "baselines": {
            "previous_lims": regression_metrics(truth, baseline_previous_lims(dataset.X)),
            "pak_nowcast": regression_metrics(truth, baseline_pak(dataset.X)),
        },
    }


def system_metrics(cfg, start, end, cycles: int) -> dict:
    """Replay decision cycles on a uniform grid inside the period."""
    system = build_system(cfg)
    lo = max(pd.Timestamp(start), system.telemetry.index.min())
    hi = min(pd.Timestamp(end), system.telemetry.index.max())
    if lo >= hi:
        return {"n_cycles": 0, "note": "период вне покрытия телеметрии"}
    stamps = pd.date_range(lo, hi, periods=max(cycles, 2))
    rows, durations = [], []
    for t in stamps:
        started = time.time()
        rec = system.decide(t, log=False)
        durations.append(time.time() - started)
        action = rec.selected_action or {}
        rows.append(
            {
                "t": str(t),
                "abstained": bool(rec.abstained),
                "mode": rec.data_freshness["TELEMETRY"]["mode"],
                "do_nothing": bool(action.get("is_do_nothing")),
                "acted": bool(action and not action.get("is_do_nothing")),
                "selected_upper": action.get("predicted_sulfur_upper"),
                "forecast_upper": rec.agent_trace["QualityAgent"]["upper"],
                "risk": rec.agent_trace["QualityAgent"]["risk_exceed_limit"],
                "confidence": rec.confidence["overall"],
                "ood": bool(rec.confidence["ood_flag"]),
                "reasons": list(rec.reasons),
                "rounds": len(rec.agent_trace["Orchestrator"].get("rounds", [])) or 1,
                "seconds": time.time() - started,
            }
        )
    frame = pd.DataFrame(rows)
    recommending = frame[~frame.abstained]
    unsafe = [
        r
        for r in rows
        if not r["abstained"]
        and r["selected_upper"] is not None
        and r["selected_upper"] > cfg.sulfur_limit
    ]
    reasons: dict[str, int] = {}
    for row in rows:
        if row["abstained"]:
            for reason in row["reasons"] or ["причина не указана"]:
                reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "n_cycles": int(len(frame)),
        "abstention_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "abstention_rate": float(frame.abstained.mean()),
        "action_rate": float(frame.acted.mean()),
        "do_nothing_rate": float(frame.do_nothing.mean()),
        "hard_constraint_violations_in_recommendations": len(unsafe),
        "cycles_at_risk": int((frame.forecast_upper.astype(float) > cfg.sulfur_limit).sum()),
        "action_rate_when_at_risk": _rate_when_at_risk(frame, cfg.sulfur_limit),
        "mean_confidence": float(frame.confidence.mean()),
        "ood_rate": float(frame.ood.mean()),
        "mode_counts": frame["mode"].value_counts().to_dict(),
        "seconds_per_cycle_mean": float(np.mean(durations)),
        "seconds_per_cycle_max": float(np.max(durations)),
        "seconds_per_cycle_p95": float(np.percentile(durations, 95)),
        "seconds_by_rounds": {
            str(int(r)): {
                "cycles": int((frame.rounds == r).sum()),
                "mean": float(frame.seconds[frame.rounds == r].mean()),
                "max": float(frame.seconds[frame.rounds == r].max()),
            }
            for r in sorted(frame.rounds.unique())
        },
        "n_recommending": int(len(recommending)),
    }


def _rate_when_at_risk(frame: pd.DataFrame, limit: float) -> float | None:
    """How often the system actually acts when the forecast breaches the limit."""
    at_risk = frame[frame.forecast_upper.astype(float) > limit]
    return None if not len(at_risk) else float(at_risk.acted.mean())


def main(start: str, end: str, cycles: int = 24) -> dict:
    cfg = load_config()
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    if lo >= hi:
        raise SystemExit(f"пустой период: --start {start} >= --end {end}")
    blocks = _blocks_touched(lo, hi, cfg)
    report = {
        "period": [str(lo), str(hi)],
        "blocks_touched": blocks,
        "honest_reading": (
            "период не пересекается с обучением: оценка честная"
            if blocks == ["test"]
            else f"период пересекает блоки {blocks}: метрики оптимистичны, "
            "это не доказательство обобщения"
        ),
        "model_version": cfg.model_version,
        "model": model_metrics(cfg, lo, hi),
        "system": system_metrics(cfg, lo, hi, cycles),
    }
    out_dir = project_root() / cfg.main["paths"]["reports"] / "periods" / f"{lo:%Y%m%d}_{hi:%Y%m%d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "backtest.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("period", "blocks_touched", "honest_reading")},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        json.dumps(
            {
                "model": report["model"].get("regression", report["model"]),
                "system": report["system"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="оценка за произвольный период")
    ap.add_argument("--start", required=True, help="начало периода, например 2026-01-01")
    ap.add_argument("--end", required=True, help="конец периода, например 2026-02-01")
    ap.add_argument("--cycles", type=int, default=24, help="сколько циклов решения проиграть")
    a = ap.parse_args()
    main(a.start, a.end, a.cycles)
