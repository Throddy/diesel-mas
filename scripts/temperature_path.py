"""Controlled check that reactor temperature reaches both models.

A decision is made by two models.  N forecasts product sulfur from the whole
feature vector; M turns a proposed move into an expected change.  This script
perturbs reactor temperature alone, holds every other feature fixed and shows
what each model returns, so the path from the control to the decision is
visible rather than assumed.

The perturbation stays inside the historical support of M.  What it measures
is a model response, not a plant response: N is fitted on a closed loop, where
the observed association between temperature and sulfur is biased towards
zero, and M carries a sign fixed by the chemistry rather than by the data.
Neither number is a causal effect, and this script does not establish one.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.config import load_config, project_root
from src.optimization.candidates import SHIFTED_SUFFIXES
from src.pipeline import build_system

TEMPERATURE_TAG = "ht:T5"


def shift_temperature(features: pd.Series, delta: float) -> pd.Series:
    """Move the level features of the temperature tag, leave the rest alone."""
    out = features.copy()
    for suffix in SHIFTED_SUFFIXES:
        key = f"{TEMPERATURE_TAG}{suffix}"
        if key in out.index and np.isfinite(out[key]):
            out[key] = float(out[key]) + delta
    return out


def untouched(before: pd.Series, after: pd.Series) -> int:
    """How many features kept their value, as evidence the test is controlled."""
    same = 0
    for key in before.index:
        a, b = before[key], after[key]
        if (pd.isna(a) and pd.isna(b)) or (np.isfinite(a) and np.isfinite(b) and a == b):
            same += 1
    return same


def main(scenario: str = "quality_risk", span_c: float = 6.0, step_c: float = 2.0) -> dict:
    cfg = load_config()
    system = build_system(cfg)
    moment = pd.Timestamp(cfg.main["demo"][scenario])
    agent = system.orchestrator.qa
    features = agent.build_features(moment)
    base_temperature = float(features[f"{TEMPERATURE_TAG}|value"])

    response = system.orchestrator.oa.whatif.model
    low, high = response.bounds[TEMPERATURE_TAG]
    deltas = [
        d for d in np.arange(-span_c, span_c + 1e-9, step_c) if low <= base_temperature + d <= high
    ]

    rows = []
    for delta in deltas:
        moved = shift_temperature(features, float(delta))
        point, lower, upper, risk, _ = agent.predict_features(moved, moment)
        rows.append(
            {
                "delta_c": round(float(delta), 3),
                "temperature_c": round(base_temperature + float(delta), 3),
                "forecast_mgkg": round(point, 4),
                "lower_mgkg": round(lower, 4),
                "upper_mgkg": round(upper, 4),
                "risk_over_limit": round(risk, 4),
                "features_unchanged": untouched(features, moved),
                "features_total": int(len(features)),
            }
        )

    state = system.orchestrator.dq.run(moment)
    before = system.orchestrator.oa._baseline_state(state)
    feed = before.get("feed_sulfur_mgkg")
    response_rows = []
    if feed is not None and np.isfinite(feed):
        for delta in deltas:
            after = {**before, "ht:T5": before["ht:T5"] + float(delta)}
            try:
                central, pessimistic = response.effect(before, after)
            except ValueError as error:
                response_rows.append(
                    {"delta_c": round(float(delta), 3), "outside_support": str(error)}
                )
                continue
            response_rows.append(
                {
                    "delta_c": round(float(delta), 3),
                    "log_change": round(central, 5),
                    "ratio": round(float(np.exp(central)), 4),
                    "pessimistic_log_change": round(pessimistic, 5),
                }
            )

    recommendation = system.decide(moment, log=False)
    action = recommendation.selected_action or {}
    checks = [
        {
            "id": c.get("constraint_id"),
            "name": c.get("name"),
            "status": c.get("status"),
            "detail": c.get("detail"),
        }
        for c in recommendation.constraint_checks
    ]

    report = {
        "moment": str(moment),
        "scenario": scenario,
        "base_temperature_c": round(base_temperature, 3),
        "support_c": [round(low, 3), round(high, 3)],
        "held_fixed": (
            "изменяются только уровневые признаки тега ht:T5; "
            "остальные признаки и все прочие теги сохраняют значения"
        ),
        "forecast_model_N": rows,
        "response_model_M": response_rows,
        "sensitivity_d_ln_s_dt": {
            "forecast_model_N": _slope(rows),
            "response_model_M": response.diagnostics["temperature_log_sensitivity_joint"],
            "response_model_M_data_only": response.diagnostics["temperature_log_sensitivity_data"],
        },
        "decision_at_the_moment": {
            "abstained": bool(recommendation.abstained),
            "headline": recommendation.headline,
            "action_id": action.get("action_id"),
            "moves": action.get("moves"),
            "moves_from": action.get("moves_from"),
            "predicted_sulfur": action.get("predicted_sulfur"),
            "predicted_sulfur_upper": action.get("predicted_sulfur_upper"),
            "constraint_checks": checks,
        },
        "reading": (
            "прогноз N отражает наблюдаемую связь на замкнутом контуре и "
            "занижает эффект; знак у M задан химией, а не данными; ни то, "
            "ни другое не является измеренным причинным эффектом"
        ),
    }
    out = project_root() / cfg.main["paths"]["reports"] / "temperature_path.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"момент {moment}, температура {base_temperature:.2f} °C, "
        f"область поддержки M от {low:.1f} до {high:.1f} °C"
    )
    print(
        f"\n{'сдвиг':>7s} {'T5, °C':>8s} {'прогноз N':>10s} {'верх':>8s} "
        f"{'риск':>6s} {'M, отношение':>13s} {'признаков без изменений':>24s}"
    )
    by_delta = {r["delta_c"]: r for r in response_rows}
    for row in rows:
        m = by_delta.get(row["delta_c"], {})
        ratio = f"{m['ratio']:13.4f}" if "ratio" in m else f"{'вне поддержки':>13s}"
        print(
            f"{row['delta_c']:7.1f} {row['temperature_c']:8.2f} "
            f"{row['forecast_mgkg']:10.3f} {row['upper_mgkg']:8.3f} "
            f"{row['risk_over_limit']:6.3f} {ratio} "
            f"{row['features_unchanged']:>10d} из {row['features_total']}"
        )
    s = report["sensitivity_d_ln_s_dt"]
    print(
        f"\nd ln S / dT: прогноз N {s['forecast_model_N']:+.5f}, отклик M "
        f"{s['response_model_M']:+.5f}, M только по данным "
        f"{s['response_model_M_data_only']:+.5f} 1/°C"
    )
    d = report["decision_at_the_moment"]
    print(f"\nрешение в этот момент: отказ={d['abstained']}, {d['headline'][:90]}")
    for c in checks:
        print(f"  {c['id']} {c['name']}: {c['status']}")
    print(f"\n-> {out}")
    return report


def _slope(rows):
    """Least-squares slope of ln(forecast) against temperature."""
    usable = [r for r in rows if r["forecast_mgkg"] > 0]
    if len(usable) < 3:
        return float("nan")
    x = np.array([r["temperature_c"] for r in usable], float)
    y = np.log(np.array([r["forecast_mgkg"] for r in usable], float))
    return float(np.polyfit(x, y, 1)[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="контролируемый тест пути влияния температуры")
    ap.add_argument(
        "--scenario",
        default="quality_risk",
        choices=["stable", "quality_risk", "bad_data", "limit_breach"],
    )
    ap.add_argument("--span-c", type=float, default=6.0)
    ap.add_argument("--step-c", type=float, default=2.0)
    main(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()})
