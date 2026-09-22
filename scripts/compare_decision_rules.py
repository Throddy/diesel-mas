"""Threshold rule against expected-loss rule, on the same cycles.

Both keep the hard constraint: a candidate whose upper bound breaches the
limit is rejected before either rule sees it.  What is compared is when they
choose to act, and how far they push.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import yaml

from src.config import load_config, project_root


def set_flag(enabled: bool, **overrides) -> None:
    """Switch the rule in config; the system reads it on the next build."""
    path = project_root() / "config" / "config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["expected_loss"]["enabled"] = enabled
    data["expected_loss"].update(overrides)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    load_config.cache_clear()


def run(cycles: int, limit: float) -> dict:
    """Replay the test block and record what the current rule decided."""
    from src.pipeline import build_system

    cfg = load_config()
    system = build_system(cfg)
    start = pd.Timestamp(cfg.main["split"]["test_start"])
    stamps = pd.date_range(start, system.telemetry.index.max(), periods=cycles)
    rows, durations = [], []
    for t in stamps:
        began = time.time()
        rec = system.decide(t, log=False)
        durations.append(time.time() - began)
        action = rec.selected_action or {}
        quality = rec.agent_trace["QualityAgent"]
        rows.append(
            {
                "t": str(t),
                "abstained": bool(rec.abstained),
                "acted": bool(action and not action.get("is_do_nothing")),
                "risk": float(quality["risk_exceed_limit"] or 0.0),
                "forecast_upper": float(quality["upper"]),
                "selected_upper": action.get("predicted_sulfur_upper"),
                "selected_point": action.get("predicted_sulfur"),
                "energy": action.get("energy_proxy"),
                "magnitude": action.get("action_magnitude"),
            }
        )
    frame = pd.DataFrame(rows)
    at_risk = frame[frame.forecast_upper > limit]
    acted = frame[frame.acted]
    overshoot = [limit - p for p in acted.selected_point.dropna().astype(float)]
    return {
        "n_cycles": int(len(frame)),
        "abstention_rate": float(frame.abstained.mean()),
        "action_rate": float(frame.acted.mean()),
        "action_rate_when_at_risk": None if not len(at_risk) else float(at_risk.acted.mean()),
        "violations": int(
            sum(
                1
                for r in rows
                if not r["abstained"]
                and r["selected_upper"] is not None
                and r["selected_upper"] > limit
            )
        ),
        "median_margin_below_limit": float(np.median(overshoot)) if overshoot else None,
        "mean_margin_below_limit": float(np.mean(overshoot)) if overshoot else None,
        "mean_energy_when_acting": (
            float(acted.energy.dropna().mean()) if len(acted.energy.dropna()) else None
        ),
        "mean_magnitude_when_acting": (
            float(acted.magnitude.dropna().mean()) if len(acted.magnitude.dropna()) else None
        ),
        "acted_at_low_risk": int(((frame.acted) & (frame.risk < 0.2)).sum()),
        "held_at_high_risk": int(((~frame.acted) & (~frame.abstained) & (frame.risk > 0.5)).sum()),
        "seconds_per_cycle_mean": float(np.mean(durations)),
    }


def closed_loop(scenario: str) -> dict:
    """Same scenario under the rule currently enabled."""
    from src.pipeline import build_system
    from src.sim.closed_loop import ClosedLoopSimulator, SimConfig

    cfg = load_config()
    system = build_system(cfg)
    start = pd.Timestamp(cfg.main["demo"][scenario])
    result = ClosedLoopSimulator(system, SimConfig(steps=12, lag_minutes=360.0)).run(start, "mas")
    summary = result.summary(cfg.sulfur_limit)
    trajectory = [r.sulfur_mgkg for r in result.records]
    return {
        "n_actions": summary["n_actions"],
        "minutes_off_spec": summary["minutes_off_spec"],
        "sulfur_start": summary["sulfur_start_mgkg"],
        "sulfur_final": summary["sulfur_final_mgkg"],
        "sulfur_min": float(min(trajectory)),
        "overshoot_below_limit": float(cfg.sulfur_limit - min(trajectory)),
    }


def main(cycles: int = 40) -> dict:
    cfg = load_config()
    limit = cfg.sulfur_limit
    original = dict(cfg.main["expected_loss"])
    try:
        set_flag(False)
        threshold = run(cycles, limit)
        threshold_loop = {s: closed_loop(s) for s in ("quality_risk", "limit_breach")}
        set_flag(True)
        expected = run(cycles, limit)
        expected_loop = {s: closed_loop(s) for s in ("quality_risk", "limit_breach")}
        sensitivity = {}
        for factor, label in (
            (2.0, "цена некондиции вдвое выше"),
            (0.5, "цена некондиции вдвое ниже"),
        ):
            set_flag(True, off_spec=original["off_spec"] * factor)
            sensitivity[label] = run(cycles, limit)
    finally:
        set_flag(original["enabled"], **{k: v for k, v in original.items() if k != "enabled"})

    report = {
        "cycles": cycles,
        "limit_mgkg": limit,
        "threshold_rule": threshold,
        "expected_loss_rule": expected,
        "closed_loop": {"threshold": threshold_loop, "expected_loss": expected_loop},
        "sensitivity": sensitivity,
        "weights": {k: v for k, v in original.items() if k not in ("enabled", "basis")},
    }
    safe = expected["violations"] <= threshold["violations"]
    acts_when_needed = (expected["action_rate_when_at_risk"] or 0) >= 0.8 * (
        threshold["action_rate_when_at_risk"] or 0
    )
    less_overshoot = (expected["median_margin_below_limit"] or 0) <= (
        threshold["median_margin_below_limit"] or 0
    )
    report["verdict"] = {
        "no_new_violations": bool(safe),
        "still_acts_when_at_risk": bool(acts_when_needed),
        "less_overshoot": bool(less_overshoot),
        "adopt": bool(safe and acts_when_needed and less_overshoot),
        "rule": (
            "внедрять, если нарушений не больше, действие при риске не упало "
            "более чем на пятую часть и избыточность снизилась"
        ),
    }
    out = project_root() / cfg.main["paths"]["reports"] / "decision_rules.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "threshold_rule": threshold,
                "expected_loss_rule": expected,
                "closed_loop": report["closed_loop"],
                "verdict": report["verdict"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="сравнение правил принятия решения")
    ap.add_argument("--cycles", type=int, default=40)
    main(**vars(ap.parse_args()))
