"""Search a period for the three demo moments the terms of reference require.

The moments are *found*, never hand-picked: the scanner replays decision
cycles on a grid and keeps the first timestamp of each class, using only the
recommendation the system itself produced.  By default it scans the untouched
test block, so the demonstration never runs on data a model was fitted on
(K-16).  Selection looks at the decision, not at the future laboratory result,
so nothing here tunes the system to the test set.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import load_config, project_root
from src.pipeline import build_system

CLASSES = ("stable", "quality_risk", "bad_data")


def classify(rec, cfg) -> str | None:
    """Label one decision cycle as a demo class, or None if it shows nothing."""
    quality = rec.agent_trace["QualityAgent"]
    mode = rec.data_freshness["TELEMETRY"]["mode"]
    data_ok = all(
        c["status"] != "FAIL" for c in rec.constraint_checks if c["constraint_id"] == "HC-04"
    )
    if rec.abstained and not data_ok:
        return "bad_data"
    if rec.abstained:
        return None
    action = rec.selected_action or {}
    if mode != "RUNNING":
        return None
    if not action.get("is_do_nothing"):
        return "quality_risk"
    margin = cfg.sulfur_limit - (quality["upper"] or cfg.sulfur_limit)
    risk = quality["risk_exceed_limit"] or 0.0
    if margin > 0 and risk < cfg.main["optimization"]["act_only_if_risk_above"]:
        return "stable"
    return None


def scan(system, start: pd.Timestamp, end: pd.Timestamp, step_hours: float) -> dict:
    """Replay the grid and collect candidate moments per class."""
    found: dict[str, list[dict]] = {c: [] for c in CLASSES}
    grid = pd.date_range(start, end, freq=pd.Timedelta(hours=step_hours))
    for t in grid:
        rec = system.decide(t, log=False)
        label = classify(rec, system.cfg)
        if label is None:
            continue
        quality = rec.agent_trace["QualityAgent"]
        found[label].append(
            {
                "timestamp": str(t),
                "headline": rec.headline,
                "point_forecast_mgkg": quality["point_forecast"],
                "upper_mgkg": quality["upper"],
                "risk_exceed_limit": quality["risk_exceed_limit"],
                "abstained": rec.abstained,
                "mode": rec.data_freshness["TELEMETRY"]["mode"],
            }
        )
    return found


def pick(found: dict) -> dict:
    """One representative per class: the most informative, not the first."""
    chosen = {}
    if found["quality_risk"]:
        chosen["quality_risk"] = max(
            found["quality_risk"], key=lambda r: r["risk_exceed_limit"] or 0.0
        )
    if found["stable"]:
        chosen["stable"] = min(found["stable"], key=lambda r: r["upper_mgkg"] or 0.0)
    if found["bad_data"]:
        chosen["bad_data"] = found["bad_data"][0]
    return chosen


def main(start: str | None, end: str | None, step_hours: float, write_config: bool) -> dict:
    cfg = load_config()
    system = build_system(cfg)
    lo = pd.Timestamp(start or cfg.main["split"]["test_start"])
    hi = pd.Timestamp(end) if end else system.telemetry.index.max()
    found = scan(system, lo, hi, step_hours)
    chosen = pick(found)
    report = {
        "period": [str(lo), str(hi)],
        "step_hours": step_hours,
        "block": "test" if lo >= pd.Timestamp(cfg.main["split"]["test_start"]) else "mixed",
        "n_scanned": len(pd.date_range(lo, hi, freq=pd.Timedelta(hours=step_hours))),
        "n_found": {k: len(v) for k, v in found.items()},
        "selected": chosen,
        "candidates": found,
    }
    out = project_root() / cfg.main["paths"]["reports"] / "demo_moments.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(
        json.dumps(
            {
                "period": report["period"],
                "n_found": report["n_found"],
                "selected": {k: v["timestamp"] for k, v in chosen.items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\n-> {out}")
    if write_config:
        _write_config(chosen)
    return report


def _write_config(chosen: dict) -> None:
    """Point config/config.yaml at the found moments, keeping the file's shape."""
    import yaml

    path = project_root() / "config" / "config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, row in chosen.items():
        data["demo"][key] = row["timestamp"][:16]
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"config/config.yaml demo section updated: {sorted(chosen)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="начало периода поиска (по умолчанию test_start)")
    ap.add_argument("--end", default=None, help="конец периода поиска")
    ap.add_argument("--step-hours", type=float, default=6.0)
    ap.add_argument(
        "--write-config", action="store_true", help="записать найденные моменты в config.yaml"
    )
    a = ap.parse_args()
    main(a.start, a.end, a.step_hours, a.write_config)
