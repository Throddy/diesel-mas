"""Capture the exact decision of every cycle on a grid, for refactor checks.

A refactor of the agent plumbing must not change what the system decides.
This writes a per-cycle fingerprint that can be diffed byte for byte before
and after the change; aggregate metrics hide single-cycle differences that
cancel out.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import load_config, project_root
from src.pipeline import build_system


def fingerprint(rec) -> dict:
    """The parts of a recommendation that define observable behaviour."""
    action = rec.selected_action or {}
    quality = rec.agent_trace["QualityAgent"]
    return {
        "t": str(rec.decision_timestamp),
        "abstained": bool(rec.abstained),
        "headline": rec.headline,
        "reasons": sorted(rec.reasons),
        "point_forecast": round(float(quality["point_forecast"]), 6),
        "upper": round(float(quality["upper"]), 6),
        "risk": round(float(quality["risk_exceed_limit"] or 0.0), 6),
        "action_id": action.get("action_id"),
        "moves": {k: round(float(v), 6) for k, v in (action.get("moves") or {}).items()},
        "selected_upper": (
            None
            if action.get("predicted_sulfur_upper") is None
            else round(float(action["predicted_sulfur_upper"]), 6)
        ),
        "feasible_ids": sorted(rec.agent_trace["SafetyAgent"]["feasible_ids"]),
        "checks": sorted(
            (c["constraint_id"], c["name"], c["status"]) for c in rec.constraint_checks
        ),
        "confidence": round(float(rec.confidence["overall"]), 6),
    }


def main(cycles: int, out: str) -> None:
    cfg = load_config()
    system = build_system(cfg)
    start = pd.Timestamp(cfg.main["split"]["test_start"])
    stamps = pd.date_range(start, system.telemetry.index.max(), periods=cycles)
    rows = [fingerprint(system.decide(t, log=False)) for t in stamps]
    path = project_root() / out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"{len(rows)} циклов -> {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="слепок решений для проверки рефакторинга")
    ap.add_argument("--cycles", type=int, default=60)
    ap.add_argument("--out", default="reports/phase_b/behaviour.json")
    a = ap.parse_args()
    main(a.cycles, a.out)
