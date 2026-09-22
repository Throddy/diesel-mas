"""Compare five architectures on one grid (site task 08).

The agents themselves are not modified: each configuration is assembled from
the same components, with one part replaced by a different stand-in.

  A  monolith      forecast plus the rule "act when the limit is breached"
  B  no guards     the full agent set without SafetyAgent and DataQualityAgent
  C  full          the system as submitted
  D  no response   the full system with the response model M disabled
  E  free response the full system with the response model fitted the ordinary
                   way, without sign constraints and without the physical prior

E is what a straightforward approach produces.  On this closed loop the
ordinary fit puts d ln S / dT at +0,0038 1/°C, the opposite sign to the
chemistry, so E is measured with an extra figure: the share of recommendations
whose direction contradicts the declared physical sign.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter

import numpy as np
import pandas as pd

from src.config import _load_yaml, load_config, project_root
from src.models.response_model import ResponseModel
from src.optimization.candidates import control_specs
from src.pipeline import build_system

UNCONSTRAINED = "response_model_unconstrained.joblib"


def physical_signs(cfg) -> dict:
    """Declared sign of d(sulfur)/d(control) for each controllable parameter."""
    variables = _load_yaml("physics.yaml")["variables"]
    out = {}
    for spec in control_specs(cfg):
        tag = spec.tags[0]
        if tag in variables and "sulfur_sign" in variables[tag]:
            out[spec.canonical_name] = (tag, int(variables[tag]["sulfur_sign"]))
    return out


def contradicts_physics(action: dict, signs: dict, tolerance: float = 1e-9) -> bool:
    """True when a move is expected to raise sulfur under the declared signs.

    Every recommendation is issued to bring sulfur down, so a move whose
    declared effect is upward points the wrong way.  A candidate may touch two
    parameters; one wrong parameter is enough.
    """
    moves = action.get("moves") or {}
    before = action.get("moves_from") or {}
    for name, proposed in moves.items():
        if name not in signs or name not in before:
            continue
        step = float(proposed) - float(before[name])
        if abs(step) <= tolerance:
            continue
        if signs[name][1] * (1 if step > 0 else -1) > 0:
            return True
    return False


class NoResponse:
    """Stand-in for the response model: every move is predicted to do nothing.

    It replaces the model the WhatIfAgent owns, which is where the response
    lives since the agents started talking over the bus.  Substituting an
    attribute that no longer exists would leave configuration D identical to
    C, and the comparison would silently say nothing.
    """

    diagnostics = {"disabled": True}
    bounds: dict = {}

    def effect(self, before, after):
        return 0.0, 0.0


def _grid(system, cycles: int) -> pd.DatetimeIndex:
    cfg = system.cfg
    start = pd.Timestamp(cfg.main["split"]["test_start"])
    return pd.date_range(start, system.telemetry.index.max(), periods=cycles)


def run_full(system, stamps, disable_response: bool = False, response=None) -> list[dict]:
    """Configuration C, or D and E when the response model is replaced."""
    if disable_response or response is not None:
        agent = system.orchestrator.oa.whatif
        assert hasattr(agent, "model"), "response model moved again: fix the substitution"
        agent.model = NoResponse() if disable_response else response
    signs = physical_signs(system.cfg)
    rows = []
    for t in stamps:
        started = time.time()
        rec = system.decide(t, log=False)
        action = rec.selected_action or {}
        quality = rec.agent_trace["QualityAgent"]
        candidates = rec.agent_trace["OptimizationAgent"]["candidates"]
        rows.append(
            {
                "abstained": bool(rec.abstained),
                "acted": bool(action and not action.get("is_do_nothing")),
                "selected_upper": action.get("predicted_sulfur_upper"),
                "forecast_upper": quality["upper"],
                "unsafe_rejected": sum(
                    1
                    for c in candidates
                    if c["predicted_sulfur_upper"] is not None
                    and c["predicted_sulfur_upper"] > system.cfg.sulfur_limit
                    and not c["feasible"]
                ),
                "against_physics": bool(
                    action
                    and not action.get("is_do_nothing")
                    and contradicts_physics(action, signs)
                ),
                "reason": (rec.reasons[0] if rec.abstained and rec.reasons else None),
                "best_upper": min(
                    (
                        c["predicted_sulfur_upper"]
                        for c in candidates
                        if not c["is_do_nothing"] and c["predicted_sulfur_upper"] is not None
                    ),
                    default=None,
                ),
                "seconds": time.time() - started,
            }
        )
    return rows


def run_no_guards(system, stamps) -> list[dict]:
    """Configuration B: the optimizer's best candidate, with no safety veto.

    Quality and reliability still run; the SafetyAgent verdict and the data
    validity gate are simply ignored, which is what an architecture without
    those two agents would do.
    """
    rows = []
    for t in stamps:
        started = time.time()
        state = system.orchestrator.dq.run(t)
        quality = system.orchestrator.qa.run(state)
        reliability = system.orchestrator.ra.run(state)
        candidates = system.orchestrator.oa.run(state, quality, reliability)
        movable = [c for c in candidates if not c.is_do_nothing and c.response_supported]
        best = min(movable, key=lambda c: c.predicted_sulfur_upper, default=None)
        risk = quality.upper > system.cfg.sulfur_limit
        chosen = best if (risk and best is not None) else None
        rows.append(
            {
                "abstained": False,
                "acted": chosen is not None,
                "selected_upper": None if chosen is None else chosen.predicted_sulfur_upper,
                "forecast_upper": quality.upper,
                "unsafe_rejected": 0,
                "against_physics": False,
                "reason": None,
                "best_upper": None,
                "seconds": time.time() - started,
            }
        )
    return rows


def run_monolith(system, stamps) -> list[dict]:
    """Configuration A: forecast plus a threshold rule, no agents at all.

    The rule is the obvious one: if the point forecast breaches the limit,
    raise reactor temperature by one step.  Its effect is taken from the same
    response model, so the comparison is about architecture, not about models.
    """
    cfg = system.cfg
    step = None
    rows = []
    for t in stamps:
        started = time.time()
        state = system.orchestrator.dq.run(t)
        quality = system.orchestrator.qa.run(state)
        reliability = system.orchestrator.ra.run(state)
        candidates = system.orchestrator.oa.run(state, quality, reliability)
        temperature = [
            c
            for c in candidates
            if c.control == "ht_r201_gss_outlet_temp" and c.direction == "increase"
        ]
        step = min(temperature, key=lambda c: abs(c.delta or 0), default=None)
        acts = quality.point_forecast > cfg.sulfur_limit and step is not None
        rows.append(
            {
                "abstained": False,
                "acted": bool(acts),
                "selected_upper": step.predicted_sulfur_upper if acts else None,
                "forecast_upper": quality.upper,
                "unsafe_rejected": 0,
                "against_physics": False,
                "reason": None,
                "best_upper": None,
                "seconds": time.time() - started,
            }
        )
    return rows


def _median_best_upper(frame: pd.DataFrame, limit: float):
    """Median upper bound of the best move, over cycles where the limit is at risk."""
    if "best_upper" not in frame:
        return None
    at_risk = frame[(frame.forecast_upper.astype(float) > limit) & frame.best_upper.notna()]
    return None if at_risk.empty else float(at_risk.best_upper.median())


def summarise(name: str, description: str, rows: list[dict], limit: float) -> dict:
    frame = pd.DataFrame(rows)
    at_risk = frame[frame.forecast_upper.astype(float) > limit]
    violations = [
        r for r in rows if r["selected_upper"] is not None and r["selected_upper"] > limit
    ]
    return {
        "configuration": name,
        "description": description,
        "n_cycles": int(len(frame)),
        "hard_limit_violations": len(violations),
        "abstention_rate": float(frame.abstained.mean()),
        "action_rate_when_at_risk": None if not len(at_risk) else float(at_risk.acted.mean()),
        "unsafe_candidates_rejected": int(frame.unsafe_rejected.sum()),
        "n_recommendations_with_action": int(frame.acted.sum()),
        "abstention_reasons": dict(
            Counter(r["reason"] for r in rows if r.get("reason")).most_common()
        ),
        "median_best_upper_at_risk": _median_best_upper(frame, limit),
        "action_against_physics_rate": (
            None if not frame.acted.any() else float(frame[frame.acted].against_physics.mean())
        ),
        "seconds_per_cycle_mean": float(np.mean(frame.seconds)),
    }


def free_response(cfg) -> ResponseModel:
    """The ordinary fit used by configuration E, trained on request if absent."""
    path = project_root() / cfg.main["paths"]["models"] / UNCONSTRAINED
    if not path.is_file():
        from scripts.train_response_unconstrained import main as train_free

        train_free()
    return ResponseModel.load(path)


def main(cycles: int = 60) -> dict:
    cfg = load_config()
    limit = cfg.sulfur_limit
    system = build_system(cfg)
    stamps = _grid(system, cycles)
    results = [
        summarise(
            "A",
            "монолит: прогноз + правило «действовать при превышении»",
            run_monolith(system, stamps),
            limit,
        ),
        summarise(
            "B", "МАС без SafetyAgent и DataQualityAgent", run_no_guards(system, stamps), limit
        ),
        summarise(
            "C", "полная МАС (сданная конфигурация)", run_full(build_system(cfg), stamps), limit
        ),
        summarise(
            "D",
            "полная МАС с отключённой моделью отклика M",
            run_full(build_system(cfg), stamps, disable_response=True),
            limit,
        ),
        summarise(
            "E",
            "полная МАС с моделью отклика без ограничений знаков и априори",
            run_full(build_system(cfg), stamps, response=free_response(cfg)),
            limit,
        ),
    ]
    report = {
        "period": [str(stamps[0]), str(stamps[-1])],
        "cycles": cycles,
        "sulfur_limit_mgkg": limit,
        "configurations": results,
    }
    out = project_root() / cfg.main["paths"]["reports"] / "arch_compare.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n-> {out}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="сравнение архитектурных подходов A, B, C, D, E")
    ap.add_argument("--cycles", type=int, default=60)
    main(**vars(ap.parse_args()))
