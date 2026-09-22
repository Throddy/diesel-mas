"""Hard-constraint evaluation, in lexicographic order."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.config import load_config
from src.models.horizons import decision_horizon
from src.schemas import CandidateAction, ConstraintCheck


def _spec(cfg, cid: str) -> dict:
    for h in cfg.constraints["hard"]:
        if h["id"] == cid:
            return h
    return {}


def check_candidate(
    cand: CandidateAction,
    *,
    cfg=None,
    bounds: Optional[Dict[str, tuple]] = None,
    ood_ok: bool = True,
    data_ok: bool = True,
    reliability_ok: bool = True,
    model_confident: bool = True,
) -> List[ConstraintCheck]:
    cfg = cfg or load_config()
    limit = cfg.sulfur_limit
    checks: List[ConstraintCheck] = []

    checks.append(
        ConstraintCheck(
            "HC-04",
            "data_validity",
            "PASS" if data_ok else "FAIL",
            (
                "critical inputs fresh, numeric and from a healthy analyser"
                if data_ok
                else "critical inputs are stale / unhealthy"
            ),
            True,
        )
    )

    lo, hi = (bounds or {}).get(cand.control, (None, None)) if cand.control else (None, None)
    if cand.is_do_nothing or cand.proposed_value is None or lo is None:
        checks.append(
            ConstraintCheck(
                "HC-03",
                "control_within_model_support",
                "NA" if cand.is_do_nothing else "FAIL",
                "no control move" if cand.is_do_nothing else "no support range available",
                False,
            )
        )
    else:
        ok = lo <= cand.proposed_value <= hi
        checks.append(
            ConstraintCheck(
                "HC-03",
                "control_within_model_support",
                "PASS" if ok else "FAIL",
                f"{cand.proposed_value:.4g} vs model support [{lo:.4g}, {hi:.4g}] "
                f"(ASSUMPTION - NOT AN INDUSTRIAL LIMIT)",
                False,
            )
        )

    for control, value in cand.moves.items():
        bounds_pair = (bounds or {}).get(control)
        ok = (
            bounds_pair is not None
            and np.isfinite(value)
            and value > 0
            and bounds_pair[0] <= value <= bounds_pair[1]
        )
        checks.append(
            ConstraintCheck(
                "HC-03",
                "component_support",
                "PASS" if ok else "FAIL",
                f"{control}: {value}; модельный диапазон {bounds_pair}",
                False,
            )
        )
    if not cand.is_do_nothing:
        ok = (
            cand.response_supported
            and 0 < cand.ramp_minutes <= cfg.main["optimization"]["planning_horizon_minutes"]
        )
        checks.append(
            ConstraintCheck(
                "HC-03",
                "response_and_rate",
                "PASS" if ok else "FAIL",
                "Применимость отклика и темп; допущения",
                False,
            )
        )
    spec = _spec(cfg, "HC-01")
    horizon = decision_horizon(cfg)
    scope = (
        f"горизонт {horizon['horizon_hours']:.0f} ч"
        if horizon["source"] == "horizon"
        else "текущий режим"
    )
    upper = cand.predicted_sulfur_upper
    if upper is None or not np.isfinite(upper):
        checks.append(
            ConstraintCheck(
                "HC-01",
                "product_sulfur_limit",
                "FAIL",
                f"нет достоверной верхней границы прогноза, {scope}",
                True,
            )
        )
    else:
        unit = spec.get("unit", "мг/кг")
        detail = f"верхняя граница {upper:.2f} {unit} против предела {limit:.1f}, {scope}"
        if cand.predicted_sulfur is not None:
            detail += f", точечный прогноз {cand.predicted_sulfur:.2f}"
        checks.append(
            ConstraintCheck(
                "HC-01", "product_sulfur_limit", "PASS" if upper <= limit else "FAIL", detail, True
            )
        )

    checks.append(
        ConstraintCheck(
            "HC-05",
            "state_in_distribution",
            "PASS" if ood_ok else "FAIL",
            (
                "process state inside the historical operating envelope"
                if ood_ok
                else "process state is out-of-distribution for the model"
            ),
            False,
        )
    )
    checks.append(
        ConstraintCheck(
            "HC-06",
            "reliability_gate",
            "PASS" if reliability_ok else "FAIL",
            (
                "operating severity acceptable"
                if reliability_ok
                else "operating severity in the critical class"
            ),
            False,
        )
    )
    if not model_confident:
        checks.append(
            ConstraintCheck(
                "HC-01",
                "forecast_confidence",
                "FAIL",
                "forecast uncertainty too high to certify the limit",
                True,
            )
        )
    return checks


def all_pass(checks: List[ConstraintCheck]) -> bool:
    return all(c.status != "FAIL" for c in checks)
