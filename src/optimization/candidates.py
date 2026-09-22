"""Candidate generation for the verified controls.

Scale of a candidate move is derived from the data, not invented: it is the
q95 of |x_t - x_{t-1}| on the TRAIN block (one 10-minute step), and it is
clipped to the model-support envelope.  This is still a modelling assumption
(A-06) and is documented as such.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import load_config, project_root
from src.schemas import CandidateAction

SHIFTED_SUFFIXES = ("|value", "|mean60", "|mean180", "|mean360", "|mean720", "|min720", "|max720")


def load_support() -> Dict[str, dict]:
    cfg = load_config()
    path = project_root() / cfg.main["paths"]["models"] / "controls_support.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass
class ControlSpec:
    canonical_name: str
    tags: List[str]
    description_ru: str
    unit: str
    composite: Optional[str]
    support: Dict[str, dict]

    def current(self, telemetry: Dict[str, Optional[float]]) -> Optional[float]:
        vals = [telemetry.get(t) for t in self.tags]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            return None
        return float(np.sum(vals)) if len(vals) > 1 else float(vals[0])

    def step(self) -> Optional[float]:
        steps = [self.support[t]["step_q95"] for t in self.tags if t in self.support]
        return float(np.sum(steps)) if steps else None

    def bounds(self) -> Tuple[Optional[float], Optional[float]]:
        lows = [self.support[t]["low"] for t in self.tags if t in self.support]
        highs = [self.support[t]["high"] for t in self.tags if t in self.support]
        if not lows:
            return None, None
        return float(np.sum(lows)), float(np.sum(highs))


def control_specs(cfg=None) -> List[ControlSpec]:
    cfg = cfg or load_config()
    support = load_support()
    out = []
    for c in cfg.controls["controls"]:
        sup = support.get(c["canonical_name"], {}).get("tags", {})
        out.append(
            ControlSpec(
                c["canonical_name"],
                list(c["tags"]),
                c["description_ru"],
                c.get("unit", "NOT CONFIRMED"),
                c.get("composite"),
                sup,
            )
        )
    return out


def generate_candidates(
    state_telemetry: Dict[str, Optional[float]], cfg=None, search=None
) -> List[CandidateAction]:
    """DO NOTHING is always present and is always the first candidate.

    ``search`` widens the space during replanning: extra step multipliers,
    permission to combine two controls, permission to cut throughput.  With
    no search options the set is exactly the default one, so a normal cycle
    is unaffected.
    """
    cfg = cfg or load_config()
    search = search or {}
    multipliers = list(cfg.main["optimization"]["step_multipliers"])
    for extra in search.get("extra_multipliers", ()):
        if extra not in multipliers:
            multipliers.append(extra)
    allow_pairs = bool(search.get("allow_pairs", True))
    allow_load_cut = bool(search.get("allow_load_cut", True))
    cands: List[CandidateAction] = [
        CandidateAction(
            action_id="A00_do_nothing",
            control=None,
            tags=[],
            current_value=None,
            proposed_value=None,
            delta=0.0,
            direction="hold",
            is_do_nothing=True,
            action_magnitude=0.0,
            notes=["baseline scenario: keep the current regime"],
        )
    ]
    i = 1
    for spec in control_specs(cfg):
        cur = spec.current(state_telemetry)
        step = spec.step()
        lo, hi = spec.bounds()
        if cur is None or step is None or not np.isfinite(step) or step <= 0:
            continue
        for multiplier in multipliers:
            for sign, direction in ((1.0, "increase"), (-1.0, "decrease")):
                proposed = cur + sign * step * multiplier
                cands.append(
                    CandidateAction(
                        action_id=f"A{i:03d}_{spec.canonical_name}_{direction}_{multiplier}",
                        control=spec.canonical_name,
                        tags=list(spec.tags),
                        current_value=cur,
                        proposed_value=float(proposed),
                        delta=float(proposed - cur),
                        direction=direction,
                        action_magnitude=(
                            abs(proposed - cur) / (hi - lo)
                            if hi and lo is not None and hi > lo
                            else 1.0
                        ),
                        moves={spec.canonical_name: float(proposed)},
                        moves_from={spec.canonical_name: float(cur)},
                        ramp_minutes=10 * multiplier,
                    )
                )
                i += 1
    from copy import deepcopy

    temperature = [
        c for c in cands if c.control == "ht_r201_gss_outlet_temp" and c.direction == "increase"
    ]
    feed = [c for c in cands if c.control == "ht_feed_flow_mass" and c.direction == "decrease"]
    if not allow_load_cut:
        feed = []
    for temp in (temperature if allow_pairs else []):
        for load in feed:
            pair = deepcopy(temp)
            pair.action_id = f"A{i:03d}_temperature_and_feed"
            pair.moves.update(load.moves)
            pair.moves_from.update(load.moves_from)
            pair.tags += load.tags
            pair.action_magnitude += load.action_magnitude
            pair.ramp_minutes = max(pair.ramp_minutes, load.ramp_minutes)
            cands.append(pair)
            i += 1
    return cands


def apply_candidate_to_features(
    features: pd.Series, cand: CandidateAction, spec_by_name: Dict[str, ControlSpec]
) -> pd.Series:
    """Counterfactual feature shift (assumption A-08).

    A held step change of size ``delta`` shifts the level-type features of the
    affected tags by ``delta`` and leaves the variability features untouched.
    For a composite control the change is distributed proportionally between
    its tags.  This is a model-based approximation inside historical support,
    NOT a causal simulation of the process.
    """
    if cand.is_do_nothing or cand.delta is None or cand.control is None:
        return features
    spec = spec_by_name.get(cand.control)
    if spec is None:
        return features
    out = features.copy()
    total = 0.0
    weights = {}
    for tag in spec.tags:
        base = features.get(f"{tag}|value", np.nan)
        weights[tag] = abs(float(base)) if np.isfinite(base) else 0.0
        total += weights[tag]
    for tag in spec.tags:
        share = (weights[tag] / total) if total > 0 else 1.0 / len(spec.tags)
        d = cand.delta * share
        for suffix in SHIFTED_SUFFIXES:
            key = f"{tag}{suffix}"
            if key in out.index and np.isfinite(out[key]):
                out[key] = float(out[key]) + d
    return out
