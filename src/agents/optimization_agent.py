"""OptimizationAgent: generates and scores model-based scenarios.

The predictions attached to a candidate are a *model-based counterfactual
approximation inside historical support*, not a proof of a causal effect.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.agents.quality_agent import QualityAgent
from src.config import load_config
from src.mas.messages import Message, MessageType
from src.optimization.candidates import control_specs, generate_candidates
from src.optimization.objectives import energy_proxy, production_proxy
from src.schemas import CandidateAction, ProcessState, QualityAssessment, ReliabilityAssessment

RESPONSE_INPUTS = ("ht:T5", "ht:F9", "ht:P3", "feed_sulfur_mgkg")


class OptimizationAgent:
    name = "OptimizationAgent"

    def __init__(
        self,
        quality_agent: QualityAgent,
        reliability_agent,
        train_medians: Dict[str, float],
        cfg=None,
        whatif_agent=None,
    ):
        self.cfg = cfg or load_config()
        self.qa = quality_agent
        self.reliability_agent = reliability_agent
        self.medians = train_medians
        self.whatif = whatif_agent
        self.specs = {s.canonical_name: s for s in control_specs(self.cfg)}

    def run(
        self,
        state: ProcessState,
        quality: QualityAssessment,
        reliability: ReliabilityAssessment,
        bus=None,
        parents: tuple = (),
        search=None,
    ) -> List[CandidateAction]:
        """Build the candidate set and price each move through what-if exchanges.

        Every effect estimate and every severity figure comes back as a
        message, so the journal shows what the optimizer asked and what it was
        told.  ``search`` widens the space during replanning; with no search
        options the candidate set is the default one.
        """
        before = self._baseline_state(state)
        tolerance = float(self.cfg.main["optimization"]["severity_increase_tolerance"])
        cands = generate_candidates(state.telemetry, self.cfg, search=search)
        for cand in cands:
            after = dict(before)
            for control, value in cand.moves.items():
                after[self.specs[control].tags[0]] = value
            effect = self._ask_what_if(cand, before, after, bus, parents)
            self._apply_effect(cand, quality, effect)
            self._price(cand, before, after)
            self._ask_severity(cand, after, reliability, tolerance, bus, parents)
            cand.notes.append(
                "Эффект — наблюдательная модель со знаками; "
                "верхняя граница учитывает блочный бутстреп."
            )
        return cands

    def _baseline_state(self, state: ProcessState) -> dict:
        """Current operating point plus the feed sulfur available as of now."""
        from src.features.quality_features import available_feature

        t = pd.Timestamp(state.timestamp)
        dq = self.cfg.main["data_quality"]
        feed = available_feature(
            [t],
            self.qa.lims_other["feed_sulfur"],
            "feed_sulfur_mgkg",
            dq["lims_delay_minutes"],
            dq["feed_lims_max_age_minutes"],
        ).iloc[0]
        before = dict(state.telemetry)
        before["feed_sulfur_mgkg"] = feed["feed_sulfur_mgkg"]
        return before

    def _ask_what_if(
        self, cand: CandidateAction, before: dict, after: dict, bus, parents: tuple
    ) -> dict:
        """Send a WhatIfRequest and record the answer."""
        request = {
            "action_id": cand.action_id,
            "is_do_nothing": cand.is_do_nothing,
            "moves": dict(cand.moves),
            "before": {k: before.get(k) for k in RESPONSE_INPUTS},
            "after": {k: after.get(k) for k in RESPONSE_INPUTS},
        }
        if bus is not None:
            sent = bus.publish(
                Message.make(
                    self.name,
                    self.whatif.name,
                    MessageType.WHAT_IF_REQUEST,
                    cand.action_id,
                    request,
                    parents,
                )
            )
            parents = (sent.id,)
        answer = self.whatif.handle(request)
        if bus is not None:
            bus.publish(
                Message.make(
                    self.whatif.name,
                    self.name,
                    MessageType.WHAT_IF_RESPONSE,
                    cand.action_id,
                    answer,
                    parents,
                )
            )
        if not answer["supported"]:
            cand.response_supported = False
            cand.notes.append(answer["note"])
        return answer

    def _apply_effect(
        self, cand: CandidateAction, quality: QualityAssessment, effect: dict
    ) -> None:
        """Turn a what-if answer into the candidate's forecast trio and risk."""
        forecast = self.whatif.sulfur_after(quality, effect["central"], effect["pessimistic"])
        cand.predicted_sulfur = forecast["predicted_sulfur"]
        cand.predicted_sulfur_lower = forecast["predicted_sulfur_lower"]
        cand.predicted_sulfur_upper = forecast["predicted_sulfur_upper"]
        cand.risk_exceed_limit = float(
            self.qa.model.calibration.exceedance_risk(
                np.array([np.log(cand.predicted_sulfur)]), self.cfg.sulfur_limit
            )[0]
        )

    def _price(self, cand: CandidateAction, before: dict, after: dict) -> None:
        """Production and energy proxies for the proposed operating point."""
        change = (after["ht:F9"] / before["ht:F9"] - 1) if before.get("ht:F9") else 0.0
        cand.production_proxy = production_proxy(after, change)
        cand.energy_proxy = energy_proxy(after, self.medians)

    def _ask_severity(
        self,
        cand: CandidateAction,
        after: dict,
        reliability: ReliabilityAssessment,
        tolerance: float,
        bus,
        parents: tuple,
    ) -> None:
        """Ask ReliabilityAgent what the proposed state would cost in severity."""
        if bus is not None:
            sent = bus.publish(
                Message.make(
                    self.name,
                    self.reliability_agent.name,
                    MessageType.WHAT_IF_REQUEST,
                    cand.action_id,
                    {"action_id": cand.action_id, "asks": "severity"},
                    parents,
                )
            )
            parents = (sent.id,)
        result = self.reliability_agent.assess_state(after)
        if bus is not None:
            bus.publish(
                Message.make(
                    self.reliability_agent.name,
                    self.name,
                    MessageType.WHAT_IF_RESPONSE,
                    cand.action_id,
                    {"action_id": cand.action_id, **result},
                    parents,
                )
            )
        cand.reliability_severity = float(result["severity_index"])
        cand.ood_score = float(result["ood_score"])
        worsens = cand.reliability_severity > (reliability.severity_index or 0.0) + tolerance
        new_ood = bool(result["ood_flag"]) and not reliability.ood_flag
        if worsens:
            cand.rejection_reasons.append(
                "CANDIDATE_RELIABILITY: вариант ухудшает тяжесть режима "
                f"({cand.reliability_severity:.3f} против текущей {reliability.severity_index:.3f})"
            )
        if new_ood:
            cand.rejection_reasons.append(
                "CANDIDATE_OOD: предлагаемое состояние выходит за область обучения модели "
                f"(оценка {cand.ood_score:.3f}), тогда как текущее состояние в ней остаётся"
            )


def rank_feasible(
    cands: List[CandidateAction], limit: float, recent_moves=None
) -> List[CandidateAction]:
    """Order the admissible actions, by threshold margin or by expected loss.

    Both rules run downstream of the hard constraints: a candidate breaching
    the limit was already removed and neither rule can bring it back.  What
    differs is when acting is worth it at all - a margin to the limit, or a
    comparison of what holding costs against what moving costs.
    """
    feas = [c for c in cands if c.feasible]
    if not feas:
        return []
    cfg = load_config()
    feas = _apply_cooldown(cands, feas, cfg, recent_moves)
    if not feas:
        return []
    if cfg.main.get("expected_loss", {}).get("enabled"):
        return _rank_by_expected_loss(cands, feas, cfg)
    w = cfg.main["optimization"]["ranking_weights"]
    base = next(c for c in cands if c.is_do_nothing)
    for c in feas:
        energy = max(0.0, (c.energy_proxy or 0.0) - (base.energy_proxy or 0.0))
        loss = max(0.0, 1 - (c.production_proxy or 0.0) / (base.production_proxy or 1.0))
        severity = max(0.0, (c.reliability_severity or 0.0) - (base.reliability_severity or 0.0))
        c.score = -(
            w["energy"] * energy
            + w["production_loss"] * loss
            + w["severity"] * severity
            + w["movement"] * c.action_magnitude
        )
    if (
        base.feasible
        and base.predicted_sulfur_upper <= limit - cfg.main["optimization"]["hold_upper_margin"]
    ):
        return [base] + sorted(
            [c for c in feas if c is not base], key=lambda c: (-c.score, c.action_id)
        )
    return sorted(feas, key=lambda c: (-c.score, c.action_magnitude, c.action_id))


def _apply_cooldown(cands, feasible, cfg, recent_moves) -> List[CandidateAction]:
    """Drop moves that stack on a control which has not answered yet.

    "Still off specification" is a fact about the product, not a risk about
    it, so it reads the point forecast rather than the upper bound: using the
    bound would lift the cooldown in every cycle that needs action at all,
    which is the same as not having one.
    """
    from src.decision.expected_loss import cooldown_block

    recent = list(recent_moves or [])
    cooldown = float(cfg.main.get("expected_loss", {}).get("cooldown_minutes", 0.0))
    if not cooldown or not recent:
        return feasible
    hold = next((c for c in cands if c.is_do_nothing), None)
    still_off_spec = bool(
        hold is not None
        and hold.predicted_sulfur is not None
        and hold.predicted_sulfur > cfg.sulfur_limit
    )
    for candidate in feasible:
        blocked = cooldown_block(candidate, recent, cooldown, still_off_spec=still_off_spec)
        if blocked:
            candidate.feasible = False
            candidate.rejection_reasons = list(candidate.rejection_reasons) + [
                "COOLDOWN: " + blocked
            ]
    return [c for c in feasible if c.feasible]


def _rank_by_expected_loss(cands, feasible, cfg) -> List[CandidateAction]:
    """Order by expected loss, holding the regime unless a move earns its cost."""
    from src.decision.expected_loss import LossWeights, evaluate

    weights = LossWeights.from_config(cfg)
    verdict = evaluate(cands, weights)
    by_id = {c.action_id: c for c in feasible}
    for row in verdict["priced"]:
        candidate = by_id.get(row["action_id"])
        if candidate is not None:
            candidate.score = -row["expected_loss"]
            candidate.notes.append(
                f"ожидаемая цена {row['expected_loss']:.2f} = риск {row['off_spec_loss']:.2f} "
                f"+ вмешательство {row['intervention_cost']:.2f} (безразмерный прокси)"
            )
    chosen = by_id.get(verdict.get("decision"))
    ordered = sorted(feasible, key=lambda c: (-(c.score or -1e9), c.action_magnitude, c.action_id))
    if chosen is None:
        return ordered
    return [chosen] + [c for c in ordered if c is not chosen]


def pareto_front(cands: List[CandidateAction], limit: float) -> List[str]:
    """Non-dominated set over (quality margin ↑, production ↑, energy ↓, severity ↓)."""
    feas = [c for c in cands if c.feasible]
    pts = []
    for c in feas:
        pts.append(
            (
                limit
                - (c.predicted_sulfur_upper if c.predicted_sulfur_upper is not None else limit),
                c.production_proxy or 0.0,
                -(c.energy_proxy or 0.0),
                -(c.reliability_severity or 0.0),
            )
        )
    front = []
    for i, p in enumerate(pts):
        dominated = any(
            all(q[k] >= p[k] for k in range(4)) and any(q[k] > p[k] for k in range(4))
            for j, q in enumerate(pts)
            if j != i
        )
        if not dominated:
            front.append(feas[i].action_id)
    return front
