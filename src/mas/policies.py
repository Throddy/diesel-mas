"""Declared policies: priority between agents and how the search widens.

Two things are written down here rather than buried in the orchestrator: the
order in which conflicting judgements are settled, and what each replanning
round is allowed to do.  Both are read by the orchestrator and reported in the
journal.

Dimensions are settled lexicographically: an earlier dimension wins outright,
so throughput or energy cannot buy back a safety or quality objection.
Replanning round 0 is the ordinary search and each later round widens it in one
declared way.  Widening only adds candidates, and every candidate is judged
again, so a safety veto cannot be overturned by a later round.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from src.mas.messages import Verdict

PRIORITY: tuple[str, ...] = ("data", "safety", "quality", "reliability", "production", "energy")

PRIORITY_RU = {
    "data": "достоверность данных",
    "safety": "безопасность и жёсткие ограничения",
    "quality": "качество продукта",
    "reliability": "надёжность оборудования",
    "production": "выпуск",
    "energy": "энергия",
}

REPLAN_ROUNDS: tuple[dict, ...] = (
    {"round": 0, "description": "обычный поиск", "search": None},
    {
        "round": 1,
        "description": "увеличенные шаги кандидатов",
        "search": {"extra_multipliers": (8, 12), "allow_pairs": True, "allow_load_cut": True},
    },
    {
        "round": 2,
        "description": "комбинации из двух управляемых параметров",
        "search": {"extra_multipliers": (8, 12, 16), "allow_pairs": True, "allow_load_cut": True},
    },
    {
        "round": 3,
        "description": "разрешено снижение нагрузки",
        "search": {
            "extra_multipliers": (8, 12, 16, 20),
            "allow_pairs": True,
            "allow_load_cut": True,
            "deep_load_cut": True,
        },
    },
)


def rank_of(dimension: str) -> int:
    """Position of a dimension in the priority order; unknown goes last."""
    return PRIORITY.index(dimension) if dimension in PRIORITY else len(PRIORITY)


def resolve_conflicts(candidates: Sequence, verdicts: Iterable[Verdict]) -> list[dict]:
    """Collect the verdicts about each candidate and record what blocked it.

    Any agent may block a candidate; no agent can unblock one.  An approval is
    therefore not a competing opinion but the absence of an objection, and the
    priority order decides only which objection is reported as the leading
    one, never whether the block holds.  The additional objections are listed
    beside it so the record shows every reason the candidate was rejected.
    """
    grouped: dict[str, list[Verdict]] = {}
    for verdict in verdicts:
        grouped.setdefault(verdict.action_id, []).append(verdict)

    resolved = []
    for cand in candidates:
        own = grouped.get(cand.action_id, [])
        own = own + [
            Verdict(
                agent="OptimizationAgent",
                action_id=cand.action_id,
                admissible=False,
                dimension="reliability",
                reason=reason,
            )
            for reason in cand.rejection_reasons
        ]
        objections = [v for v in own if not v.admissible]
        approvals = [v for v in own if v.admissible]
        contested = bool(objections) and bool(approvals)
        if objections:
            leading = min(objections, key=lambda v: (rank_of(v.dimension), v.agent))
            others = [v for v in objections if v is not leading]
            dimension_ru = PRIORITY_RU.get(leading.dimension, leading.dimension)
            resolved.append(
                {
                    "action_id": cand.action_id,
                    "admissible": False,
                    "blocked_by": leading.agent,
                    "dimension": leading.dimension,
                    "dimension_ru": dimension_ru,
                    "why": f"запрет по измерению {dimension_ru}; снять запрет "
                    "не может ни один агент",
                    "contested": contested,
                    "additional_objections": [
                        {
                            "agent": v.agent,
                            "dimension_ru": PRIORITY_RU.get(v.dimension, v.dimension),
                            "reason": v.reason,
                        }
                        for v in others
                    ],
                    "passed_checks_of": sorted({v.agent for v in approvals}),
                    "reasons": [v.reason for v in objections if v.reason],
                }
            )
        else:
            resolved.append(
                {
                    "action_id": cand.action_id,
                    "admissible": True,
                    "blocked_by": None,
                    "dimension": None,
                    "dimension_ru": None,
                    "why": "запретов нет",
                    "contested": False,
                    "additional_objections": [],
                    "passed_checks_of": sorted({v.agent for v in approvals}),
                    "reasons": [],
                }
            )
    return resolved
