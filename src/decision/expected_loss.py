"""Decide by comparing expected losses instead of crossing a threshold.

The threshold rule answers a yes-or-no question: is the upper bound above the
limit?  It treats a 12 % chance of breaching and an 80 % chance the same way,
and it cannot express that a small correction now is cheaper than a large one
later.  The closed loop showed the consequence: the system drove sulfur to
4.21 mg/kg against a limit of 10, paying temperature for margin nobody asked
for.

Here a candidate is priced.  Holding the regime costs the chance of going
off-specification; acting costs energy, throughput, added severity and the
wear of moving equipment.  The action is taken when the loss it removes
exceeds what it costs, by a margin.

Every coefficient is dimensionless and lives in config.  The package contains
no prices, so this is a proxy for cost, not cost, and it is named that way
wherever it is shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class LossWeights:
    """Relative prices, all dimensionless (A-27)."""

    off_spec: float = 100.0
    energy: float = 1.0
    production_loss: float = 3.0
    severity: float = 1.0
    movement: float = 0.2
    margin: float = 0.05

    @classmethod
    def from_config(cls, cfg) -> "LossWeights":
        section = dict(cfg.main.get("expected_loss", {}))
        known = {f: section[f] for f in cls.__dataclass_fields__ if f in section}
        return cls(**known)


def intervention_cost(candidate, baseline, weights: LossWeights) -> float:
    """What a move costs, measured against holding the current regime.

    Only the excess counts: energy and severity the regime already spends are
    not charged to the action that leaves them unchanged.
    """
    if candidate.is_do_nothing:
        return 0.0
    energy = max(0.0, (candidate.energy_proxy or 0.0) - (baseline.energy_proxy or 0.0))
    throughput = max(
        0.0, 1.0 - (candidate.production_proxy or 0.0) / (baseline.production_proxy or 1.0)
    )
    severity = max(
        0.0, (candidate.reliability_severity or 0.0) - (baseline.reliability_severity or 0.0)
    )
    return float(
        weights.energy * energy
        + weights.production_loss * throughput
        + weights.severity * severity
        + weights.movement * candidate.action_magnitude
    )


def off_spec_loss(candidate, weights: LossWeights) -> Optional[float]:
    """Expected cost of the product going off-specification under this candidate."""
    risk = candidate.risk_exceed_limit
    if risk is None or not np.isfinite(risk):
        return None
    return float(weights.off_spec * risk)


def expected_loss(candidate, baseline, weights: LossWeights) -> Optional[float]:
    """Total expected cost of choosing this candidate."""
    quality = off_spec_loss(candidate, weights)
    if quality is None:
        return None
    return float(quality + intervention_cost(candidate, baseline, weights))


def evaluate(candidates, weights: LossWeights) -> dict:
    """Price every candidate and say which one the loss comparison prefers.

    The hard constraint is not part of the arithmetic: candidates whose upper
    bound breaches the limit were already rejected by SafetyAgent, and nothing
    here can bring them back.  What this decides is whether to act at all, and
    how far.
    """
    feasible = [c for c in candidates if c.feasible]
    baseline = next((c for c in candidates if c.is_do_nothing), None)
    if baseline is None or not feasible:
        return {"decision": "нет допустимых вариантов", "priced": []}

    priced = []
    for candidate in feasible:
        total = expected_loss(candidate, baseline, weights)
        if total is None:
            continue
        priced.append(
            {
                "action_id": candidate.action_id,
                "is_do_nothing": bool(candidate.is_do_nothing),
                "risk": float(candidate.risk_exceed_limit),
                "off_spec_loss": off_spec_loss(candidate, weights),
                "intervention_cost": intervention_cost(candidate, baseline, weights),
                "expected_loss": total,
            }
        )
    if not priced:
        return {"decision": "ожидаемые потери не считаются", "priced": []}

    hold = next((p for p in priced if p["is_do_nothing"]), None)
    moves = [p for p in priced if not p["is_do_nothing"]]
    best_move = min(moves, key=lambda p: p["expected_loss"]) if moves else None

    if hold is None:
        chosen, reason = best_move, "удержание режима недопустимо"
    elif best_move is None:
        chosen, reason = hold, "исполнимых воздействий нет"
    else:
        advantage = hold["expected_loss"] - best_move["expected_loss"]
        required = weights.margin * max(abs(hold["expected_loss"]), 1e-9)
        if advantage > required:
            chosen = best_move
            reason = (
                f"ожидаемая цена бездействия {hold['expected_loss']:.2f} выше цены "
                f"действия {best_move['expected_loss']:.2f} на {advantage:.2f}, "
                f"порог запаса {required:.2f}"
            )
        else:
            chosen = hold
            reason = (
                f"выигрыш от вмешательства {advantage:.2f} не превышает порог запаса "
                f"{required:.2f}: удержание режима дешевле"
            )

    return {
        "decision": chosen["action_id"],
        "acts": not chosen["is_do_nothing"],
        "reason": reason,
        "hold": hold,
        "best_move": best_move,
        "priced": sorted(priced, key=lambda p: p["expected_loss"]),
        "weights": vars(weights),
    }


@dataclass(frozen=True)
class RecentMove:
    """A control moved earlier, and how long ago."""

    control: str
    minutes_ago: float
    delta: float


def cooldown_block(
    candidate,
    recent: list[RecentMove],
    cooldown_minutes: float,
    effect_seen: bool = False,
    still_off_spec: bool = False,
) -> Optional[str]:
    """Refuse to stack another move on a control that has not answered yet.

    With a transport delay longer than the decision interval, a system that
    keeps acting is acting blind, and the closed loop showed the cost: off-spec
    time nearly doubled when the plant was slower than assumed.

    The hold is lifted in two cases.  If the effect has started to show there
    is no longer anything to wait for.  And if the product is still off
    specification, waiting is the more expensive mistake: the first move was
    evidently not enough, and the rule exists to stop over-tightening a regime
    that is already in range, not to stop a recovery.
    """
    if candidate.is_do_nothing or effect_seen or still_off_spec:
        return None
    for move in recent:
        if move.control in candidate.moves and move.minutes_ago < cooldown_minutes:
            return (
                f"{move.control}: предыдущее воздействие {move.minutes_ago:.0f} мин назад, "
                f"эффект ещё не проявился (нужно {cooldown_minutes:.0f} мин)"
            )
    return None
