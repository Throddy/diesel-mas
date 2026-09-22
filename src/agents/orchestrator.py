"""Deterministic orchestration over a message bus.

The orchestrator owns the working set of candidates.  Agents publish verdicts;
the orchestrator applies them, resolves conflicts by the declared policy and
decides whether to replan.  A safety veto can only narrow the feasible set,
never widen it, and no later round can overturn it.
"""

from __future__ import annotations

import json

import pandas as pd

from src.agents.explainer_agent import ExplainerAgent
from src.agents.optimization_agent import pareto_front, rank_feasible
from src.config import load_config
from src.mas.bus import MessageBus
from src.mas.messages import BROADCAST, Message, MessageType
from src.mas.policies import REPLAN_ROUNDS, resolve_conflicts
from src.schemas import _clean


class Orchestrator:
    name = "Orchestrator"

    def __init__(
        self, dq_agent, quality_agent, reliability_agent, optimization_agent, safety_agent, cfg=None
    ):
        self.cfg = cfg or load_config()
        self.dq, self.qa, self.ra = dq_agent, quality_agent, reliability_agent
        self.oa, self.sa = optimization_agent, safety_agent
        self.explainer = ExplainerAgent(self.cfg)

    def _apply_verdicts(self, candidates, safety, conflicts) -> None:
        """Write the agreed verdict onto the working set; only here.

        Candidates belong to the orchestrator.  Agents return judgements, and
        this is the single place where a judgement becomes a candidate's
        feasibility (K-23).
        """
        resolution = {c["action_id"]: c for c in conflicts}
        for cand in candidates:
            verdict = resolution[cand.action_id]
            cand.feasible = verdict["admissible"]
            reasons = list(cand.rejection_reasons)
            reasons += [r for r in verdict["reasons"] if r not in reasons]
            cand.rejection_reasons = reasons

    def _plan_round(self, state, quality, reliability, bus, parents, round_index):
        """One planning round: candidates, safety verdicts, conflict resolution."""
        search = REPLAN_ROUNDS[round_index]["search"] if round_index else None
        candidates = self.oa.run(
            state, quality, reliability, bus=bus, parents=parents, search=search
        )
        candidate_msg = bus.publish(
            Message.make(
                self.oa.name,
                self.sa.name,
                MessageType.CANDIDATE_SET,
                state.timestamp,
                {
                    "round": round_index,
                    "n": len(candidates),
                    "ids": [c.action_id for c in candidates],
                    "search": search or "по умолчанию",
                },
                parents,
            )
        )

        safety = self.sa.run(candidates, state, quality, reliability)
        safety_msg = bus.publish(
            Message.make(
                self.sa.name,
                self.name,
                MessageType.SAFETY_VERDICT,
                state.timestamp,
                {
                    "round": round_index,
                    "veto": safety["veto"],
                    "admissible": safety["admissible_ids"],
                    "verdicts": [v.to_dict() for v in safety["verdicts"]],
                },
                (candidate_msg.id,),
            )
        )

        conflicts = resolve_conflicts(candidates, safety["verdicts"])
        contested = [c for c in conflicts if c["contested"]]
        if contested:
            bus.publish(
                Message.make(
                    self.name,
                    BROADCAST,
                    MessageType.CONFLICT_RESOLVED,
                    state.timestamp,
                    {"round": round_index, "resolved": contested},
                    (safety_msg.id,),
                )
            )
        self._apply_verdicts(candidates, safety, conflicts)
        return candidates, safety, conflicts, safety_msg

    def decide(self, t, log: bool = True, recent_moves=None):
        t = pd.Timestamp(t)
        self._recent_moves = list(recent_moves or [])
        bus = MessageBus()

        state = self.dq.run(t)
        state_msg = bus.publish(
            Message.make(
                self.dq.name,
                BROADCAST,
                MessageType.STATE_READY,
                t,
                {
                    "mode": state.quality_sources["mode"],
                    "telemetry_timestamp": state.quality_sources["telemetry_timestamp"],
                },
            )
        )
        dq_msg = bus.publish(
            Message.make(
                self.dq.name,
                BROADCAST,
                MessageType.DATA_QUALITY_VERDICT,
                t,
                state.data_quality.to_dict(),
                (state_msg.id,),
            )
        )

        quality = self.qa.run(state)
        quality_msg = bus.publish(
            Message.make(
                self.qa.name,
                self.oa.name,
                MessageType.QUALITY_ESTIMATE,
                t,
                quality.to_dict(),
                (dq_msg.id,),
            )
        )

        reliability = self.ra.run(state)
        reliability_msg = bus.publish(
            Message.make(
                self.ra.name,
                self.oa.name,
                MessageType.RELIABILITY_VERDICT,
                t,
                reliability.to_dict(),
                (dq_msg.id,),
            )
        )

        parents = (quality_msg.id, reliability_msg.id)
        rounds = []
        max_rounds = int(self.cfg.main["optimization"].get("replan_max_rounds", len(REPLAN_ROUNDS)))
        for round_index in range(min(max_rounds, len(REPLAN_ROUNDS))):
            if round_index:
                bus.publish(
                    Message.make(
                        self.name,
                        self.oa.name,
                        MessageType.REPLAN_REQUEST,
                        t,
                        {
                            "round": round_index,
                            "reason": "после вердикта безопасности допустимых "
                            "вариантов не осталось",
                            "widening": REPLAN_ROUNDS[round_index]["description"],
                        },
                        parents,
                    )
                )
            candidates, safety, conflicts, safety_msg = self._plan_round(
                state, quality, reliability, bus, parents, round_index
            )
            ranked = rank_feasible(
                candidates, self.cfg.sulfur_limit, recent_moves=getattr(self, "_recent_moves", None)
            )
            rounds.append(
                {
                    "round": round_index,
                    "widening": REPLAN_ROUNDS[round_index]["description"],
                    "n_candidates": len(candidates),
                    "n_feasible": len(safety["admissible_ids"]),
                    "selected": ranked[0].action_id if ranked else None,
                }
            )
            parents = (safety_msg.id,)
            if ranked or safety["veto"]:
                break

        trace = self._trace(
            state, quality, reliability, candidates, safety, conflicts, ranked, rounds
        )
        trace["messages"] = bus.to_list()
        trace["message_ids"] = bus.index()
        rec = self.explainer.build(
            t, state, quality, reliability, candidates, ranked, safety, trace
        )
        bus.publish(
            Message.make(
                self.name,
                self.explainer.name,
                MessageType.DECISION,
                t,
                {"abstained": rec.abstained, "headline": rec.headline, "rounds": rounds},
                parents,
            )
        )
        bus.publish(
            Message.make(
                self.explainer.name,
                "operator",
                MessageType.EXPLANATION,
                t,
                {"headline": rec.headline, "reasons": rec.reasons},
                bus.latest_id(),
            )
        )
        trace["messages"] = bus.to_list()
        trace["message_ids"] = bus.index()

        if log:
            path = self.cfg.path("artifacts") / "decisions.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        _clean(
                            {
                                "input_state": state.to_dict(),
                                "recommendation": rec.to_dict(),
                                "journal": trace["messages"],
                            }
                        ),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
        return rec

    def _trace(
        self, state, quality, reliability, candidates, safety, conflicts, ranked, rounds
    ) -> dict:
        return {
            "DataQualityAgent": state.data_quality.to_dict(),
            "QualityAgent": quality.to_dict(),
            "ReliabilityAgent": reliability.to_dict(),
            "OptimizationAgent": {
                "n_candidates": len(candidates),
                "candidates": [c.to_dict() for c in candidates],
                "pareto_front": pareto_front(candidates, self.cfg.sulfur_limit),
                "response": self.oa.whatif.diagnostics,
            },
            "SafetyAgent": {
                "veto": safety["veto"],
                "feasible_ids": [c.action_id for c in candidates if c.feasible],
                "checks": safety["checks"],
            },
            "Orchestrator": {
                "ranked": [c.action_id for c in ranked],
                "rounds": rounds,
                "conflicts": [c for c in conflicts if c["contested"]],
            },
        }
