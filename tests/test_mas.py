"""Tests for the message layer: immutability, journal, replanning, conflicts.

These check the properties the architecture claims, not the numbers a model
happens to produce: a message cannot be altered, every decision can be rebuilt
from its journal, replanning fires only when the first round found nothing,
and a conflict is settled by the declared priority.
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import pytest

from src.config import load_config, project_root
from src.mas.bus import MessageBus
from src.mas.messages import BROADCAST, Message, MessageType, Verdict, canonical_json
from src.mas.policies import PRIORITY, REPLAN_ROUNDS, resolve_conflicts

CFG = load_config()


@pytest.fixture(scope="module")
def system():
    if not (project_root() / CFG.main["paths"]["interim"] / "telemetry.parquet").exists():
        pytest.skip("canonical layer not built")
    from src.pipeline import build_system

    return build_system(CFG)


def test_messages_cannot_be_mutated():
    message = Message.make("A", "B", MessageType.STATE_READY, "2026-01-01", {"x": 1})
    with pytest.raises(dataclasses.FrozenInstanceError):
        message.sender = "C"
    with pytest.raises(dataclasses.FrozenInstanceError):
        message.payload_json = "{}"


def test_message_id_is_content_addressed():
    """Same content gives the same id; any change gives a different one."""
    first = Message.make("A", "B", MessageType.QUALITY_ESTIMATE, "t", {"v": 1})
    same = Message.make("A", "B", MessageType.QUALITY_ESTIMATE, "t", {"v": 1})
    other = Message.make("A", "B", MessageType.QUALITY_ESTIMATE, "t", {"v": 2})
    assert first.id == same.id
    assert first.id != other.id


def test_payload_decoding_does_not_leak_into_the_message():
    """A caller may edit the decoded payload without touching the message."""
    message = Message.make("A", "B", MessageType.STATE_READY, "t", {"items": [1, 2]})
    decoded = message.payload
    decoded["items"].append(3)
    assert message.payload["items"] == [1, 2]


def test_non_finite_numbers_serialise_as_null():
    """A missing measurement must survive the journal round trip."""
    encoded = canonical_json({"a": float("nan"), "b": float("inf"), "c": 1.5})
    assert json.loads(encoded) == {"a": None, "b": None, "c": 1.5}


def test_bus_keeps_publication_order_and_addressing():
    bus = MessageBus()
    first = bus.publish(Message.make("DQ", BROADCAST, MessageType.STATE_READY, "t", {}))
    second = bus.publish(
        Message.make("QA", "Opt", MessageType.QUALITY_ESTIMATE, "t", {}, (first.id,))
    )
    assert [m.id for m in bus.journal] == [first.id, second.id]
    assert second in bus.inbox("Opt")
    assert first in bus.inbox("Opt")
    assert second not in bus.inbox("Safety")
    assert [m.sender for m in bus.trace(second.id)] == ["DQ", "QA"]


def test_every_decision_is_reconstructible_from_its_journal(system):
    """The journal alone must explain the decision it accompanies."""
    recommendation = system.decide(pd.Timestamp(CFG.main["demo"]["quality_risk"]), log=False)
    journal = recommendation.agent_trace["messages"]
    assert journal, "a cycle must leave a journal"

    types = [m["type"] for m in journal]
    for required in (
        MessageType.STATE_READY,
        MessageType.DATA_QUALITY_VERDICT,
        MessageType.QUALITY_ESTIMATE,
        MessageType.RELIABILITY_VERDICT,
        MessageType.CANDIDATE_SET,
        MessageType.WHAT_IF_REQUEST,
        MessageType.WHAT_IF_RESPONSE,
        MessageType.SAFETY_VERDICT,
        MessageType.DECISION,
        MessageType.EXPLANATION,
    ):
        assert required.value in types, f"{required.value} missing from the journal"

    asked = {m["timestamp"] for m in journal if m["type"] == MessageType.WHAT_IF_REQUEST.value}
    answered = {m["timestamp"] for m in journal if m["type"] == MessageType.WHAT_IF_RESPONSE.value}
    assert asked == answered

    decision = [m for m in journal if m["type"] == MessageType.DECISION.value][-1]
    payload = json.loads(decision["payload_json"])
    assert payload["abstained"] == recommendation.abstained
    assert payload["headline"] == recommendation.headline

    ids = {m["id"] for m in journal}
    for message in journal:
        for parent in message["parent_ids"]:
            assert parent in ids


def test_replanning_only_runs_when_the_first_round_found_nothing(system):
    """A cycle solved straight away must not widen the search."""
    recommendation = system.decide(pd.Timestamp(CFG.main["demo"]["quality_risk"]), log=False)
    rounds = recommendation.agent_trace["Orchestrator"]["rounds"]
    if rounds[0]["selected"] is not None:
        assert len(rounds) == 1
        assert not [
            m
            for m in recommendation.agent_trace["messages"]
            if m["type"] == MessageType.REPLAN_REQUEST.value
        ]


def test_replanning_widens_the_search_and_keeps_the_veto(system):
    """A cycle rescued by replanning shows more candidates in the later round."""
    rescued = system.decide(pd.Timestamp("2025-12-13 13:25:25"), log=False)
    rounds = rescued.agent_trace["Orchestrator"]["rounds"]
    if len(rounds) < 2:
        pytest.skip("this moment no longer needs replanning")
    assert rounds[0]["n_feasible"] == 0
    assert rounds[-1]["n_candidates"] > rounds[0]["n_candidates"]
    action = rescued.selected_action
    if action and not action.get("is_do_nothing"):
        assert action["predicted_sulfur_upper"] <= CFG.sulfur_limit


def test_replanning_is_bounded(system):
    """The orchestrator never runs more rounds than the policy declares."""
    for scenario in ("stable", "quality_risk", "bad_data"):
        recommendation = system.decide(pd.Timestamp(CFG.main["demo"][scenario]), log=False)
        assert len(recommendation.agent_trace["Orchestrator"]["rounds"]) <= len(REPLAN_ROUNDS)


def test_conflict_is_resolved_by_the_declared_priority():
    """The highest-priority objection wins and is named in the resolution."""
    candidate = type("C", (), {"action_id": "A1", "rejection_reasons": []})()
    verdicts = [
        Verdict("SafetyAgent", "A1", True, "safety"),
        Verdict("ReliabilityAgent", "A1", False, "reliability", "severity rises"),
    ]
    resolved = resolve_conflicts([candidate], verdicts)[0]
    assert resolved["admissible"] is False
    assert resolved["contested"] is True
    assert resolved["blocked_by"] == "ReliabilityAgent"
    assert resolved["passed_checks_of"] == ["SafetyAgent"]
    assert "снять запрет не может ни один агент" in resolved["why"]


def test_safety_objection_outranks_a_lower_dimension():
    candidate = type("C", (), {"action_id": "A1", "rejection_reasons": []})()
    verdicts = [
        Verdict("SafetyAgent", "A1", False, "safety", "limit breached"),
        Verdict("EnergyAgent", "A1", False, "energy", "expensive"),
    ]
    resolved = resolve_conflicts([candidate], verdicts)[0]
    assert resolved["blocked_by"] == "SafetyAgent"
    assert resolved["additional_objections"][0]["agent"] == "EnergyAgent"
    assert PRIORITY.index("safety") < PRIORITY.index("energy")


def test_uncontested_candidate_is_admissible():
    candidate = type("C", (), {"action_id": "A1", "rejection_reasons": []})()
    resolved = resolve_conflicts([candidate], [Verdict("SafetyAgent", "A1", True, "safety")])[0]
    assert resolved["admissible"] is True
    assert resolved["contested"] is False


def test_two_identical_cycles_produce_identical_journals(system):
    """Same input, same messages: ids are content-addressed, so they must match."""
    t = pd.Timestamp(CFG.main["demo"]["stable"])
    first = system.decide(t, log=False).agent_trace["messages"]
    second = system.decide(t, log=False).agent_trace["messages"]
    assert [m["id"] for m in first] == [m["id"] for m in second]


def test_clean_unwraps_numpy_before_checking_finiteness():
    """np.float64('nan') must become null, not a bare NaN token (schemas._clean)."""
    import numpy as np

    from src.schemas import _clean

    assert _clean(np.float64("nan")) is None
    assert _clean(np.float64("inf")) is None
    assert _clean(np.float32("-inf")) is None
    assert _clean(np.float64(1.5)) == 1.5
    assert _clean(np.int64(3)) == 3
    assert json.dumps(_clean({"a": np.float64("nan")}), allow_nan=False) == '{"a": null}'
