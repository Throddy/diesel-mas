"""Typed, immutable messages exchanged by the agents.

Every handoff between agents is a message: it names its sender and recipient,
carries a canonical payload, and points at the messages it answers.  Messages
are frozen and content-addressed, so a decision journal can be replayed and
any number in a card can be traced to the message that produced it.

An agent returns new messages.  It never mutates a message, nor an object
another agent owns; the orchestrator is the only component that applies a
verdict to the working set of candidates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class MessageType(str, Enum):
    """The vocabulary of the decision cycle."""

    STATE_READY = "StateReady"
    DATA_QUALITY_VERDICT = "DataQualityVerdict"
    QUALITY_ESTIMATE = "QualityEstimate"
    RELIABILITY_VERDICT = "ReliabilityVerdict"
    CANDIDATE_SET = "CandidateSet"
    WHAT_IF_REQUEST = "WhatIfRequest"
    WHAT_IF_RESPONSE = "WhatIfResponse"
    SAFETY_VERDICT = "SafetyVerdict"
    REPLAN_REQUEST = "ReplanRequest"
    CONFLICT_RESOLVED = "ConflictResolved"
    DECISION = "Decision"
    EXPLANATION = "Explanation"


BROADCAST = "*"


def sanitize(value: Any) -> Any:
    """Make a payload JSON-safe without losing the fact that data are missing.

    Non-finite floats become null rather than the non-standard ``NaN`` token,
    so any parser can read the journal back.  A missing measurement is a real
    state of the plant and has to survive the round trip.
    """
    import math

    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json(payload: Any) -> str:
    """Stable serialisation: the same payload always yields the same string."""
    return json.dumps(
        sanitize(payload), ensure_ascii=False, sort_keys=True, default=str, allow_nan=False
    )


@dataclass(frozen=True)
class Message:
    """One immutable handoff between two agents."""

    id: str
    sender: str
    recipient: str
    type: MessageType
    timestamp: str
    parent_ids: tuple[str, ...]
    payload_json: str

    @classmethod
    def make(
        cls,
        sender: str,
        recipient: str,
        type: MessageType,
        timestamp: Any,
        payload: Mapping[str, Any],
        parent_ids: tuple[str, ...] = (),
    ) -> "Message":
        """Build a message; its id is the digest of everything it carries."""
        encoded = canonical_json(payload)
        digest = hashlib.sha256(
            "|".join(
                [sender, recipient, str(type.value), str(timestamp), ",".join(parent_ids), encoded]
            ).encode()
        ).hexdigest()[:16]
        return cls(
            id=digest,
            sender=sender,
            recipient=recipient,
            type=type,
            timestamp=str(timestamp),
            parent_ids=tuple(parent_ids),
            payload_json=encoded,
        )

    @property
    def payload(self) -> Any:
        """The decoded payload; decoding never mutates the message."""
        return json.loads(self.payload_json)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["type"] = self.type.value
        data["parent_ids"] = list(self.parent_ids)
        return data

    def describe(self) -> str:
        """One line for the journal view in the dashboard."""
        return f"{self.sender} -> {self.recipient}: {self.type.value} [{self.id}]"


@dataclass(frozen=True)
class Verdict:
    """One agent's judgement about one candidate.

    The verdict is data.  Whether a candidate ends up feasible is decided by
    the orchestrator applying the policy to all verdicts, not by the agent
    reaching into the candidate (K-23).
    """

    agent: str
    action_id: str
    admissible: bool
    dimension: str
    reason: str = ""
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "action_id": self.action_id,
            "admissible": self.admissible,
            "dimension": self.dimension,
            "reason": self.reason,
            "detail": dict(self.detail),
        }
