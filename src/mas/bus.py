"""Message bus: delivery and an ordered journal, nothing else.

The bus carries no decision logic.  It accepts messages, keeps them in the
order they were published, and lets a component read what was addressed to it.
Keeping the journal here means the decision log is a by-product of the run
rather than something assembled afterwards from memory.
"""

from __future__ import annotations

from typing import Iterable, Iterator

from src.mas.messages import BROADCAST, Message, MessageType


class MessageBus:
    """An append-only log of one decision cycle."""

    def __init__(self) -> None:
        self._journal: list[Message] = []

    def publish(self, message: Message) -> Message:
        """Append one message and return it, so callers can chain parents."""
        self._journal.append(message)
        return message

    def publish_all(self, messages: Iterable[Message]) -> tuple[Message, ...]:
        return tuple(self.publish(m) for m in messages)

    def __len__(self) -> int:
        return len(self._journal)

    def __iter__(self) -> Iterator[Message]:
        return iter(tuple(self._journal))

    @property
    def journal(self) -> tuple[Message, ...]:
        """The full cycle in publication order."""
        return tuple(self._journal)

    def inbox(self, recipient: str) -> tuple[Message, ...]:
        """Messages addressed to one agent, broadcasts included."""
        return tuple(m for m in self._journal if m.recipient in (recipient, BROADCAST))

    def by_type(self, type: MessageType) -> tuple[Message, ...]:
        return tuple(m for m in self._journal if m.type is type)

    def last(self, type: MessageType) -> Message | None:
        found = self.by_type(type)
        return found[-1] if found else None

    def latest_id(self) -> tuple[str, ...]:
        """Id of the last published message, as a parent tuple."""
        return (self._journal[-1].id,) if self._journal else ()

    def to_list(self) -> list[dict]:
        return [m.to_dict() for m in self._journal]

    def index(self) -> dict[str, str]:
        """Sender to id of its latest message, for citing numbers in the card."""
        out: dict[str, str] = {}
        for message in self._journal:
            out[message.sender] = message.id
        return out

    def trace(self, message_id: str) -> tuple[Message, ...]:
        """One message and everything it answers, ancestors first."""
        by_id = {m.id: m for m in self._journal}
        seen: list[Message] = []
        pending = [message_id]
        while pending:
            current = pending.pop()
            message = by_id.get(current)
            if message is None or message in seen:
                continue
            seen.append(message)
            pending.extend(message.parent_ids)
        return tuple(reversed(seen))
