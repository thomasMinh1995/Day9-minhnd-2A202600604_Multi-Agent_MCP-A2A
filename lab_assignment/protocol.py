"""Local A2A-style protocol for the Supervisor-Workers lab assignment."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


PROTOCOL_VERSION = "a2a-supervisor-workers-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class WorkerMessage:
    """Message passed between the supervisor and workers."""

    sender: str
    receiver: str
    intent: str
    payload: dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    in_reply_to: str | None = None
    protocol_version: str = PROTOCOL_VERSION
    created_at: str = field(default_factory=_utc_now)
    trace: list[str] = field(default_factory=list)

    def reply(
        self,
        *,
        sender: str,
        intent: str,
        payload: dict[str, Any],
        receiver: str | None = None,
    ) -> "WorkerMessage":
        return WorkerMessage(
            sender=sender,
            receiver=receiver or self.sender,
            intent=intent,
            payload=payload,
            conversation_id=self.conversation_id,
            in_reply_to=self.message_id,
            trace=[*self.trace, f"{self.sender}->{self.receiver}:{self.intent}"],
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Worker(Protocol):
    """Protocol implemented by every worker."""

    name: str
    role: str
    capabilities: tuple[str, ...]

    def handle(self, message: WorkerMessage) -> WorkerMessage:
        ...

    def worker_card(self) -> dict[str, Any]:
        ...


class WorkerBus:
    """In-memory bus used by the supervisor to call workers."""

    def __init__(self) -> None:
        self._workers: dict[str, Worker] = {}
        self.message_log: list[WorkerMessage] = []

    def register(self, worker: Worker) -> None:
        self._workers[worker.name] = worker

    def discover(self) -> list[dict[str, Any]]:
        return [worker.worker_card() for worker in self._workers.values()]

    def send(self, message: WorkerMessage) -> WorkerMessage:
        worker = self._workers.get(message.receiver)
        if worker is None:
            raise ValueError(f"Unknown worker: {message.receiver}")

        self.message_log.append(message)
        response = worker.handle(message)
        self.message_log.append(response)
        return response

    def request(
        self,
        *,
        sender: str,
        receiver: str,
        intent: str,
        payload: dict[str, Any],
        conversation_id: str,
        trace: list[str] | None = None,
        in_reply_to: str | None = None,
    ) -> WorkerMessage:
        return self.send(
            WorkerMessage(
                sender=sender,
                receiver=receiver,
                intent=intent,
                payload=payload,
                conversation_id=conversation_id,
                trace=trace or [],
                in_reply_to=in_reply_to,
            )
        )
