"""Supervisor orchestrator for the legal worker agents."""

from __future__ import annotations

import uuid
from typing import Any

from .protocol import PROTOCOL_VERSION, WorkerBus
from .workers import ComplianceWorker, DraftingWorker, EvidenceWorker, LegalAnalysisWorker


class LegalSupervisorSystem:
    """Supervisor-Workers implementation of the legal multi-agent workflow."""

    supervisor_name = "legal_supervisor"

    def __init__(self) -> None:
        self.bus = WorkerBus()
        self.bus.register(EvidenceWorker())
        self.bus.register(LegalAnalysisWorker())
        self.bus.register(DraftingWorker())
        self.bus.register(ComplianceWorker())

    def discover_workers(self) -> list[dict[str, Any]]:
        return self.bus.discover()

    def build_plan(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Return the supervisor's explicit execution plan."""
        return [
            {
                "step": 1,
                "worker": "evidence_worker",
                "intent": "retrieve_evidence",
                "reason": "Find source passages before any legal answer is drafted.",
                "payload": {"query": query, "top_k": top_k},
            },
            {
                "step": 2,
                "worker": "legal_analysis_worker",
                "intent": "analyze_issue",
                "reason": "Classify the legal issue and inspect evidence coverage.",
                "payload_from": "evidence_result",
            },
            {
                "step": 3,
                "worker": "drafting_worker",
                "intent": "draft_answer",
                "reason": "Draft only from retrieved evidence and attach citations.",
                "payload_from": "analysis_result + evidence_result",
            },
            {
                "step": 4,
                "worker": "compliance_worker",
                "intent": "audit_answer",
                "reason": "Check citations and grounding before returning to user.",
                "payload_from": "draft_result",
            },
        ]

    def ask(self, query: str, top_k: int = 5) -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            raise ValueError("query must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        conversation_id = str(uuid.uuid4())
        plan = self.build_plan(query, top_k)

        evidence_response = self.bus.request(
            sender=self.supervisor_name,
            receiver="evidence_worker",
            intent="retrieve_evidence",
            payload={"query": query, "top_k": top_k},
            conversation_id=conversation_id,
            trace=["user->legal_supervisor:ask"],
        )
        evidence = list(evidence_response.payload.get("evidence") or [])

        analysis_response = self.bus.request(
            sender=self.supervisor_name,
            receiver="legal_analysis_worker",
            intent="analyze_issue",
            payload={"query": query, "evidence": evidence},
            conversation_id=conversation_id,
            trace=evidence_response.trace,
            in_reply_to=evidence_response.message_id,
        )
        analysis = dict(analysis_response.payload)

        draft_response = self.bus.request(
            sender=self.supervisor_name,
            receiver="drafting_worker",
            intent="draft_answer",
            payload={"query": query, "evidence": evidence, "analysis": analysis},
            conversation_id=conversation_id,
            trace=analysis_response.trace,
            in_reply_to=analysis_response.message_id,
        )
        draft = dict(draft_response.payload)

        audit_response = self.bus.request(
            sender=self.supervisor_name,
            receiver="compliance_worker",
            intent="audit_answer",
            payload={
                "query": query,
                "answer": draft.get("answer", ""),
                "sources": draft.get("sources", []),
            },
            conversation_id=conversation_id,
            trace=draft_response.trace,
            in_reply_to=draft_response.message_id,
        )

        return {
            "ok": True,
            "pattern": "Supervisor-Workers",
            "answer": draft.get("answer", ""),
            "sources": draft.get("sources", []),
            "analysis": analysis,
            "audit": audit_response.payload,
            "supervisor": {
                "name": self.supervisor_name,
                "protocol": PROTOCOL_VERSION,
                "conversation_id": conversation_id,
                "plan": plan,
                "workers": [card["name"] for card in self.discover_workers()],
                "message_count": len(self.bus.message_log),
                "trace": audit_response.trace,
            },
        }


def answer_legal_question(query: str, top_k: int = 5) -> dict[str, Any]:
    return LegalSupervisorSystem().ask(query, top_k=top_k)
