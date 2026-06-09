"""Worker agents for the legal Supervisor-Workers system."""

from __future__ import annotations

import re
from typing import Any, Callable

from .protocol import WorkerMessage

try:
    from src.task9_retrieval_pipeline import retrieve
except Exception:  # pragma: no cover
    retrieve = None  # type: ignore

try:
    from src.task10_generation import (
        INSUFFICIENT_EVIDENCE_MESSAGE,
        format_context,
        generate_extractive_fallback,
        reorder_for_llm,
    )
except Exception:  # pragma: no cover
    INSUFFICIENT_EVIDENCE_MESSAGE = "Tôi không thể xác minh thông tin này từ nguồn hiện có"

    def reorder_for_llm(chunks: list[dict]) -> list[dict]:
        return chunks

    def format_context(chunks: list[dict]) -> str:
        return "\n\n".join(str(chunk.get("content", "")) for chunk in chunks)

    def generate_extractive_fallback(query: str, chunks: list[dict]) -> str:
        return INSUFFICIENT_EVIDENCE_MESSAGE


LEGAL_TERMS = {
    "điều",
    "khoản",
    "luật",
    "nghị",
    "định",
    "hình",
    "sự",
    "phạt",
    "tù",
    "ma",
    "tuý",
    "túy",
    "chất",
    "cấm",
    "cai",
    "nghiện",
    "bắt",
    "tàng",
    "trữ",
}


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", (text or "").lower(), flags=re.UNICODE))


def _safe_payload(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _source_name(source: dict, index: int) -> str:
    metadata = source.get("metadata") or {}
    return (
        metadata.get("source")
        or metadata.get("filename")
        or metadata.get("title")
        or metadata.get("path")
        or f"source_{index}"
    )


class BaseWorker:
    """Common worker metadata and error handling."""

    name: str
    role: str
    capabilities: tuple[str, ...]

    def __init__(self, name: str, role: str, capabilities: tuple[str, ...]) -> None:
        self.name = name
        self.role = role
        self.capabilities = capabilities

    def worker_card(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "capabilities": list(self.capabilities),
        }

    def unsupported(self, message: WorkerMessage) -> WorkerMessage:
        return message.reply(
            sender=self.name,
            intent="worker_error",
            payload={"ok": False, "error": f"Unsupported intent: {message.intent}"},
        )


class EvidenceWorker(BaseWorker):
    """Worker responsible for retrieval from the RAG pipeline."""

    def __init__(self) -> None:
        super().__init__(
            name="evidence_worker",
            role="retrieve evidence from legal/news corpus",
            capabilities=("retrieve_evidence", "hybrid_search", "pageindex_fallback"),
        )

    def handle(self, message: WorkerMessage) -> WorkerMessage:
        if message.intent != "retrieve_evidence":
            return self.unsupported(message)

        query = str(message.payload.get("query", "")).strip()
        top_k = int(message.payload.get("top_k", 5))

        def run() -> dict[str, Any]:
            if retrieve is None:
                return {"ok": False, "error": "retrieval pipeline is unavailable", "evidence": []}
            evidence = retrieve(query, top_k=top_k)
            return {"ok": True, "evidence": evidence, "count": len(evidence)}

        return message.reply(
            sender=self.name,
            intent="evidence_result",
            payload=_safe_payload(run),
        )


class LegalAnalysisWorker(BaseWorker):
    """Worker responsible for issue classification and evidence review."""

    def __init__(self) -> None:
        super().__init__(
            name="legal_analysis_worker",
            role="classify legal issue and inspect evidence coverage",
            capabilities=("analyze_issue", "classify_domain", "inspect_sources"),
        )

    def handle(self, message: WorkerMessage) -> WorkerMessage:
        if message.intent != "analyze_issue":
            return self.unsupported(message)

        query = str(message.payload.get("query", ""))
        evidence = list(message.payload.get("evidence") or [])
        terms = _tokenize(query)

        if {"phạt", "tù", "hình"} & terms:
            domain = "criminal_penalty"
        elif {"cai", "nghiện"} & terms:
            domain = "rehabilitation"
        elif {"nghệ", "sĩ", "báo", "tin"} & terms:
            domain = "legal_news"
        else:
            domain = "general_drug_law"

        payload = {
            "ok": True,
            "domain": domain,
            "legal_terms_found": sorted(terms & LEGAL_TERMS),
            "evidence_count": len(evidence),
            "evidence_sources": [_source_name(item, i) for i, item in enumerate(evidence, 1)],
            "coverage": {
                "has_evidence": bool(evidence),
                "has_legal_document": any(
                    (item.get("metadata") or {}).get("type") == "legal"
                    for item in evidence
                ),
                "has_news_document": any(
                    (item.get("metadata") or {}).get("type") == "news"
                    for item in evidence
                ),
            },
        }
        return message.reply(sender=self.name, intent="analysis_result", payload=payload)


class DraftingWorker(BaseWorker):
    """Worker responsible for cited answer drafting."""

    def __init__(self) -> None:
        super().__init__(
            name="drafting_worker",
            role="write grounded Vietnamese answer with citations",
            capabilities=("draft_answer", "format_context", "cite_sources"),
        )

    def handle(self, message: WorkerMessage) -> WorkerMessage:
        if message.intent != "draft_answer":
            return self.unsupported(message)

        query = str(message.payload.get("query", ""))
        evidence = list(message.payload.get("evidence") or [])
        analysis = dict(message.payload.get("analysis") or {})
        ordered_sources = reorder_for_llm(evidence)
        context = format_context(ordered_sources)
        answer = generate_extractive_fallback(query, ordered_sources)

        payload = {
            "ok": True,
            "answer": answer,
            "sources": ordered_sources,
            "context": context,
            "analysis": analysis,
            "style": "Vietnamese legal answer with [D#] citations",
        }
        return message.reply(sender=self.name, intent="draft_result", payload=payload)


class ComplianceWorker(BaseWorker):
    """Worker responsible for grounding and citation checks."""

    def __init__(self) -> None:
        super().__init__(
            name="compliance_worker",
            role="audit citations and evidence grounding",
            capabilities=("audit_answer", "check_citations", "check_grounding"),
        )

    def handle(self, message: WorkerMessage) -> WorkerMessage:
        if message.intent != "audit_answer":
            return self.unsupported(message)

        answer = str(message.payload.get("answer", ""))
        sources = list(message.payload.get("sources") or [])
        citations = re.findall(r"\[D\d+\]", answer)
        warnings: list[str] = []

        if not sources:
            warnings.append("No evidence sources returned.")
        if sources and not citations and not answer.startswith(INSUFFICIENT_EVIDENCE_MESSAGE):
            warnings.append("Answer has sources but no [D#] citation.")
        if answer.startswith(INSUFFICIENT_EVIDENCE_MESSAGE):
            warnings.append("Answer reports insufficient evidence.")

        payload = {
            "ok": True,
            "grounded": bool(sources) and not warnings,
            "citation_count": len(citations),
            "source_count": len(sources),
            "warnings": warnings,
        }
        return message.reply(sender=self.name, intent="audit_result", payload=payload)
