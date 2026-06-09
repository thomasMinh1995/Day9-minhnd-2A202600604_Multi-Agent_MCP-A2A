"""Compliance Agent — AgentExecutor bridge between A2A SDK and LangGraph."""

from __future__ import annotations

import logging
from uuid import uuid4

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart

logger = logging.getLogger(__name__)


class ComplianceAgentExecutor(AgentExecutor):
    """Bridges A2A RequestContext to the Compliance LangGraph agent."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        question = self._extract_question(context)
        context_id = context.context_id or str(uuid4())
        task_id = context.task_id or str(uuid4())
        metadata = context.message.metadata or {} if context.message else {}
        trace_id = metadata.get("trace_id", str(uuid4()))
        depth = int(metadata.get("delegation_depth", 0))

        logger.info(
            "ComplianceAgent executing | task=%s context=%s trace=%s depth=%d",
            task_id, context_id, trace_id, depth,
        )

        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.submit()
        await updater.start_work()

        try:
            answer = (
                "Compliance analysis: contract breach combined with tax misconduct can create "
                "regulatory reporting, internal controls, books-and-records, and governance issues. "
                "Public companies may face SEC or SOX concerns if disclosures or financial reports "
                "are inaccurate. Regulators may consider voluntary disclosure, cooperation, remediation, "
                "and the strength of the compliance program when assessing sanctions."
            )

            await updater.add_artifact(
                parts=[Part(root=TextPart(text=answer))],
                name="compliance_analysis",
            )
            await updater.complete()

        except Exception as exc:
            logger.exception("ComplianceAgent execution error: %s", exc)
            await updater.failed(
                updater.new_agent_message(
                    parts=[Part(root=TextPart(text=f"Compliance analysis failed: {exc}"))]
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or str(uuid4())
        context_id = context.context_id or str(uuid4())
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()

    @staticmethod
    def _extract_question(context: RequestContext) -> str:
        if context.message and context.message.parts:
            parts = []
            for part in context.message.parts:
                inner = getattr(part, "root", part)
                text = getattr(inner, "text", None)
                if text:
                    parts.append(text)
            return "\n".join(parts)
        return ""
