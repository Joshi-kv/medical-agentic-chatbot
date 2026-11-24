from collections.abc import AsyncIterator
from uuid import UUID, uuid4

from langgraph.graph.state import CompiledStateGraph

from agent.workflows.medical_agent.state import (
    MedicalAgentState,
    answer_result_from_state,
)
from services.answering import (
    AnswerCompleted,
    AnswerDelta,
    AnswerProgress,
    AnswerResult,
    AnswerStreamItem,
)
from services.conversation_context import ContextMessage


class MedicalAgent:
    def __init__(self, graph: CompiledStateGraph) -> None:
        self._graph = graph

    async def answer(
        self,
        query: str,
        tenant_id: str,
        document_id: UUID | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        conversation_history: list[ContextMessage] | None = None,
    ) -> AnswerResult:
        state = self._initial_state(
            query=query,
            tenant_id=tenant_id,
            document_id=document_id,
            conversation_id=conversation_id,
            run_id=run_id,
            conversation_history=conversation_history,
            streaming=False,
        )
        result = await self._graph.ainvoke(
            state,
            config=self._config(tenant_id=tenant_id, run_id=run_id),
        )
        return answer_result_from_state(result)

    async def stream_answer(
        self,
        query: str,
        tenant_id: str,
        document_id: UUID | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        conversation_history: list[ContextMessage] | None = None,
    ) -> AsyncIterator[AnswerStreamItem]:
        state = self._initial_state(
            query=query,
            tenant_id=tenant_id,
            document_id=document_id,
            conversation_id=conversation_id,
            run_id=run_id,
            conversation_history=conversation_history,
            streaming=True,
        )

        async for event in self._graph.astream(
            state,
            config=self._config(tenant_id=tenant_id, run_id=run_id),
            stream_mode="custom",
        ):
            if not isinstance(event, dict):
                continue

            event_type = event.get("type")
            if event_type == "progress":
                yield AnswerProgress(
                    stage=str(event["stage"]),
                    message=str(event["message"]),
                )
            elif event_type == "answer_delta":
                yield AnswerDelta(text=str(event["text"]))
            elif event_type == "answer_completed":
                result_state = event.get("result")
                if not isinstance(result_state, dict):
                    raise RuntimeError(
                        "Medical-agent graph returned an invalid result."
                    )
                yield AnswerCompleted(result=answer_result_from_state(result_state))

    def _initial_state(
        self,
        query: str,
        tenant_id: str,
        document_id: UUID | None,
        conversation_id: str | None,
        run_id: str | None,
        conversation_history: list[ContextMessage] | None,
        streaming: bool,
    ) -> MedicalAgentState:
        history = [
            {"role": message.role, "content": message.content}
            for message in conversation_history or []
        ]

        return {
            "query": query,
            "tenant_id": tenant_id,
            "document_id": str(document_id) if document_id else None,
            "conversation_id": conversation_id,
            "run_id": run_id,
            "conversation_history": history,
            "streaming": streaming,
        }

    def _config(
        self,
        tenant_id: str,
        run_id: str | None,
    ) -> dict[str, dict[str, str]]:
        graph_run_id = run_id or str(uuid4())
        return {
            "configurable": {
                "thread_id": f"medical-agent:{tenant_id}:{graph_run_id}"
            }
        }
