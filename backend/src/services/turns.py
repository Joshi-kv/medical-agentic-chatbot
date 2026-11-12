import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from uuid import UUID, uuid4

from services.answering import (
    AnswerCompleted,
    AnswerDelta,
    AnswerProgress,
    AnswerProvider,
    AnswerResult,
)
from services.conversation import (
    CitationRecord,
    CompleteTurnCommand,
    ConversationService,
    ConversationStore,
    Message,
)
from services.human_review import HumanReviewCase, HumanReviewService

MAX_TURN_QUERY_LENGTH = 2000
logger = logging.getLogger(__name__)


class TurnError(Exception):
    pass


class InvalidTurnInput(TurnError):
    pass


@dataclass(frozen=True)
class TurnResult:
    run_id: str
    message: Message
    citations: list[CitationRecord]
    sufficient_evidence: bool
    safety_notice: str
    route: str = "curated_documents"
    review_case_id: str | None = None


@dataclass(frozen=True)
class TurnStreamEvent:
    event: str
    data: dict[str, object]


class TurnService:
    def __init__(
        self,
        conversations: ConversationService,
        store: ConversationStore,
        answering: AnswerProvider,
        model_name: str,
        human_review: HumanReviewService | None = None,
    ) -> None:
        self._conversations = conversations
        self._store = store
        self._answering = answering
        self._model_name = model_name
        self._human_review = human_review

    async def run(
        self,
        conversation_id: str,
        tenant_id: str,
        query: str,
        document_id: UUID | None = None,
    ) -> TurnResult:
        normalized_query = self._validate_query(query)

        await self._conversations.get(conversation_id, tenant_id)

        user_message = Message(
            message_id=str(uuid4()),
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            role="user",
            content=normalized_query,
            created_at=datetime.now(timezone.utc),
        )

        try:
            started_turn = await self._store.start_turn(user_message)
        except Exception as error:
            raise TurnError("Unable to start conversation turn.") from error

        started_at = monotonic()
        try:
            answer = await self._answering.answer(
                query=normalized_query,
                tenant_id=tenant_id,
                document_id=document_id,
            )
            review_case = await self._create_review_case(
                route=answer.route,
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                run_id=started_turn.run_id,
                query=normalized_query,
            )

            assistant_message = Message(
                message_id=str(uuid4()),
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                role="assistant",
                content=answer.answer,
                created_at=datetime.now(timezone.utc),
            )
            citations = [
                CitationRecord(
                    citation_id=str(uuid4()),
                    tenant_id=tenant_id,
                    assistant_message_id=assistant_message.message_id,
                    citation_label=item.citation_id,
                    document_id=item.document_id,
                    chunk_id=item.chunk_id,
                    page_number=item.page_number,
                    source=item.source,
                    title=item.title,
                )
                for item in answer.citations
            ]
            command = CompleteTurnCommand(
                run_id=started_turn.run_id,
                tenant_id=tenant_id,
                assistant_message=assistant_message,
                citations=citations,
                model=self._model_name,
                route=answer.route,
                latency_ms=int((monotonic() - started_at) * 1000),
            )
            await self._store.complete_turn(command)
        except Exception as error:
            await self._fail_quietly(started_turn.run_id, tenant_id)
            raise TurnError("Conversation turn failed.") from error

        return TurnResult(
            run_id=started_turn.run_id,
            message=assistant_message,
            citations=citations,
            sufficient_evidence=answer.sufficient_evidence,
            safety_notice=answer.safety_notice,
            route=answer.route,
            review_case_id=(review_case.case_id if review_case is not None else None),
        )

    async def stream(
        self,
        conversation_id: str,
        tenant_id: str,
        query: str,
        document_id: UUID | None = None,
    ) -> AsyncIterator[TurnStreamEvent]:
        normalized_query = self._validate_query(query)

        await self._conversations.get(conversation_id, tenant_id)

        user_message = Message(
            message_id=str(uuid4()),
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            role="user",
            content=normalized_query,
            created_at=datetime.now(timezone.utc),
        )

        try:
            started_turn = await self._store.start_turn(user_message)
        except Exception as error:
            raise TurnError("Unable to start conversation turn.") from error

        yield TurnStreamEvent(
            event="turn_started",
            data={"run_id": started_turn.run_id},
        )

        started_at = monotonic()
        final_answer: AnswerResult | None = None

        try:
            async for item in self._answering.stream_answer(
                query=normalized_query,
                tenant_id=tenant_id,
                document_id=document_id,
            ):
                if isinstance(item, AnswerProgress):
                    yield TurnStreamEvent(
                        event="agent_status",
                        data={
                            "stage": item.stage,
                            "message": item.message,
                        },
                    )
                elif isinstance(item, AnswerDelta):
                    yield TurnStreamEvent(
                        event="answer_delta",
                        data={"delta": item.text},
                    )
                elif isinstance(item, AnswerCompleted):
                    final_answer = item.result

            if final_answer is None:
                raise TurnError("Answer stream ended without a result.")

            review_case = await self._create_review_case(
                route=final_answer.route,
                tenant_id=tenant_id,
                conversation_id=conversation_id,
                run_id=started_turn.run_id,
                query=normalized_query,
            )

            assistant_message = Message(
                message_id=str(uuid4()),
                conversation_id=conversation_id,
                tenant_id=tenant_id,
                role="assistant",
                content=final_answer.answer,
                created_at=datetime.now(timezone.utc),
            )

            citations = [
                CitationRecord(
                    citation_id=str(uuid4()),
                    tenant_id=tenant_id,
                    assistant_message_id=assistant_message.message_id,
                    citation_label=citation.citation_id,
                    document_id=citation.document_id,
                    chunk_id=citation.chunk_id,
                    page_number=citation.page_number,
                    source=citation.source,
                    title=citation.title,
                )
                for citation in final_answer.citations
            ]

            command = CompleteTurnCommand(
                run_id=started_turn.run_id,
                tenant_id=tenant_id,
                assistant_message=assistant_message,
                citations=citations,
                model=self._model_name,
                route=final_answer.route,
                latency_ms=int((monotonic() - started_at) * 1000),
            )

            await self._store.complete_turn(command)

            yield TurnStreamEvent(
                event="citations",
                data={
                    "items": [
                        {
                            "citation_label": citation.citation_label,
                            "document_id": citation.document_id,
                            "chunk_id": citation.chunk_id,
                            "page_number": citation.page_number,
                            "source": citation.source,
                            "title": citation.title,
                        }
                        for citation in citations
                    ],
                },
            )

            yield TurnStreamEvent(
                event="turn_completed",
                data={
                    "run_id": started_turn.run_id,
                    "message_id": assistant_message.message_id,
                    "sufficient_evidence": final_answer.sufficient_evidence,
                    "safety_notice": final_answer.safety_notice,
                    "route": final_answer.route,
                    "review_case_id": (
                        review_case.case_id if review_case is not None else None
                    ),
                },
            )

        except asyncio.CancelledError:
            await self._fail_quietly(
                run_id=started_turn.run_id,
                tenant_id=tenant_id,
            )
            raise

        except Exception:
            logger.exception(
                "Conversation turn stream failed",
                extra={"run_id": started_turn.run_id},
            )
            await self._fail_quietly(
                run_id=started_turn.run_id,
                tenant_id=tenant_id,
            )

            yield TurnStreamEvent(
                event="error",
                data={
                    "code": "turn_failed",
                    "message": "The answer could not be completed.",
                    "discard_partial": True,
                },
            )

    async def _create_review_case(
        self,
        route: str,
        tenant_id: str,
        conversation_id: str,
        run_id: str,
        query: str,
    ) -> HumanReviewCase | None:
        if route != "human_review":
            return None

        if self._human_review is None:
            raise TurnError("Human-review service is unavailable.")

        return await self._human_review.request_review(
            tenant_id=tenant_id,
            conversation_id=conversation_id,
            run_id=run_id,
            query=query,
            review_type="clinical",
        )

    async def _fail_quietly(self, run_id: str, tenant_id: str) -> None:
        try:
            await self._store.fail_turn(
                run_id=run_id,
                tenant_id=tenant_id,
                error_code="turn_failed",
            )
        except Exception:
            return

    def _validate_query(self, query: str) -> str:
        normalized_query = query.strip()

        if not normalized_query:
            raise InvalidTurnInput("query is required.")

        if len(normalized_query) > MAX_TURN_QUERY_LENGTH:
            raise InvalidTurnInput(
                f"query must be at most {MAX_TURN_QUERY_LENGTH} characters."
            )

        return normalized_query
