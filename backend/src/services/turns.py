from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic
from uuid import UUID, uuid4

from services.answering import AnsweringService
from services.conversation import (
    CitationRecord,
    CompleteTurnCommand,
    ConversationService,
    ConversationStore,
    Message,
)

MAX_TURN_QUERY_LENGTH = 2000


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


class TurnService:
    def __init__(
        self,
        conversations: ConversationService,
        store: ConversationStore,
        answering: AnsweringService,
        model_name: str,
    ) -> None:
        self._conversations = conversations
        self._store = store
        self._answering = answering
        self._model_name = model_name

    async def run(
        self,
        conversation_id: str,
        tenant_id: str,
        query: str,
        document_id: UUID | None = None,
    ) -> TurnResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise InvalidTurnInput("query is required.")
        if len(normalized_query) > MAX_TURN_QUERY_LENGTH:
            raise InvalidTurnInput(
                f"query must be at most {MAX_TURN_QUERY_LENGTH} characters."
            )

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
                route="curated_documents",
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
