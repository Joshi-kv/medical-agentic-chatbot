import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies.conversations import (
    get_conversation_service,
    get_turn_service,
)
from api.dependencies.security import require_scope
from api.schemas.conversations import (
    CitationResponse,
    ConversationResponse,
    CreateConversationRequest,
    CreateTurnRequest,
    MessageResponse,
    TurnResponse,
)
from api.security.api_key import APIPrincipal
from services.answering import AnsweringError
from services.conversation import (
    Conversation,
    ConversationError,
    ConversationNotFound,
    ConversationService,
    InvalidConversationInput,
    Message,
)
from services.retrieval import RetrievalServiceError
from services.turns import InvalidTurnInput, TurnError, TurnService

router = APIRouter(prefix="/v1/conversations", tags=["conversations"])
logger = logging.getLogger(__name__)


def _format_sse(event: str, data: dict[str, object]) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    request: CreateConversationRequest,
    principal: Annotated[
        APIPrincipal,
        Depends(require_scope("conversations:write")),
    ],
    service: Annotated[
        ConversationService,
        Depends(get_conversation_service),
    ],
) -> ConversationResponse:
    try:
        conversation = await service.create(
            tenant_id=principal.tenant_id,
            actor_id=principal.key_id,
            title=request.title,
        )
    except InvalidConversationInput as error:
        raise _bad_request(str(error)) from error
    except ConversationError as error:
        raise _unavailable() from error
    return _conversation_response(conversation)


@router.post(
    "/{conversation_id}/turns/stream",
    response_class=StreamingResponse,
)
async def stream_turn(
    conversation_id: UUID,
    request: CreateTurnRequest,
    principal: Annotated[
        APIPrincipal,
        Depends(require_scope("conversations:write")),
    ],
    service: Annotated[
        TurnService,
        Depends(get_turn_service),
    ],
) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[str]:
        try:
            async for event in service.stream(
                conversation_id=str(conversation_id),
                tenant_id=principal.tenant_id,
                query=request.query,
                document_id=request.document_id,
            ):
                yield _format_sse(event=event.event, data=event.data)
        except InvalidTurnInput:
            yield _format_sse(
                event="error",
                data={
                    "code": "invalid_turn",
                    "message": "The submitted question is invalid.",
                    "discard_partial": True,
                },
            )
        except ConversationNotFound:
            yield _format_sse(
                event="error",
                data={
                    "code": "conversation_not_found",
                    "message": "Conversation not found.",
                    "discard_partial": True,
                },
            )
        except (ConversationError, TurnError):
            yield _format_sse(
                event="error",
                data={
                    "code": "stream_unavailable",
                    "message": "The answer stream could not be started.",
                    "discard_partial": True,
                },
            )
        except Exception:
            logger.exception("Unexpected conversation stream failure")
            yield _format_sse(
                event="error",
                data={
                    "code": "stream_unavailable",
                    "message": "The answer stream could not be started.",
                    "discard_partial": True,
                },
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: UUID,
    principal: Annotated[
        APIPrincipal,
        Depends(require_scope("conversations:read")),
    ],
    service: Annotated[
        ConversationService,
        Depends(get_conversation_service),
    ],
) -> ConversationResponse:
    try:
        conversation = await service.get(
            conversation_id=str(conversation_id),
            tenant_id=principal.tenant_id,
        )
    except ConversationNotFound as error:
        raise _not_found() from error
    except ConversationError as error:
        raise _unavailable() from error
    return _conversation_response(conversation)


@router.get(
    "/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
async def list_messages(
    conversation_id: UUID,
    principal: Annotated[
        APIPrincipal,
        Depends(require_scope("conversations:read")),
    ],
    service: Annotated[
        ConversationService,
        Depends(get_conversation_service),
    ],
) -> list[MessageResponse]:
    try:
        messages = await service.list_messages(
            conversation_id=str(conversation_id),
            tenant_id=principal.tenant_id,
        )
    except ConversationNotFound as error:
        raise _not_found() from error
    except ConversationError as error:
        raise _unavailable() from error
    return [_message_response(message) for message in messages]


@router.post(
    "/{conversation_id}/turns",
    response_model=TurnResponse,
)
async def create_turn(
    conversation_id: UUID,
    request: CreateTurnRequest,
    principal: Annotated[
        APIPrincipal,
        Depends(require_scope("conversations:write")),
    ],
    service: Annotated[TurnService, Depends(get_turn_service)],
) -> TurnResponse:
    try:
        result = await service.run(
            conversation_id=str(conversation_id),
            tenant_id=principal.tenant_id,
            query=request.query,
            document_id=request.document_id,
        )
    except (InvalidTurnInput, InvalidConversationInput) as error:
        raise _bad_request(str(error)) from error
    except ConversationNotFound as error:
        raise _not_found() from error
    except (
        AnsweringError,
        ConversationError,
        RetrievalServiceError,
        TurnError,
    ) as error:
        raise _unavailable() from error

    return TurnResponse(
        run_id=result.run_id,
        message_id=result.message.message_id,
        answer=result.message.content,
        route=result.route,
        review_case_id=result.review_case_id,
        citations=[
            CitationResponse(
                citation_label=item.citation_label,
                document_id=item.document_id,
                chunk_id=item.chunk_id,
                page_number=item.page_number,
                source=item.source,
                title=item.title,
            )
            for item in result.citations
        ],
        sufficient_evidence=result.sufficient_evidence,
        safety_notice=result.safety_notice,
    )


def _conversation_response(conversation: Conversation) -> ConversationResponse:
    return ConversationResponse(
        conversation_id=conversation.conversation_id,
        title=conversation.title,
        summary=conversation.summary,
        status=conversation.status,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def _message_response(message: Message) -> MessageResponse:
    return MessageResponse(
        message_id=message.message_id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
    )


def _bad_request(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail={"code": "invalid_conversation_request", "message": message},
    )


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"code": "conversation_not_found", "message": "Conversation not found."},
    )


def _unavailable() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "code": "conversation_unavailable",
            "message": "Conversation service is temporarily unavailable.",
        },
    )
