from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from services.answering import AnswerResult
from services.conversation import Message, StartedTurn
from services.evidence import Evidence
from services.turns import InvalidTurnInput, TurnError, TurnService


def make_started_turn() -> StartedTurn:
    message = Message(
        message_id="22222222-2222-4222-8222-222222222222",
        conversation_id="11111111-1111-4111-8111-111111111111",
        tenant_id="tenant-1",
        role="user",
        content="What is hypertension?",
        created_at=datetime.now(timezone.utc),
    )
    return StartedTurn(
        run_id="33333333-3333-4333-8333-333333333333",
        user_message=message,
    )


def make_service() -> tuple[TurnService, AsyncMock, AsyncMock, AsyncMock]:
    conversations = AsyncMock()
    store = AsyncMock()
    store.start_turn.return_value = make_started_turn()
    answering = AsyncMock()
    service = TurnService(
        conversations=conversations,
        store=store,
        answering=answering,
        model_name="nvidia/test-model",
    )
    return service, conversations, store, answering


@pytest.mark.anyio
async def test_run_turn_persists_answer_and_citations() -> None:
    service, conversations, store, answering = make_service()
    evidence = Evidence(
        citation_id="C1",
        chunk_id="chunk-1",
        text="Hypertension is persistently elevated blood pressure.",
        document_id="document-1",
        page_number=12,
        source="medical-book.pdf",
        score=0.92,
        title="Medical Book",
    )
    answering.answer.return_value = AnswerResult(
        answer="Hypertension is persistently elevated blood pressure [C1].",
        citations=[evidence],
        sufficient_evidence=True,
        safety_notice="Educational only.",
    )
    document_id = UUID("55555555-5555-4555-8555-555555555555")

    result = await service.run(
        conversation_id="11111111-1111-4111-8111-111111111111",
        tenant_id="tenant-1",
        query="  What is hypertension?  ",
        document_id=document_id,
    )

    conversations.get.assert_awaited_once_with(
        "11111111-1111-4111-8111-111111111111",
        "tenant-1",
    )
    answering.answer.assert_awaited_once_with(
        query="What is hypertension?",
        tenant_id="tenant-1",
        document_id=document_id,
    )
    command = store.complete_turn.await_args.args[0]
    assert command.tenant_id == "tenant-1"
    assert command.model == "nvidia/test-model"
    assert command.route == "curated_documents"
    assert command.assistant_message.role == "assistant"
    assert command.citations[0].citation_label == "C1"
    assert result.run_id == "33333333-3333-4333-8333-333333333333"
    assert result.message.content.endswith("[C1].")
    assert result.citations == command.citations


@pytest.mark.anyio
@pytest.mark.parametrize("query", ["", "   ", "a" * 2001])
async def test_run_turn_rejects_invalid_query(query: str) -> None:
    service, conversations, store, answering = make_service()

    with pytest.raises(InvalidTurnInput):
        await service.run("conversation-1", "tenant-1", query)

    conversations.get.assert_not_awaited()
    store.start_turn.assert_not_awaited()
    answering.answer.assert_not_awaited()


@pytest.mark.anyio
async def test_run_turn_marks_started_run_failed_when_answering_fails() -> None:
    service, _, store, answering = make_service()
    provider_error = RuntimeError("provider unavailable")
    answering.answer.side_effect = provider_error

    with pytest.raises(TurnError, match="Conversation turn failed") as exc_info:
        await service.run(
            "11111111-1111-4111-8111-111111111111",
            "tenant-1",
            "What is hypertension?",
        )

    assert exc_info.value.__cause__ is provider_error
    store.fail_turn.assert_awaited_once_with(
        run_id="33333333-3333-4333-8333-333333333333",
        tenant_id="tenant-1",
        error_code="turn_failed",
    )
    store.complete_turn.assert_not_awaited()


@pytest.mark.anyio
async def test_run_turn_wraps_start_persistence_failure() -> None:
    service, _, store, answering = make_service()
    store.start_turn.side_effect = RuntimeError("database unavailable")

    with pytest.raises(TurnError, match="Unable to start") as exc_info:
        await service.run(
            "11111111-1111-4111-8111-111111111111",
            "tenant-1",
            "What is hypertension?",
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    answering.answer.assert_not_awaited()
    store.fail_turn.assert_not_awaited()


@pytest.mark.anyio
async def test_run_turn_does_not_mask_original_error_when_fail_update_fails() -> None:
    service, _, store, answering = make_service()
    provider_error = RuntimeError("provider unavailable")
    answering.answer.side_effect = provider_error
    store.fail_turn.side_effect = RuntimeError("database unavailable")

    with pytest.raises(TurnError) as exc_info:
        await service.run(
            "11111111-1111-4111-8111-111111111111",
            "tenant-1",
            "What is hypertension?",
        )

    assert exc_info.value.__cause__ is provider_error
